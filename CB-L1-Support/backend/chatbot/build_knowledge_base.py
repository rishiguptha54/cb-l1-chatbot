"""Stage 1-4: build the chatbot knowledge base from existing Jira data.

Loads ``raw_defects.json`` + ``issue_comments.json`` + ``llm_analysis.json``,
normalizes heterogeneous Jira/MCP fields, merges comments, masks sensitive
data, and derives the enriched columns the retriever and answer generator rely
on. Outputs ``defect_knowledge_base.json`` and ``.csv`` plus a helper
artifact (``synonym_dictionary.json``).

Run with:  python -m chatbot.build_knowledge_base
"""

from __future__ import annotations

import csv
import re
import sys
from typing import Any

import config
from chatbot import utils
from chatbot.schemas import DefectRecord


# ── Custom-field logical names (resolved to ids via raw_defects["custom_fields"]) ──
_CF_LOGICAL = {
    "environment": "customfield_10167",
    "product_module": "customfield_11907",
    "team": "customfield_12801",
    "scrum_team": "customfield_10001",
    "root_cause": "customfield_10580",
    "reported_by": "customfield_12051",
}

# ── Error-signature detection ──
_ERROR_PATTERNS = [
    r"\b\d{3}\s*(?:bad request|error|internal server error|not found|unauthorized|forbidden|timeout)\b",
    r"\b(?:http\s*)?[45]\d{2}\b",
    r"\b[A-Z][A-Za-z]*(?:Exception|Error|Failure|Timeout)\b",
    r"\bnull\s*pointer\b", r"\bnullpointer\b",
    r"\b(?:publish|sync|mapping|validation|login|deployment|api|find\s*api|job|export)\s*(?:failed|failure|error|missing|rejected)\b",
    r"\b(?:failed|failure|rejected|missing|mismatch|duplicate|not\s+(?:populating|loading|reflecting|syncing|available|associated))\b",
    r"\btimed?\s*out\b", r"\btimeout\b",
]
_ERROR_RE = re.compile("|".join(_ERROR_PATTERNS), re.IGNORECASE)

# ── Fix-pattern keyword cues (ordered; first strong hit wins ties via scoring) ──
_FIX_PATTERN_CUES: dict[str, list[str]] = {
    "Data Sync-Integration": ["sync", "integration", "find api", "findapi", "publish", "eom",
                               "upstream", "downstream", "payload", "mapping", "remap", "api call"],
    "Config-Environment": ["config", "configuration", "feature flag", "ff ", "environment",
                            "stage", "prod", "setting", "permission flag", "toggle", "variation"],
    "Data Cleanup": ["data mismatch", "duplicate", "bad record", "correct data", "data correction",
                     "cleanup", "data issue", "incorrect data", "wrong data", "stale data"],
    "Null Handling": ["null", "empty", "missing value", "nullpointer", "null pointer", "blank",
                      "not set", "undefined"],
    "Access-Authorization": ["permission", "access", "role", "token", "auth", "authorization",
                             "unauthorized", "forbidden", "ad ", "active directory"],
    "Performance": ["slow", "performance", "timeout", "timed out", "delay", "latency", "load time",
                    "optimi", "throughput"],
    "Job-Workflow-Pipeline": ["job", "scheduler", "pipeline", "workflow", "rerun", "batch",
                              "trigger", "cron", "task failed"],
    "UI-Frontend": ["ui", "dashboard", "widget", "screen", "page", "display", "render", "frontend",
                    "button", "grid", "label not", "not displaying"],
    "Deployment/Version": ["deploy", "deployment", "release", "version", "build", "rollback", "patch"],
    "External Dependency": ["third party", "third-party", "external", "vendor", "nanoprecise",
                            "skyspark", "sfdc", "salesforce"],
    "Logic Correction": ["logic", "calculation", "incorrect result", "algorithm", "rule", "condition",
                         "wrong value", "computation"],
    "Workaround-Manual": ["workaround", "manual", "manually", "temporary fix", "temp fix"],
}

# ── Defect-type keyword cues ──
_DEFECT_TYPE_CUES: dict[str, list[str]] = {
    "data issue": ["data mismatch", "duplicate", "data not", "incorrect data", "missing data",
                   "wrong value", "stale"],
    "configuration issue": ["config", "feature flag", "setting", "mapping", "not configured"],
    "environment issue": ["environment", "prod only", "only in prod", "stage", "deployment"],
    "integration issue": ["sync", "integration", "api", "find api", "publish", "payload", "external"],
    "performance issue": ["slow", "performance", "timeout", "delay", "latency"],
    "access issue": ["permission", "access", "role", "token", "auth", "unauthorized"],
    "functional bug": ["error", "exception", "failed", "incorrect", "not working", "broken", "bug"],
}


def _build_cf_resolver(raw: dict) -> dict[str, str]:
    """Map logical custom-field names -> actual field ids using the file's own
    ``custom_fields`` block, falling back to known defaults."""
    resolver = dict(_CF_LOGICAL)
    cf_block = raw.get("custom_fields", {})
    if isinstance(cf_block, dict):
        for logical, meta in cf_block.items():
            if isinstance(meta, dict) and meta.get("id"):
                resolver[logical] = meta["id"]
    return resolver


def _comment_insight(insights: dict, key: str) -> dict[str, Any]:
    ci = insights.get(key) if isinstance(insights, dict) else None
    return ci if isinstance(ci, dict) else {}


def _llm_fix_pattern_for(key: str, llm_fix_patterns: dict[str, str]) -> str:
    return llm_fix_patterns.get(key, "")


def _detect_error_signature(text: str) -> str:
    if not text:
        return ""
    hits: list[str] = []
    seen = set()
    for m in _ERROR_RE.finditer(text):
        frag = m.group(0).strip()
        low = frag.lower()
        if low not in seen:
            seen.add(low)
            hits.append(frag)
        if len(hits) >= 6:
            break
    return "; ".join(hits)


def _score_category(text: str, cues: dict[str, list[str]]) -> str:
    text_l = (text or "").lower()
    best, best_score = "", 0
    for category, words in cues.items():
        score = sum(text_l.count(w) for w in words)
        if score > best_score:
            best, best_score = category, score
    return best


def _classify_fix_pattern(text: str, llm_hint: str) -> str:
    cat = _score_category(text, _FIX_PATTERN_CUES)
    if cat:
        return cat
    # Map an llm_analysis pattern string onto our taxonomy as a fallback hint.
    hint = (llm_hint or "").lower()
    if "sync" in hint or "model" in hint or "integration" in hint:
        return "Data Sync-Integration"
    if "config" in hint or "environment" in hint:
        return "Config-Environment"
    if "ui" in hint or "frontend" in hint:
        return "UI-Frontend"
    if "data" in hint:
        return "Data Cleanup"
    return "Unknown"


def _classify_defect_type(text: str) -> str:
    cat = _score_category(text, _DEFECT_TYPE_CUES)
    return cat or "unknown"


def _join_nonempty(parts: list[str], sep: str = " ") -> str:
    return sep.join(p.strip() for p in parts if p and p.strip())


def _compute_quality(rec: DefectRecord) -> float:
    """Heuristic completeness/usefulness score in [0, 1]."""
    score = 0.0
    if rec.summary:
        score += 0.15
    if len(rec.problem_text) > 40:
        score += 0.15
    if rec.comments:
        score += 0.15
    if rec.root_cause_extracted:
        score += 0.18
    if rec.fix_applied_extracted:
        score += 0.18
    if rec.components:
        score += 0.05
    if rec.error_signature:
        score += 0.06
    if rec.is_fixed:
        score += 0.08
    if rec.is_cancelled and not rec.root_cause_extracted and not rec.fix_applied_extracted:
        score -= 0.20  # cancelled with no explanation = low value
    return max(0.0, min(1.0, round(score, 3)))


def _normalize_defect(
    issue: dict,
    cf: dict[str, str],
    comments_map: dict,
    insights: dict,
    llm_fix_patterns: dict[str, str],
    mask: bool,
) -> DefectRecord:
    key = issue.get("key") or utils.value_name(utils.field_get(issue, "key"))
    rec = DefectRecord(issue_key=key)

    rec.summary = utils.clean_text(utils.value_name(utils.field_get(issue, "summary")), mask)
    rec.description = utils.clean_text(
        utils.value_name(utils.field_get(issue, "description")), mask
    )
    rec.status = utils.value_name(utils.field_get(issue, "status"))
    rec.resolution = utils.value_name(utils.field_get(issue, "resolution"))
    rec.priority = utils.value_name(utils.field_get(issue, "priority"))
    rec.issue_type = utils.value_name(utils.field_get(issue, "issuetype", "issue_type"), "Defect")
    rec.components = utils.value_list(utils.field_get(issue, "components"))
    rec.labels = utils.value_list(utils.field_get(issue, "labels"))
    rec.environment = utils.value_name(
        utils.field_get(issue, cf.get("environment", ""), "environment")
    )
    rec.product = utils.value_name(utils.field_get(issue, cf.get("product_module", "")))
    rec.team = utils.value_name(
        utils.field_get(issue, cf.get("team", ""), cf.get("scrum_team", ""))
    )
    rec.project = utils.project_from_key(key)
    rec.reporter = utils.value_name(utils.field_get(issue, "reporter"))
    rec.assignee = utils.value_name(utils.field_get(issue, "assignee"), "Unassigned")
    rec.created = utils.value_name(utils.field_get(issue, "created"))[:10]
    rec.updated = utils.value_name(utils.field_get(issue, "updated"))[:10]
    rec.resolved = utils.value_name(
        utils.field_get(issue, "resolutiondate", "resolution_date", "resolved")
    )[:10]
    rec.root_cause_raw = utils.clean_text(
        utils.value_name(utils.field_get(issue, cf.get("root_cause", ""))), mask
    )
    rec.resolution_description_raw = utils.clean_text(
        utils.value_name(utils.field_get(issue, "resolution_description", "fix_description")), mask
    )
    rec.steps_to_reproduce = utils.clean_text(
        utils.value_name(utils.field_get(issue, "steps_to_reproduce", "customfield_steps")), mask
    )
    rec.linked_issues = _extract_linked(issue)

    # ── Merge comments ──
    raw_comments = comments_map.get(key, []) if isinstance(comments_map, dict) else []
    norm_comments: list[dict[str, Any]] = []
    for c in raw_comments or []:
        if not isinstance(c, dict):
            continue
        body = utils.clean_text(c.get("body", ""), mask)
        if not body:
            continue
        norm_comments.append({
            "author": utils.value_name(c.get("author"), "Unknown"),
            "created": str(c.get("created", ""))[:10],
            "body": body,
        })
    rec.comments = norm_comments
    comments_blob = "\n".join(c["body"] for c in norm_comments)

    # ── Pull pre-extracted insights from llm_analysis (preferred evidence) ──
    ci = _comment_insight(insights, key)
    ci_root = utils.clean_text(ci.get("root_cause") or "", mask)
    ci_fix = utils.clean_text(ci.get("fix_applied") or "", mask)
    ci_finding = utils.clean_text(ci.get("key_finding") or "", mask)
    ci_workaround = utils.clean_text(ci.get("workaround") or "", mask)

    rec.status  # noqa: keep ordering readable
    rec.is_fixed = utils.classify_fixed(rec.status, rec.resolution)
    rec.is_cancelled = utils.classify_cancelled(rec.status, rec.resolution)

    # ── Enriched text columns ──
    rec.problem_text = utils.truncate(_join_nonempty([
        rec.summary, rec.description, rec.steps_to_reproduce,
        f"Components: {', '.join(rec.components)}" if rec.components else "",
        f"Environment: {rec.environment}" if rec.environment else "",
        f"Labels: {', '.join(rec.labels)}" if rec.labels else "",
        ci_finding,
    ], sep=". "), 1200)
    rec.symptom_text = utils.truncate(_join_nonempty([rec.summary, ci_finding], sep=". "), 320)
    rec.error_signature = _detect_error_signature(
        _join_nonempty([rec.summary, rec.description, comments_blob])
    )

    # Root cause priority: explicit field -> comment insight -> resolution desc -> comments
    rec.root_cause_extracted = utils.truncate(
        rec.root_cause_raw or ci_root or rec.resolution_description_raw
        or _first_sentence_with(comments_blob, ("root cause", "because", "due to", "caused by")),
        400,
    )
    # Fix priority: resolution desc -> comment insight -> comments
    rec.fix_applied_extracted = utils.truncate(
        ci_fix or rec.resolution_description_raw
        or _first_sentence_with(comments_blob, ("fixed", "resolved", "corrected", "updated",
                                                "added", "enabled", "rerun", "remap", "deployed")),
        400,
    )
    rec.workaround_extracted = utils.truncate(
        ci_workaround
        or _first_sentence_with(comments_blob, ("workaround", "temporar", "manually", "for now")),
        300,
    )
    rec.key_findings = utils.truncate(ci_finding, 400)
    rec.failure_area = _join_nonempty(
        [", ".join(rec.components), rec.product, rec.team, rec.environment], sep=" | "
    )

    classify_blob = _join_nonempty([
        rec.summary, rec.problem_text, rec.error_signature,
        rec.root_cause_extracted, rec.fix_applied_extracted, comments_blob,
    ])
    rec.fix_pattern = _classify_fix_pattern(classify_blob, _llm_fix_pattern_for(key, llm_fix_patterns))
    rec.defect_type = _classify_defect_type(classify_blob)

    # ── Combined search + solution text ──
    rec.solution_text = utils.truncate(_join_nonempty([
        rec.root_cause_extracted, rec.fix_applied_extracted, rec.workaround_extracted,
        rec.resolution_description_raw,
    ], sep=". "), 900)
    rec.search_text = utils.truncate(_join_nonempty([
        rec.summary, rec.problem_text, rec.symptom_text, rec.error_signature,
        rec.root_cause_extracted, rec.fix_applied_extracted, rec.workaround_extracted,
        rec.key_findings, ", ".join(rec.components), ", ".join(rec.labels),
        rec.environment, rec.product, rec.team,
    ], sep=". "), 2000)

    rec.quality_score = _compute_quality(rec)
    return rec


def _extract_linked(issue: dict) -> list[str]:
    links = utils.field_get(issue, "issuelinks", "linked_issues", default=[])
    out: list[str] = []
    if isinstance(links, list):
        for link in links:
            if isinstance(link, str):
                out.extend(utils.extract_jira_keys(link))
            elif isinstance(link, dict):
                for side in ("inwardIssue", "outwardIssue", "inward_issue", "outward_issue"):
                    node = link.get(side)
                    if isinstance(node, dict) and node.get("key"):
                        out.append(node["key"])
    return list(dict.fromkeys(out))


def _first_sentence_with(text: str, cues: tuple[str, ...]) -> str:
    """Return the first sentence containing any cue keyword (evidence snippet)."""
    if not text:
        return ""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        low = sentence.lower()
        if any(cue in low for cue in cues) and len(sentence.strip()) > 12:
            return sentence.strip()
    return ""


def build_knowledge_base() -> list[dict[str, Any]]:
    """Build and persist the enriched defect knowledge base."""
    raw = utils.load_json(config.RAW_DEFECTS_PATH, default={}) or {}
    issues = raw.get("issues", []) if isinstance(raw, dict) else (raw or [])
    comments_map = utils.load_json(config.ISSUE_COMMENTS_PATH, default={}) or {}
    llm = utils.load_json(config.LLM_ANALYSIS_PATH, default={}) or {}
    insights = llm.get("comment_insights", {}) if isinstance(llm, dict) else {}

    # Flatten llm_analysis fix_patterns -> {key: pattern} hint map.
    llm_fix_patterns: dict[str, str] = {}
    for fp in (llm.get("fix_patterns", []) if isinstance(llm, dict) else []):
        pattern = fp.get("pattern", "")
        for k in fp.get("keys", []):
            llm_fix_patterns.setdefault(k, pattern)

    cf = _build_cf_resolver(raw)
    mask = config.MASK_SENSITIVE_DATA

    records: list[DefectRecord] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        key = issue.get("key")
        if not key:
            continue
        records.append(
            _normalize_defect(issue, cf, comments_map, insights, llm_fix_patterns, mask)
        )

    # Exclude cancelled (won't-fix / rejected) defects from the retrieval KB and
    # embeddings so "how to fix" answers only surface real defects.
    total_records = len(records)
    if config.CHATBOT_EXCLUDE_CANCELLED:
        records = [r for r in records if not r.is_cancelled]
        print(f"[kb] excluded {total_records - len(records)} cancelled defects from retrieval KB")

    kb = [r.to_dict() for r in records]
    utils.save_json(config.DEFECT_KB_PATH, kb)
    _write_csv(kb, config.DEFECT_KB_CSV_PATH)
    _write_synonyms()

    fixed = sum(1 for r in records if r.is_fixed)
    with_rc = sum(1 for r in records if r.root_cause_extracted)
    with_fix = sum(1 for r in records if r.fix_applied_extracted)
    print(f"[kb] {len(kb)} defects | fixed={fixed} | root_cause={with_rc} | fix={with_fix}")
    print(f"[kb] wrote {config.DEFECT_KB_PATH}")
    print(f"[kb] wrote {config.DEFECT_KB_CSV_PATH}")
    return kb


def _write_csv(kb: list[dict], path: str) -> None:
    if not kb:
        return
    cols = [
        "issue_key", "summary", "status", "resolution", "priority", "project",
        "components", "labels", "environment", "product", "team",
        "root_cause_extracted", "fix_applied_extracted", "workaround_extracted",
        "key_findings", "error_signature", "fix_pattern", "defect_type",
        "failure_area", "quality_score", "is_fixed", "is_cancelled",
        "created", "resolved",
    ]
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for rec in kb:
            row = dict(rec)
            row["components"] = "; ".join(rec.get("components", []))
            row["labels"] = "; ".join(rec.get("labels", []))
            writer.writerow(row)


def _write_synonyms() -> None:
    """Domain synonym dictionary used to expand keyword matching/intent."""
    synonyms = {
        "eom": ["end of month", "eom publish", "publish"],
        "ff": ["feature flag", "flag"],
        "find api": ["findapi", "find-api"],
        "rc": ["root cause"],
        "prod": ["production"],
        "config": ["configuration", "setting"],
        "ui": ["dashboard", "widget", "screen", "frontend"],
        "sync": ["synchronization", "integration"],
        "ad": ["active directory"],
        "sfdc": ["salesforce"],
    }
    utils.save_json(config.SYNONYM_DICT_PATH, synonyms)


if __name__ == "__main__":
    try:
        build_knowledge_base()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"[kb] ERROR: {exc}", file=sys.stderr)
        raise
