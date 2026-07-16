"""Diagnostic answer generation: LLM synthesis with a deterministic fallback.

When ``USE_LLM`` is on and Azure OpenAI is configured, the retrieved evidence is
synthesized by the model under the grounding system prompt. Otherwise (or on any
LLM error) a template-based answer is built directly from the retrieved metadata
so the assistant always returns a structured, evidence-cited response.
"""

from __future__ import annotations

import re
from typing import Any

import config
from chatbot import prompts
from chatbot.retriever import RetrievedDefect

# Fix-step templates keyed by fix_pattern (used by the deterministic fallback).
_FIX_STEP_TEMPLATES: dict[str, list[str]] = {
    "Config-Environment": [
        "Verify the relevant configuration/feature-flag values for the affected customer or site.",
        "Compare configuration between the working and failing environments (stage vs prod).",
        "Update the incorrect configuration / enable the correct flag.",
        "Rerun the affected job or republish, then validate logs.",
    ],
    "Data Cleanup": [
        "Identify the bad or duplicate records driving the symptom.",
        "Correct or remove the affected data.",
        "Rerun the dependent process so downstream data refreshes.",
        "Add validation to prevent the bad data from recurring.",
    ],
    "Null Handling": [
        "Reproduce the case that produces the null/empty value.",
        "Inspect which field is null and why it is not populated upstream.",
        "Add a guard / default value and handle the null path safely.",
        "Add a regression test for the null scenario.",
    ],
    "Data Sync-Integration": [
        "Check the upstream/downstream sync status and the integration (e.g. Find API) health.",
        "Retry the failed message/job and confirm the payload mapping is correct.",
        "Re-map or re-publish the affected entity so data propagates.",
        "Validate the data now appears end-to-end.",
    ],
    "Access-Authorization": [
        "Verify the role / permission / token for the affected user or service.",
        "Check for duplicate or stale identity records (e.g. in AD).",
        "Update access or correct the identity mapping.",
        "Re-test the previously failing action.",
    ],
    "Performance": [
        "Review logs and timings for the slow operation.",
        "Identify the slow query/job/batch step.",
        "Optimize the query/batch or add appropriate retries/limits.",
        "Re-measure to confirm the latency is acceptable.",
    ],
    "Job-Workflow-Pipeline": [
        "Check the scheduler/job status and its dependencies.",
        "Rerun the failed step once blockers are cleared.",
        "Validate the job output and downstream artifacts.",
        "Add monitoring/alerting for the failure point.",
    ],
    "UI-Frontend": [
        "Reproduce the UI symptom and check the browser/network/API responses.",
        "Confirm the backing data/API returns the expected values.",
        "Fix the rendering/state-binding defect in the component.",
        "Verify the UI updates correctly after the change.",
    ],
    "Deployment/Version": [
        "Confirm the deployed version on the affected environment.",
        "Check the deployment/release notes for the relevant fix.",
        "Deploy the corrected version or roll back if needed.",
        "Validate the fix post-deployment.",
    ],
    "External Dependency": [
        "Confirm whether the failure originates in an external/third-party system.",
        "Engage the dependency owner with the captured evidence.",
        "Apply the agreed fix or workaround on the integration boundary.",
        "Validate end-to-end once the dependency is healthy.",
    ],
    "Logic Correction": [
        "Reproduce the incorrect result with a concrete example.",
        "Trace the rule/calculation producing the wrong value.",
        "Correct the logic and cover edge cases.",
        "Add a test asserting the correct result.",
    ],
    "Workaround-Manual": [
        "Apply the documented manual workaround to unblock the user.",
        "Capture the manual steps and affected scope.",
        "Plan a permanent fix to remove the manual effort.",
        "Track recurrence until the permanent fix ships.",
    ],
    "Unknown": [
        "Triage using the most similar historical defects below.",
        "Reproduce the issue and capture logs/error signatures.",
        "Compare with the cited fixed defects and apply the closest fix.",
        "Validate and, if unresolved, escalate with the gathered evidence.",
    ],
}

# Most-likely root cause phrasing per fix_pattern (used when no explicit
# comment-derived root cause exists, so the assistant still commits to a cause).
_PATTERN_ROOT_CAUSE: dict[str, str] = {
    "Config-Environment": "a configuration or feature-flag difference between the affected environment/data center and the working ones (e.g. a missing or incorrect setting).",
    "Data Cleanup": "bad, duplicate, or stale records driving the incorrect behavior.",
    "Null Handling": "a missing/null value that is not populated upstream and is not safely guarded.",
    "Data Sync-Integration": "a data synchronization or integration gap so the expected data did not propagate end-to-end.",
    "Access-Authorization": "an incorrect role/permission/token or a stale identity mapping.",
    "Performance": "an unoptimized query/job/batch step causing slow or timing-sensitive behavior.",
    "Job-Workflow-Pipeline": "a failed or stuck scheduled job/workflow step or an unmet dependency.",
    "UI-Frontend": "a rendering or state-binding defect, or the backing API returning unexpected values.",
    "Deployment/Version": "a version/deployment mismatch where the fix is missing on the affected environment.",
    "External Dependency": "a fault originating in an external/third-party system the feature depends on.",
    "Logic Correction": "an incorrect rule/calculation producing the wrong result for certain cases.",
    "Workaround-Manual": "a gap that currently requires a manual step, pointing to an underlying process/data issue.",
    "Unknown": "a configuration, data, or integration issue specific to the affected environment, based on the closest historical defects below.",
}



def generate_diagnostic_answer(
    question: str,
    current_defect: dict[str, Any] | None,
    similar: list[RetrievedDefect],
    history: list[dict[str, str]] | None = None,
    sections: list[str] | None = None,
) -> str:
    """Return a markdown diagnostic answer (LLM if enabled, else fallback).

    ``sections`` (optional) restricts the narrative to a subset — any of
    ``"root_cause"`` / ``"resolve"``. When set, the output is filtered to only
    those sections as a safety net (in case the model emits extra headers).
    """
    similar_dicts = [s.to_dict() for s in similar]

    if config.USE_LLM:
        try:
            md = _llm_answer(question, current_defect, similar_dicts, history, sections)
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[answer] LLM failed ({exc}); using deterministic fallback.")
            md = _fallback_answer(question, current_defect, similar)
    else:
        md = _fallback_answer(question, current_defect, similar)

    if sections:
        md = _keep_sections(md, _wanted_headers(sections, bool(current_defect)))
    return md


# Section header titles per selectable option (the Similar Fixed Defects table is
# rendered separately by the caller, so it is never produced here).
_SECTION_HEADERS: dict[str, list[str]] = {
    "root_cause": ["Current Defect", "Root Causes"],
    "resolve": ["How Similar Defects Were Fixed Earlier", "Suggested Fix Steps"],
}


def _wanted_headers(sections: list[str], has_current: bool) -> list[str]:
    wanted: list[str] = []
    for sec in sections:
        for h in _SECTION_HEADERS.get(sec, []):
            if h == "Current Defect" and not has_current:
                continue
            wanted.append(h)
    return wanted


_HEADER_RE = re.compile(r"^#{1,3}\s+(.*)$")


def _keep_sections(md: str, wanted: list[str]) -> str:
    """Keep only the markdown ``##`` sections whose title matches ``wanted``.

    Robust to the model emitting extra sections: blocks are split on heading
    lines and only requested ones are retained, preserving order.
    """
    if not wanted:
        return ""
    wanted_l = [w.lower() for w in wanted]
    blocks: list[tuple[str | None, list[str]]] = []
    title: str | None = None
    buf: list[str] = []
    for line in md.splitlines():
        m = _HEADER_RE.match(line.strip())
        if m:
            if buf:
                blocks.append((title, buf))
            title = m.group(1).strip()
            buf = [line]
        else:
            buf.append(line)
    if buf:
        blocks.append((title, buf))

    out: list[str] = []
    for t, blk in blocks:
        if t is None:
            continue
        tl = t.lower()
        if any(w in tl for w in wanted_l):
            out.append("\n".join(blk).rstrip())
    return "\n\n".join(out).strip()


def render_similar_table(similar: list[RetrievedDefect]) -> str:
    """Public renderer for the Similar Fixed Defects markdown table."""
    fixed = [s for s in similar if s.is_fixed]
    primary = fixed or [s for s in similar if not s.is_cancelled] or similar
    return _render_similar_table(primary)


# ─────────────────────────────────────────────────────────────────────────────
#  LLM path
# ─────────────────────────────────────────────────────────────────────────────
def _llm_answer(
    question: str, current_defect: dict | None, similar: list[dict],
    history: list[dict[str, str]] | None = None,
    sections: list[str] | None = None,
) -> str:
    from llm_provider import chat

    user_prompt = prompts.build_user_prompt(question, current_defect, similar, history, sections)
    return chat(
        [
            {"role": "system", "content": prompts.DIAGNOSTIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stage="complex",
        temperature=0.2,
        max_tokens=1800,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Streaming path
# ─────────────────────────────────────────────────────────────────────────────
def stream_diagnostic_answer(
    question: str,
    current_defect: dict[str, Any] | None,
    similar: list[RetrievedDefect],
    history: list[dict[str, str]] | None = None,
):
    """Yield the diagnostic answer incrementally as text chunks.

    Uses Azure OpenAI streaming when enabled; otherwise streams the deterministic
    fallback in word-sized chunks so the UI behaves identically either way.
    """
    similar_dicts = [s.to_dict() for s in similar]

    if config.USE_LLM:
        try:
            yield from _stream_llm_answer(question, current_defect, similar_dicts, history)
            return
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[answer] LLM stream failed ({exc}); using deterministic fallback.")

    # Deterministic fallback streamed word-by-word.
    full = _fallback_answer(question, current_defect, similar)
    yield from _chunk_text(full)


def _stream_llm_answer(question: str, current_defect: dict | None, similar: list[dict],
                       history: list[dict[str, str]] | None = None):
    from llm_provider import chat

    user_prompt = prompts.build_user_prompt(question, current_defect, similar, history)
    yield from chat(
        [
            {"role": "system", "content": prompts.DIAGNOSTIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stage="complex",
        temperature=0.2,
        max_tokens=1800,
        stream=True,
    )


def _chunk_text(text: str, size: int = 6):
    """Split text into small word groups for a streaming-like cadence."""
    words = text.split(" ")
    for i in range(0, len(words), size):
        yield " ".join(words[i : i + size]) + (" " if i + size < len(words) else "")


# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic fallback
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_answer(
    question: str, current_defect: dict | None, similar: list[RetrievedDefect]
) -> str:
    fixed = [s for s in similar if s.is_fixed]
    primary = fixed or [s for s in similar if not s.is_cancelled] or similar

    # No similar/matching defect at all → be honest, don't infer a cause.
    if not primary and not current_defect:
        return (
            "## No Matching Defect Found\n"
            "I couldn't find any similar or matching defect for this in the "
            "knowledge base, so I can't suggest a grounded root cause or fix.\n\n"
            "To help me find one, try adding more detail — the exact error "
            "message, the affected component/feature, or a Jira key "
            "(e.g. `HCBS-12345`)."
        )

    parts: list[str] = []

    if current_defect:
        parts.append(_render_current_defect(current_defect))

    # Root Causes
    parts.append("## Root Causes")
    causes = [(s.issue_key, s.root_cause_extracted) for s in primary if s.root_cause_extracted]
    if causes:
        for key, rc in causes[:5]:
            parts.append(f"- {key}: {rc}")
    else:
        pattern = _dominant_pattern(primary)
        likely = _PATTERN_ROOT_CAUSE.get(pattern, _PATTERN_ROOT_CAUSE["Unknown"])
        parts.append(f"- **Most likely cause ({pattern}):** {likely}")
        for s in primary[:3]:
            if s.summary:
                parts.append(f"- Inferred from similar defect {s.issue_key}: {s.summary}")

    # How Similar Defects Were Fixed Earlier
    parts.append("\n## How Similar Defects Were Fixed Earlier")
    fixes = [(s.issue_key, s.fix_applied_extracted) for s in primary if s.fix_applied_extracted]
    if fixes:
        for key, fx in fixes[:5]:
            parts.append(f"- {key}: {fx}")
    else:
        parts.append("- No fix text was recorded on the closest defects; apply the inferred "
                     "fix steps below, validated against the cited defects.")

    # Suggested Fix Steps (driven by dominant fix_pattern)
    parts.append("\n## Suggested Fix Steps")
    pattern = _dominant_pattern(primary)
    steps = _FIX_STEP_TEMPLATES.get(pattern, _FIX_STEP_TEMPLATES["Unknown"])
    for i, step in enumerate(steps, 1):
        parts.append(f"{i}. {step}")
    parts.append(f"{len(steps) + 1}. Validate against the cited fixed defects below before closing.")

    # Similar Fixed Defects table
    parts.append("\n## Similar Fixed Defects")
    parts.append(_render_similar_table(primary))

    # Surface cancelled matches as warnings only.
    cancelled = [s for s in similar if s.is_cancelled]
    if cancelled:
        parts.append("\n> ⚠️ Dead-ends (cancelled/rejected): "
                     + ", ".join(s.issue_key for s in cancelled[:5]))

    return "\n".join(parts)


def _render_current_defect(d: dict) -> str:
    return (
        "## Current Defect\n"
        f"- **Key:** {d.get('issue_key')}\n"
        f"- **Summary:** {d.get('summary') or 'n/a'}\n"
        f"- **Status:** {d.get('status') or 'n/a'}\n"
        f"- **Priority:** {d.get('priority') or 'n/a'}\n"
        f"- **Known root cause:** {d.get('root_cause_extracted') or 'n/a'}\n"
        f"- **Known fix:** {d.get('fix_applied_extracted') or 'n/a'}\n"
    )


def _render_similar_table(items: list[RetrievedDefect]) -> str:
    if not items:
        return "No strong historical match found."
    base = config.JIRA_BASE_URL
    lines = [
        "| Key | Summary | Status | Priority | Root Cause | Fix Applied | Relevance |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in items[: config.MAX_CONTEXT_DEFECTS]:
        key_link = f"[{s.issue_key}]({base}{s.issue_key})" if base else s.issue_key
        lines.append(
            f"| {key_link} | {_cell(s.summary)} | {s.status} | {s.priority} | "
            f"{_cell(s.root_cause_extracted)} | {_cell(s.fix_applied_extracted)} | "
            f"{round(s.relevance_score, 3)} |"
        )
    return "\n".join(lines)


def _dominant_pattern(items: list[RetrievedDefect]) -> str:
    from collections import Counter

    counts = Counter(s.fix_pattern for s in items if s.fix_pattern and s.fix_pattern != "Unknown")
    return counts.most_common(1)[0][0] if counts else "Unknown"


def _cell(text: str, limit: int = 90) -> str:
    text = (text or "").replace("|", "\\|").replace("\n", " ").strip()
    return (text[: limit - 1] + "…") if len(text) > limit else (text or "n/a")
