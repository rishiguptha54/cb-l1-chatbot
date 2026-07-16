"""
defect_ingest.py
─────────────────
On-demand ingestion of new defects, either ONE at a time by Jira key or in
BULK via a JQL query. Both paths fetch live from Jira, normalize with the
SAME extraction pipeline used for the bulk knowledge base
(``build_knowledge_base._normalize_defect`` — root cause / fix / fix-pattern
/ quality-score heuristics), chunk with the same 5-chunk scheme
(``build_chunks._chunks_for``), embed with the same provider, and upsert into
the defect Qdrant collection. This mirrors the documentation RAG's on-demand
PDF ingestion (``rag_docs/ingest.py``) — same "complete pipeline runs, same
collection" pattern, just for defects.

Requires ``JIRA_URL`` / ``JIRA_USERNAME`` / ``JIRA_API_TOKEN`` (already used
for the offline bulk rebuild — see config.py). JQL search uses Jira Cloud's
current ``POST /rest/api/3/search/jql`` endpoint (the classic ``/search``
endpoint was retired by Atlassian).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, Optional

import config
from chatbot import utils
from chatbot.build_chunks import _chunks_for
from chatbot.build_knowledge_base import _build_cf_resolver, _normalize_defect
from chatbot.schemas import DefectRecord

_JIRA_KEY_RE = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")

# Standard fields requested for JQL bulk search (custom field ids are appended
# from the resolver at query time — see ``_search_fields``).
_STANDARD_SEARCH_FIELDS = [
    "summary", "description", "status", "resolution", "priority", "issuetype",
    "components", "labels", "created", "updated", "resolutiondate",
    "assignee", "reporter", "comment", "issuelinks",
]

_cf_resolver: Optional[dict] = None


def _get_cf_resolver() -> dict:
    """Reuse the SAME custom-field id mapping the bulk build used, read once
    from the committed raw_defects.json's own ``custom_fields`` block (these
    ids are stable per-Jira-instance, not per-fetch)."""
    global _cf_resolver
    if _cf_resolver is None:
        raw = utils.load_json(config.RAW_DEFECTS_PATH, default={}) or {}
        _cf_resolver = _build_cf_resolver(raw)
    return _cf_resolver


def _search_fields(cf: dict) -> list[str]:
    return list(dict.fromkeys(_STANDARD_SEARCH_FIELDS + [v for v in cf.values() if v]))


def _adf_to_text(node: Any) -> str:
    """Flatten Atlassian Document Format (Jira Cloud v3 rich text) into plain
    text. ``_normalize_defect`` expects plain strings for description/comment
    bodies; the live REST API returns ADF instead."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_adf_to_text(n) for n in node)
    if isinstance(node, dict):
        ntype = node.get("type")
        parts: list[str] = []
        if ntype == "text":
            parts.append(node.get("text", ""))
        for child in node.get("content", []) or []:
            parts.append(_adf_to_text(child))
        if ntype in ("paragraph", "heading", "listItem", "tableRow", "tableCell"):
            parts.append("\n")
        return "".join(parts)
    return ""


def _jira_base_and_auth() -> tuple[str, tuple[str, str]]:
    if not (config.JIRA_URL and config.JIRA_USERNAME and config.JIRA_API_TOKEN):
        raise ValueError(
            "Jira is not configured on this server (missing JIRA_URL / "
            "JIRA_USERNAME / JIRA_API_TOKEN)."
        )
    return config.JIRA_URL.rstrip("/"), (config.JIRA_USERNAME, config.JIRA_API_TOKEN)


def fetch_jira_issue(issue_key: str) -> dict:
    """Fetch one issue via the Jira Cloud REST API v3."""
    import httpx

    base, auth = _jira_base_and_auth()
    url = f"{base}/rest/api/3/issue/{issue_key}"
    resp = httpx.get(url, auth=auth, timeout=30, verify=False)
    if resp.status_code == 404:
        raise ValueError(f"Jira issue {issue_key} was not found.")
    resp.raise_for_status()
    return resp.json()


def fetch_jira_comments(issue_key: str) -> list[dict]:
    import httpx

    base, auth = _jira_base_and_auth()
    url = f"{base}/rest/api/3/issue/{issue_key}/comment"
    resp = httpx.get(url, auth=auth, timeout=30, verify=False)
    resp.raise_for_status()
    return (resp.json() or {}).get("comments", [])


def search_jira_issues_by_jql(
    jql: str, max_results: int, fields: list[str]
) -> list[dict]:
    """Search Jira via the current ``POST /rest/api/3/search/jql`` endpoint
    (the classic ``/rest/api/3/search`` was retired by Atlassian in 2025),
    paginating with ``nextPageToken`` until ``max_results`` is reached.

    Raises ``ValueError`` with Jira's own error message for an invalid JQL
    query (e.g. a syntax error or an unbounded query with no restriction).
    """
    import httpx

    base, auth = _jira_base_and_auth()
    url = f"{base}/rest/api/3/search/jql"

    issues: list[dict] = []
    next_token: Optional[str] = None
    page_size = 100
    while len(issues) < max_results:
        body: dict[str, Any] = {
            "jql": jql,
            "maxResults": min(page_size, max_results - len(issues)),
            "fields": fields,
        }
        if next_token:
            body["nextPageToken"] = next_token
        resp = httpx.post(url, auth=auth, json=body, timeout=30, verify=False)
        if resp.status_code == 400:
            try:
                messages = resp.json().get("errorMessages") or [resp.text]
            except Exception:
                messages = [resp.text]
            raise ValueError(f"Invalid JQL query: {'; '.join(messages)}")
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("issues", [])
        issues.extend(batch)
        if data.get("isLast", True) or not batch:
            break
        next_token = data.get("nextPageToken")
        if not next_token:
            break
    return issues[:max_results]


def _prepare_issue_for_normalize(issue_json: dict) -> dict:
    """Convert the ADF description to plain text in-place so
    ``_normalize_defect`` (which expects a plain string) works unchanged."""
    fields = issue_json.get("fields", {})
    if isinstance(fields, dict) and isinstance(fields.get("description"), (dict, list)):
        fields["description"] = _adf_to_text(fields["description"])
    return issue_json


def _prepare_comments(raw_comments: list[dict]) -> list[dict]:
    out: list[dict] = []
    for c in raw_comments:
        body = c.get("body")
        text = _adf_to_text(body) if isinstance(body, (dict, list)) else (body or "")
        out.append({"author": c.get("author", {}), "created": c.get("created", ""), "body": text})
    return out


def _normalize_and_chunk(
    issue_json: dict, comments_map: dict, cf: dict
) -> tuple[DefectRecord, dict, list[dict]]:
    """Shared core: normalize one issue into a DefectRecord (deriving root
    cause / fix / fix_pattern / quality_score / is_fixed / is_cancelled — the
    same enriched columns the bulk knowledge base has) and build its chunks."""
    rec = _normalize_defect(
        issue_json, cf, comments_map, insights={}, llm_fix_patterns={},
        mask=config.MASK_SENSITIVE_DATA,
    )
    rec_dict = rec.to_dict()
    chunks = [c.to_dict() for c in _chunks_for(rec_dict)]
    return rec, rec_dict, chunks


def _process_search_result_issue(issue_json: dict, cf: dict) -> tuple[DefectRecord, dict, list[dict]]:
    """Normalize+chunk one issue as returned by ``search_jira_issues_by_jql``
    (comments come embedded in ``fields.comment.comments`` rather than a
    separate per-issue API call, so bulk imports stay fast)."""
    issue_key = issue_json.get("key", "")
    issue_json = _prepare_issue_for_normalize(issue_json)
    fields = issue_json.get("fields", {}) or {}
    raw_comments = (fields.get("comment") or {}).get("comments", [])
    comments_map = {issue_key: _prepare_comments(raw_comments)}
    return _normalize_and_chunk(issue_json, comments_map, cf)


def _store_records(
    records: list[dict], chunks: list[dict], vectors: Any
) -> None:
    """Shared persistence: upsert chunks to Qdrant, update the local
    knowledge base file, and refresh the live retriever singleton in-memory
    (kb, kb_by_key, and the incremental lexical/IDF table) — used by both the
    single-key and JQL bulk paths so both are immediately queryable."""
    from chatbot.defect_vector_store import get_defect_vector_store
    from chatbot.retriever import get_retriever

    vector_store = get_defect_vector_store()
    keys = {r["issue_key"] for r in records}
    for key in keys:
        vector_store.delete_by_issue_key(key)  # idempotent re-add/update
    vector_store.upsert_chunks(chunks, vectors)

    kb = utils.load_json(config.DEFECT_KB_PATH, default=[]) or []
    kb = [r for r in kb if r.get("issue_key") not in keys]
    kb.extend(records)
    utils.save_json(config.DEFECT_KB_PATH, kb)

    retriever = get_retriever()
    retriever.kb = kb
    retriever.kb_by_key = {r["issue_key"]: r for r in kb}
    for rec_dict in records:
        retriever.add_lexical_doc(rec_dict["issue_key"], rec_dict.get("search_text", ""))


def add_defect_stages(issue_key: str) -> Iterator[Dict[str, Any]]:
    """Generator form of the SINGLE-defect ingestion pipeline: yields a
    progress event before each stage, then a final ``{"stage": "done",
    "result": {...}}``.

    Raises ``ValueError`` for a configuration/input problem (missing Jira
    creds, bad key format, issue not found, too little content to embed).
    """
    issue_key = (issue_key or "").strip().upper()
    if not _JIRA_KEY_RE.match(issue_key):
        raise ValueError(f"'{issue_key}' does not look like a Jira key (e.g. HCBS-12345).")

    from chatbot.build_embeddings import EmbeddingProvider
    from chatbot.defect_vector_store import get_defect_vector_store

    yield {"stage": "fetching", "message": f"🔎 Fetching {issue_key} from Jira…"}
    issue_json = fetch_jira_issue(issue_key)
    raw_comments = fetch_jira_comments(issue_key)

    issue_json = _prepare_issue_for_normalize(issue_json)
    comments_map = {issue_key: _prepare_comments(raw_comments)}

    yield {
        "stage": "normalizing",
        "message": "🧩 Extracting root cause, fix, and classification…",
    }
    cf = _get_cf_resolver()
    rec, rec_dict, chunks = _normalize_and_chunk(issue_json, comments_map, cf)
    if not chunks:
        raise ValueError(
            f"{issue_key} has too little content to build embeddings from "
            "(no summary/description/comments)."
        )
    yield {"stage": "chunking", "message": f"✂️ Built {len(chunks)} embedding chunks…"}

    yield {
        "stage": "embedding",
        "message": f"🧠 Embedding {len(chunks)} chunks and storing in Qdrant…",
    }
    provider = EmbeddingProvider()
    texts = [c.get("text", "") for c in chunks]
    vectors = provider.embed(texts)

    _store_records([rec_dict], chunks, vectors)

    result = {
        "issue_key": issue_key,
        "collection": get_defect_vector_store().collection_name,
        "chunks": len(chunks),
        "is_fixed": rec.is_fixed,
        "is_cancelled": rec.is_cancelled,
        "fix_pattern": rec.fix_pattern,
        "quality_score": rec.quality_score,
    }
    yield {"stage": "done", "result": result}


def add_defect(issue_key: str) -> Dict[str, Any]:
    """Non-streaming convenience wrapper around ``add_defect_stages()``."""
    result: Dict[str, Any] = {}
    for event in add_defect_stages(issue_key):
        if event["stage"] == "done":
            result = event["result"]
    return result


def add_defects_by_jql_stages(
    jql: str, max_results: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """Generator form of the BULK ingestion pipeline: run a JQL query,
    normalize + chunk + embed EVERY matching defect (deriving the same
    enriched columns as the bulk knowledge base — root cause, fix applied,
    fix pattern, defect type, quality score, is_fixed/is_cancelled), and
    upsert them all into the defect Qdrant collection.

    Yields a progress event before each stage, then a final
    ``{"stage": "done", "result": {...}}`` with a summary. Raises
    ``ValueError`` for an empty/invalid JQL query or when nothing matched.
    """
    jql = (jql or "").strip()
    if not jql:
        raise ValueError("JQL query must not be empty.")
    max_results = max_results or config.JQL_ADD_MAX_RESULTS

    from chatbot.build_embeddings import EmbeddingProvider

    cf = _get_cf_resolver()
    fields = _search_fields(cf)

    yield {"stage": "searching", "message": f'🔎 Searching Jira: "{jql}"…'}
    issues = search_jira_issues_by_jql(jql, max_results, fields)
    if not issues:
        raise ValueError("No defects matched this JQL query.")

    yield {
        "stage": "found",
        "message": (
            f"📋 Found {len(issues)} defect(s) — extracting root cause, fix, "
            "and classification for each…"
        ),
    }

    records: list[dict] = []
    all_chunks: list[dict] = []
    for i, issue_json in enumerate(issues, start=1):
        _rec, rec_dict, chunks = _process_search_result_issue(issue_json, cf)
        if chunks:
            records.append(rec_dict)
            all_chunks.extend(chunks)
        if i % 10 == 0 or i == len(issues):
            yield {
                "stage": "normalizing",
                "message": f"🧩 Processed {i}/{len(issues)} defects…",
            }

    if not all_chunks:
        raise ValueError("None of the matched defects had enough content to embed.")

    yield {
        "stage": "embedding",
        "message": (
            f"🧠 Embedding {len(all_chunks)} chunks from {len(records)} defects "
            "and storing in Qdrant…"
        ),
    }
    provider = EmbeddingProvider()
    texts = [c.get("text", "") for c in all_chunks]
    vectors = provider.embed(texts)

    _store_records(records, all_chunks, vectors)

    from chatbot.defect_vector_store import get_defect_vector_store

    result = {
        "jql": jql,
        "collection": get_defect_vector_store().collection_name,
        "matched": len(issues),
        "processed": len(records),
        "chunks": len(all_chunks),
        "issue_keys": sorted(r["issue_key"] for r in records),
    }
    yield {"stage": "done", "result": result}


def add_defects_by_jql(jql: str, max_results: Optional[int] = None) -> Dict[str, Any]:
    """Non-streaming convenience wrapper around ``add_defects_by_jql_stages()``."""
    result: Dict[str, Any] = {}
    for event in add_defects_by_jql_stages(jql, max_results):
        if event["stage"] == "done":
            result = event["result"]
    return result

