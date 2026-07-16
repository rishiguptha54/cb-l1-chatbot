"""In-process client for the document-RAG pipeline.

Originally this module called a separate FastAPI RAG server over HTTP. It now
runs the extracted RAG pipeline (see the top-level ``rag_docs`` package) directly
*in-process*, querying the same Qdrant document knowledge base. This removes the
need to host/run a second service.

The public interface is unchanged so the rest of the chatbot keeps working:

    rag_health() -> bool
    ask_rag(question, source_filter=None, conversation_history=None) -> str | None

Every failure path (RAG disabled, pipeline not configured, Qdrant unreachable,
LLM error) returns ``None`` / ``False`` instead of raising, so the defect
chatbot always keeps working from its own defect evidence even when the
document-RAG side is misconfigured or unavailable.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from typing import Any

import config

# Shared executor for bounding the (potentially slow) RAG call. A module-level
# executor lets a timed-out call return immediately at the cap: the orphaned
# worker keeps running in the background and its result is simply discarded,
# rather than blocking on a context-manager exit (which waits for the thread).
_RAG_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag")

# ── "No answer" checkpoint ──
# The RAG answer-generation prompt (rag_docs/answer_generator.py) always appends
# a "**Sources:**" block, even when it had to say the documentation contains no
# relevant information. Citing sources for a "not found" answer is misleading —
# it looks like those documents *do* contain the fix. This checkpoint runs on
# every RAG answer before it is fused into the final response: when the answer
# indicates no real information was found, the sources block is dropped and the
# answer is replaced with a single, clear "no solution in documentation" line.
_SOURCES_BLOCK_RE = re.compile(r"\n*-{2,}\s*\n+\s*\*\*Sources:?\*\*.*", re.IGNORECASE | re.DOTALL)

_NO_ANSWER_PHRASES = (
    "does not specify",
    "does not contain",
    "does not mention",
    "does not include",
    "does not provide",
    "does not describe",
    "does not cover",
    "does not address",
    "no relevant information",
    "no information is available",
    "no information was found",
    "not specified in the document",
    "not mentioned in the document",
    "not covered in the document",
    "not available in the document",
    "not found in the document",
    "could not find any information",
    "could not find relevant information",
    "unable to find any information",
    "unable to find relevant information",
)


def _strip_sources_when_no_answer(answer: str) -> str:
    """Checkpoint: if the RAG answer says the docs have nothing relevant, drop
    the sources block and normalize to a plain "no solution" message. Otherwise
    return the answer unchanged (sources kept)."""
    body = _SOURCES_BLOCK_RE.sub("", answer).strip()
    lowered = body.lower()
    if any(phrase in lowered for phrase in _NO_ANSWER_PHRASES):
        return "The documentation does not contain a solution for this."
    return answer


def _map_history(history: Any) -> list[dict[str, str]]:
    """Convert this app's ``[{question, answer}]`` turns into the pipeline's
    ``[{role, content}]`` message list (chronological order)."""
    messages: list[dict[str, str]] = []
    if not isinstance(history, list):
        return messages
    for turn in history:
        if not isinstance(turn, dict):
            continue
        q = str(turn.get("question", "")).strip()
        a = str(turn.get("answer", "")).strip()
        if q:
            messages.append({"role": "user", "content": q})
        if a:
            messages.append({"role": "assistant", "content": a})
    return messages


# ── Documentation query bridge ──
# The document RAG answers "how does X work / how is X configured" DOCUMENTATION
# questions, not raw defect symptoms ("X not showing"). A verbatim symptom —
# especially its negation — retrieves poorly. This reframes the symptom into a
# neutral, general documentation question so that, WHEN the docs cover the topic,
# the answer is actually found. When they don't, the RAG's own honest "not
# covered" path still applies. Falls back to the raw symptom on any error, so the
# RAG is always queried with at least the original text (no regression).
_DOC_QUERY_SYSTEM = (
    "You convert an L1 support symptom into ONE general documentation search "
    "question about the underlying product feature or process.\n"
    "Rules:\n"
    "- Identify the core feature/topic from the symptom (use the category hint "
    "only to disambiguate).\n"
    "- Ask how that feature/process works, is configured, or is displayed, in "
    "broad terms that would match a user-guide section.\n"
    "- Use only common product vocabulary. Do NOT add narrow component, "
    "environment, or portal names, or specific jargon that may not appear in a "
    "user guide.\n"
    "- Do NOT mention 'defect', 'bug', 'error', 'issue', 'fail', or ticket keys.\n"
    "- Output ONE concise question and nothing else."
)


def build_doc_query(symptom: str, pattern_hint: str = "Unknown") -> str:
    """Reframe a defect symptom into a documentation-style question.

    ``pattern_hint`` is the dominant fix_pattern of the retrieved defects, used
    only as a light category cue. Returns the reframed question, or the original
    ``symptom`` unchanged on any error so the RAG is always queried with at
    least the raw text.
    """
    symptom = (symptom or "").strip()
    if not symptom:
        return symptom
    try:
        from llm_provider import chat

        user = (
            f"User symptom: {symptom}\n\n"
            f"Category hint (fix pattern): {pattern_hint or 'Unknown'}\n\n"
            f"Documentation question:"
        )
        out = chat(
            [
                {"role": "system", "content": _DOC_QUERY_SYSTEM},
                {"role": "user", "content": user},
            ],
            stage="standard",
            temperature=0.2,
            max_tokens=80,
        ).strip()
        return out or symptom
    except Exception as exc:  # LLM/network error → fall back to raw symptom
        print(f"[rag] doc-query bridge failed ({exc}); using raw symptom")
        return symptom


def rag_health() -> bool:
    """Return True when the in-process RAG pipeline can be constructed
    (env vars present + Qdrant reachable)."""
    if not config.USE_RAG_DOCS:
        return False
    try:
        from rag_docs import pipeline_health
        return pipeline_health()
    except Exception as exc:  # missing deps, bad config, Qdrant down, ...
        print(f"[rag] health check failed: {exc}")
        return False


def ask_rag(
    question: str,
    source_filter: str | None = None,
    conversation_history: Any = None,
) -> str | None:
    """Answer ``question`` from the document knowledge base via the in-process
    RAG pipeline.

    Args:
        question: The user's question (required, non-empty).
        source_filter: Optional PDF filename to restrict the search.
        conversation_history: Optional recent turns; accepted either as this
            app's ``[{question, answer}]`` shape or already as
            ``[{role, content}]`` messages.

    Returns:
        The answer string, or ``None`` when RAG is disabled or the call fails
        for any reason (never raises).
    """
    if not config.USE_RAG_DOCS:
        return None

    question = (question or "").strip()
    if not question:
        return None

    try:
        from rag_docs import answer_question
        history = _map_history(conversation_history)

        # Run the (potentially slow) in-process pipeline under a hard timeout so
        # the defect assistant is never blocked for long. Locally every Azure /
        # Qdrant call goes through the corporate proxy, so the full pipeline can
        # take a long time; if it exceeds RAG_TIMEOUT we simply drop the
        # documentation answer and let the defect evidence stand on its own.
        fut = _RAG_EXECUTOR.submit(
            answer_question,
            question,
            source_filter,
            history or None,
        )
        try:
            answer, _timing = fut.result(timeout=config.RAG_TIMEOUT)
        except _FuturesTimeout:
            print(f"[rag] timed out after {config.RAG_TIMEOUT}s; skipping doc answer")
            return None
    except Exception as exc:  # missing deps, Qdrant/LLM error, ...
        print(f"[rag] in-process query failed: {exc}")
        return None

    answer = (answer or "").strip()
    if not answer:
        return None
    return _strip_sources_when_no_answer(answer)
