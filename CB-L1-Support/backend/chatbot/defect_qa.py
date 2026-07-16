"""Top-level question-answering orchestration.

Routes a question by intent, runs hybrid retrieval + synthesis, and returns a
uniform response dict used by both the API and the CLI.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import config
from chatbot import answer_generator, intent_router, rag_client
from chatbot.retriever import RetrievedDefect, get_retriever

# Source-attribution headers so the user always sees which knowledge each part
# of the answer came from: historical defect chats vs. product documentation.
_DEFECT_HEADER = "### 🗂️ From Historical Defects (defect knowledge base)"
_DOC_HEADER = "### 📄 From Documentation (RAG knowledge base)"

_HELP_TEXT = (
    "I'm the **Defect Intelligence Assistant**. I can:\n\n"
    "- Diagnose defects and recommend fixes from historical evidence "
    "(e.g. *How to fix EOM publish failure?*, *Root cause of HCBS-95506?*, "
    "*Show similar fixed defects for login errors*).\n\n"
    "Mention a Jira key (like `HCBS-95506`) for a specific defect, or describe the "
    "symptom and I'll find similar fixed defects."
)


def _similar_payload(items: list[RetrievedDefect]) -> list[dict[str, Any]]:
    return [
        {
            "issue_key": s.issue_key,
            "summary": s.summary,
            "status": s.status,
            "resolution": s.resolution,
            "priority": s.priority,
            "root_cause": s.root_cause_extracted,
            "fix_applied": s.fix_applied_extracted,
            "relevance_score": round(s.relevance_score, 4),
            "quality_score": round(s.quality_score, 4),
        }
        for s in items
    ]


# ── Documentation RAG fusion helpers ──
def _pattern_hint(similar: list[RetrievedDefect]) -> str:
    """Dominant fix_pattern across the retrieved defects — a light category cue
    for the documentation-query bridge."""
    from collections import Counter

    counts = Counter(
        s.fix_pattern for s in similar if s.fix_pattern and s.fix_pattern != "Unknown"
    )
    return counts.most_common(1)[0][0] if counts else "Unknown"


def _bridge_and_ask(
    symptom: str, similar: list[RetrievedDefect], ctx: list[dict[str, str]] | None
) -> str | None:
    """Reframe the symptom into a documentation question (using the retrieved
    defects' dominant fix_pattern as a hint), then query the documentation RAG.
    Runs entirely inside the background thread."""
    doc_query = rag_client.build_doc_query(symptom, _pattern_hint(similar))
    return rag_client.ask_rag(doc_query, None, ctx)


def _start_rag(
    symptom: str,
    similar: list[RetrievedDefect],
    ctx: list[dict[str, str]] | None,
) -> tuple[ThreadPoolExecutor | None, Future | None]:
    """Kick off (query-bridge → documentation RAG) in a background thread so its
    latency overlaps the defect-answer synthesis. Must be called AFTER retrieval
    so the bridge can use the dominant fix_pattern hint. Returns
    ``(executor, future)`` or ``(None, None)`` when the RAG integration is
    disabled."""
    if not config.USE_RAG_DOCS:
        return None, None
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_bridge_and_ask, symptom, similar, ctx)
    return executor, future


def _collect_rag(
    executor: ThreadPoolExecutor | None, future: Future | None
) -> str | None:
    """Wait for the background RAG call and return its answer (or ``None``)."""
    if future is None:
        return None
    try:
        return future.result()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[rag] background call failed: {exc}")
        return None
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def _doc_block(doc_answer: str | None) -> str:
    """Render the documentation section (with separator), or '' when empty."""
    if not doc_answer:
        return ""
    return f"\n\n---\n\n{_DOC_HEADER}\n\n{doc_answer}"


def _fuse(defect_answer: str, doc_answer: str | None) -> str:
    """Combine the defect-derived answer and the documentation answer, labeling
    each source. When there's no documentation answer, return the defect answer
    unchanged (no labels needed)."""
    if not doc_answer:
        return defect_answer
    return f"{_DEFECT_HEADER}\n\n{defect_answer}{_doc_block(doc_answer)}"


# ── Interactive section selection (defect questions) ──
# Defect-family intents first offer the user a choice of what to see, instead of
# always returning every section. The selected options are sent back as
# ``sections`` on the follow-up request.
_DEFECT_FAMILY = (
    intent_router.DEFECT_BY_KEY,
    intent_router.DEFECT_DIAGNOSTIC,
    intent_router.SIMILAR_DEFECTS,
)
_DEFECT_SECTIONS = ("root_cause", "resolve", "similar")

_DEFECT_OPTIONS = [
    {"id": "root_cause", "label": "Root cause", "hint": "Most likely cause(s) from similar defects"},
    {
        "id": "resolve",
        "label": "How to resolve it",
        "hint": "How similar defects were fixed, suggested fix steps, and documentation steps",
    },
    {"id": "similar", "label": "Earlier similar defects", "hint": "Table of fixed defects with Jira keys"},
]

_OPTIONS_PROMPT = (
    "Got it — this looks like a **defect question**. What would you like me to "
    "pull together? Tick the options you want and I'll fetch just those."
)


def _normalize_sections(sections: Any) -> list[str] | None:
    """Return the valid selected section ids, or ``None`` if none were provided."""
    if not isinstance(sections, list):
        return None
    valid = [s for s in sections if s in _DEFECT_SECTIONS]
    return valid or None


def _options_response(intent: str, question: str) -> dict[str, Any]:
    """The interactive 'pick what you want' response for a defect question."""
    return {
        "intent": intent,
        "mode": "options",
        "answer": _OPTIONS_PROMPT,
        "options": _DEFECT_OPTIONS,
        "question": question,
        "similar_defects": [],
    }


def _compose_sections(
    question: str,
    current_defect: dict | None,
    similar: list[RetrievedDefect],
    ctx: list[dict[str, str]] | None,
    sections: list[str],
    doc_answer: str | None,
) -> str:
    """Build a defect answer containing only the user-selected sections."""
    blocks: list[str] = []

    if ("root_cause" in sections) or ("resolve" in sections):
        narrative = answer_generator.generate_diagnostic_answer(
            question, current_defect, similar, ctx, sections=sections
        )
        if narrative.strip():
            blocks.append(narrative)

    if "similar" in sections:
        table = answer_generator.render_similar_table(similar)
        blocks.append(f"## Similar Fixed Defects\n{table}")

    # Documentation (RAG) goes LAST so it appears after the defect evidence.
    if "resolve" in sections and doc_answer:
        blocks.append(f"{_DOC_HEADER}\n\n{doc_answer}")

    body = "\n\n---\n\n".join(b for b in blocks if b.strip())
    if not body:
        body = "_No content was available for the selected options._"
    return f"{_DEFECT_HEADER}\n\n{body}"


def _resolve_intent(question: str, mode: str | None) -> str:
    """Pick an intent, honoring an explicit user-selected ``mode``.

    ``mode`` may be ``"defect"`` (force the defect/diagnostic path) or
    ``None``/``"auto"`` (use the classifier).
    """
    mode = (mode or "auto").lower()

    if mode == "defect":
        # Keep a specific-ticket or similar-defects request if the text implies
        # it; otherwise default to the diagnostic retrieval path.
        intent = intent_router.classify(question)
        if intent in (intent_router.DEFECT_BY_KEY, intent_router.SIMILAR_DEFECTS):
            return intent
        return intent_router.DEFECT_DIAGNOSTIC

    return intent_router.classify(question)


def ask(question: str, mode: str | None = None, history: Any = None,
        sections: Any = None) -> dict[str, Any]:
    """Answer a question and return ``{intent, answer, similar_defects}``.

    For defect-family questions, the first call (no ``sections``) returns an
    interactive ``mode="options"`` response; the follow-up call carries the
    user-selected ``sections`` and returns only those parts.
    """
    question = (question or "").strip()
    if not question:
        return {"intent": intent_router.GENERAL_HELP, "answer": _HELP_TEXT, "similar_defects": []}

    # Every question is answered independently — no prior conversation context.
    ctx = None
    routing_q = question

    intent = _resolve_intent(routing_q, mode)

    if intent == intent_router.GENERAL_HELP:
        return {"intent": intent, "answer": _HELP_TEXT, "similar_defects": []}

    # Defect-family: offer the option picker first, then answer the selection.
    selected = _normalize_sections(sections)
    if selected is None:
        return _options_response(intent, question)

    # Only fetch documentation when the user wants resolution guidance.
    want_rag = "resolve" in selected

    retriever = get_retriever()
    if not retriever.ready:
        return {
            "intent": intent,
            "answer": (
                "The defect knowledge base / index is not built yet. "
                "Run `python run_chatbot.py --build` first."
            ),
            "similar_defects": [],
        }

    current_defect: dict | None = None
    note = ""
    if intent == intent_router.DEFECT_BY_KEY:
        key = intent_router.extract_key(routing_q)
        current_defect, similar = retriever.retrieve_for_key(key, top_k=config.TOP_K_RESULTS)
        if current_defect is None:
            note = f"_Defect {key} was not found in the knowledge base; showing closest matches._\n\n"
    else:
        similar = retriever.retrieve(routing_q, top_k=config.TOP_K_RESULTS, diagnostic=True)

    # Start (query-bridge → documentation RAG) AFTER retrieval so the bridge can
    # use the dominant fix_pattern hint; it runs in the background while the
    # defect answer is synthesized below.
    rag_exec, rag_future = _start_rag(routing_q, similar, ctx) if want_rag else (None, None)

    doc_answer = _collect_rag(rag_exec, rag_future) if want_rag else None
    answer = note + _compose_sections(question, current_defect, similar, ctx, selected, doc_answer)
    return {
        "intent": intent,
        "answer": answer,
        "similar_defects": _similar_payload(similar),
    }


def ask_stream(question: str, mode: str | None = None, history: Any = None,
               sections: Any = None):
    """Yield streaming events for a question.

    Event shapes:
        {"type": "meta",    "intent": str}
        {"type": "options", "intent": str, "options": list, "question": str}
        {"type": "token",   "text": str}
        {"type": "sources", "similar_defects": list}
        {"type": "done"}
    """
    question = (question or "").strip()
    if not question:
        yield {"type": "meta", "intent": intent_router.GENERAL_HELP}
        for piece in answer_generator._chunk_text(_HELP_TEXT):
            yield {"type": "token", "text": piece}
        yield {"type": "sources", "similar_defects": []}
        yield {"type": "done"}
        return

    # Every question is answered independently — no prior conversation context.
    ctx = None
    routing_q = question

    intent = _resolve_intent(routing_q, mode)
    yield {"type": "meta", "intent": intent}

    # Deterministic, non-retrieval intents stream their text directly.
    if intent == intent_router.GENERAL_HELP:
        for piece in answer_generator._chunk_text(_HELP_TEXT):
            yield {"type": "token", "text": piece}
        yield {"type": "sources", "similar_defects": []}
        yield {"type": "done"}
        return

    # Defect-family: offer the option picker first, then answer the selection.
    selected = _normalize_sections(sections)
    if selected is None:
        opts = _options_response(intent, question)
        yield {
            "type": "options",
            "intent": intent,
            "options": opts["options"],
            "question": question,
        }
        yield {"type": "done"}
        return

    retriever = get_retriever()
    if not retriever.ready:
        msg = (
            "The defect knowledge base / index is not built yet. "
            "Run `python run_chatbot.py --build` first."
        )
        for piece in answer_generator._chunk_text(msg):
            yield {"type": "token", "text": piece}
        yield {"type": "sources", "similar_defects": []}
        yield {"type": "done"}
        return

    # Only fetch documentation when the user wants resolution guidance.
    want_rag = "resolve" in selected

    current_defect: dict | None = None
    note = ""
    if intent == intent_router.DEFECT_BY_KEY:
        key = intent_router.extract_key(routing_q)
        current_defect, similar = retriever.retrieve_for_key(key, top_k=config.TOP_K_RESULTS)
        if current_defect is None:
            note = f"_Defect {key} was not found in the knowledge base; showing closest matches._\n\n"
    else:
        similar = retriever.retrieve(routing_q, top_k=config.TOP_K_RESULTS, diagnostic=True)

    # Start (query-bridge → documentation RAG) AFTER retrieval so the bridge can
    # use the dominant fix_pattern hint; it runs in the background while the
    # defect sections stream below.
    rag_exec, rag_future = _start_rag(routing_q, similar, ctx) if want_rag else (None, None)

    # Emit sources first so the UI can render the panel while tokens stream.
    yield {"type": "sources", "similar_defects": _similar_payload(similar)}

    # 1) Stream the defect sections IMMEDIATELY — do not wait for the RAG
    #    documentation lookup. The doc section is appended at the very end
    #    (below), once it has been fetched, so the user sees the defect
    #    evidence right away instead of waiting on the slow RAG call.
    defect_blocks: list[str] = []
    if ("root_cause" in selected) or ("resolve" in selected):
        narrative = answer_generator.generate_diagnostic_answer(
            question, current_defect, similar, ctx, sections=selected
        )
        if narrative.strip():
            defect_blocks.append(narrative)
    if "similar" in selected:
        table = answer_generator.render_similar_table(similar)
        defect_blocks.append(f"## Similar Fixed Defects\n{table}")

    defect_body = "\n\n---\n\n".join(b for b in defect_blocks if b.strip())
    if not defect_body:
        defect_body = "_No content was available for the selected options._"
    for piece in answer_generator._chunk_text(f"{note}{_DEFECT_HEADER}\n\n{defect_body}"):
        yield {"type": "token", "text": piece}

    # 2) Only when the user asked how to resolve it: wait for the documentation
    #    RAG answer and append it LAST. If it times out / returns nothing, the
    #    user already has the full defect answer.
    if want_rag:
        # Tell the UI a documentation lookup is in progress so it can show a
        # "fetching from documentation…" note below the defect answer.
        yield {"type": "status", "text": "📄 Also fetching from documentation…"}
        doc_answer = _collect_rag(rag_exec, rag_future)
        if doc_answer:
            for piece in answer_generator._chunk_text(
                f"\n\n---\n\n{_DOC_HEADER}\n\n{doc_answer}"
            ):
                yield {"type": "token", "text": piece}

    yield {"type": "done"}
