"""LLM prompt templates for the diagnostic assistant.

The system prompt enforces evidence-grounding (no invented keys/numbers/fixes)
and the fixed required answer structure. The context builder assembles a compact,
retrieval-scoped payload so the model never receives the entire dataset.
"""

from __future__ import annotations

from typing import Any

DIAGNOSTIC_SYSTEM_PROMPT = (
    "You are a Defect Intelligence Assistant for Jira defects. Use only the "
    "provided context. Do not invent defect keys, counts, owners, or dates. "
    "Prefer fixed/resolved historical defects when recommending fixes. Prefer "
    "comment-derived and resolution-derived evidence over generic reasoning. "
    "The user may ask about a brand-new defect that is not in the knowledge "
    "base — that is expected. In that case, INFER the most probable root "
    "cause(s) and fixes from the most similar historical defects and present "
    "them confidently as the likely cause, citing the actual Jira keys they "
    "came from. NEVER reply that the root cause is 'not provided', 'not "
    "explicitly provided', 'unclear', or 'unavailable' — always give the most "
    "likely root cause(s) and concrete next steps grounded in the similar "
    "defects. You may note that a cause is 'likely' or 'based on similar "
    "defects', but always commit to an answer. Always cite actual Jira keys "
    "from the provided context. Do not calculate aggregate counts or statistics. EXCEPTION: "
    "if the RETRIEVED SIMILAR DEFECTS section is empty or marked (none), do NOT "
    "infer or guess a cause — instead clearly state that no similar or matching "
    "defect was found in the knowledge base, and ask the user for more detail "
    "(symptom, error text, or a Jira key). For diagnostic questions, answer in "
    "the required structure: Root Causes, How Similar Defects Were Fixed "
    "Earlier, Suggested Fix Steps, Similar Fixed Defects."
)

ANSWER_STRUCTURE_HINT = (
    "Respond in GitHub-flavored markdown using exactly these sections in order:\n"
    "## Root Causes\n"
    "## How Similar Defects Were Fixed Earlier\n"
    "## Suggested Fix Steps\n"
    "## Similar Fixed Defects\n"
    "If a current defect key was provided, add a `## Current Defect` section first.\n"
    "In Root Causes, always state the most likely root cause(s) inferred from "
    "the similar defects (cite their keys). Do NOT say the root cause is not "
    "provided or unavailable — commit to the most probable cause(s).\n"
    "The Similar Fixed Defects section must be a markdown table with columns: "
    "Key | Summary | Status | Priority | Root Cause | Fix Applied | Relevance. "
    "Only use keys present in the context. If the matches are weak, still give "
    "the closest ones and your best inferred guidance."
)


def _fmt_defect_line(d: dict[str, Any]) -> str:
    return (
        f"- {d.get('issue_key')} | status={d.get('status')} | "
        f"priority={d.get('priority')} | fix_pattern={d.get('fix_pattern')} | "
        f"relevance={d.get('relevance_score')}\n"
        f"  summary: {d.get('summary')}\n"
        f"  root_cause: {d.get('root_cause_extracted') or 'n/a'}\n"
        f"  fix_applied: {d.get('fix_applied_extracted') or 'n/a'}\n"
        f"  workaround: {d.get('workaround_extracted') or 'n/a'}\n"
        f"  key_findings: {d.get('key_findings') or 'n/a'}"
    )


def build_user_prompt(
    question: str,
    current_defect: dict[str, Any] | None,
    similar: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
    sections: list[str] | None = None,
) -> str:
    """Assemble the retrieval-scoped user prompt.

    ``sections`` (optional) restricts the answer to a subset of the narrative
    sections — any of ``"root_cause"`` / ``"resolve"``. ``None`` keeps the full
    default structure. The Similar Fixed Defects table is rendered separately by
    the caller, so it is never requested here when ``sections`` is set.
    """
    parts: list[str] = []

    if history:
        convo: list[str] = []
        for turn in history:
            uq = (turn.get("question") or "").strip()
            ua = (turn.get("answer") or "").strip()
            if len(ua) > 600:
                ua = ua[:600] + "…"
            if uq:
                convo.append(f"User: {uq}\nAssistant: {ua}")
        if convo:
            parts.append(
                "RECENT CONVERSATION (context only — use it to resolve references "
                "like 'it'/'that defect'; do not repeat it verbatim):\n"
                + "\n\n".join(convo)
                + "\n"
            )

    parts.append(f"QUESTION:\n{question}\n")

    if current_defect:
        parts.append(
            "CURRENT DEFECT:\n"
            f"- {current_defect.get('issue_key')} | status={current_defect.get('status')} | "
            f"priority={current_defect.get('priority')}\n"
            f"  summary: {current_defect.get('summary')}\n"
            f"  problem: {current_defect.get('problem_text')}\n"
            f"  known_root_cause: {current_defect.get('root_cause_extracted') or 'n/a'}\n"
            f"  known_fix: {current_defect.get('fix_applied_extracted') or 'n/a'}\n"
        )

    if similar:
        lines = "\n".join(_fmt_defect_line(d) for d in similar)
        parts.append(f"RETRIEVED SIMILAR DEFECTS (evidence):\n{lines}\n")
    else:
        parts.append("RETRIEVED SIMILAR DEFECTS (evidence):\n(none)\n")

    parts.append(_structure_hint(sections, bool(current_defect)))
    return "\n".join(parts)


def _structure_hint(sections: list[str] | None, has_current: bool) -> str:
    """Build the answer-structure instruction, scoped to ``sections`` if given."""
    if not sections:
        return ANSWER_STRUCTURE_HINT

    wanted: list[str] = []
    if "root_cause" in sections:
        if has_current:
            wanted.append("## Current Defect")
        wanted.append("## Root Causes")
    if "resolve" in sections:
        wanted.append("## How Similar Defects Were Fixed Earlier")
        wanted.append("## Suggested Fix Steps")

    if not wanted:
        # Only the similar-defects table was requested; no narrative needed.
        return "Do not write any prose. Output nothing."

    hint = (
        "Respond in GitHub-flavored markdown. Output ONLY the following sections, "
        "in this exact order, and NOTHING else — no other headers, and DO NOT "
        "include a Similar Fixed Defects table:\n"
        + "\n".join(wanted)
        + "\n"
    )
    if "root_cause" in sections:
        hint += (
            "In Root Causes, always state the most likely root cause(s) inferred "
            "from the similar defects (cite their keys). Do NOT say the root cause "
            "is not provided or unavailable — commit to the most probable cause(s).\n"
        )
    return hint
