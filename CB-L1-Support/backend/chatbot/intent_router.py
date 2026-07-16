"""Deterministic-first intent router.

Classifies a question into one of:
    DEFECT_BY_KEY, DEFECT_DIAGNOSTIC, SIMILAR_DEFECTS, GENERAL_HELP

Rules run in priority order. A recognized Jira key wins (DEFECT_BY_KEY).
"""

from __future__ import annotations

import re

from chatbot import utils

DEFECT_BY_KEY = "DEFECT_BY_KEY"
DEFECT_DIAGNOSTIC = "DEFECT_DIAGNOSTIC"
SIMILAR_DEFECTS = "SIMILAR_DEFECTS"
GENERAL_HELP = "GENERAL_HELP"

# Diagnostic/fix cues -> retrieval + synthesis. Includes common defect SYMPTOM
# phrasings ("not showing", "missing", "blank") so symptom reports are treated
# as defects rather than analytics.
_DIAGNOSTIC_CUES = [
    "root cause", "why is", "why does", "why are", "why did", "how to fix",
    "how do i fix", "how was", "fix steps", "steps to fix", "failing", "failure",
    "error", "exception", "resolve", "resolution for", "workaround", "broken",
    "not working", "issue with", "problem with", "troubleshoot", "diagnose",
    "earlier fix", "historical fix", "fixed earlier", "fixed before",
    # ── defect symptom / negation phrasings ──
    "not showing", "not show", "not shown", "not displaying", "not displayed",
    "not visible", "not appearing", "not appear", "not loading", "not load",
    "not populating", "not populated", "not coming", "not reflecting",
    "not updating", "not updated", "not generated", "not getting",
    "doesn't show", "does not show", "won't show", "will not show",
    "can't see", "cannot see", "unable to see", "no data", "missing",
    "blank", "empty", "stuck", "crash", "hang", "slow", "incorrect",
    "wrong", "mismatch", "duplicate",
]

_SIMILAR_CUES = [
    "similar defect", "similar issue", "similar fixed", "like this", "related defect",
    "defects like", "comparable", "same kind of", "other defects", "previous defects",
    "similar to", "defects similar", "issues similar", "anything similar",
]

_GENERIC_HELP_CUES = [
    "what can you do", "help", "who are you", "what are you", "how do you work",
    "capabilities", "hello", "hi ", "hey",
]

_TECH_HINT = re.compile(
    r"\b(api|publish|sync|mapping|widget|dashboard|config|deploy|null|timeout|"
    r"job|pipeline|asset|rule|eom|login|permission|integration|data|service|"
    r"fault|case|error|fail|defect|bug)\b",
    re.IGNORECASE,
)


def _has_cue(text: str, cues: list[str]) -> bool:
    return any(cue in text for cue in cues)


def classify(question: str) -> str:
    """Return the intent label for a question (deterministic rules)."""
    if not question or not question.strip():
        return GENERAL_HELP

    q = question.lower().strip()
    keys = utils.extract_jira_keys(question)

    # 1) Explicit Jira key dominates.
    if keys:
        return DEFECT_BY_KEY

    # 2) Similar-defects requests.
    if _has_cue(q, _SIMILAR_CUES):
        return SIMILAR_DEFECTS

    # 3) Diagnostic/fix questions.
    if _has_cue(q, _DIAGNOSTIC_CUES):
        return DEFECT_DIAGNOSTIC

    # 4) Generic help / greetings (only when nothing technical is mentioned).
    if _has_cue(q, _GENERIC_HELP_CUES) and not _TECH_HINT.search(q):
        return GENERAL_HELP

    # 5) Fallback: a very short, non-technical input is treated as a greeting /
    #    help request; everything else is treated as a defect query, since the
    #    vast majority of questions to this assistant are about defects.
    if len(q.split()) <= 2 and not _TECH_HINT.search(q):
        return GENERAL_HELP
    return DEFECT_DIAGNOSTIC


def extract_key(question: str) -> str | None:
    keys = utils.extract_jira_keys(question)
    return keys[0] if keys else None
