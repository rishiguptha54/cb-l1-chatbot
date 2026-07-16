"""
query_processor.py
──────────────────
Improved query-time processing pipeline components.

Components
──────────
QueryNormalizer  – Rule-based query cleaning (no LLM, zero latency).
QueryEnhancer    – LLM-based query rewriting, multi-query variant
                   generation, and synonym/domain expansion.
FallbackHandler  – Returns structured, honest fallback messages when
                   retrieval confidence is too low to answer safely.
"""

import ast
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_factory import build_chat_llm

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# QueryNormalizer
# ─────────────────────────────────────────────────────────────────────────────

# Common filler phrase prefixes to strip
_FILLER_PATTERNS = [
    r"^(can you (please )?|could you (please )?|please |kindly )",
    r"^(i (was |am )?wondering (if |about |what |how )?)",
    r"^(i need to know |i want to know |i would like to know )",
    r"^(tell me (about |what |how )?)",
    r"^(explain (to me )?(what |how |why )?)",
    r"^(um+[,\s]*|uh+[,\s]*|so[,\s]+|like[,\s]+)",
    r"^(hey[,\s]+|hi[,\s]+|hello[,\s]+)",
]

_CONTRACTIONS: Dict[str, str] = {
    "don't": "do not",    "doesn't": "does not",  "didn't": "did not",
    "won't": "will not",  "can't": "cannot",       "shouldn't": "should not",
    "wouldn't": "would not", "isn't": "is not",   "aren't": "are not",
    "wasn't": "was not",  "weren't": "were not",   "I'm": "I am",
    "I've": "I have",     "I'd": "I would",        "I'll": "I will",
    "it's": "it is",      "that's": "that is",     "there's": "there is",
    "what's": "what is",  "where's": "where is",   "how's": "how is",
    "who's": "who is",    "they're": "they are",   "we're": "we are",
    "you're": "you are",  "they've": "they have",  "we've": "we have",
}


class QueryNormalizer:
    """
    Rule-based query normalizer — no LLM, no external calls.

    Applies:
      • Contraction expansion  ("don't" → "do not")
      • Filler phrase removal  ("can you please tell me about X" → "X")
      • Whitespace collapsing
      • First-letter capitalisation
    """

    _FILLER_RE = re.compile(
        "|".join(_FILLER_PATTERNS),
        re.IGNORECASE,
    )
    # Build contraction regex once at class level
    _CONTRACTION_RE = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b",
        re.IGNORECASE,
    )

    def _expand_contraction(self, match: re.Match) -> str:
        word = match.group(0)
        # Try exact-case lookup first, then lower-case
        return _CONTRACTIONS.get(word, _CONTRACTIONS.get(word.lower(), word))

    def normalize(self, query: str) -> str:
        """Return a cleaned, normalised version of *query*."""
        if not query or not query.strip():
            return query

        result = query.strip()

        # Expand contractions
        result = self._CONTRACTION_RE.sub(self._expand_contraction, result)

        # Strip leading filler phrases (iterate because stacking is common)
        for _ in range(3):
            new = self._FILLER_RE.sub("", result).strip()
            if new == result:
                break
            result = new

        # Collapse multiple whitespace
        result = re.sub(r"\s{2,}", " ", result).strip()

        # Convert trailing colon / "Include:" / "Are:" patterns into a proper question.
        # e.g. "Baseline Onboarding Steps Include:" → "What are the Baseline Onboarding Steps?"
        # e.g. "Configuration steps:" → "What are the Configuration steps?"
        _COLON_SUFFIX_RE = re.compile(
            r"\s*(include[s]?|are|is|consist[s]? of|contain[s]?)?\s*:+\s*$",
            re.IGNORECASE,
        )
        if _COLON_SUFFIX_RE.search(result):
            result = _COLON_SUFFIX_RE.sub("", result).strip()
            if result and not re.search(r"[?]$", result):
                result = "What are the " + result[0].lower() + result[1:] + "?"

        # Capitalise first character
        if result:
            result = result[0].upper() + result[1:]

        if result != query:
            logger.debug(f"[NORMALIZER] '{query}' → '{result}'")

        return result or query  # safety: never return empty string


# ─────────────────────────────────────────────────────────────────────────────
# QueryEnhancer
# ─────────────────────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = """\
You are a query optimization assistant for a document retrieval system.
Rewrite the user question into a self-contained, specific, declarative
retrieval query that will return the best matching results from a vector
database of business and technical documents.

Rules:
- Resolve all pronouns and elliptical references using the conversation
  history when provided.
- Remove filler phrases and conversational noise.
- Be specific; include domain context if it is clearly inferable.
- Do NOT answer the question — only rewrite it.
- Output a single rewritten query string with no quotes, no explanation."""

_VARIANTS_SYSTEM = """\
Generate exactly {n} semantically diverse paraphrases of the given query.
Each paraphrase must:
- Preserve the original meaning exactly.
- Use different vocabulary, sentence structure, or framing.
- Be suitable for searching a vector database of technical documents.

Respond with ONLY a Python-style list of strings. Example:
["paraphrase 1", "paraphrase 2", "paraphrase 3"]
No explanation. No numbering outside the list."""

_SYNONYMS_SYSTEM = """\
You are a domain terminology expert for business, technical, and software documents.
Given a query, append at most 5 relevant synonyms or domain-specific alternate
terms that would commonly appear in formal documents for the same concept.

Rules:
- Do NOT change the original query meaning.
- Only add genuinely applicable synonyms — do not pad with generic words.
- Separate appended terms from the query with a comma.
- If no useful synonyms exist, return the original query unchanged.
- Output only the expanded query string. No explanation."""

# Single combined prompt — replaces 3 separate calls (rewrite + synonyms + variants).
# Returns a JSON object so one round-trip does all query enhancement work.
_COMBINED_ENHANCE_SYSTEM = """\
You are a query optimization assistant for a document retrieval system.
Given a user question and optional conversation history, produce a JSON object
with exactly two keys:

  "rewritten"  – A single self-contained, specific, declarative retrieval query.
                 Resolve ALL pronouns and elliptical references using the
                 conversation history. Remove filler phrases. Be specific.
                 Do NOT answer the question — only rewrite it.

  "variants"   – A JSON array of {n} semantically diverse paraphrases of the
                 rewritten query. Each paraphrase must:
                   • Preserve the exact meaning of "rewritten".
                   • Use different vocabulary, sentence structure, or framing.
                   • Naturally incorporate relevant domain synonyms or alternate
                     terms where appropriate (e.g. CDD / cooling degree day /
                     cooling load — pick whichever fits each variant).
                   • Be suitable for searching a vector database of technical
                     business documents.

Output ONLY valid JSON — no markdown fences, no explanation, no extra keys.

Example:
{{
  "rewritten": "What are the configuration steps for enabling the baseline app flow?",
  "variants": [
    "How do I configure the baseline application flow feature?",
    "Steps to set up and enable baseline app flow",
    "Baseline app flow configuration and setup procedure"
  ]
}}"""


class QueryEnhancer:
    """
    LLM-powered query enhancement using Azure OpenAI.

    Three capabilities
    ──────────────────
    rewrite()          – Rewrite into a retrieval-optimised declarative form.
    generate_variants() – Produce N semantically diverse paraphrases.
    expand_synonyms()  – Append domain-relevant alternate terms.

    All methods fail gracefully: on any LLM / network error they log a
    warning and return the original query unchanged.
    """

    def __init__(
        self,
        azure_openai_api_key:     str,
        azure_openai_endpoint:    str,
        azure_chat_deployment:    str,
        azure_openai_api_version: str = "2024-02-01",
        n_variants:               int = 3,
    ) -> None:
        self._n_variants = n_variants
        self._llm = build_chat_llm(
            azure_deployment=azure_chat_deployment,
            azure_endpoint=azure_openai_endpoint,
            azure_api_key=azure_openai_api_key,
            azure_api_version=azure_openai_api_version,
            max_tokens=1024,  # increased: combined JSON output needs more room
            timeout=60,  # bypass corporate SSL proxy; 60s for slow proxy
        )
        logger.info(
            f"[QUERY_ENHANCER] ✓ Ready │ "
            f"deployment={azure_chat_deployment}, n_variants={n_variants}"
        )

    # ── Public methods ────────────────────────────────────────────────────────

    def rewrite_and_expand(
        self,
        query:                str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple:
        """
        ONE LLM call that returns (rewritten_query, variants_list).

        Replaces the old 3-call sequence:
            rewrite() → expand_synonyms() → generate_variants()

        By batching everything into a single JSON-returning prompt, we save
        2 full Zscaler round-trips (~15–17s on a corporate proxy).

        The quality is identical — the model does the same work, just asked
        all at once instead of three separate times.

        Returns
        -------
        (rewritten: str, variants: List[str])
        Falls back to (original_query, []) on any error.
        """
        import json as _json

        history_text = self._format_history(conversation_history)
        user_content = (
            f"Conversation history:\n{history_text}\n\n"
            if history_text else ""
        ) + f"User question: {query}"

        system = _COMBINED_ENHANCE_SYSTEM.format(n=self._n_variants)

        try:
            response = self._llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user_content),
            ])
            raw = (response.content or "").strip()
            logger.debug(f"[QUERY_ENHANCER] Raw LLM response: {raw!r}")

            # Strip markdown code fences if the model wrapped the JSON
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```\s*$", "", raw).strip()

            # If still empty after stripping, try to extract JSON object via regex
            if not raw:
                logger.warning("[QUERY_ENHANCER] LLM returned empty content after fence stripping.")
                return query, []

            # Attempt direct parse first; if it fails, try to pull the JSON
            # object out of surrounding prose (model sometimes adds explanations)
            try:
                data = _json.loads(raw)
            except _json.JSONDecodeError:
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    data = _json.loads(match.group())
                else:
                    raise  # re-raise original; will be caught by outer except

            rewritten = str(data.get("rewritten", "")).strip().strip('"').strip("'")
            if len(rewritten) < 5:
                rewritten = query

            variants_raw = data.get("variants", [])
            if not isinstance(variants_raw, list):
                variants_raw = []

            # Deduplicate: remove any variant identical to the rewritten query
            seen = {rewritten.strip().lower(), query.strip().lower()}
            variants: List[str] = []
            for v in variants_raw[: self._n_variants]:
                v = str(v).strip()
                if v and v.lower() not in seen:
                    seen.add(v.lower())
                    variants.append(v)

            logger.info(
                f"[QUERY_ENHANCER] Combined enhance: '{query}' → '{rewritten}' "
                f"+ {len(variants)} variant(s)"
            )
            return rewritten, variants

        except Exception as exc:
            logger.warning(
                f"[QUERY_ENHANCER] rewrite_and_expand failed ({exc}). "
                "Falling back to original query with no variants."
            )
            return query, []

    def rewrite(
        self,
        query:                str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Rewrite *query* into a self-contained retrieval query.

        Uses the last 3 turns of *conversation_history* (if provided) to
        resolve pronouns and elliptical references.
        """
        history_text = self._format_history(conversation_history)
        user_content = (
            f"Conversation history:\n{history_text}\n\n"
            if history_text else ""
        ) + f"Original question: {query}"

        try:
            response = self._llm.invoke([
                SystemMessage(content=_REWRITE_SYSTEM),
                HumanMessage(content=user_content),
            ])
            rewritten = response.content.strip().strip('"').strip("'")
            if len(rewritten) < 5:
                return query
            logger.debug(f"[QUERY_ENHANCER] Rewrite: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as exc:
            logger.warning(f"[QUERY_ENHANCER] Rewrite failed ({exc}). Using original.")
            return query

    def generate_variants(self, query: str) -> List[str]:
        """
        Generate up to *n_variants* semantically diverse paraphrases.
        Returns an empty list if the LLM call fails.
        """
        system_msg = _VARIANTS_SYSTEM.format(n=self._n_variants)
        try:
            response = self._llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=f"Query: {query}"),
            ])
            variants = self._parse_list(response.content.strip())
            if not variants:
                return []
            # Remove blanks and near-duplicates of the source query
            seen = {query.strip().lower()}
            filtered: List[str] = []
            for v in variants[: self._n_variants]:
                v = v.strip()
                if v and v.lower() not in seen:
                    seen.add(v.lower())
                    filtered.append(v)
            logger.debug(
                f"[QUERY_ENHANCER] Generated {len(filtered)} variant(s) for: '{query}'"
            )
            return filtered
        except Exception as exc:
            logger.warning(f"[QUERY_ENHANCER] Variant generation failed ({exc}). Skipping.")
            return []

    def expand_synonyms(self, query: str) -> str:
        """
        Append domain-relevant synonyms / alternate terms to *query*.
        Returns the original *query* unchanged if the call fails or
        if the LLM returns nothing useful.
        """
        try:
            response = self._llm.invoke([
                SystemMessage(content=_SYNONYMS_SYSTEM),
                HumanMessage(content=f"Query: {query}"),
            ])
            expanded = response.content.strip().strip('"').strip("'")
            # Sanity check: expanded must be at least as long as original
            if len(expanded) < len(query):
                return query
            logger.debug(f"[QUERY_ENHANCER] Expanded: '{query}' → '{expanded}'")
            return expanded
        except Exception as exc:
            logger.warning(f"[QUERY_ENHANCER] Synonym expansion failed ({exc}). Skipping.")
            return query

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_history(history: Optional[List[Dict[str, str]]]) -> str:
        if not history:
            return ""
        lines = []
        for turn in history[-6:]:  # last 6 messages = up to 3 Q&A pairs
            role    = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a Python-style list of strings from LLM output."""
        # Try ast.literal_eval on the whole response
        try:
            result = ast.literal_eval(text)
            if isinstance(result, list):
                return [str(x) for x in result]
        except Exception:
            pass
        # Fallback: extract all quoted strings
        matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        return [m[0] or m[1] for m in matches if (m[0] or m[1]).strip()]




# ─────────────────────────────────────────────────────────────────────────────
# FallbackHandler
# ─────────────────────────────────────────────────────────────────────────────

class FallbackHandler:
    """
    Returns structured, honest fallback messages when the retrieval
    pipeline cannot produce a confident answer.

    Four standard reasons are defined as class-level constants.
    Call ``get_message(reason)`` to retrieve the appropriate message.
    """

    REASON_NO_CHUNKS    = "no_chunks_found"
    REASON_LOW_SCORE    = "low_confidence_retrieval"
    REASON_OUT_OF_SCOPE = "out_of_scope"
    REASON_AMBIGUOUS    = "unclear_intent"

    _MESSAGES: Dict[str, str] = {
        REASON_NO_CHUNKS: (
            "I searched the available documents but could not find any relevant "
            "information to answer your question.\n\n"
            "**Suggestions:**\n"
            "- Try rephrasing your question using different terminology.\n"
            "- Specify which document, section, or feature you are referring to.\n"
            "- Confirm that the topic is covered in the uploaded documents."
        ),
        REASON_LOW_SCORE: (
            "I found some potentially related content in the documents, but I am "
            "not confident it fully answers your question.\n\n"
            "**Suggestions:**\n"
            "- Try a more specific question.\n"
            "- Reference a specific document, section, feature name, or flag value.\n"
            "- Verify the details directly in the source document."
        ),
        REASON_OUT_OF_SCOPE: (
            "This question appears to be outside the scope of the documents I have access to.\n\n"
            "I can answer questions about the content of the uploaded documents, such as "
            "feature flags, widget configurations, procedures, and technical specifications."
        ),
        REASON_AMBIGUOUS: (
            "Your question is a bit ambiguous. Could you provide more detail?\n\n"
            "**For example:**\n"
            "- Which document or section are you referring to?\n"
            "- What specific value, flag, or condition are you asking about?\n"
            "- Are you asking about a feature, a process, or a configuration?"
        ),
    }

    def get_message(self, reason: str, extra: str = "") -> str:
        """Return the formatted fallback message for *reason*."""
        base = self._MESSAGES.get(reason, self._MESSAGES[self.REASON_NO_CHUNKS])
        return f"{base}\n\n{extra}".strip() if extra else base
