"""
answer_generator.py
───────────────────
Stage 6 of the RAG pipeline.

Receives an ordered, deduplicated list of context chunks (text + table rows)
and calls the Azure OpenAI chat model to produce a cited, factual answer.

Responsibilities
────────────────
• Format context chunks into numbered blocks with full metadata.
• Build system + human messages and invoke the LLM.
• Return the model's response string (already contains inline citations
  in the format  [Source: filename.pdf, Page: X]  as mandated by the
  system prompt).
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_factory import build_chat_llm

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a document-grounded RAG assistant.
Answer the user's question using the retrieved context chunks provided below.

========================
ANSWERING RULES
========================

1. USE RETRIEVED CONTEXT — ANSWERING POLICY
   - Always start from the retrieved context chunks.
   - If the context contains FULL information about the question → answer from it directly.
   - If the context contains PARTIAL or RELATED information → use that information as the
     foundation and build on it using ONLY what the other retrieved chunks contain.
     Do NOT fill gaps with external general knowledge — stay within the document.
   - ONLY say "The document does not specify this." when the retrieved context contains
     absolutely NO relevant information about the topic at all (zero mention of it).
   - Never say "The document does not specify" just because the exact phrasing of the
     question was not found — use reasoning and related context to answer.
   - NEVER add facts, steps, values, or explanations that are not present in the chunks.
   - When the question is a keyword phrase or an incomplete sentence (e.g. "Baseline
     Onboarding Steps Include:"), treat it exactly the same as "What are the [topic]?" and
     answer from the retrieved chunks — do NOT say the document does not specify.

2. ANSWER STYLE
   - Write the answer in clear, easy-to-understand language.
   - Begin with a one-sentence **direct answer**, then elaborate with well-structured content
     using only what the retrieved chunks contain.
   - Do NOT add external context, background knowledge, or general industry information.
   - NEVER write a list of items as a single run-on sentence separated by semicolons or commas.
     Always break them into separate bullet points, one item per line.
   - Use **bullet points** for multi-part or list-style answers.
   - Use **numbered steps** (1. 2. 3.) for procedural, sequential, or how-to answers.
   - For color-coded or status indicators, format each on its own line with the color/status bolded:
       - 🔴 **Red:** <meaning>
       - 🟡 **Yellow:** <meaning>
       - 🟢 **Green:** <meaning>
   - For table results with **multiple matching rows**, render a **markdown table**:
       | Column A | Column B | Column C |
       |----------|----------|----------|
       | value    | value    | value    |
   - For a **single matching row**, use clearly labelled bullets:
       **Column A:** value
       **Column B:** value
   - **Bold** all feature flag names, widget names, configuration values, and key terms.
   - Use a `### ` heading to separate distinct answer sections when covering more than one topic.
   - Add a blank line between each bullet point or section for readability.
   - Do NOT add inline citations inside the answer body.

3. SOURCES BLOCK (at the end only)
   After the answer, add a single sources block in this exact format:

   ---
   **Sources:**
   - <file_name_1>, Pages: <comma-separated or range, e.g. 4, 7-9, 12>
   - <file_name_2>, Pages: <pages>

   Rules for the sources block:
   - List each unique source file ONCE.
   - Merge all page numbers for that file into a compact range (e.g. 4, 7-9, 12).
   - Sort page numbers ascending.
   - Do NOT list the same file multiple times.
   - Do NOT put any source references inside the answer body.

4. TABLE ANSWERING RULES
   - Identify the exact row(s) that match the question conditions.
   - Use the "Table Headers" prefix to understand each column's meaning.
   - Do NOT combine values from different rows.
   - Preserve exact cell values – do not paraphrase table data.
   - If multiple rows match, show all of them in a markdown table.
   - If required conditions are missing, list all possible matching rows.

5. FEATURE FLAG RULES
   - Return the exact feature flag name as written in the source.
   - Include the level (org / site / user) if stated.
   - Do not rename or normalise flags.

6. CONVERSATION CONTINUITY — FOLLOW-UP QUESTIONS
   - The conversation history injected above (prior HumanMessage/AIMessage turns) shows
     what has already been discussed. You MUST use it to resolve all references.
   - "it", "this", "that", "the same", "elaborate", "more detail", "explain further",
     "what about X" → always resolve by looking at the previous assistant turn.
   - NEVER ask the user to clarify a pronoun if the previous turn already establishes
     the topic. Assume the follow-up refers to the most recent subject discussed.
   - Example: if the previous turn answered about CDD (Cooling Degree Days) and the
     user now says "explain it in detail", answer in detail about CDD.

7. AMBIGUOUS QUESTION RULE
   - Only ask for clarification when the question is genuinely ambiguous AND the
     conversation history provides NO clue about what was being discussed.
   - Provide all valid interpretations from the retrieved context if possible.

8. TABLE_FULL CITATION RULE
   - When a TABLE_FULL chunk is present in the context, reference the table by its
     name (e.g., "the Feature Flag Options table" or "the CEM Widgets matrix").
   - If the table spans multiple pages, mention the page range.
   - When TABLE_FULL and TABLE_ROW chunks are both present, prefer to cite row-level
     values from TABLE_ROW chunks and the table's overall structure from TABLE_FULL.
"""


class AnswerGenerator:
    """
    Stage 6 – LLM answer generation with mandatory source citations.

    Parameters
    ──────────
    azure_openai_api_key      : str – Azure OpenAI API key
    azure_openai_endpoint     : str – Azure OpenAI endpoint URL
    azure_chat_deployment     : str – Deployment name for the chat model
    azure_openai_api_version  : str – Azure OpenAI API version
    """

    def __init__(
        self,
        azure_openai_api_key:     str,
        azure_openai_endpoint:    str,
        azure_chat_deployment:    str,
        azure_openai_api_version: str = "2024-02-01",
    ) -> None:
        logger.info(
            f"[ANSWER_GEN] ── Initialising AnswerGenerator "
            f"(deployment='{azure_chat_deployment}')..."
        )
        self._llm = build_chat_llm(
            azure_deployment=azure_chat_deployment,
            azure_endpoint=azure_openai_endpoint,
            azure_api_key=azure_openai_api_key,
            azure_api_version=azure_openai_api_version,
            timeout=120,  # bypass corporate SSL proxy; 120s for slow proxy
        )
        logger.info(
            f"[ANSWER_GEN] ✓ AnswerGenerator ready │ deployment={azure_chat_deployment}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        question:             str,
        chunks:               List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] | None = None,
    ) -> str:
        """
        Format *chunks* into labelled SOURCE_ID context blocks and call the LLM.
        Uses content_for_display (exact source text) for each chunk.
        Includes up to 3 prior Q&A turns so the LLM can handle follow-up questions.
        """
        context_str = self._build_context(chunks)
        messages    = self._build_messages(question, context_str, conversation_history)

        logger.info(
            f"[ANSWER_GEN]   Context window: {len(chunks)} chunks, "
            f"{len(context_str):,} characters total."
        )
        response = self._llm.invoke(messages)
        return response.content

    def generate_stream(
        self,
        question:             str,
        chunks:               List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] | None = None,
    ):
        """
        Same as generate() but yields text chunks incrementally via LLM streaming.
        Allows the UI to display tokens as they arrive instead of waiting for the
        full response (~18–22s with Zscaler).

        Yields
        ------
        str – successive text chunks from the model (empty chunks are skipped).
        """
        context_str = self._build_context(chunks)
        messages    = self._build_messages(question, context_str, conversation_history)

        logger.info(
            f"[ANSWER_GEN]   Streaming: {len(chunks)} chunks, "
            f"{len(context_str):,} characters total."
        )
        for chunk in self._llm.stream(messages):
            text = chunk.content
            if text:
                yield text

    # Minimum similarity score a candidate follow-up must achieve in the
    # vector store to be considered "answerable" by the knowledge base.
    _FOLLOWUP_SCORE_THRESHOLD: float = 0.62
    _FOLLOWUP_MIN_CHUNKS: int = 2  # need at least 2 strong chunks

    def suggest_followups(
        self,
        question: str,
        answer:   str,
        vector_store=None,
    ) -> List[str]:
        """
        Generate follow-up questions that the system can actually answer.

        1. Ask the LLM to propose 6 candidate follow-up questions.
        2. For each candidate, run a lightweight similarity search against
           the vector store and keep only those with a top-1 score above
           ``_FOLLOWUP_SCORE_THRESHOLD``.
        3. Return up to 3 validated questions.

        If no vector_store is provided, falls back to returning unvalidated
        suggestions (original behaviour).
        """
        prompt = (
            "Based on the following question and answer, suggest exactly 6 short, "
            "relevant follow-up questions that the user might want to ask next. "
            "The questions should be closely related to the topic discussed and "
            "should be answerable from the same document knowledge base.\n\n"
            "Rules:\n"
            "- Each question must be concise (under 15 words).\n"
            "- Questions should explore different aspects of the same topic.\n"
            "- Do NOT repeat the original question.\n"
            "- Do NOT ask generic or open-ended questions.\n"
            "- Questions must be specific and factual.\n"
            "- Return ONLY the 6 questions, one per line, numbered 1. 2. 3. 4. 5. 6.\n\n"
            f"Original Question: {question}\n\n"
            f"Answer: {answer[:1500]}\n\n"
            "Suggested follow-up questions:"
        )
        try:
            response = self._llm.invoke([
                SystemMessage(content="You generate concise follow-up questions that can be answered from a technical document knowledge base."),
                HumanMessage(content=prompt),
            ])
            lines = [
                line.strip().lstrip("0123456789.)-– ").strip()
                for line in response.content.strip().split("\n")
                if line.strip()
            ]
            candidates = [q for q in lines if q][:6]

            # If no vector store provided, return top 3 unvalidated (fallback)
            if vector_store is None:
                return candidates[:3]

            # Validate each candidate against the knowledge base
            validated = []
            for candidate in candidates:
                if len(validated) >= 3:
                    break
                try:
                    results = vector_store.similarity_search(candidate, top_k=3)
                    strong_hits = [
                        r for r in results
                        if r.get("_score", 0) >= self._FOLLOWUP_SCORE_THRESHOLD
                    ]
                    if len(strong_hits) >= self._FOLLOWUP_MIN_CHUNKS:
                        validated.append(candidate)
                    else:
                        logger.debug(
                            f"[ANSWER_GEN] Dropped follow-up ({len(strong_hits)} "
                            f"chunks >= {self._FOLLOWUP_SCORE_THRESHOLD}): {candidate}"
                        )
                except Exception:
                    continue

            logger.info(
                f"[ANSWER_GEN] Follow-up validation: {len(validated)}/{len(candidates)} "
                f"candidates passed (threshold={self._FOLLOWUP_SCORE_THRESHOLD}, "
                f"min_chunks={self._FOLLOWUP_MIN_CHUNKS})"
            )
            return validated

        except Exception as e:
            logger.warning(f"[ANSWER_GEN] Failed to generate follow-ups: {e}")
            return []

    def generate_strict(
        self,
        question: str,
        chunks:   List[Dict[str, Any]],
    ) -> str:
        """
        Stricter re-generation used by the validation step.
        Adds an explicit instruction to use ONLY the provided row data.
        """
        context_str = self._build_context(chunks)
        strict_note = (
            "\n\nIMPORTANT: Answer STRICTLY from the single table row above. "
            "Do not use any other knowledge. "
            "If a value is not in that row, say 'not specified in this row'."
        )
        messages = self._build_messages(question, context_str + strict_note)
        response = self._llm.invoke(messages)
        return response.content

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: List[Dict[str, Any]]) -> str:
        """Format chunk list into SOURCE_ID-labelled context blocks."""
        blocks: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            sid          = f"S{i}"
            asset_type   = chunk.get("asset_type", "TEXT")
            source_file  = chunk.get("source_file", "unknown")
            pdf_page     = chunk.get("pdf_page_number", chunk.get("page_number", "not provided"))
            sec_id       = chunk.get("section_id", "")
            sec_title    = chunk.get("section_title", "")
            section      = f"{sec_id} {sec_title}".strip() or "N/A"
            display_text = (
                chunk.get("content_for_display")
                or chunk.get("content", "")
            )

            if asset_type in ("TABLE_ROW", "TABLE", "TABLE_FRAGMENT"):
                table_name = chunk.get("table_name", "")
                row_id     = chunk.get("row_id", "")
                row_label  = chunk.get("row_label", "")
                block = (
                    f"[SOURCE_ID: {sid}]\n"
                    f"source_file:    {source_file}\n"
                    f"pdf_page_number: {pdf_page}\n"
                    f"section:        {section}\n"
                    f"asset_type:     {asset_type}\n"
                    f"table_name:     {table_name}\n"
                    f"row_id:         {row_id}\n"
                    f"row_label:      {row_label}\n"
                    f"content:\n{display_text}\n"
                )

            elif asset_type == "TABLE_FULL":
                table_name    = chunk.get("table_name", "")
                page_start    = chunk.get("page_start", pdf_page)
                page_end      = chunk.get("page_end", pdf_page)
                is_multi_page = chunk.get("is_multi_page", False)
                page_range    = f"{page_start}–{page_end}" if is_multi_page else str(page_start)
                block = (
                    f"[SOURCE_ID: {sid}]\n"
                    f"source_file:    {source_file}\n"
                    f"pdf_page_number: {page_range}\n"
                    f"section:        {section}\n"
                    f"asset_type:     TABLE_FULL\n"
                    f"table_name:     {table_name}\n"
                    f"multi_page:     {is_multi_page}\n"
                    f"content:\n{display_text}\n"
                )

            elif asset_type == "IMAGE":
                import json as _json
                image_type    = chunk.get("image_type", "")
                ocr_text      = chunk.get("ocr_text", "")
                ocr_engine    = chunk.get("ocr_engine_used", "")
                ocr_conf      = chunk.get("ocr_confidence", 1.0)
                desc_raw      = chunk.get("image_description_json", "")
                low_conf_note = ""
                if ocr_conf < 0.7 and ocr_text:
                    low_conf_note = (
                        "NOTE: OCR confidence is low — extracted text may contain errors.\n"
                    )
                desc_text = ""
                if desc_raw:
                    try:
                        desc_obj = _json.loads(desc_raw) if isinstance(desc_raw, str) else desc_raw
                        desc_text = desc_obj.get("description") or desc_obj.get("summary") or str(desc_obj)
                    except Exception:
                        desc_text = str(desc_raw)
                block = (
                    f"[SOURCE_ID: {sid}]\n"
                    f"source_file:    {source_file}\n"
                    f"pdf_page_number: {pdf_page}\n"
                    f"section:        {section}\n"
                    f"asset_type:     IMAGE\n"
                    f"image_type:     {image_type}\n"
                    f"ocr_engine:     {ocr_engine}\n"
                    f"ocr_confidence: {ocr_conf:.2f}\n"
                    f"{low_conf_note}"
                    f"ocr_text:\n{ocr_text}\n"
                    f"visual_description:\n{desc_text}\n"
                    f"embedding_text:\n{display_text}\n"
                )

            else:
                chunk_type = chunk.get("chunk_type", "")
                block = (
                    f"[SOURCE_ID: {sid}]\n"
                    f"source_file:    {source_file}\n"
                    f"pdf_page_number: {pdf_page}\n"
                    f"section:        {section}\n"
                    f"asset_type:     TEXT\n"
                    f"chunk_type:     {chunk_type}\n"
                    f"content:\n{display_text}\n"
                )
            blocks.append(block)

        return "\n" + ("-" * 50 + "\n").join(blocks)

    @staticmethod
    def _build_messages(
        question:             str,
        context_str:          str,
        conversation_history: List[Dict[str, str]] | None = None,
    ) -> list:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages: list = [SystemMessage(content=_SYSTEM_PROMPT)]

        # ── Inject prior conversation turns ───────────────────────────────────
        # Use up to 3 complete Q&A pairs (6 messages) so the LLM understands
        # follow-up references like "elaborate", "what about X?", "same for Y?"
        if conversation_history:
            # Exclude the current user message (last item) – it's handled below
            prior = conversation_history[:-1] if conversation_history else []
            for turn in prior[-6:]:   # last 6 = up to 3 Q&A pairs
                role    = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))

        # ── Current question with retrieved context ────────────────────────────
        messages.append(HumanMessage(
            content=(
                "========================\n"
                "RETRIEVED CONTEXT\n"
                "========================\n"
                f"{context_str}\n\n"
                "========================\n"
                "USER QUESTION\n"
                "========================\n\n"
                f"{question}\n\n"
                "========================\n"
                "FINAL ANSWER\n"
                "========================"
            )
        ))
        return messages
