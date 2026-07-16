"""
rag_agent.py
────────────
Stage 5 (query routing + context retrieval) of the RAG pipeline.

For every user question the agent:
  1. Classifies the query into a route (feature_flag_table | widget_matrix |
     text_section | hybrid) using QueryRouter.
  2. Runs route-specific retrieval against Qdrant:
       feature_flag_table → filtered_table_search on TABLE_ROW / feature_flags
       widget_matrix      → filtered_table_search with exact condition filters
       text_section       → similarity_search filtered to TEXT chunks only
       hybrid             → similarity_search across all chunk types
  3. For non-widget routes, expands context by fetching the ±1 sequence
     neighbours of any TABLE_ROW chunks.
  4. Logs the route, filters, and top retrieved chunks for debuggability.
  5. Delegates answer generation (Stage 6) to AnswerGenerator.
  6. Runs a lightweight post-generation validation pass.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .answer_generator import AnswerGenerator
from .query_processor  import (
    FallbackHandler,
    QueryEnhancer,
    QueryNormalizer,
)
from .query_router     import (
    QueryRouter,
    ROUTE_FEATURE_FLAG,
    ROUTE_WIDGET_MATRIX,
    ROUTE_TEXT_SECTION,
    ROUTE_TABLE_FULL,
    ROUTE_HYBRID,
    _WIDGET_REQUIRED_CONDITIONS,
)
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Image route constant (images disabled in chunker but kept for future use)
ROUTE_IMAGE_QUERY = "image_query"


class RAGAgent:
    """
    Orchestrates query routing, retrieval, context expansion, reranking,
    and answer generation for the full RAG pipeline.

    Improved pipeline (query-time only)
    ────────────────────────────────────
    1. Normalize  – Strip filler phrases / expand contractions (QueryNormalizer).
    2. Route      – Classify intent and extract widget conditions (QueryRouter).
    3. Rewrite    – LLM rewrites query into retrieval-optimised form (QueryEnhancer).
    4. Variants   – Generate N diverse paraphrases for multi-query search.
    5. Retrieve   – Multi-query vector search + deduplication (VectorStore).
    6. Expand     – Fetch ±1 sequence neighbours for TABLE_ROW chunks.
    7. Generate   – Azure GPT-4o produces a grounded, cited answer.
    8. Validate   – Lightweight widget-row value check; regenerate if needed.
    9. Fallback   – Honest message when retrieval confidence is too low.

    Parameters
    ──────────
    vector_store     : VectorStore          – pre-initialised vector store
    answer_generator : AnswerGenerator      – pre-initialised Stage 6
    top_k            : int                  – final chunks returned to the LLM (default 7)
    query_enhancer   : QueryEnhancer | None – LLM query rewriter + multi-query generator;
                                              pass None to skip query enhancement
    """

    _LOW_SCORE_THRESHOLD: float = 0.0

    def __init__(
        self,
        vector_store:      VectorStore,
        answer_generator:  AnswerGenerator,
        top_k:             int                     = 7,
        query_enhancer:    Optional[QueryEnhancer] = None,
    ) -> None:
        logger.info("[RAG_AGENT] ── Initialising RAGAgent (improved pipeline)...")
        self._vector_store     = vector_store
        self._answer_generator = answer_generator
        self._router           = QueryRouter()
        self._normalizer       = QueryNormalizer()
        self._fallback         = FallbackHandler()
        self._enhancer         = query_enhancer   # optional – None = disabled
        self.top_k             = top_k

        enhancer_status = "enabled" if query_enhancer else "disabled"
        logger.info(
            f"[RAG_AGENT] ✓ RAGAgent ready │ top_k={top_k} │ enhancer={enhancer_status}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def ask(
        self,
        question:             str,
        source_filter:        Optional[str]           = None,
        conversation_history: Optional[List[Dict]]    = None,
        progress_callback:    Optional[Any]           = None,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Full improved RAG pipeline for a single natural-language question.

        Parameters
        ──────────
        question             : raw user question
        source_filter        : restrict retrieval to this PDF filename (optional)
        conversation_history : list of {"role": str, "content": str} dicts
                               used for pronoun/reference resolution during
                               query rewriting (optional)

        Returns
        ───────
        Tuple of (answer_string, timing_dict).
        timing_dict keys (all in seconds): normalize, route, rewrite,
        query_expansion, retrieval, context_expansion, answer_generation,
        validation, total.
        """
        sep = "═" * 60
        logger.info(f"\n[RAG_AGENT] {sep}")
        logger.info(f'[RAG_AGENT] Question (raw): "{question}"')

        timing: Dict[str, float] = {}
        _t_total = time.perf_counter()

        # ── Step 1: Normalize ─────────────────────────────────────────────────
        _t = time.perf_counter()
        clean_question = self._normalizer.normalize(question)
        timing["normalize"] = time.perf_counter() - _t
        if clean_question != question:
            logger.info(f'[RAG_AGENT] Normalized: "{clean_question}"')

        # ── Step 2: Route classification (on normalized question) ─────────────
        _t = time.perf_counter()
        routing    = self._router.classify(clean_question)
        route      = routing["route"]
        conditions = routing["conditions"]
        missing    = routing["missing_conditions"]
        timing["route"] = time.perf_counter() - _t
        logger.info(
            f"[RAG_AGENT] Route: {route} │ "
            f"conditions={conditions} │ missing={missing}"
        )

        # ── Steps 3+4: Query rewrite + expansion (ONE LLM call) ──────────────
        # rewrite_and_expand() combines what used to be 3 separate LLM calls:
        #   rewrite() + expand_synonyms() + generate_variants()
        # → saves ~2 Zscaler round-trips (~15–17s on a corporate proxy).
        if progress_callback:
            progress_callback("rewrite")
        _t = time.perf_counter()
        exact_widget_match = (route == ROUTE_WIDGET_MATRIX and not missing)

        if self._enhancer:
            prior_history = conversation_history[:-1] if conversation_history else None
            search_query, variants = self._enhancer.rewrite_and_expand(
                clean_question, prior_history
            )
            logger.info(f'[RAG_AGENT] Rewritten:  "{search_query}"')
            if variants:
                logger.info(f"[RAG_AGENT] Variants ({len(variants)}): {variants}")
        else:
            search_query = clean_question
            variants = []

        # Build the final deduplicated query list for multi-query retrieval.
        # Widget exact-match skips variants (exact filter already pins the row).
        seen: set = {search_query.strip().lower()}
        query_variants: List[str] = [search_query]
        if not exact_widget_match:
            for v in variants:
                if v.strip().lower() not in seen:
                    query_variants.append(v)
                    seen.add(v.strip().lower())

        rewrite_expand_time = time.perf_counter() - _t
        timing["rewrite"]         = rewrite_expand_time
        timing["query_expansion"] = 0.0   # merged — kept for UI compatibility

        logger.info(f"[RAG_AGENT] Total queries for retrieval: {len(query_variants)}")

        # ── Step 5: Route-specific retrieval ──────────────────────────────────
        if progress_callback:
            progress_callback("retrieval")
        _t = time.perf_counter()
        if route == ROUTE_WIDGET_MATRIX:
            chunks = self._retrieve_widget_matrix(
                search_query, conditions, missing, source_filter
            )
        elif route == ROUTE_FEATURE_FLAG:
            chunks = self._retrieve_feature_flags_multi(query_variants, source_filter)
        elif route == ROUTE_TEXT_SECTION:
            chunks = self._retrieve_text_section_multi(query_variants, source_filter)
        elif route == ROUTE_TABLE_FULL:
            chunks = self._retrieve_full_table(query_variants, source_filter)
        else:
            chunks = self._retrieve_hybrid_multi(query_variants, source_filter)
        timing["retrieval"] = time.perf_counter() - _t

        # ── Step 6: Context expansion (±1 sequence neighbours for TABLE rows) ─
        _t = time.perf_counter()
        if route != ROUTE_WIDGET_MATRIX:
            chunks = self._expand_table_context(chunks)
        if route == ROUTE_TEXT_SECTION:
            chunks = self._expand_text_context(chunks)
        timing["context_expansion"] = time.perf_counter() - _t

        # ── Step 7: Handle missing widget conditions ───────────────────────────
        if route == ROUTE_WIDGET_MATRIX and missing and not chunks:
            missing_str = ", ".join(missing)
            timing["total"] = time.perf_counter() - _t_total
            return (
                f"To answer this widget question I need more information.\n"
                f"Please also specify: **{missing_str}**\n\n"
                f"Required conditions: User Feature Flag, Coexist Feature Flag, "
                f"Energy Default Baseline, EO Default Baseline.",
                timing,
            )

        # ── Step 8: Fallback – no chunks returned ────────────────────────────
        if not chunks:
            logger.warning("[RAG_AGENT] No chunks retrieved. Returning fallback.")
            timing["total"] = time.perf_counter() - _t_total
            return self._fallback.get_message(FallbackHandler.REASON_NO_CHUNKS), timing

        # ── Step 8b: Filter out Table-of-Contents noise ──────────────────────
        # ToC entries (just dots + page numbers) score high in similarity but
        # contain zero useful content.  Remove them before the smart cap.
        chunks = [c for c in chunks if not self._is_toc_chunk(c)]

        if not chunks:
            logger.warning("[RAG_AGENT] All chunks filtered as ToC. Returning fallback.")
            timing["total"] = time.perf_counter() - _t_total
            return self._fallback.get_message(FallbackHandler.REASON_NO_CHUNKS), timing

        # ── Step 9: Route-aware smart cap ────────────────────────────────────
        # IMPORTANT: multi-query + context expansion can return 20-30 chunks,
        # dominated by TABLE_ROW.  Without a route-aware cap, text_budget = 0
        # for hybrid/text questions — the LLM receives only table rows and
        # produces wrong answers.  Fix: text always gets at least half the slots
        # for non-table routes.
        #
        # Sort each category by relevance score (desc) BEFORE capping so the
        # most relevant chunks win their slot — not just first-by-sequence-id.
        _TABLE_TYPES = {"TABLE_ROW", "TABLE", "TABLE_FRAGMENT"}
        _score_key = lambda c: c.get("_score", 0)
        table_rows   = sorted([c for c in chunks if c.get("asset_type") in _TABLE_TYPES], key=_score_key, reverse=True)
        full_tables  = sorted([c for c in chunks if c.get("asset_type") == "TABLE_FULL"], key=_score_key, reverse=True)
        image_chunks = sorted([c for c in chunks if c.get("asset_type") == "IMAGE"], key=_score_key, reverse=True)
        text_chunks  = sorted([c for c in chunks if c.get("asset_type") == "TEXT"], key=_score_key, reverse=True)

        if route == ROUTE_TEXT_SECTION:
            # For procedural/step queries, text is the primary content.
            # Give text the full budget; only include tables if slots remain.
            text_slots  = max(self.top_k, 6)
            table_slots = max(self.top_k - text_slots + 2, 2)
            chunks = text_chunks[:text_slots] + table_rows[:table_slots]
        elif route == ROUTE_HYBRID:
            # Balanced: text always gets ≥ half the budget so general
            # questions are not drowned out by table rows.
            text_slots  = max(self.top_k // 2, 6)
            table_slots = max(self.top_k - text_slots, 2)
            chunks = table_rows[:table_slots] + text_chunks[:text_slots]
        elif route == ROUTE_IMAGE_QUERY:
            img_slots  = max(self.top_k // 2, 2)
            text_slots = self.top_k - img_slots
            chunks = image_chunks[:img_slots] + text_chunks[:text_slots]
        else:
            # TABLE-heavy routes (FEATURE_FLAG, TABLE_FULL, WIDGET_MATRIX):
            # tables fill most slots; always leave 2 for supporting text.
            text_budget = max(2, self.top_k - len(table_rows) - len(full_tables))
            chunks = table_rows + full_tables + text_chunks[:text_budget]

        if len(chunks) > 50:   # hard safety cap for context budget
            chunks = chunks[:50]
        chunks.sort(key=lambda c: c.get("sequence_id", 0))  # restore reading order

        # ── Step 11: Generate answer ──────────────────────────────────────────
        if progress_callback:
            progress_callback("answer")
        _t = time.perf_counter()
        answer = self._answer_generator.generate(
            clean_question, chunks, conversation_history=conversation_history
        )
        timing["answer_generation"] = time.perf_counter() - _t

        # ── Step 12: Lightweight validation (widget rows only) ───────────────
        _t = time.perf_counter()
        answer = self._validate_answer(answer, chunks, route, conditions)
        timing["validation"] = time.perf_counter() - _t

        timing["total"] = time.perf_counter() - _t_total
        logger.info(
            f"[RAG_AGENT] ✓ Done in {timing['total']:.2f}s │ "
            f"rewrite={timing['rewrite']:.2f}s  expansion={timing['query_expansion']:.2f}s  "
            f"retrieval={timing['retrieval']:.2f}s  llm={timing['answer_generation']:.2f}s"
        )
        return answer, timing

    def ask_stream(
        self,
        question:             str,
        source_filter:        Optional[str]        = None,
        conversation_history: Optional[List[Dict]] = None,
        progress_callback:    Optional[Any]        = None,
    ):
        """
        Streaming variant of ask().

        Runs steps 1–10 (normalise → route → rewrite → retrieval → context
        expansion → fallback checks) identically to ask(), then returns a
        3-tuple so the caller can stream the answer generation step:

            partial_timing  – dict with all keys populated EXCEPT
                              answer_generation, validation, total
            t_total_start   – perf_counter() value at the start of the call,
                              so the caller can compute total after streaming
            text_stream     – generator that yields str chunks from the LLM

        The caller is responsible for:
          • Consuming text_stream (e.g. via a Streamlit write loop).
          • Setting timing["answer_generation"] = elapsed time for streaming.
          • Setting timing["total"] = time.perf_counter() - t_total_start.

        Falls back gracefully: on any error during the prepare phase, returns
        (partial_timing, t_start, iter(["⚠️ Error: ..."]) so the UI still works.
        """
        sep = "═" * 60
        logger.info(f"\n[RAG_AGENT] {sep}")
        logger.info(f'[RAG_AGENT] (stream) Question (raw): "{question}"')

        timing: Dict[str, float] = {}
        t_total_start = time.perf_counter()

        # ── Step 1 ────────────────────────────────────────────────────────────
        _t = time.perf_counter()
        clean_question = self._normalizer.normalize(question)
        timing["normalize"] = time.perf_counter() - _t

        # ── Step 2 ────────────────────────────────────────────────────────────
        _t = time.perf_counter()
        routing    = self._router.classify(clean_question)
        route      = routing["route"]
        conditions = routing["conditions"]
        missing    = routing["missing_conditions"]
        timing["route"] = time.perf_counter() - _t

        # ── Steps 3+4: Rewrite + expand ───────────────────────────────────────
        if progress_callback:
            progress_callback("rewrite")
        _t = time.perf_counter()
        exact_widget_match = (route == ROUTE_WIDGET_MATRIX and not missing)
        if self._enhancer:
            prior_history = conversation_history[:-1] if conversation_history else None
            search_query, variants = self._enhancer.rewrite_and_expand(
                clean_question, prior_history
            )
        else:
            search_query = clean_question
            variants = []

        seen: set = {search_query.strip().lower()}
        query_variants: List[str] = [search_query]
        if not exact_widget_match:
            for v in variants:
                if v.strip().lower() not in seen:
                    query_variants.append(v)
                    seen.add(v.strip().lower())
        timing["rewrite"]         = time.perf_counter() - _t
        timing["query_expansion"] = 0.0

        # ── Step 5: Retrieval ─────────────────────────────────────────────────
        if progress_callback:
            progress_callback("retrieval")
        _t = time.perf_counter()
        if route == ROUTE_WIDGET_MATRIX:
            chunks = self._retrieve_widget_matrix(
                search_query, conditions, missing, source_filter
            )
        elif route == ROUTE_FEATURE_FLAG:
            chunks = self._retrieve_feature_flags_multi(query_variants, source_filter)
        elif route == ROUTE_TEXT_SECTION:
            chunks = self._retrieve_text_section_multi(query_variants, source_filter)
        elif route == ROUTE_TABLE_FULL:
            chunks = self._retrieve_full_table(query_variants, source_filter)
        else:
            chunks = self._retrieve_hybrid_multi(query_variants, source_filter)
        timing["retrieval"] = time.perf_counter() - _t

        # ── Step 6: Context expansion ─────────────────────────────────────────
        _t = time.perf_counter()
        if route != ROUTE_WIDGET_MATRIX:
            chunks = self._expand_table_context(chunks)
        if route == ROUTE_TEXT_SECTION:
            chunks = self._expand_text_context(chunks)
        timing["context_expansion"] = time.perf_counter() - _t

        # ── Step 7: Missing widget conditions ─────────────────────────────────
        if route == ROUTE_WIDGET_MATRIX and missing and not chunks:
            missing_str = ", ".join(missing)
            timing["answer_generation"] = 0.0
            timing["validation"]        = 0.0
            timing["total"]             = time.perf_counter() - t_total_start
            msg = (
                f"To answer this widget question I need more information.\n"
                f"Please also specify: **{missing_str}**\n\n"
                f"Required conditions: User Feature Flag, Coexist Feature Flag, "
                f"Energy Default Baseline, EO Default Baseline."
            )
            return timing, t_total_start, iter([msg]), []

        # ── Step 8: No chunks fallback ────────────────────────────────────────
        if not chunks:
            timing["answer_generation"] = 0.0
            timing["validation"]        = 0.0
            timing["total"]             = time.perf_counter() - t_total_start
            return timing, t_total_start, iter([self._fallback.get_message(FallbackHandler.REASON_NO_CHUNKS)]), []

        # ── Step 8b: Filter out Table-of-Contents noise ───────────────────────
        chunks = [c for c in chunks if not self._is_toc_chunk(c)]

        if not chunks:
            timing["answer_generation"] = 0.0
            timing["validation"]        = 0.0
            timing["total"]             = time.perf_counter() - t_total_start
            return timing, t_total_start, iter([self._fallback.get_message(FallbackHandler.REASON_NO_CHUNKS)]), []

        # ── Step 9: Route-aware smart cap (mirrors ask()) ──────────────────
        # Sort by score before capping so highest-relevance chunks win.
        _TABLE_TYPES = {"TABLE_ROW", "TABLE", "TABLE_FRAGMENT"}
        _score_key = lambda c: c.get("_score", 0)
        table_rows   = sorted([c for c in chunks if c.get("asset_type") in _TABLE_TYPES], key=_score_key, reverse=True)
        full_tables  = sorted([c for c in chunks if c.get("asset_type") == "TABLE_FULL"], key=_score_key, reverse=True)
        text_chunks  = sorted([c for c in chunks if c.get("asset_type") == "TEXT"], key=_score_key, reverse=True)

        if route == ROUTE_TEXT_SECTION:
            text_slots  = max(self.top_k, 6)
            table_slots = max(self.top_k - text_slots + 2, 2)
            chunks = text_chunks[:text_slots] + table_rows[:table_slots]
        elif route == ROUTE_HYBRID:
            text_slots  = max(self.top_k // 2, 6)
            table_slots = max(self.top_k - text_slots, 2)
            chunks = table_rows[:table_slots] + text_chunks[:text_slots]
        else:
            text_budget = max(2, self.top_k - len(table_rows) - len(full_tables))
            chunks = table_rows + full_tables + text_chunks[:text_budget]

        if len(chunks) > 50:
            chunks = chunks[:50]
        chunks.sort(key=lambda c: c.get("sequence_id", 0))  # restore reading order
        self._log_retrieved_chunks(chunks)

        # ── Step 11: Return stream ────────────────────────────────────────────
        if progress_callback:
            progress_callback("answer")

        text_stream = self._answer_generator.generate_stream(
            clean_question, chunks, conversation_history=conversation_history
        )
        return timing, t_total_start, text_stream, chunks

    # ──────────────────────────────────────────────────────────────────────────
    # Route-specific retrieval  (multi-query versions)
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve_widget_matrix(
        self,
        question:      str,
        conditions:    Dict[str, str],
        missing:       List[str],
        source_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Widget-matrix retrieval.  Uses a single query (not multi-query) because
        the conditions must be extracted from the original question text and
        passed as exact metadata filters — diversifying queries could introduce
        conditions that were never stated.

        • All 4 conditions present → exact metadata filter (returns ≤1 row)
        • Partial / no conditions  → semantic search with partial filters
        """
        logger.info(
            f"[RAG_AGENT]   Widget matrix retrieval │ "
            f"conditions={conditions} missing={missing}"
        )
        if not missing:
            results = self._vector_store.filtered_table_search(
                query=question,
                table_type="conditional_matrix",
                conditions=conditions,
                top_k=5,
                source_filter=source_filter,
            )
            if results:
                logger.info(
                    f"[RAG_AGENT]   Exact-match widget row found: "
                    f"row_id={results[0].get('row_id', '?')}"
                )
                return results

        return self._vector_store.filtered_table_search(
            query=question,
            table_type="conditional_matrix",
            conditions=conditions,
            top_k=self.top_k,
            source_filter=source_filter,
        )

    def _complete_table_rows(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        After a semantic search retrieves at least one TABLE_ROW chunk, fetch
        ALL sibling rows from the same table (source_file + table_name) so that
        cross-page table continuations are never silently dropped.

        Only issues one Qdrant scroll call per unique (source_file, table_name)
        pair, so the overhead is minimal (typically 1-2 calls per query).
        """
        seen_seqs: set = {c.get("sequence_id") for c in chunks}
        extra: List[Dict[str, Any]] = []

        # Collect unique (source_file, table_name) pairs from TABLE_ROW chunks
        table_keys: set = set()
        for c in chunks:
            if c.get("asset_type") in ("TABLE_ROW", "TABLE"):
                sf = c.get("source_file", "")
                tn = c.get("table_name", "")
                if sf and tn:
                    table_keys.add((sf, tn))

        for source_file, table_name in table_keys:
            all_rows = self._vector_store.fetch_table_rows_by_name(source_file, table_name)
            for row in all_rows:
                if row.get("sequence_id") not in seen_seqs:
                    extra.append(row)
                    seen_seqs.add(row.get("sequence_id"))

        if extra:
            logger.info(
                "[RAG_AGENT] Table completeness: +%d sibling row(s) from cross-page continuation.",
                len(extra),
            )
        return chunks + extra

    def _retrieve_feature_flags_multi(
        self,
        queries:       List[str],
        source_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Multi-query feature-flag retrieval with hybrid fallback."""
        logger.info("[RAG_AGENT]   Feature flag retrieval (multi-query)...")
        # Larger pool catches more candidates before dedup + smart cap
        pool_size = max(50, self.top_k + 30)
        results = self._vector_store.multi_filtered_table_search(
            queries=queries,
            table_type="feature_flags",
            top_k_per_query=pool_size,
            source_filter=source_filter,
        )
        if not results:
            logger.info("[RAG_AGENT]   No feature_flags rows found – falling back to hybrid.")
            results = self._vector_store.multi_similarity_search(
                queries=queries,
                top_k_per_query=pool_size,
                source_filter=source_filter,
            )
        # Fetch ALL sibling rows so cross-page table splits are never missed
        results = self._complete_table_rows(results)
        return results

    def _retrieve_text_section_multi(
        self,
        queries:       List[str],
        source_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Multi-query retrieval for procedural/how-to questions.

        Searches ALL chunk types (not just TEXT) so that answers which live in
        tables are not silently missed.  Step 9 in ask() / ask_stream() gives
        text chunks their fair share via the route-aware smart cap.
        """
        logger.info("[RAG_AGENT]   Text-section retrieval (multi-query, all types)...")
        pool_size = max(50, self.top_k + 30)
        results = self._vector_store.multi_similarity_search(
            queries=queries,
            top_k_per_query=pool_size,
            source_filter=source_filter,
            # No asset_type_filter — include tables so config-guide answers
            # stored in table cells are reachable from procedural questions.
        )
        full  = [r for r in results if r.get("chunk_type") == "section_full"]
        parts = [r for r in results if r.get("chunk_type") != "section_full"]
        return (full + parts)[: self.top_k * 3]

    def _retrieve_hybrid_multi(
        self,
        queries:       List[str],
        source_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Multi-query hybrid retrieval across all chunk types."""
        logger.info("[RAG_AGENT]   Hybrid retrieval (multi-query, all chunk types)...")
        pool_size = max(50, self.top_k + 30)
        results = self._vector_store.multi_similarity_search(
            queries=queries,
            top_k_per_query=pool_size,
            source_filter=source_filter,
        )
        # Complete cross-page table rows if any TABLE_ROW chunks were retrieved
        if any(c.get("asset_type") in ("TABLE_ROW", "TABLE") for c in results):
            results = self._complete_table_rows(results)
        return results[: self.top_k * 3]

    # ── Legacy single-query helpers kept for backwards compatibility ──────────

    def _retrieve_feature_flags(
        self, question: str, source_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        return self._retrieve_feature_flags_multi([question], source_filter)

    def _retrieve_text_section(
        self, question: str, source_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        return self._retrieve_text_section_multi([question], source_filter)

    def _retrieve_hybrid(
        self, question: str, source_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        return self._retrieve_hybrid_multi([question], source_filter)

    # ── New route retrieval methods (v2) ──────────────────────────────────────

    def _retrieve_full_table(
        self,
        queries:       List[str],
        source_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        TABLE_FULL route: search for TABLE_FULL chunks first, then supplement
        with TABLE_ROW chunks so the LLM can cite individual rows.
        """
        logger.info("[RAG_AGENT]   Full-table retrieval...")
        pool_size = max(50, self.top_k + 20)
        # Dense search across TABLE_FULL chunks
        full_chunks = self._vector_store.multi_similarity_search(
            queries=queries,
            top_k_per_query=pool_size,
            source_filter=source_filter,
            asset_type_filter="TABLE_FULL",
        )
        # Also fetch some TABLE_ROW chunks for detail
        row_chunks = self._vector_store.multi_similarity_search(
            queries=queries,
            top_k_per_query=pool_size,
            source_filter=source_filter,
            asset_type_filter="TABLE_ROW",
        )
        # Expand to all sibling rows for any group found
        row_chunks = self._complete_table_rows(row_chunks)
        results = full_chunks + row_chunks
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for c in results:
            key = self._vector_store._chunk_key(c)
            if key not in seen:
                deduped.append(c)
                seen.add(key)
        return deduped

    # ──────────────────────────────────────────────────────────────────────────

    def _expand_text_context(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        For every TEXT chunk retrieved, ensure ALL surrounding steps/paragraphs
        that belong to the same section are included, even if they scored low
        in similarity search.

        Two-pass strategy:
        ──────────────────
        Pass 1 — Sequence-range expansion (PRIMARY):
            Fetch every chunk whose sequence_id falls within ±SEQ_WINDOW of
            any retrieved TEXT chunk's sequence_id.  Since steps on pages 3-5
            have consecutive sequence_ids, this reliably pulls in all of them
            regardless of whether section_title metadata is set correctly.

        Pass 2 — Section-title expansion (SECONDARY / fallback):
            For any TEXT chunk that has a non-empty section_title, fetch ALL
            chunks with the same section_title from the same file.  This catches
            cases where a section restarts after a large gap (e.g. appendix).
        """
        # How many sequence positions to expand in each direction per chunk
        SEQ_WINDOW = 5

        seen_seq_ids: set  = {c.get("sequence_id") for c in chunks}
        seen_sections: set = set()
        extra: List[Dict]  = []

        # Group TEXT chunks by source_file to batch sequence-range calls
        text_chunks_by_file: Dict[str, List[int]] = {}
        for chunk in chunks:
            if chunk.get("asset_type") != "TEXT":
                continue
            sf  = chunk.get("source_file", "")
            sid = chunk.get("sequence_id", 0)
            text_chunks_by_file.setdefault(sf, []).append(sid)

        # Pass 1: sequence-range expansion
        for source_file, seq_ids in text_chunks_by_file.items():
            if not seq_ids:
                continue
            # Build the minimal [min-WINDOW, max+WINDOW] range covering all hits
            seq_from = max(1, min(seq_ids) - SEQ_WINDOW)
            seq_to   = max(seq_ids) + SEQ_WINDOW
            neighbours = self._vector_store.fetch_sequence_range(
                source_file=source_file,
                seq_from=seq_from,
                seq_to=seq_to,
            )
            for nb in neighbours:
                if nb.get("sequence_id") not in seen_seq_ids:
                    extra.append(nb)
                    seen_seq_ids.add(nb.get("sequence_id"))

        # Pass 2: section-title expansion (catches non-contiguous continuations)
        all_text = [c for c in chunks if c.get("asset_type") == "TEXT"] + \
                   [c for c in extra  if c.get("asset_type") == "TEXT"]
        for chunk in all_text:
            source_file   = chunk.get("source_file", "")
            section_title = chunk.get("section_title", "").strip()
            if not section_title or (source_file, section_title) in seen_sections:
                continue
            seen_sections.add((source_file, section_title))

            siblings = self._vector_store.fetch_chunks_by_section(
                source_file=source_file,
                section_title=section_title,
            )
            for sibling in siblings:
                if sibling.get("sequence_id") not in seen_seq_ids:
                    extra.append(sibling)
                    seen_seq_ids.add(sibling.get("sequence_id"))

        if extra:
            logger.info(
                "[RAG_AGENT] Text section expansion: +%d chunk(s) added.", len(extra)
            )
        merged = chunks + extra
        merged.sort(key=lambda c: c.get("sequence_id", 0))
        return merged

    def _expand_table_context(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        For every TABLE_ROW or TABLE_FRAGMENT chunk:
        1. Fetch ±3 sequence neighbours (catches multi-page table rows even
           when table_group_id linking failed).
        2. Fetch all sibling rows in the same table_group_id.
        3. Fetch the TABLE_FULL chunk for the same table_group_id.
        """
        seen_seq_ids: set        = {c.get("sequence_id") for c in chunks}
        seen_group_ids: set      = set()
        extra:        List[Dict] = []

        for chunk in chunks:
            asset = chunk.get("asset_type", "")
            if asset not in ("TABLE_ROW", "TABLE", "TABLE_FRAGMENT"):
                continue

            source_file = chunk.get("source_file", "")
            seq_id      = chunk.get("sequence_id", 0)

            # ± 3 sequence neighbours (covers multi-page tables)
            for neighbour_seq in range(seq_id - 3, seq_id + 4):
                if neighbour_seq < 1 or neighbour_seq in seen_seq_ids:
                    continue
                neighbour = self._vector_store.fetch_by_sequence_id(
                    source_file, neighbour_seq
                )
                if neighbour:
                    extra.append(neighbour)
                    seen_seq_ids.add(neighbour_seq)

            # Sibling rows + TABLE_FULL (by table_group_id)
            group_id = chunk.get("table_group_id", "")
            if group_id and group_id not in seen_group_ids:
                seen_group_ids.add(group_id)

                sibling_rows = self._vector_store.fetch_table_group_rows(
                    table_group_id=group_id, source_file=source_file
                )
                for row in sibling_rows:
                    if row.get("sequence_id") not in seen_seq_ids:
                        extra.append(row)
                        seen_seq_ids.add(row.get("sequence_id"))

                full_table = self._vector_store.fetch_full_table(
                    table_group_id=group_id, source_file=source_file
                )
                if full_table and full_table.get("sequence_id") not in seen_seq_ids:
                    extra.append(full_table)
                    seen_seq_ids.add(full_table.get("sequence_id"))

        if extra:
            logger.info(
                "[RAG_AGENT] Context expansion: +%d additional chunks.", len(extra)
            )
        merged = chunks + extra
        merged.sort(key=lambda c: c.get("sequence_id", 0))
        return merged

    # ──────────────────────────────────────────────────────────────────────────
    # ToC chunk detection (query-time filter)
    # ──────────────────────────────────────────────────────────────────────────

    _TOC_LINE_RE = re.compile(
        r'^.*[.\u2026·]{4,}\s*\d+\s*$'   # line with 4+ dots/periods then page number
    )

    @staticmethod
    def _is_toc_chunk(chunk: Dict[str, Any]) -> bool:
        """Return True if a chunk looks like a Table-of-Contents entry."""
        content = chunk.get("content", "") or ""
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if not lines:
            return False
        toc_count = sum(1 for ln in lines if RAGAgent._TOC_LINE_RE.match(ln))
        return toc_count / len(lines) > 0.5

    # Debug logging
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _log_retrieved_chunks(chunks: List[Dict[str, Any]]) -> None:
        """Log enough metadata per chunk to debug wrong retrievals."""
        logger.info(f"[RAG_AGENT] ── Retrieved {len(chunks)} context chunks:")
        for i, c in enumerate(chunks, 1):
            display = (c.get("content_for_display") or c.get("content", ""))[:300]
            logger.info(
                f"[RAG_AGENT]   [{i}] score={c.get('_score', 'N/A')} │ "
                f"type={c.get('asset_type', '?')} │ "
                f"page={c.get('pdf_page_number', c.get('page_number', '?'))} │ "
                f"section={c.get('section_id', '')} {c.get('section_title', '')} │ "
                f"table={c.get('table_name', '')} │ "
                f"row_id={c.get('row_id', '')} │ "
                f"row_label={c.get('row_label', '')} │ "
                f"content_preview={display!r}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Lightweight answer validation
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_answer(
        self,
        answer:     str,
        chunks:     List[Dict[str, Any]],
        route:      str,
        conditions: Dict[str, str],
    ) -> str:
        """
        Simple validation pass:
        • For widget_matrix answers: verify that cell values in the answer
          match the retrieved row metadata (not values from a different row).
        If validation fails once, regenerate with a stricter prompt.
        """
        if route != ROUTE_WIDGET_MATRIX or not conditions:
            return answer

        # Find the widget row in chunks that matches all conditions
        matching_row = next(
            (c for c in chunks
             if all(str(c.get(k, "")).lower() == v.lower()
                    for k, v in conditions.items())),
            None,
        )
        if matching_row is None:
            return answer   # no exact row to validate against

        # Check that widget values in the answer come from the matching row
        row_values = {
            "energy_widget",
            "carbon_widget",
            "eo_widget_all_utilities",
            "eo_widget_electricity",
            "eo_widget_gas",
            "eo_widget_hot_water",
        }
        mismatches: List[str] = []
        for field in row_values:
            row_val = matching_row.get(field, "")
            if row_val and row_val.lower() not in answer.lower():
                mismatches.append(f"{field}={row_val}")

        if mismatches:
            logger.warning(
                f"[RAG_AGENT] ⚠ Answer validation failed for fields: {mismatches}. "
                "Regenerating with stricter instruction..."
            )
            strict_chunks = [matching_row]
            answer = self._answer_generator.generate_strict(
                question=f"Based ONLY on the following exact table row, answer: "
                         f"What are the widget values when {conditions}?",
                chunks=strict_chunks,
            )

        return answer

