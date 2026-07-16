"""Hybrid retriever for diagnostic questions, backed by Qdrant.

Combines four signals over defect chunks stored in the Qdrant "defect"
collection (see ``chatbot/defect_vector_store.py``) and aggregates to the
defect level:

    final = 0.45*semantic + 0.25*keyword + 0.15*metadata + 0.10*quality + 0.05*fixed_boost

Semantic candidates come from Qdrant vector search; keyword candidates come
from Qdrant's full-text index (exact/lexical term matches) merged in, then
re-scored locally with an in-memory BM25-style IDF table built from the local
knowledge base (updated incrementally when a defect is added — see
``add_lexical_doc()``). This keeps the precision of the original local-BM25
design while staying correct for defects added after the collection was
first built (no local index rebuild needed).

Key behaviours:
- Exact Jira-key lookup overrides semantic search (returns the defect itself plus
  similar fixed defects seeded from its own text).
- Scores are aggregated by ``issue_key`` (best chunk wins) so duplicate chunks of
  one defect never crowd out other defects.
- For diagnostic/fix queries, fixed/resolved defects get a boost while cancelled
  defects are demoted (kept only as warnings).
- Both the initially bulk-migrated defects AND any defect added later via the
  "Add defect" button live in the same Qdrant collection, so retrieval is
  identical for both — no local index rebuild/restart needed after an add.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import config
from chatbot import utils
from chatbot.build_embeddings import EmbeddingProvider
from chatbot.defect_vector_store import get_defect_vector_store

# ── Scoring weights (sum of first four ≈ 0.95; fixed boost adds up to 0.05) ──
W_SEMANTIC = 0.45
W_KEYWORD = 0.25
W_METADATA = 0.15
W_QUALITY = 0.10
W_FIXED = 0.05

_SEM_CANDIDATES = 60
_KW_CANDIDATES = 60
_KW_FILTER_TOKENS = 8  # max distinct tokens sent to the Qdrant text filter


@dataclass
class RetrievedDefect:
    issue_key: str
    summary: str = ""
    status: str = ""
    resolution: str = ""
    priority: str = ""
    components: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    environment: str = ""
    root_cause_extracted: str = ""
    fix_applied_extracted: str = ""
    workaround_extracted: str = ""
    key_findings: str = ""
    fix_pattern: str = "Unknown"
    defect_type: str = "unknown"
    quality_score: float = 0.0
    relevance_score: float = 0.0
    is_fixed: bool = False
    is_cancelled: bool = False
    matched_chunk_types: list[str] = field(default_factory=list)
    matched_text_snippets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


class ChatbotRetriever:
    """Loads the local knowledge base once and serves Qdrant-backed semantic
    retrieval + exact-key lookup."""

    def __init__(self) -> None:
        self.kb = utils.load_json(config.DEFECT_KB_PATH, default=[]) or []
        self.kb_by_key = {r["issue_key"]: r for r in self.kb}
        self._provider: EmbeddingProvider | None = None
        self._vector_store = None
        self._vector_store_error: str | None = None
        try:
            self._vector_store = get_defect_vector_store()
        except Exception as exc:  # pragma: no cover - env dependent
            self._vector_store_error = f"{type(exc).__name__}: {exc}"

        # In-memory BM25-style IDF table over defect-level text, so keyword
        # scoring works without a separately-built/stale local index.
        self._lexical_doc_freq: Counter = Counter()
        self._lexical_idf: dict[str, float] = {}
        self._lexical_n = 0
        self._lexical_avgdl = 0.0
        self._build_lexical_index()

    def _build_lexical_index(self) -> None:
        doc_lens: list[int] = []
        doc_freq: Counter = Counter()
        for rec in self.kb:
            text = rec.get("search_text") or rec.get("problem_text") or rec.get("summary") or ""
            tokens = set(utils.tokenize(text))
            doc_lens.append(len(tokens))
            for t in tokens:
                doc_freq[t] += 1
        n = len(self.kb)
        self._lexical_doc_freq = doc_freq
        self._lexical_n = n
        self._lexical_avgdl = (sum(doc_lens) / n) if n else 0.0
        self._lexical_idf = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()
        }

    def add_lexical_doc(self, issue_key: str, text: str) -> None:
        """Incrementally fold a newly-added defect's text into the in-memory
        IDF table, so keyword scoring is correct for it immediately — called
        by ``chatbot/defect_ingest.py`` after a new defect is stored."""
        tokens = set(utils.tokenize(text or ""))
        prev_n = self._lexical_n
        self._lexical_n = prev_n + 1
        for t in tokens:
            self._lexical_doc_freq[t] += 1
        n = self._lexical_n
        self._lexical_avgdl = ((self._lexical_avgdl * prev_n) + len(tokens)) / n if n else 0.0
        self._lexical_idf = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for t, df in self._lexical_doc_freq.items()
        }

    # ── Lazy provider so the API can boot even if the model is slow to import ──
    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = EmbeddingProvider()
        return self._provider

    @property
    def ready(self) -> bool:
        return bool(self.kb) and self._vector_store is not None

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────
    def get_defect(self, issue_key: str) -> dict | None:
        return self.kb_by_key.get(issue_key.upper())

    def defect_from_key(self, issue_key: str, score: float = 0.0) -> RetrievedDefect | None:
        """Build a RetrievedDefect straight from the KB record (no scoring)."""
        rec = self.kb_by_key.get(issue_key.upper())
        if not rec:
            return None
        return RetrievedDefect(
            issue_key=rec.get("issue_key", issue_key.upper()),
            summary=rec.get("summary", ""),
            status=rec.get("status", ""),
            resolution=rec.get("resolution", ""),
            priority=rec.get("priority", ""),
            components=rec.get("components", []),
            labels=rec.get("labels", []),
            environment=rec.get("environment", ""),
            root_cause_extracted=rec.get("root_cause_extracted", ""),
            fix_applied_extracted=rec.get("fix_applied_extracted", ""),
            workaround_extracted=rec.get("workaround_extracted", ""),
            key_findings=rec.get("key_findings", ""),
            fix_pattern=rec.get("fix_pattern", "Unknown"),
            defect_type=rec.get("defect_type", "unknown"),
            quality_score=float(rec.get("quality_score", 0.0)),
            relevance_score=score,
            is_fixed=bool(rec.get("is_fixed", False)),
            is_cancelled=bool(rec.get("is_cancelled", False)),
        )

    def retrieve(
        self, query: str, top_k: int | None = None, diagnostic: bool = True
    ) -> list[RetrievedDefect]:
        """Hybrid retrieval (Qdrant semantic + Qdrant full-text keyword)
        aggregated to defect level."""
        top_k = top_k or config.TOP_K_RESULTS
        if not self.ready:
            return []

        qvec = self.provider.embed_query(query)
        sem_hits = self._vector_store.search(qvec, top_k=_SEM_CANDIDATES)

        query_tokens_list = utils.tokenize(query)
        query_tokens = set(query_tokens_list)
        filter_tokens = sorted(query_tokens, key=len, reverse=True)[:_KW_FILTER_TOKENS]
        kw_hits = (
            self._vector_store.keyword_search(filter_tokens, top_k=_KW_CANDIDATES)
            if filter_tokens else []
        )

        # Merge + dedupe by chunk_id (semantic hits keep their real score;
        # keyword-only hits carry _score=0.0 and get a semantic score of 0).
        merged: dict[str, dict] = {}
        for chunk in sem_hits + kw_hits:
            cid = chunk.get("chunk_id") or f"{chunk.get('issue_key', '')}::{chunk.get('chunk_type', '')}"
            merged.setdefault(cid, chunk)

        # Local BM25-lite keyword scores, normalized to [0, 1] like the old
        # local-BM25 index did (divide by the max raw score among candidates).
        raw_lexical = {
            cid: self._lexical_raw_score(query_tokens_list, chunk.get("text", ""))
            for cid, chunk in merged.items()
        }
        max_raw = max(raw_lexical.values()) if raw_lexical else 0.0

        # Per-defect aggregation.
        agg: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"score": -1.0, "chunks": [], "snippets": []}
        )
        for cid, chunk in merged.items():
            key = chunk.get("issue_key", "")
            if not key:
                continue
            sem = _clamp01(float(chunk.get("_score", 0.0)))
            kw = (raw_lexical[cid] / max_raw) if max_raw > 0 else 0.0
            meta = self._metadata_score(query_tokens, chunk)
            quality = float(chunk.get("quality_score", 0.0))
            fixed_boost = self._fixed_boost(chunk, diagnostic)

            score = (
                W_SEMANTIC * sem + W_KEYWORD * kw + W_METADATA * meta
                + W_QUALITY * quality + fixed_boost
            )
            if diagnostic and chunk.get("is_cancelled"):
                score -= 0.15  # demote dead-ends

            bucket = agg[key]
            bucket["chunks"].append(chunk.get("chunk_type", ""))
            if sem > 0.15 or kw > 0.15:
                bucket["snippets"].append(utils.truncate(chunk.get("text", ""), 220))
            if score > bucket["score"]:
                bucket["score"] = score

        results = [self._to_result(k, v) for k, v in agg.items()]
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        results = self._filter_by_relevance(results)
        return results[:top_k]

    def _lexical_raw_score(self, query_tokens: list[str], chunk_text: str) -> float:
        """Unnormalized BM25-style score of ``chunk_text`` against
        ``query_tokens``, using the in-memory IDF table."""
        if not query_tokens or not chunk_text:
            return 0.0
        tokens = utils.tokenize(chunk_text)
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        dl = len(tokens)
        avgdl = self._lexical_avgdl or 1.0
        k1, b = 1.5, 0.75
        raw = 0.0
        for t in query_tokens:
            tf = counts.get(t, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            raw += self._lexical_idf.get(t, 0.0) * (tf * (k1 + 1)) / denom
        return raw

    def _filter_by_relevance(
        self, results: list[RetrievedDefect]
    ) -> list[RetrievedDefect]:
        """Drop weak matches so a diagnostic answer isn't forced to use a full
        ``TOP_K`` of loosely-related defects (which yields misleading fix steps).

        A defect is kept only when its relevance score clears BOTH:
          • an absolute floor (``RELEVANCE_MIN_SCORE``) — removes global noise, and
          • a relative margin of the best match (``RELEVANCE_REL_MARGIN`` × top)
            — trims the tail when the query has only a few genuine matches.

        Returns ``[]`` when even the best match is below the floor, so the answer
        layer can honestly report that no similar defect was found instead of
        inventing fixes from unrelated defects. Results must be sorted desc.
        """
        if not results or config.RELEVANCE_MIN_SCORE <= 0:
            return results
        top = results[0].relevance_score
        threshold = max(
            config.RELEVANCE_MIN_SCORE, config.RELEVANCE_REL_MARGIN * top
        )
        return [r for r in results if r.relevance_score >= threshold]

    def retrieve_for_key(
        self, issue_key: str, top_k: int | None = None
    ) -> tuple[dict | None, list[RetrievedDefect]]:
        """Return the exact defect plus similar fixed defects seeded from it."""
        issue_key = issue_key.upper()
        defect = self.get_defect(issue_key)
        if not defect:
            return None, self.retrieve(issue_key, top_k=top_k, diagnostic=True)

        seed = defect.get("search_text") or defect.get("problem_text") or defect.get("summary")
        similar = self.retrieve(seed, top_k=(top_k or config.TOP_K_RESULTS) + 1, diagnostic=True)
        similar = [r for r in similar if r.issue_key != issue_key]
        return defect, similar[: (top_k or config.TOP_K_RESULTS)]

    # ─────────────────────────────────────────────────────────────────────────
    #  Scoring internals
    # ─────────────────────────────────────────────────────────────────────────
    def _metadata_score(self, query_tokens: set[str], chunk: dict) -> float:
        """Reward overlap between query tokens and structured metadata."""
        if not query_tokens:
            return 0.0
        fields_tokens: list[str] = []
        for comp in chunk.get("components", []):
            fields_tokens += utils.tokenize(comp)
        for lab in chunk.get("labels", []):
            fields_tokens += utils.tokenize(lab)
        for val in (
            chunk.get("environment", ""), chunk.get("product", ""),
            chunk.get("team", ""), chunk.get("fix_pattern", ""),
            chunk.get("defect_type", ""), chunk.get("priority", ""),
            chunk.get("status", ""),
        ):
            fields_tokens += utils.tokenize(val)
        if not fields_tokens:
            return 0.0
        overlap = query_tokens & set(fields_tokens)
        return min(1.0, len(overlap) / 3.0)

    def _fixed_boost(self, chunk: dict, diagnostic: bool) -> float:
        if not diagnostic:
            return 0.0
        if chunk.get("is_fixed"):
            return W_FIXED
        if chunk.get("is_cancelled"):
            return 0.0
        return W_FIXED * 0.3  # open but not cancelled: small partial boost

    def _to_result(self, key: str, bucket: dict) -> RetrievedDefect:
        rec = self.kb_by_key.get(key, {})
        chunk_types = list(dict.fromkeys(bucket["chunks"]))
        snippets = list(dict.fromkeys(bucket["snippets"]))[:3]
        return RetrievedDefect(
            issue_key=key,
            summary=rec.get("summary", ""),
            status=rec.get("status", ""),
            resolution=rec.get("resolution", ""),
            priority=rec.get("priority", ""),
            components=rec.get("components", []),
            labels=rec.get("labels", []),
            environment=rec.get("environment", ""),
            root_cause_extracted=rec.get("root_cause_extracted", ""),
            fix_applied_extracted=rec.get("fix_applied_extracted", ""),
            workaround_extracted=rec.get("workaround_extracted", ""),
            key_findings=rec.get("key_findings", ""),
            fix_pattern=rec.get("fix_pattern", "Unknown"),
            defect_type=rec.get("defect_type", "unknown"),
            quality_score=float(rec.get("quality_score", 0.0)),
            relevance_score=round(float(bucket["score"]), 4),
            is_fixed=bool(rec.get("is_fixed", False)),
            is_cancelled=bool(rec.get("is_cancelled", False)),
            matched_chunk_types=chunk_types,
            matched_text_snippets=snippets,
        )


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


# ── Module-level singleton (loaded lazily by the API/CLI) ──
_RETRIEVER: ChatbotRetriever | None = None


def get_retriever() -> ChatbotRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = ChatbotRetriever()
    return _RETRIEVER
