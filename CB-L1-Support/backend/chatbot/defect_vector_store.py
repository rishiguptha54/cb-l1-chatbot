"""
defect_vector_store.py
───────────────────────
Qdrant-backed vector store for the DEFECT knowledge base.

Uses the SAME Qdrant cluster as the documentation RAG (``rag_docs``) but a
separate collection (default ``"defect"``, see ``config.DEFECT_QDRANT_COLLECTION``)
so the two knowledge bases never mix. Mirrors ``rag_docs/vector_store.py``'s
connection/upsert/search pattern (corporate-proxy-tolerant retries, idempotent
content-hash point IDs) with a defect-specific payload schema instead of the
documentation chunk schema.

Payload fields stored per point (one point per defect chunk — see
``chatbot/build_chunks.py``'s 5 chunk types): ``chunk_id``, ``issue_key``,
``chunk_type``, ``text``, plus the defect metadata copied onto every chunk
(``status``, ``resolution``, ``priority``, ``summary``, ``components``,
``labels``, ``environment``, ``product``, ``team``, ``fix_pattern``,
``defect_type``, ``failure_area``, ``root_cause_extracted``,
``fix_applied_extracted``, ``workaround_extracted``, ``quality_score``,
``is_fixed``, ``is_cancelled``, ``created``, ``resolved``).
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Fields indexed for exact-match filtering (all other payload fields are still
# stored and returned, just not filterable).
_KEYWORD_FIELDS = (
    "issue_key", "chunk_type", "status", "priority", "resolution",
    "fix_pattern", "defect_type", "components", "labels", "product",
    "team", "environment",
)
_BOOL_FIELDS = ("is_fixed", "is_cancelled")


class DefectVectorStore:
    """Qdrant connection + ingest/search for the defect knowledge base."""

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "defect",
        vector_dim: int = 3072,
    ) -> None:
        from qdrant_client import QdrantClient

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL / QDRANT_API_KEY are not configured (required "
                "for the defect Qdrant collection)."
            )

        self.collection_name = collection_name
        self.vector_dim = vector_dim
        logger.info(
            "[DEFECT_VS] Connecting to Qdrant collection '%s'...", collection_name
        )
        self._client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
            verify=False,              # bypass corporate SSL inspection proxy
            check_compatibility=False,
        )
        self._ensure_collection()
        self._vector_name = self._detect_vector_name()

    # ── Collection management ──
    def _ensure_collection(self) -> None:
        max_retries, retry_wait = 5, 10
        existing_names: list[str] = []
        for attempt in range(1, max_retries + 1):
            try:
                existing_names = [
                    c.name for c in self._client.get_collections().collections
                ]
                break
            except Exception as exc:
                if attempt == max_retries:
                    logger.error(
                        "[DEFECT_VS] Cannot reach Qdrant after %d attempts: %s",
                        max_retries, exc,
                    )
                    raise
                logger.warning(
                    "[DEFECT_VS] Qdrant connection failed (attempt %d/%d): %s. "
                    "Retrying in %ds...", attempt, max_retries, exc, retry_wait,
                )
                time.sleep(retry_wait)

        if self.collection_name not in existing_names:
            from qdrant_client.models import Distance, VectorParams

            logger.info("[DEFECT_VS] Creating collection '%s'...", self.collection_name)
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
        else:
            logger.info(
                "[DEFECT_VS] Collection '%s' already exists — reusing.",
                self.collection_name,
            )

        # Always (re-)ensure indexes — safe/idempotent even on an existing
        # collection, and lets a newly-added index (e.g. the full-text "text"
        # index, added after the collection already existed) get created too.
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        from qdrant_client.models import PayloadSchemaType, TextIndexParams, TextIndexType, TokenizerType

        for field in _KEYWORD_FIELDS:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # pragma: no cover - index may already exist
                logger.debug("[DEFECT_VS] Index '%s' skipped: %s", field, exc)
        for field in _BOOL_FIELDS:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.BOOL,
                )
            except Exception as exc:  # pragma: no cover
                logger.debug("[DEFECT_VS] Index '%s' skipped: %s", field, exc)

        # Full-text index on the chunk's free text — powers keyword_search()
        # (exact/lexical term matching), recovering BM25-like precision on
        # top of semantic search without a separate local BM25 index.
        try:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=25,
                    lowercase=True,
                ),
            )
        except Exception as exc:  # pragma: no cover - index may already exist
            logger.debug("[DEFECT_VS] Text index skipped: %s", exc)

        logger.info("[DEFECT_VS] Indexes ready for '%s'.", self.collection_name)

    # ── Ingestion ──
    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Deterministic UUID from a chunk's stable id, so re-adding the same
        defect (same chunk_id values) overwrites rather than duplicates."""
        digest = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()
        return str(uuid.UUID(digest[:32]))

    def _detect_vector_name(self) -> Optional[str]:
        """Some Qdrant clusters auto-assign a name to the default vector
        (e.g. "defect-vector") instead of leaving it unnamed. Detect that name
        here once so upsert/search calls address the right vector field on
        ANY cluster, old (unnamed) or new (named)."""
        try:
            vectors_cfg = self._client.get_collection(self.collection_name).config.params.vectors
        except Exception:
            return None
        if isinstance(vectors_cfg, dict):
            name = next(iter(vectors_cfg.keys()), "")
            return name or None
        return None

    def upsert_chunks(self, chunks: list[dict[str, Any]], vectors: Any) -> None:
        """Upsert pre-embedded chunks. ``vectors`` is row-aligned with ``chunks``
        (a numpy array or list of vectors)."""
        from qdrant_client.models import PointStruct

        points = []
        for chunk, vector in zip(chunks, vectors):
            vec = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            if self._vector_name:
                vec = {self._vector_name: vec}
            point_id = self._point_id(chunk.get("chunk_id", "") or str(uuid.uuid4()))
            points.append(PointStruct(id=point_id, vector=vec, payload=dict(chunk)))

        max_retries, retry_wait = 4, 15
        for attempt in range(1, max_retries + 1):
            try:
                self._client.upsert(
                    collection_name=self.collection_name, points=points, wait=True
                )
                return
            except Exception as exc:
                if attempt == max_retries:
                    logger.error(
                        "[DEFECT_VS] Upsert failed after %d attempts: %s",
                        max_retries, exc,
                    )
                    raise
                logger.warning(
                    "[DEFECT_VS] Upsert attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt, max_retries, exc, retry_wait,
                )
                time.sleep(retry_wait)

    def delete_by_issue_key(self, issue_key: str) -> None:
        """Remove all existing chunks for a defect (used before re-adding it,
        so updates never leave stale/duplicate points behind)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="issue_key", match=MatchValue(value=issue_key))]
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[DEFECT_VS] delete_by_issue_key(%s) failed: %s", issue_key, exc
            )

    # ── Retrieval ──
    def search(self, query_vector: Any, top_k: int = 60) -> list[dict[str, Any]]:
        """Return the top-K chunk payloads (each with an added ``_score``)."""
        vec = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vec,
            using=self._vector_name,
            limit=top_k,
            with_payload=True,
        )
        results: list[dict[str, Any]] = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            payload["_score"] = round(hit.score, 4)
            results.append(payload)
        return results

    def keyword_search(self, tokens: list[str], top_k: int = 60) -> list[dict[str, Any]]:
        """Return chunk payloads whose ``text`` field contains ANY of
        ``tokens`` (via the full-text index), unranked (no ``_score`` — the
        caller re-scores candidates locally with its own IDF weighting).

        This recovers exact-term-match candidates that a pure vector search
        can miss/underrank, using Qdrant's own tokenizer/full-text index
        instead of a separate local BM25 file — so it stays correct for
        chunks added after the collection was first built.
        """
        if not tokens:
            return []
        from qdrant_client.models import FieldCondition, Filter, MatchText

        try:
            points, _ = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    should=[FieldCondition(key="text", match=MatchText(text=t)) for t in tokens]
                ),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:  # pragma: no cover - text index missing/env dependent
            logger.warning("[DEFECT_VS] keyword_search failed: %s", exc)
            return []

        results: list[dict[str, Any]] = []
        for point in points:
            payload = dict(point.payload or {})
            payload["_score"] = 0.0  # no vector similarity for this path
            results.append(payload)
        return results

    def count(self) -> int:
        try:
            info = self._client.get_collection(self.collection_name)
            return int(info.points_count or 0)
        except Exception:
            return 0


_store: Optional[DefectVectorStore] = None


def get_defect_vector_store() -> DefectVectorStore:
    """Return the cached DefectVectorStore, building it on first use."""
    global _store
    if _store is None:
        import config

        _store = DefectVectorStore(
            qdrant_url=config.QDRANT_URL,
            qdrant_api_key=config.QDRANT_API_KEY,
            collection_name=config.DEFECT_QDRANT_COLLECTION,
            vector_dim=config.DEFECT_VECTOR_DIM,
        )
    return _store
