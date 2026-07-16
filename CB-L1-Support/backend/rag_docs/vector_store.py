"""
vector_store.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Stage 3 (embedding + storage) and Stage 4 (retrieval) of the RAG pipeline.

Responsibilities
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
â€¢ Connect to Qdrant Cloud via URL + API key.
â€¢ Create the collection and four payload indexes on first run
  (source_file, page_number, asset_type, sequence_id).
â€¢ Embed chunks in small batches (``EMBED_BATCH_SIZE``) using Azure OpenAI's
  text-embedding-3-small deployment, with a configurable sleep between
  batches to respect API rate limits.
â€¢ Upsert points idempotently: each point ID is a deterministic UUID derived
  from a SHA-256 hash of the chunk content, so re-ingesting the same PDF
  never creates duplicate vectors.
â€¢ Provide similarity_search() for top-K semantic retrieval with optional
  payload filtering by source_file.
â€¢ Provide fetch_by_sequence_id() used by the RAGAgent for Â±1 context
  expansion around TABLE chunks.
"""

import hashlib
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from .llm_factory import build_embeddings

logger = logging.getLogger(__name__)

# â”€â”€ Batching constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EMBED_BATCH_SIZE: int = 20   # chunks per OpenAI embedding call
EMBED_SLEEP_SECS: int = 5    # pause between batches (rate-limit safety)


class VectorStore:
    """
    Manages embedding generation and Qdrant Cloud storage/retrieval.

    Parameters
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    qdrant_url                  : str â€“ Qdrant Cloud cluster URL
    qdrant_api_key              : str â€“ Qdrant Cloud API key
    azure_openai_api_key        : str â€“ Azure OpenAI API key
    azure_openai_endpoint       : str â€“ Azure OpenAI endpoint URL
    azure_embedding_deployment  : str â€“ Deployment name for the embedding model
    azure_openai_api_version    : str â€“ Azure OpenAI API version (e.g. "2024-02-01")
    collection_name             : str â€“ Qdrant collection name (default "rag_collection")
    vector_dim                  : int â€“ embedding dimension (text-embedding-3-small = 1536)
    """

    def __init__(
        self,
        qdrant_url:                 str,
        qdrant_api_key:             str,
        azure_openai_api_key:       str,
        azure_openai_endpoint:      str,
        azure_embedding_deployment: str,
        azure_openai_api_version:   str = "2024-02-01",
        collection_name:            str = "rag_collection",
        vector_dim:                 int = 3072,
    ) -> None:
        logger.info("[VECTOR_STORE] â”€â”€ Initialising VectorStore...")

        self.collection_name = collection_name
        self.vector_dim      = vector_dim

        # â”€â”€ Qdrant Cloud client â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logger.info(f"[VECTOR_STORE] Connecting to Qdrant Cloud: {qdrant_url}")
        self._client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
            verify=False,             # bypass corporate SSL inspection proxy
            check_compatibility=False, # suppress version-mismatch warning
        )

        # â”€â”€ Azure OpenAI embeddings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logger.info(
            f"[VECTOR_STORE] Initialising Azure OpenAI embeddings "
            f"(deployment='{azure_embedding_deployment}')..."
        )
        self._embeddings = build_embeddings(
            azure_deployment=azure_embedding_deployment,
            azure_endpoint=azure_openai_endpoint,
            azure_api_key=azure_openai_api_key,
            azure_api_version=azure_openai_api_version,
            timeout=60,  # bypass corporate SSL proxy; 60s for slow proxy
        )

        self._ensure_collection()
        self._vector_name = self._detect_vector_name()
        logger.info(
            f"[VECTOR_STORE] âœ“ VectorStore ready â”‚ Collection: '{collection_name}'"
        )

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Collection management
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _detect_vector_name(self):
        """Some Qdrant clusters auto-assign a name to the default vector
        (e.g. "rag-dense-vector") instead of leaving it unnamed. Detect that
        name here once so upsert/search calls address the right vector field
        on ANY cluster, old (unnamed) or new (named)."""
        try:
            vectors_cfg = self._client.get_collection(self.collection_name).config.params.vectors
        except Exception:
            return None
        if isinstance(vectors_cfg, dict):
            name = next(iter(vectors_cfg.keys()), "")
            return name or None
        return None

    def _ensure_collection(self) -> None:
        """Creates the Qdrant collection and payload indexes if absent.
        Retries on transient corporate-proxy connection errors."""
        max_retries = 5
        retry_wait  = 10
        for attempt in range(1, max_retries + 1):
            try:
                existing_names = [
                    c.name for c in self._client.get_collections().collections
                ]
                break   # success
            except Exception as exc:
                if attempt == max_retries:
                    logger.error(
                        f"[VECTOR_STORE] âœ— Cannot reach Qdrant after {max_retries} "
                        f"attempts: {exc}"
                    )
                    raise
                logger.warning(
                    f"[VECTOR_STORE] âš  Qdrant connection failed "
                    f"(attempt {attempt}/{max_retries}): {exc}. "
                    f"Retrying in {retry_wait}s..."
                )
                time.sleep(retry_wait)

        if self.collection_name in existing_names:
            logger.info(
                f"[VECTOR_STORE] Collection '{self.collection_name}' already exists."
                "  Skipping creation â€“ existing data will be preserved (incremental mode)."
            )
            return

        logger.info(
            f"[VECTOR_STORE] Collection '{self.collection_name}' not found. Creating..."
        )
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=Distance.COSINE,
            ),
        )
        self._create_payload_indexes()
        logger.info(
            f"[VECTOR_STORE] âœ“ Collection '{self.collection_name}' created with indexes."
        )

    def _create_payload_indexes(self) -> None:
        """Creates JSON payload indexes for all indexed metadata fields."""
        logger.info("[VECTOR_STORE] Creating payload indexes...")
        index_schema: Dict[str, Any] = {
            # â”€â”€ Core â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "source_file":              PayloadSchemaType.KEYWORD,
            "pdf_page_number":          PayloadSchemaType.INTEGER,
            "asset_type":               PayloadSchemaType.KEYWORD,
            "sequence_id":              PayloadSchemaType.INTEGER,
            "chunk_type":               PayloadSchemaType.KEYWORD,
            # â”€â”€ Section â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "section_id":               PayloadSchemaType.KEYWORD,
            "section_title":            PayloadSchemaType.KEYWORD,
            "parent_path":              PayloadSchemaType.KEYWORD,
            "heading_h1":               PayloadSchemaType.KEYWORD,
            "heading_h2":               PayloadSchemaType.KEYWORD,
            "heading_h3":               PayloadSchemaType.KEYWORD,
            # â”€â”€ Table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "table_type":               PayloadSchemaType.KEYWORD,
            "table_name":               PayloadSchemaType.KEYWORD,
            "table_id":                 PayloadSchemaType.KEYWORD,
            "table_group_id":           PayloadSchemaType.KEYWORD,
            "fragment_id":              PayloadSchemaType.KEYWORD,
            "header_signature":         PayloadSchemaType.KEYWORD,
            "page_start":               PayloadSchemaType.INTEGER,
            "page_end":                 PayloadSchemaType.INTEGER,
            # â”€â”€ Widget matrix conditions (preserved from v1) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "user_feature_flag":        PayloadSchemaType.KEYWORD,
            "coexist_feature_flag":     PayloadSchemaType.KEYWORD,
            "energy_default_baseline":  PayloadSchemaType.KEYWORD,
            "eo_default_baseline":      PayloadSchemaType.KEYWORD,
            # â”€â”€ Image â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "image_id":                 PayloadSchemaType.KEYWORD,
            "image_type":               PayloadSchemaType.KEYWORD,
            "image_hash":               PayloadSchemaType.KEYWORD,
            "ocr_engine_used":          PayloadSchemaType.KEYWORD,
        }
        for field, schema_type in index_schema.items():
            try:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema_type,
                )
                logger.info(
                    "[VECTOR_STORE]   Index ready: '%s' (%s)",
                    field, schema_type.value,
                )
            except Exception as exc:
                logger.debug(
                    "[VECTOR_STORE]   Index '%s' already exists or failed: %s",
                    field, exc,
                )

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Ingestion  (embed + upload)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def ingest_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embeds all chunks and upserts them into Qdrant in batches.

        Process per batch
        â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        1. Collect chunk texts for the batch.
        2. Call OpenAI to embed them.
        3. Build PointStruct objects with full metadata payload.
        4. Upsert into Qdrant (idempotent via content-hash UUIDs).
        5. Sleep ``EMBED_SLEEP_SECS`` seconds before the next batch.

        Incremental safety: content-hash UUIDs mean re-running on the same
        PDF simply overwrites identical points â€“ no duplicates are created.
        """
        total    = len(chunks)
        batches  = [
            chunks[i : i + EMBED_BATCH_SIZE]
            for i in range(0, total, EMBED_BATCH_SIZE)
        ]
        n_batches = len(batches)

        logger.info(
            f"[VECTOR_STORE] â”€â”€ Starting ingestion â”‚ "
            f"{total} chunks, {n_batches} batches of â‰¤{EMBED_BATCH_SIZE}..."
        )

        for batch_num, batch in enumerate(batches, start=1):
            logger.info(
                f"[VECTOR_STORE] â”€â”€ Batch {batch_num}/{n_batches}: "
                f"Generating embeddings for {len(batch)} chunks..."
            )

            # Use the embedding-optimised text (natural-language descriptions
            # for TABLE rows; raw content for TEXT chunks)
            texts = [c.get("content_for_embedding") or c.get("content", "") for c in batch]
            try:
                vectors = self._embeddings.embed_documents(texts)
            except Exception as exc:
                logger.error(
                    f"[VECTOR_STORE] âœ— Embedding failed on batch {batch_num}: {exc}"
                )
                raise

            points = self._build_points(batch, vectors)

            logger.info(
                f"[VECTOR_STORE] Batch {batch_num} complete.  Uploading to Qdrant..."
            )

            # â”€â”€ Retry loop for Qdrant upsert (corporate proxy drops connections) â”€â”€
            max_retries = 4
            retry_wait  = 15   # seconds between retries
            for attempt in range(1, max_retries + 1):
                try:
                    self._client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True,
                    )
                    break   # success â€“ exit retry loop
                except Exception as upsert_exc:
                    if attempt == max_retries:
                        logger.error(
                            f"[VECTOR_STORE] âœ— Batch {batch_num} failed after "
                            f"{max_retries} attempts: {upsert_exc}"
                        )
                        raise
                    logger.warning(
                        f"[VECTOR_STORE] âš  Batch {batch_num} upload failed "
                        f"(attempt {attempt}/{max_retries}): {upsert_exc}. "
                        f"Retrying in {retry_wait}s..."
                    )
                    time.sleep(retry_wait)
            logger.info(
                f"[VECTOR_STORE] âœ“ Batch {batch_num}/{n_batches} uploaded "
                f"({len(points)} points)."
            )

            if batch_num < n_batches:
                logger.info(
                    f"[VECTOR_STORE] Pausing {EMBED_SLEEP_SECS}s "
                    "to respect API rate limits..."
                )
                time.sleep(EMBED_SLEEP_SECS)

        logger.info(
            f"[VECTOR_STORE] âœ“ Ingestion complete â”‚ "
            f"{total} chunks stored in '{self.collection_name}'."
        )

    def _build_points(
        self, batch: List[Dict[str, Any]], vectors: List[List[float]]
    ) -> List[PointStruct]:
        """Builds PointStruct objects from chunk metadata and their embedding vectors."""
        # Fields stored as indexed payload (filterable in Qdrant)
        _INDEXED = {
            "source_file", "pdf_page_number", "asset_type", "sequence_id",
            "chunk_type",
            # Section
            "section_id", "section_title", "parent_path",
            "heading_h1", "heading_h2", "heading_h3",
            # Table
            "table_type", "table_name", "table_id", "table_group_id",
            "fragment_id", "header_signature",
            "page_start", "page_end",
            # Widget matrix conditions
            "user_feature_flag", "coexist_feature_flag",
            "energy_default_baseline", "eo_default_baseline",
            # Image
            "image_id", "image_type", "image_hash", "ocr_engine_used",
        }
        # Fields always stored (not necessarily indexed, but returned in payloads)
        _ALWAYS = {
            "content_for_display",
            # Table row fields
            "row_id", "row_label", "headers", "row_index",
            "description", "product", "applicable",
            "feature_flags", "enablement_levels",
            "energy_widget", "carbon_widget",
            "eo_widget_all_utilities", "eo_widget_electricity",
            "eo_widget_gas", "eo_widget_hot_water",
            # Table fragment / full
            "raw_table_markdown", "full_table_markdown", "full_table_json",
            "column_count", "row_count",
            "is_multi_page", "continuation_confidence", "continuation_reason",
            # Image
            "short_caption", "detailed_description",
            "raw_ocr_text", "normalized_ocr_text", "ocr_confidence",
            "has_ocr_text", "has_visual_description",
            "image_description_json", "surrounding_context",
            "extraction_method",
            # Provenance
            "image_subtype",
        }

        points = []
        for chunk, vector in zip(batch, vectors):
            embed_text  = chunk.get("content_for_embedding") or chunk.get("content", "")
            source_file = chunk.get("source_file", "")
            # Include source_file prefix so chunks from different PDFs never share an ID
            point_id = VectorStore._content_hash_uuid(source_file + "||" + embed_text)

            payload: Dict[str, Any] = {}
            for f in _INDEXED:
                if f in chunk:
                    payload[f] = chunk[f]
            for f in _ALWAYS:
                if f in chunk:
                    payload[f] = chunk[f]
            payload["content_for_display"] = (
                chunk.get("content_for_display") or chunk.get("content", "")
            )
            vec = {self._vector_name: vector} if self._vector_name else vector
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))
        return points

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Retrieval
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def similarity_search(
        self,
        query:              str,
        top_k:              int           = 5,
        source_filter:      Optional[str] = None,
        asset_type_filter:  Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embeds ``query`` and returns the top-K most similar chunks.

        Parameters
        â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        query             : natural-language question
        top_k             : number of results to return
        source_filter     : restrict to chunks from this PDF filename
        asset_type_filter : restrict to "TEXT" or "TABLE_ROW" only
        """
        logger.info(
            f"[VECTOR_STORE] Similarity search (top_k={top_k}, "
            f"asset_type={asset_type_filter})..."
        )
        query_vector = self._embeddings.embed_query(query)

        must_conditions = []
        if source_filter:
            must_conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=source_filter))
            )
        if asset_type_filter:
            must_conditions.append(
                FieldCondition(key="asset_type", match=MatchValue(value=asset_type_filter))
            )

        search_filter: Optional[Filter] = (
            Filter(must=must_conditions) if must_conditions else None
        )

        for _attempt in range(2):  # retry once on transient connect failures (Zscaler proxy)
            try:
                response = self._client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=self._vector_name,
                    limit=top_k,
                    query_filter=search_filter,
                    with_payload=True,
                )
                break
            except Exception as _exc:
                if _attempt == 0 and "ConnectTimeout" in type(_exc).__name__ or "10060" in str(_exc):
                    logger.warning("[VECTOR_STORE] ConnectTimeout on similarity_search â€“ retrying once...")
                    time.sleep(2)
                    continue
                raise

        results = []
        for hit in response.points:
            payload = dict(hit.payload)
            # Normalise: always expose content_for_display (fall back to any stored content)
            payload.setdefault(
                "content_for_display",
                payload.get("content", "")
            )
            payload["_score"] = round(hit.score, 4)
            results.append(payload)

        logger.info(
            f"[VECTOR_STORE] âœ“ Retrieved {len(results)} chunks "
            f"(scores: {[r['_score'] for r in results]})."
        )
        return results

    def filtered_table_search(
        self,
        query:            str,
        table_type:       Optional[str] = None,
        asset_type:       str           = "TABLE_ROW",
        conditions:       Optional[Dict[str, str]] = None,
        top_k:            int           = 10,
        source_filter:    Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search restricted to TABLE_ROW chunks, with optional
        exact-match filters on structured metadata columns.

        Parameters
        â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        query         : natural-language question for semantic ranking
        table_type    : "feature_flags" or "conditional_matrix" (optional)
        conditions    : dict of {field_name: value} for exact metadata filters
                        e.g. {"user_feature_flag": "Enabled",
                              "energy_default_baseline": "No"}
        top_k         : max results to return
        source_filter : restrict to one PDF filename

        Returns
        â”€â”€â”€â”€â”€â”€â”€
        List of payload dicts sorted by relevance.  When all conditions are
        provided for a conditional_matrix row, only the exact-match row is
        returned (if found).
        """
        query_vector = self._embeddings.embed_query(query)

        must_conditions = [
            FieldCondition(key="asset_type", match=MatchValue(value=asset_type))
        ]
        if table_type:
            must_conditions.append(
                FieldCondition(key="table_type", match=MatchValue(value=table_type))
            )
        if source_filter:
            must_conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=source_filter))
            )
        # Apply exact metadata filters (for conditional_matrix condition matching)
        if conditions:
            for field, value in conditions.items():
                if value:
                    must_conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )

        search_filter = Filter(must=must_conditions)

        for _attempt in range(2):  # retry once on transient connect failures (Zscaler proxy)
            try:
                response = self._client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=self._vector_name,
                    limit=top_k,
                    query_filter=search_filter,
                    with_payload=True,
                )
                break
            except Exception as _exc:
                if _attempt == 0 and ("ConnectTimeout" in type(_exc).__name__ or "10060" in str(_exc)):
                    logger.warning("[VECTOR_STORE] ConnectTimeout on filtered_table_search â€“ retrying once...")
                    time.sleep(2)
                    continue
                raise

        results = []
        for hit in response.points:
            payload = dict(hit.payload)
            payload.setdefault("content_for_display", payload.get("content", ""))
            payload["_score"] = round(hit.score, 4)
            results.append(payload)

        logger.info(
            f"[VECTOR_STORE] filtered_table_search â”‚ "
            f"table_type={table_type} conditions={conditions} "
            f"â†’ {len(results)} results."
        )
        return results

    def fetch_sequence_range(
        self,
        source_file:  str,
        seq_from:     int,
        seq_to:       int,
        limit:        int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all chunks for *source_file* whose sequence_id falls in
        [seq_from, seq_to] (inclusive), sorted by sequence_id.

        Used for sequence-based context expansion: when a chunk at position N
        is retrieved, fetching seq_from=N-4 to seq_to=N+4 guarantees that
        surrounding steps or table rows that scored low in similarity search
        are still included.
        """
        from qdrant_client.models import Range
        must = [
            FieldCondition(key="source_file",  match=MatchValue(value=source_file)),
            FieldCondition(key="sequence_id",  range=Range(gte=seq_from, lte=seq_to)),
        ]
        records: List[Dict[str, Any]] = []
        offset = None
        while len(records) < limit:
            batch, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=must),
                limit=min(50, limit - len(records)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in batch:
                payload = dict(rec.payload)
                payload.setdefault("content_for_display", payload.get("content", ""))
                payload.setdefault("_score", 0.5)
                records.append(payload)
            if next_offset is None:
                break
            offset = next_offset
        records.sort(key=lambda c: c.get("sequence_id", 0))
        return records

    def fetch_by_sequence_id(
        self,
        source_file: str,
        sequence_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single chunk by (source_file, sequence_id).

        Used by the RAGAgent to expand context around TABLE chunks
        by fetching sequence_id Â± 1 neighbours.

        Returns ``None`` if no matching chunk is found.
        """
        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file),
                    ),
                    FieldCondition(
                        key="sequence_id",
                        match=MatchValue(value=sequence_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if records:
            return dict(records[0].payload)
        return None

    def fetch_table_rows_by_name(
        self,
        source_file: str,
        table_name: str,
        limit: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        Fetch ALL TABLE_ROW chunks belonging to a specific table, identified by
        (source_file, table_name).

        This is used after a semantic search finds at least one row from a table
        to ensure that rows on subsequent PDF pages (cross-page table splits
        caused by LlamaParse's per-page processing) are not missed.

        Returns rows sorted by sequence_id (document reading order).
        """
        records: List[Dict[str, Any]] = []
        offset = None

        while len(records) < limit:
            batch, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=source_file),
                        ),
                        FieldCondition(
                            key="table_name",
                            match=MatchValue(value=table_name),
                        ),
                        FieldCondition(
                            key="asset_type",
                            match=MatchValue(value="TABLE_ROW"),
                        ),
                    ]
                ),
                limit=min(100, limit - len(records)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in batch:
                payload = dict(rec.payload)
                payload.setdefault("content_for_display", payload.get("content", ""))
                payload.setdefault("_score", 0.5)   # direct fetch â€” no cosine score
                records.append(payload)
            if next_offset is None:
                break
            offset = next_offset

        records.sort(key=lambda c: c.get("sequence_id", 0))
        logger.info(
            "[VECTOR_STORE] fetch_table_rows_by_name: "
            "'%s' in '%s' â†’ %d rows.",
            table_name,
            source_file,
            len(records),
        )
        return records

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Multi-query retrieval  (used by the improved query pipeline)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def multi_similarity_search(
        self,
        queries:           List[str],
        top_k_per_query:   int           = 20,
        source_filter:     Optional[str] = None,
        asset_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run ``similarity_search`` for every query in *queries*, merge results,
        and return a deduplicated list sorted by the best score seen for each
        unique chunk.

        Parameters
        â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        queries           : list of query strings (rewritten + variants)
        top_k_per_query   : how many chunks to retrieve per query
        source_filter     : restrict to one PDF filename (optional)
        asset_type_filter : restrict to "TEXT" or "TABLE_ROW" (optional)

        Returns
        â”€â”€â”€â”€â”€â”€â”€
        Deduplicated list of chunk payloads, best score first.
        """
        best: Dict[str, Dict[str, Any]] = {}   # chunk_key â†’ highest-scoring payload

        for query in queries:
            results = self.similarity_search(
                query=query,
                top_k=top_k_per_query,
                source_filter=source_filter,
                asset_type_filter=asset_type_filter,
            )
            for chunk in results:
                key = self._chunk_key(chunk)
                if key not in best or chunk.get("_score", 0) > best[key].get("_score", 0):
                    best[key] = chunk

        merged = sorted(best.values(), key=lambda c: c.get("_score", 0), reverse=True)
        logger.info(
            f"[VECTOR_STORE] multi_similarity_search: "
            f"{len(queries)} quer(ies) â†’ {len(merged)} unique chunks."
        )
        return merged

    def multi_filtered_table_search(
        self,
        queries:         List[str],
        table_type:      Optional[str]           = None,
        conditions:      Optional[Dict[str, str]] = None,
        top_k_per_query: int                      = 10,
        source_filter:   Optional[str]            = None,
    ) -> List[Dict[str, Any]]:
        """
        Run ``filtered_table_search`` for every query in *queries*, merge
        results, and return a deduplicated list sorted by best score.

        Parameters
        â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        queries         : list of query strings
        table_type      : "feature_flags" or "conditional_matrix" (optional)
        conditions      : exact metadata filter conditions (optional)
        top_k_per_query : results per query
        source_filter   : restrict to one PDF filename (optional)

        Returns
        â”€â”€â”€â”€â”€â”€â”€
        Deduplicated list of table-row payloads, best score first.
        """
        best: Dict[str, Dict[str, Any]] = {}

        for query in queries:
            results = self.filtered_table_search(
                query=query,
                table_type=table_type,
                conditions=conditions,
                top_k=top_k_per_query,
                source_filter=source_filter,
            )
            for chunk in results:
                key = self._chunk_key(chunk)
                if key not in best or chunk.get("_score", 0) > best[key].get("_score", 0):
                    best[key] = chunk

        merged = sorted(best.values(), key=lambda c: c.get("_score", 0), reverse=True)
        logger.info(
            f"[VECTOR_STORE] multi_filtered_table_search: "
            f"{len(queries)} quer(ies) â†’ {len(merged)} unique chunks."
        )
        return merged

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # New retrieval methods (v2)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def search_images(
        self,
        query:         str,
        top_k:         int           = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search restricted to IMAGE asset_type chunks.

        The image chunks embed OCR text + VLM captions + descriptions,
        so semantic search finds relevant diagrams, screenshots, and charts.
        """
        return self.similarity_search(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
            asset_type_filter="IMAGE",
        )

    def fetch_chunks_by_section(
        self,
        source_file:   str,
        section_title: str,
        limit:         int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch ALL TEXT chunks that belong to the same section (by section_title
        and source_file), sorted by sequence_id (reading order).

        Used by RAGAgent._expand_text_context() to ensure multi-page sections
        (e.g. a step list spread across pages 3-5) are fully retrieved even
        when later pages score lower in similarity search.
        """
        if not section_title:
            return []
        must = [
            FieldCondition(key="source_file",   match=MatchValue(value=source_file)),
            FieldCondition(key="section_title", match=MatchValue(value=section_title)),
            FieldCondition(key="asset_type",    match=MatchValue(value="TEXT")),
        ]
        records: List[Dict[str, Any]] = []
        offset = None
        while len(records) < limit:
            batch, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=must),
                limit=min(100, limit - len(records)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in batch:
                payload = dict(rec.payload)
                payload.setdefault("content_for_display", payload.get("content", ""))
                payload.setdefault("_score", 0.5)
                records.append(payload)
            if next_offset is None:
                break
            offset = next_offset
        records.sort(key=lambda c: c.get("sequence_id", 0))
        logger.info(
            "[VECTOR_STORE] fetch_chunks_by_section: "
            "section='%s' in '%s' â†’ %d chunk(s).",
            section_title, source_file, len(records),
        )
        return records

    def fetch_full_table(
        self,
        table_group_id: str,
        source_file:    Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch the TABLE_FULL chunk for a given table_group_id.

        Returns the first matching chunk payload, or None if not found.
        """
        must = [
            FieldCondition(key="asset_type",      match=MatchValue(value="TABLE_FULL")),
            FieldCondition(key="table_group_id",  match=MatchValue(value=table_group_id)),
        ]
        if source_file:
            must.append(
                FieldCondition(key="source_file", match=MatchValue(value=source_file))
            )
        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if records:
            payload = dict(records[0].payload)
            payload.setdefault("content_for_display", "")
            payload.setdefault("_score", 0.5)
            return payload
        return None

    def fetch_table_group_rows(
        self,
        table_group_id: str,
        source_file:    Optional[str] = None,
        limit:          int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Fetch ALL TABLE_ROW chunks that belong to a given table_group_id.
        Useful for expanding context after finding a single row match.

        Returns rows sorted by sequence_id (reading order).
        """
        must = [
            FieldCondition(key="asset_type",     match=MatchValue(value="TABLE_ROW")),
            FieldCondition(key="table_group_id", match=MatchValue(value=table_group_id)),
        ]
        if source_file:
            must.append(
                FieldCondition(key="source_file", match=MatchValue(value=source_file))
            )
        records: List[Dict[str, Any]] = []
        offset = None
        while len(records) < limit:
            batch, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=must),
                limit=min(100, limit - len(records)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in batch:
                payload = dict(rec.payload)
                payload.setdefault("content_for_display", "")
                payload.setdefault("_score", 0.5)
                records.append(payload)
            if next_offset is None:
                break
            offset = next_offset
        records.sort(key=lambda r: r.get("sequence_id", 0))
        logger.info(
            "[VECTOR_STORE] fetch_table_group_rows: group=%s â†’ %d rows.",
            table_group_id[:8], len(records),
        )
        return records

    def hybrid_search(
        self,
        query:         str,
        top_k:         int                    = 10,
        source_filter: Optional[str]          = None,
        asset_types:   Optional[List[str]]    = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: dense vector search + local BM25 keyword scoring,
        fused with Reciprocal Rank Fusion (RRF).

        No external BM25 service required â€“ BM25 is computed in-memory over
        the results returned by a wide dense search (top_k * 5 candidates).

        Parameters
        â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        query        : natural-language question
        top_k        : final number of results to return
        source_filter: restrict to one PDF filename (optional)
        asset_types  : restrict to specific asset_type values (optional)
        """
        import math

        # â”€â”€ Step 1: Wide dense retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        must_conditions = []
        if source_filter:
            must_conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=source_filter))
            )
        if asset_types:
            from qdrant_client.models import MatchAny
            must_conditions.append(
                FieldCondition(key="asset_type", match=MatchAny(any=asset_types))
            )

        query_vector = self._embeddings.embed_query(query)
        search_filter = Filter(must=must_conditions) if must_conditions else None

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self._vector_name,
            limit=max(top_k * 5, 50),
            query_filter=search_filter,
            with_payload=True,
        )
        candidates = []
        for hit in response.points:
            payload = dict(hit.payload)
            payload["content_for_display"] = payload.get("content_for_display", "")
            payload["_dense_score"] = hit.score
            candidates.append(payload)

        if not candidates:
            return []

        # â”€â”€ Step 2: Local BM25 over candidates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        bm25_scores = self._bm25_scores(
            query,
            [c.get("content_for_display", "") or c.get("content_for_embedding", "")
             for c in candidates],
        )

        # â”€â”€ Step 3: RRF fusion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        k_rrf = 60  # standard RRF constant
        # Dense rank
        dense_sorted = sorted(
            range(len(candidates)),
            key=lambda i: candidates[i]["_dense_score"],
            reverse=True,
        )
        # BM25 rank
        bm25_sorted = sorted(
            range(len(candidates)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )
        dense_rank = {idx: rank + 1 for rank, idx in enumerate(dense_sorted)}
        bm25_rank  = {idx: rank + 1 for rank, idx in enumerate(bm25_sorted)}

        for i, cand in enumerate(candidates):
            rrf = 1 / (k_rrf + dense_rank[i]) + 1 / (k_rrf + bm25_rank[i])
            cand["_score"] = round(rrf, 6)

        results = sorted(candidates, key=lambda c: c["_score"], reverse=True)[:top_k]
        logger.info(
            "[VECTOR_STORE] hybrid_search: %d candidates â†’ %d final (RRF).",
            len(candidates), len(results),
        )
        return results

    @staticmethod
    def _bm25_scores(query: str, documents: List[str]) -> List[float]:
        """
        Simple in-memory BM25 scoring for a list of document texts.

        Uses standard BM25 parameters: k1=1.5, b=0.75.
        """
        import math
        import re

        k1, b = 1.5, 0.75
        query_terms = re.findall(r"\w+", query.lower())
        tokenized   = [re.findall(r"\w+", d.lower()) for d in documents]
        dl  = [len(doc) for doc in tokenized]
        avgdl = sum(dl) / len(dl) if dl else 1

        N   = len(tokenized)
        scores = [0.0] * N

        for term in query_terms:
            df = sum(1 for doc in tokenized if term in doc)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for i, doc in enumerate(tokenized):
                tf = doc.count(term)
                if tf == 0:
                    continue
                score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl[i] / avgdl))
                scores[i] += score
        return scores

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Helpers
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def list_source_files(self) -> List[str]:
        """Return a sorted list of distinct source_file values in the collection."""
        try:
            result = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=1,  # we just need to trigger the facet call
                with_payload=False,
                with_vectors=False,
            )
            # Use facet counting on the indexed 'source_file' keyword field
            from qdrant_client.models import FieldCondition, MatchValue, Filter
            # Scroll all points but only fetch source_file payload field
            all_files: set = set()
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    limit=100,
                    offset=offset,
                    with_payload=["source_file"],
                    with_vectors=False,
                )
                for p in points:
                    sf = p.payload.get("source_file", "")
                    if sf:
                        all_files.add(sf)
                if offset is None:
                    break
            return sorted(all_files)
        except Exception as exc:
            logger.warning("[VECTOR_STORE] Could not list source files: %s", exc)
            return []

    @staticmethod
    def _chunk_key(chunk: Dict[str, Any]) -> str:
        """
        Generate a stable deduplication key for *chunk* from its two most
        reliable unique identifiers: source_file and sequence_id.
        Falls back to a hash of ``content_for_display`` when these are absent.
        """
        source = chunk.get("source_file", "")
        seq_id = chunk.get("sequence_id", "")
        if source or seq_id:
            return f"{source}||{seq_id}"
        # Last-resort fallback: hash the display text
        import hashlib
        text = chunk.get("content_for_display") or chunk.get("content", "")
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _content_hash_uuid(content: str) -> str:
        """
        Generates a deterministic UUID string from the SHA-256 hash of
        ``content``.  Callers should prefix content with ``source_file + '||'``
        so chunks from different PDFs with identical text never share an ID.
          â€¢ Same source + same content  â†’ same ID   (idempotent upserts)
          â€¢ Different source or content â†’ unique ID
        """
        hash_hex = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return str(uuid.UUID(hash_hex[:32]))
