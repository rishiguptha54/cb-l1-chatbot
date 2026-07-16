"""
ingest.py
─────────
On-demand PDF ingestion for the document-RAG pipeline.

This is the SAME parsing → chunking pipeline used to originally build the
Qdrant knowledge base (LlamaParse → SemanticChunker → VectorStore), copied
from the source project so a PDF uploaded from the header button is
processed identically to the original offline ingestion — full section-aware
TEXT chunking plus TABLE_ROW / TABLE_FRAGMENT / TABLE_FULL table chunking
(not a simplified text-only split). See ``document_ingestor.py`` (Stage 1,
LlamaParse parsing) and ``semantic_chunker.py`` (Stage 2, chunking) for the
detailed pipeline docs.

Requires the ``LLAMA_CLOUD_API_KEY`` environment variable (free tier at
https://cloud.llamaindex.ai) for the LlamaParse cloud parsing step.

Idempotent: point IDs are a content hash of (source_file + embedding text), so
re-uploading the same or a revised PDF overwrites its previous chunks instead
of duplicating them (see ``VectorStore._build_points``).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Iterator, Optional

from .document_ingestor import DocumentIngestor
from .semantic_chunker import SemanticChunker

_ingestor: Optional[DocumentIngestor] = None
_chunker: Optional[SemanticChunker] = None


def _get_ingestor() -> DocumentIngestor:
    """Return the cached DocumentIngestor, building it (and validating
    LLAMA_CLOUD_API_KEY) on first use."""
    global _ingestor
    if _ingestor is None:
        _ingestor = DocumentIngestor()
    return _ingestor


def _get_chunker() -> SemanticChunker:
    """Return the cached SemanticChunker, building it on first use."""
    global _chunker
    if _chunker is None:
        _chunker = SemanticChunker()
    return _chunker


def ingest_pdf_stages(filename: str, pdf_bytes: bytes) -> Iterator[Dict[str, Any]]:
    """Generator form of the ingestion pipeline: yields a progress event before
    each stage, then a final ``{"stage": "done", "result": {...}}`` event.

    Progress events: ``{"stage": "parsing" | "chunking" | "embedding",
    "message": str}``. Used by the streaming API route so the UI can show
    exactly what the backend is doing; ``ingest_pdf()`` below just drains this
    generator and returns the final result for non-streaming callers.

    Raises the same exceptions as ``ingest_pdf()`` (``ValueError`` for a
    configuration/content problem, or any pipeline exception).
    """
    from .pipeline import get_vector_store

    ingestor = _get_ingestor()
    chunker = _get_chunker()

    yield {
        "stage": "parsing",
        "message": f'📄 Parsing "{filename}" via LlamaParse…',
    }
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(pdf_bytes)
        elements = ingestor.parse_pdf(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not elements:
        raise ValueError(
            "No content could be parsed from this PDF. It may be empty, "
            "corrupted, or a scanned image with no text layer."
        )

    yield {
        "stage": "chunking",
        "message": f"✂️ Chunking {len(elements)} parsed elements (text + tables)…",
    }
    chunks = chunker.chunk_elements(elements)
    if not chunks:
        raise ValueError(
            "Parsing succeeded but no chunkable content was produced (the "
            "PDF may contain only images, boilerplate headers/footers, or "
            "a table of contents)."
        )

    # Safety override: every chunk must be attributed to the uploaded
    # filename (not a stale name from a previous run), matching the original
    # ingestion CLI's behaviour.
    for chunk in chunks:
        chunk["source_file"] = filename

    pages = len({e.get("pdf_page_number", 0) for e in elements})

    yield {
        "stage": "embedding",
        "message": f"🧠 Embedding {len(chunks)} chunks and storing in Qdrant…",
    }
    # Reuses the SAME cached VectorStore/collection the query-time pipeline
    # searches (get_vector_store() -> get_agent()._vector_store, a module-level
    # singleton in pipeline.py) — so uploaded chunks are stored with the exact
    # same schema/payload fields as the existing corpus and land in the same
    # Qdrant collection (QDRANT_COLLECTION, e.g. "chatbot"), never a separate one.
    vector_store = get_vector_store()
    vector_store.ingest_chunks(chunks)

    def _count(asset_type: str) -> int:
        return sum(1 for c in chunks if c.get("asset_type") == asset_type)

    result = {
        "source_file": filename,
        "collection": vector_store.collection_name,
        "pages": pages,
        "chunks": len(chunks),
        "text_chunks": _count("TEXT"),
        "table_row_chunks": _count("TABLE_ROW"),
        "table_fragment_chunks": _count("TABLE_FRAGMENT"),
        "table_full_chunks": _count("TABLE_FULL"),
    }
    yield {"stage": "done", "result": result}


def ingest_pdf(filename: str, pdf_bytes: bytes) -> Dict[str, Any]:
    """Parse (LlamaParse) → chunk (SemanticChunker) → embed + upsert
    (VectorStore) a PDF into the document-RAG Qdrant collection.

    Non-streaming convenience wrapper around ``ingest_pdf_stages()`` — runs
    every stage and returns only the final summary: ``{"source_file",
    "collection", "pages", "chunks", "text_chunks", "table_row_chunks",
    "table_fragment_chunks", "table_full_chunks"}``.

    Raises ``ValueError`` for a configuration problem (e.g. missing
    ``LLAMA_CLOUD_API_KEY``) or when parsing/chunking yields no usable
    content; any other exception indicates a genuine pipeline failure
    (Qdrant/embedding error). The API layer maps these to clear HTTP
    responses instead of silently no-oping.
    """
    result: Dict[str, Any] = {}
    for event in ingest_pdf_stages(filename, pdf_bytes):
        if event["stage"] == "done":
            result = event["result"]
    return result

