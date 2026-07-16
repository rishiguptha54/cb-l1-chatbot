"""
models.py
─────────
Typed data models shared across the RAG pipeline.

All models use dataclasses (stdlib only) for zero extra dependencies and
easy JSON serialisation.  Every model that needs stable IDs exposes a
``make_ids()`` method that computes them deterministically from content.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# ID helpers (also used by table_parser.py and image_extractor.py)
# ─────────────────────────────────────────────────────────────────────────────

def stable_id(*parts: str, prefix: str = "") -> str:
    """Deterministic 16-char hex ID: SHA-256 of concatenated parts."""
    key = "||".join(str(p) for p in parts)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{h}" if prefix else h


# ─────────────────────────────────────────────────────────────────────────────
# PageElement
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageElement:
    """
    Normalized intermediate representation for one parsed element on a PDF page.

    Every element produced by DocumentIngestor is converted to this form
    before being handed to SemanticChunker, TableParser, or ImageExtractor.
    The ``element_type`` field governs which downstream processor handles it.
    """
    element_type: str           # TEXT | TABLE | IMAGE | OTHER
    content: str                # raw text or markdown table string
    source_file: str
    pdf_page_number: int
    page_local_order: int       # 0-based position within this page

    # ── Stable identifiers ────────────────────────────────────────────────────
    element_id: str = ""
    document_id: str = ""
    page_id: str = ""

    # ── Section context ───────────────────────────────────────────────────────
    heading_h1: str = ""
    heading_h2: str = ""
    heading_h3: str = ""
    section_title: str = ""     # deepest non-empty heading
    parent_path: str = ""       # "H1 > H2 > H3"
    section_id: str = ""        # numeric prefix, e.g. "1.2.3"

    # ── Provenance ────────────────────────────────────────────────────────────
    raw_markdown: str = ""
    extraction_method: str = "llamaparse"  # llamaparse | pdf_extract | fallback

    # ── Surrounding text snippets (for table/image context embedding) ─────────
    preceding_text: str = ""
    following_text: str = ""

    # ── Image-specific (only set when element_type == "IMAGE") ───────────────
    image_bytes: Optional[bytes] = None
    image_width: int = 0
    image_height: int = 0
    image_hash: str = ""
    bounding_box: Optional[Dict[str, float]] = None

    # ─────────────────────────────────────────────────────────────────────────
    def make_ids(self) -> None:
        """Populate document_id, page_id, element_id from source metadata."""
        self.document_id = stable_id(self.source_file, prefix="doc_")
        self.page_id = stable_id(
            self.source_file, str(self.pdf_page_number), prefix="pg_"
        )
        self.element_id = stable_id(
            self.source_file,
            str(self.pdf_page_number),
            str(self.page_local_order),
            self.element_type,
            prefix="el_",
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Return a dict compatible with the pre-model SemanticChunker interface."""
        return {
            "content": self.content,
            "element_type": self.element_type,
            "pdf_page_number": self.pdf_page_number,
            "llamaparse_page_number": self.pdf_page_number,
            "source_file": self.source_file,
            "heading_h1": self.heading_h1,
            "heading_h2": self.heading_h2,
            "heading_h3": self.heading_h3,
            "section_title": self.section_title,
            "parent_path": self.parent_path,
            "section_id": self.section_id,
            "preceding_text": self.preceding_text,
            "following_text": self.following_text,
            "element_id": self.element_id,
            "document_id": self.document_id,
            "page_id": self.page_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Table models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TableFragment:
    """
    One page-level fragment of a (possibly multi-page) table.

    Produced by TableParser.parse() for every TABLE element encountered.
    Multiple fragments may belong to the same logical table when it spans pages.
    """
    fragment_id: str
    table_id: str
    table_group_id: str
    raw_markdown: str
    headers: List[str]              # normalised (stripped) header labels
    raw_headers: List[str]          # exact header text as parsed
    rows: List[List[str]]           # data rows: list of cell lists
    page_number: int
    source_file: str

    section_title: str = ""
    parent_path: str = ""
    preceding_text: str = ""
    following_text: str = ""
    column_count: int = 0
    row_count: int = 0
    table_type: str = "generic_table"   # feature_flags | conditional_matrix | generic_table
    table_name: str = ""
    fragment_index: int = 0             # 0-based position in the page
    heading_h1: str = ""
    heading_h2: str = ""
    heading_h3: str = ""
    header_signature: str = ""          # normalised fingerprint for continuation detection

    def compute_header_signature(self) -> str:
        """Return and store a normalised fingerprint for the header row."""
        sig = "|".join(h.strip().lower() for h in self.headers)
        self.header_signature = sig
        return sig


@dataclass
class FullTable:
    """
    Reconstructed full table, potentially spanning multiple pages.

    Produced by TableReconstructor.reconstruct() from one or more TableFragments.
    Stores both the raw fragments (lossless) and the merged representation.
    """
    table_id: str
    table_group_id: str
    table_name: str
    table_type: str
    headers: List[str]
    rows: List[Dict[str, Any]]          # each row: {row_index, fragment_index, fragment_row_index, page_number, cells:[]}
    fragments: List[TableFragment]
    page_start: int
    page_end: int
    source_file: str

    section_title: str = ""
    parent_path: str = ""
    is_multi_page: bool = False
    continuation_confidence: float = 0.0
    continuation_reason: str = ""
    full_table_markdown: str = ""
    full_table_json: Optional[Dict[str, Any]] = None
    heading_h1: str = ""
    heading_h2: str = ""
    heading_h3: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    def build_markdown(self) -> str:
        """Render the merged table as a Markdown table string."""
        if not self.headers:
            return ""
        sep = "| " + " | ".join("---" for _ in self.headers) + " |"
        header_line = "| " + " | ".join(self.headers) + " |"
        row_lines: List[str] = []
        for row in self.rows:
            cells = row.get("cells", [])
            padded = (list(cells) + [""] * len(self.headers))[: len(self.headers)]
            row_lines.append("| " + " | ".join(str(c) for c in padded) + " |")
        return "\n".join([header_line, sep] + row_lines)

    def build_json(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict for Qdrant payload storage."""
        return {
            "table_id": self.table_id,
            "table_group_id": self.table_group_id,
            "table_name": self.table_name,
            "table_type": self.table_type,
            "headers": self.headers,
            "rows": self.rows,
            "page_span": [self.page_start, self.page_end],
            "source_file": self.source_file,
            "section_title": self.section_title,
            "parent_path": self.parent_path,
            "is_multi_page": self.is_multi_page,
            "continuation_confidence": round(self.continuation_confidence, 4),
            "continuation_reason": self.continuation_reason,
            "raw_fragments": [f.raw_markdown for f in self.fragments],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Image models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageAsset:
    """
    An extracted image with OCR and visual analysis payloads.

    Produced by ImageExtractor and enriched by OCRAdapter + VisualAnalyzer.
    Becomes one or more IMAGE chunks in the vector store.
    """
    image_id: str
    source_file: str
    pdf_page_number: int
    page_local_order: int
    image_hash: str

    # ── Raw image data ────────────────────────────────────────────────────────
    image_bytes: Optional[bytes] = None
    image_width: int = 0
    image_height: int = 0
    extraction_method: str = ""         # pdf_extract | page_render

    # ── OCR results ───────────────────────────────────────────────────────────
    ocr_engine_used: str = ""
    raw_ocr_text: str = ""
    normalized_ocr_text: str = ""
    ocr_confidence: float = 0.0
    has_ocr_text: bool = False

    # ── VLM analysis results ──────────────────────────────────────────────────
    image_type: str = "other"           # diagram|chart|screenshot|table_image|flowchart|ui|photo|other
    image_subtype: str = ""
    short_caption: str = ""
    detailed_description: str = ""
    image_description_json: Optional[Dict[str, Any]] = None
    has_visual_description: bool = False

    # ── Context from surrounding document ────────────────────────────────────
    surrounding_context: str = ""
    section_title: str = ""
    parent_path: str = ""
    heading_h1: str = ""
    heading_h2: str = ""
    heading_h3: str = ""
