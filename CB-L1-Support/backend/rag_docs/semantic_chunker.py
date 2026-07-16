"""
semantic_chunker.py
───────────────────
Stage 2 of the RAG pipeline.

Chunking strategy
─────────────────
TEXT elements
  1. MarkdownHeaderTextSplitter  →  splits on #/##/### headings.
  2. If a section is ≤ SECTION_FULL_MAX_CHARS (1500 chars), store the whole
     section as one ``section_full`` chunk.
  3. Longer sections split by RecursiveCharacterTextSplitter (700, 150)
     and labelled ``section_part``.

TABLE elements  (THREE representations, all indexed — lossless guarantee)
  TABLE_FRAGMENT  – one chunk per page-level fragment (raw markdown)
  TABLE_ROW       – one chunk per data row (natural-language embedding)
  TABLE_FULL      – one chunk per reconstructed full table (covers multi-page)

IMAGE elements
  IMAGE           – one chunk per extracted image (OCR + VLM description)

Changes from v1
───────────────
• TABLE elements now emit TABLE_FRAGMENT, TABLE_ROW *and* TABLE_FULL chunks.
• Multi-page table continuation detection via TableContinuationDetector.
• IMAGE elements handled via new _chunk_image() method.
• New section context fields (heading_h1/h2/h3) propagated to all chunks.
• preceding_text / following_text attached to TABLE and IMAGE chunks.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .models import FullTable, TableFragment
from .table_parser import (
    TableContinuationDetector,
    TableParser,
    TableReconstructor,
)

logger = logging.getLogger(__name__)

# Sections with ≤ this many characters are stored as a single chunk
SECTION_FULL_MAX_CHARS: int = 3500


# ── Module-level helpers ──────────────────────────────────────────────────────

def _slugify(text: str, max_len: int = 80) -> str:
    """Stable slug for row_id values."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")[:max_len]


def _extract_section_id(title: str) -> "tuple[str, str]":
    """
    Split "1.1.1 Feature Flag Options" → ("1.1.1", "Feature Flag Options").
    Returns ("", title) when no leading numeric section ID is found.
    """
    m = re.match(r"^(\d[\d.]*)\s+(.*)", title.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return "", title.strip()


# Regex to detect Table-of-Contents entries:
# - Primarily dots/periods (≥5) optionally followed by a page number
# - OR just a bare page number (1-4 digits)
_TOC_LINE_RE = re.compile(
    r"^[\s.·…\-_]{5,}\s*\d{0,4}\s*$"  # lines of dots + optional number
    r"|^\d{1,4}\s*$",                    # bare page number
    re.MULTILINE,
)


def _is_toc_entry(content: str) -> bool:
    """
    Return True if *content* looks like a Table-of-Contents entry.

    ToC entries after header splitting typically contain only:
      - Dot leaders: "...........30"
      - Or very short text that is just a page number
      - Or multiple ToC lines stacked together

    Real content has actual sentences/paragraphs/bullet points.
    """
    # Strip the content and check if it's all ToC-like lines
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    if not lines:
        return True  # empty = skip

    toc_lines = 0
    for line in lines:
        # Line is mostly dots/periods (>50% of chars are dots) + optional number
        dot_count = line.count('.') + line.count('·') + line.count('…')
        if dot_count > len(line) * 0.4:
            toc_lines += 1
        elif _TOC_LINE_RE.match(line):
            toc_lines += 1

    # If >70% of lines are ToC-like, it's a ToC entry
    return toc_lines / len(lines) > 0.7


# Regex for repeated document headers/footers that appear on every page
_PAGE_HEADER_RE = re.compile(
    r"^(bas[ei]line\s+app\s+configuration\s+guide|"
    r"\d{1,3}\s+31-\d{2,5}[\w-]*|"
    r"31-\d{2,5}[\w-]*\s*\d{0,3}|"
    r"honeywell\s+forge\s+sustainability|"
    r"configuration\s+guide)$",
    re.IGNORECASE,
)


def _is_page_header(text: str) -> bool:
    """Return True if text is a repeated page header/footer (document title, page numbers)."""
    clean = text.strip()
    # Single-line check
    if _PAGE_HEADER_RE.match(clean):
        return True
    # Multi-line: all lines are headers/page numbers
    lines = [l.strip() for l in clean.splitlines() if l.strip()]
    if not lines:
        return True
    return all(
        _PAGE_HEADER_RE.match(l) or re.match(r'^\d{1,3}$', l)
        for l in lines
    )


# ── Main class ────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Converts raw document elements into enriched, sequence-stamped chunks.

    Parameters
    ──────────
    chunk_size    : int  – max chars per section_part chunk  (default 700)
    chunk_overlap : int  – overlap between section_part chunks (default 150)
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        self._char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._sequence_counter: int = 0
        # Running section tracker – updated as TEXT elements are read
        self._current_section: "dict[str, str]" = {
            "h1": "", "h2": "", "h3": "",
            "section_id": "", "section_title": "", "parent_path": "",
        }
        # TableParser for new multi-representation table handling
        self._table_parser     = TableParser()
        self._table_detector   = TableContinuationDetector()
        self._table_reconstructor = TableReconstructor()
        logger.info(
            f"[CHUNKER] SemanticChunker initialised │ "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def chunk_elements(
        self, elements: "list[dict]"
    ) -> "list[dict]":
        """
        Main entry point.  Returns a flat, ordered list of chunk dicts.

        Processing order
        ────────────────
        1. TEXT elements   → merged by section across pages, then chunked
        2. TABLE elements  → TABLE_FRAGMENT + TABLE_ROW chunks (in document order)
                             Then TABLE_FULL chunks appended at end

        Each chunk contains at minimum:
            content_for_embedding  str  – text used to create the vector
            content_for_display    str  – text shown to the LLM
            asset_type             str  – TEXT | TABLE_ROW | TABLE_FRAGMENT | TABLE_FULL | IMAGE
            chunk_type             str  – section_full | section_part | table_row |
                                          table_fragment | table_full | image_chunk | table_fallback
            source_file            str  – origin PDF filename
            pdf_page_number        int  – true PDF page number
            sequence_id            int  – global position in document
            section_id             str  – e.g. "1.1.1"
            section_title          str  – e.g. "Feature Flag Options"
            parent_path            str  – "H1 > H2 > H3"
        """
        logger.info(
            f"[CHUNKER] ── Starting chunking of {len(elements)} elements..."
        )
        text_chunks:       "list[dict]" = []    # TEXT chunks (in order)
        frag_row_chunks:   "list[dict]" = []    # TABLE_FRAGMENT + TABLE_ROW (in order)
        all_fragments:     "list[TableFragment]" = []  # collected for TABLE_FULL pass

        # ── Merge consecutive TEXT elements from the same section ──────────────
        # This ensures multi-page procedural steps/sections are chunked together
        # instead of being split at page boundaries.
        merged_text_elements = self._merge_text_elements(elements)

        for element in merged_text_elements:
            etype = element.get("element_type", "TEXT")

            if etype == "TABLE":
                # Skip tables that look like UI screenshots (file browsers,
                # navigation menus, dialog boxes) — they add noise.
                if self._is_ui_screenshot_table(element.get("content", "")):
                    logger.debug(
                        "[CHUNKER] Skipping UI-screenshot table on page %d.",
                        element.get("pdf_page_number", 0),
                    )
                    continue
                frag, frchunks = self._chunk_table_new(element)
                frag_row_chunks.extend(frchunks)
                if frag is not None:
                    all_fragments.append(frag)

            elif etype == "IMAGE":
                pass  # image extraction disabled

            else:
                text_chunks.extend(self._chunk_text(element))

        # ── TABLE_FULL pass ───────────────────────────────────────────────────
        full_table_chunks: "list[dict]" = []
        if all_fragments:
            groups    = self._table_detector.detect_groups(all_fragments)
            full_tables = self._table_reconstructor.reconstruct(groups)
            for ft in full_tables:
                full_table_chunks.extend(self._chunk_full_table(ft))

        all_chunks = text_chunks + frag_row_chunks + full_table_chunks

        text_c  = sum(1 for c in all_chunks if c["asset_type"] == "TEXT")
        row_c   = sum(1 for c in all_chunks if c["asset_type"] == "TABLE_ROW")
        frag_c  = sum(1 for c in all_chunks if c["asset_type"] == "TABLE_FRAGMENT")
        full_c  = sum(1 for c in all_chunks if c["asset_type"] == "TABLE_FULL")
        logger.info(
            "[CHUNKER] ✓ Chunking complete │ %d elements → %d chunks "
            "(%d text, %d table_rows, %d table_frags, %d table_full).",
            len(elements), len(all_chunks),
            text_c, row_c, frag_c, full_c,
        )
        return all_chunks

    def reset_sequence(self) -> None:
        """Reset the global sequence counter to 0."""
        self._sequence_counter = 0
        logger.info("[CHUNKER] Sequence counter reset to 0.")

    # ──────────────────────────────────────────────────────────────────────────
    # Multi-page TEXT merging
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_text_elements(elements: "list[dict]") -> "list[dict]":
        """
        Merge consecutive TEXT elements that belong to the same parent section.

        When a procedure (e.g. "1.2.3 Download Data") contains sub-steps
        (Step 1, Step 2, Step 3) across pages, LlamaParse returns them as
        separate TEXT elements with different section_titles.  This method
        uses the **parent heading (H2 or H1)** as the merge key so that all
        sub-steps are chunked together, preserving procedural continuity.

        TABLE elements are passed through unchanged and act as section-break
        boundaries (a TABLE between two TEXT elements prevents merging).

        Returns a new list of elements (TEXT elements merged, TABLE elements intact).
        """
        if not elements:
            return []

        merged: "list[dict]" = []
        pending_text: "Optional[dict]" = None

        def _parent_key(el: "dict") -> str:
            """Key that identifies a logical parent section for merging.

            Uses H2 (or H1 if no H2) as the merge boundary.  This means
            all sub-headings (H3, Step N) under the same H2 get merged.
            """
            h2 = el.get("heading_h2") or ""
            h1 = el.get("heading_h1") or ""
            return h2 or h1 or el.get("section_title", "")

        for el in elements:
            etype = el.get("element_type", "TEXT")

            if etype != "TEXT":
                # Flush pending text before adding the non-text element
                if pending_text is not None:
                    merged.append(pending_text)
                    pending_text = None
                merged.append(el)
                continue

            # TEXT element: try to merge with pending
            if pending_text is None:
                pending_text = dict(el)  # shallow copy
                continue

            # Merge if same parent section and same source_file
            same_parent = (
                _parent_key(el) == _parent_key(pending_text)
                and _parent_key(el) != ""
                and el.get("source_file") == pending_text.get("source_file")
            )

            if same_parent:
                # Append content (with page break marker for display)
                pending_text["content"] = (
                    pending_text["content"] + "\n\n" + el["content"]
                )
                # Track page range
                if "page_end" not in pending_text:
                    pending_text["page_end"] = el["pdf_page_number"]
                else:
                    pending_text["page_end"] = max(
                        pending_text["page_end"], el["pdf_page_number"]
                    )
            else:
                # Different section: flush pending and start new
                merged.append(pending_text)
                pending_text = dict(el)

        # Flush last pending
        if pending_text is not None:
            merged.append(pending_text)

        n_orig_text = sum(1 for e in elements if e.get("element_type") == "TEXT")
        n_merged_text = sum(1 for e in merged if e.get("element_type") == "TEXT")
        if n_merged_text < n_orig_text:
            logger.info(
                "[CHUNKER] Merged %d TEXT elements → %d (multi-page sections joined).",
                n_orig_text, n_merged_text,
            )
        return merged

    # ──────────────────────────────────────────────────────────────────────────
    # Section tracker
    # ──────────────────────────────────────────────────────────────────────────

    def _update_section_from_text(self, text: str) -> None:
        """
        Scan markdown headers in *text* and update the running section tracker.
        Called before chunking each TEXT element so TABLE elements that follow
        inherit the correct section context.
        """
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("### "):
                self._current_section["h3"] = stripped[4:].strip()
                self._rebuild_section_meta()
            elif stripped.startswith("## "):
                self._current_section["h2"] = stripped[3:].strip()
                self._current_section["h3"] = ""
                self._rebuild_section_meta()
            elif stripped.startswith("# "):
                self._current_section["h1"] = stripped[2:].strip()
                self._current_section["h2"] = ""
                self._current_section["h3"] = ""
                self._rebuild_section_meta()

    def _rebuild_section_meta(self) -> None:
        h1 = self._current_section["h1"]
        h2 = self._current_section["h2"]
        h3 = self._current_section["h3"]
        active = h3 or h2 or h1
        sec_id, sec_title = _extract_section_id(active)
        self._current_section["section_id"]    = sec_id
        self._current_section["section_title"] = sec_title or active
        self._current_section["parent_path"]   = " > ".join(p for p in [h1, h2, h3] if p)

    def _section_snapshot(self) -> "dict[str, str]":
        return {
            "section_id":    self._current_section["section_id"],
            "section_title": self._current_section["section_title"],
            "parent_path":   self._current_section["parent_path"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TEXT path
    # ──────────────────────────────────────────────────────────────────────────

    def _chunk_text(self, element: "dict") -> "list[dict]":
        """
        Chunk a TEXT element.  Keeps heading + all body content together as
        ONE chunk.  Only splits if the content exceeds SECTION_FULL_MAX_CHARS.

        When split is needed, each part still carries the section context so
        the LLM knows which section it belongs to.
        """
        source_file = element["source_file"]
        page_number = element["pdf_page_number"]
        page_end    = element.get("page_end", page_number)
        raw_text    = element["content"]

        # Update section tracker so TABLE elements that follow get the right section
        self._update_section_from_text(raw_text)

        # Skip empty / noise content
        content = raw_text.strip()
        if not content:
            return []

        # Skip Table-of-Contents entries
        if _is_toc_entry(content):
            return []

        # Skip repeated page headers / footers and content with < 50 chars of
        # real text (just headings like "# Output Systems" with no body content)
        content_stripped = re.sub(r'#+ ', '', content).strip()
        if len(content_stripped) < 50 or _is_page_header(content_stripped):
            return []

        # Extract section metadata from the content's headings
        h1 = self._current_section["h1"]
        h2 = self._current_section["h2"]
        h3 = self._current_section["h3"]
        active = h3 or h2 or h1
        sec_id, sec_title = _extract_section_id(active)
        parent_path = " > ".join(p for p in [h1, h2, h3] if p) or source_file

        # Page info: single page or range
        page_info = f"{page_number}" if page_number == page_end else f"{page_number}-{page_end}"

        # Build embedding text: rich semantic context for vector search
        # Include section hierarchy + source so searches like "how to upload data"
        # match content in the "Data Upload" section even if exact words differ
        embed_parts = []
        if sec_title:
            embed_parts.append(f"Section: {sec_title}.")
        if parent_path and parent_path != sec_title and parent_path != source_file:
            embed_parts.append(f"Path: {parent_path}.")
        embed_parts.append(f"Source: {source_file}, Page {page_info}.")
        embed_parts.append(content)
        embed_text = "\n".join(embed_parts)

        chunks: "list[dict]" = []

        if len(content) <= SECTION_FULL_MAX_CHARS:
            # Store as one chunk — heading + all body together
            self._sequence_counter += 1
            chunks.append({
                "content_for_embedding": embed_text,
                "content_for_display":   content,
                "asset_type":      "TEXT",
                "chunk_type":      "section_full",
                "source_file":     source_file,
                "pdf_page_number": page_number,
                "page_end":        page_end,
                "page_info":       page_info,
                "sequence_id":     self._sequence_counter,
                "section_id":      sec_id,
                "section_title":   sec_title or active,
                "parent_path":     parent_path,
                "heading_h1":      h1,
                "heading_h2":      h2,
                "heading_h3":      h3,
            })
        else:
            # Content too large — split by character limit but keep section context
            for sub in self._char_splitter.split_text(content):
                sub = sub.strip()
                if not sub:
                    continue
                # Build per-part embedding with section context
                part_embed_parts = []
                if sec_title:
                    part_embed_parts.append(f"Section: {sec_title}.")
                if parent_path and parent_path != sec_title and parent_path != source_file:
                    part_embed_parts.append(f"Path: {parent_path}.")
                part_embed_parts.append(f"Source: {source_file}, Page {page_info}.")
                part_embed_parts.append(sub)
                part_embed = "\n".join(part_embed_parts)

                self._sequence_counter += 1
                chunks.append({
                    "content_for_embedding": part_embed,
                    "content_for_display":   sub,
                    "asset_type":      "TEXT",
                    "chunk_type":      "section_part",
                    "source_file":     source_file,
                    "pdf_page_number": page_number,
                    "page_end":        page_end,
                    "page_info":       page_info,
                    "sequence_id":     self._sequence_counter,
                    "section_id":      sec_id,
                    "section_title":   sec_title or active,
                    "parent_path":     parent_path,
                    "heading_h1":      h1,
                    "heading_h2":      h2,
                    "heading_h3":      h3,
                })

        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    # TABLE path – new three-representation approach
    # ──────────────────────────────────────────────────────────────────────────

    def _chunk_table_new(
        self, element: "dict"
    ) -> "tuple[Optional[TableFragment], list[dict]]":
        """
        Parse a TABLE element and store the ENTIRE table as ONE chunk.
        The complete table markdown is preserved in content_for_display,
        and a rich semantic summary is generated for content_for_embedding.

        Returns (fragment_or_None, [chunks]).
        TABLE_FULL chunks are emitted separately after all fragments are collected.
        """
        source_file = element["source_file"]
        page_number = element["pdf_page_number"]
        sec         = self._section_snapshot()
        prec        = element.get("preceding_text", "")
        foll        = element.get("following_text", "")
        table_md    = element["content"]

        frag = self._table_parser.parse(
            element,
            fragment_index=0,
            preceding_text=prec,
            following_text=foll,
        )

        chunks: "list[dict]" = []

        if frag is None:
            # Parser failed → store whole table as one chunk
            chunks.extend(self._chunk_table_as_whole(element))
            return None, chunks

        # ── Build rich embedding text for the whole table ─────────────────────
        headers, data_rows = self._parse_markdown_table(table_md)
        table_type = self._detect_table_type(
            headers or frag.raw_headers,
            sec.get("section_title", "")
        )
        table_name = self._get_table_name(table_type, sec.get("section_title", ""))

        embed_text = self._build_whole_table_embedding(
            headers=headers or frag.raw_headers,
            data_rows=data_rows,
            table_name=table_name,
            table_type=table_type,
            section_title=sec.get("section_title", ""),
            page_number=page_number,
        )

        # ── Store entire table as ONE chunk ───────────────────────────────────
        self._sequence_counter += 1
        table_chunk: "dict" = {
            "content_for_embedding": embed_text,
            "content_for_display":   table_md,
            "asset_type":            "TABLE_ROW",
            "chunk_type":            "table_whole",
            "source_file":           source_file,
            "pdf_page_number":       page_number,
            "sequence_id":           self._sequence_counter,
            "raw_table_markdown":    table_md,
            "table_id":              frag.table_id,
            "table_group_id":        frag.table_group_id,
            "fragment_id":           frag.fragment_id,
            "table_name":            table_name,
            "table_type":            table_type,
            "header_signature":      frag.header_signature,
            "column_count":          frag.column_count,
            "row_count":             frag.row_count,
            "headers":               headers or frag.raw_headers,
            "heading_h1":            frag.heading_h1,
            "heading_h2":            frag.heading_h2,
            "heading_h3":            frag.heading_h3,
            **sec,
        }
        chunks.append(table_chunk)

        # Also store as TABLE_FRAGMENT for the TABLE_FULL reconstruction pass
        frag_embed = embed_text[:500]  # reuse same embedding summary
        self._sequence_counter += 1
        frag_chunk: "dict" = {
            "content_for_embedding": frag_embed,
            "content_for_display":   frag.raw_markdown,
            "asset_type":            "TABLE_FRAGMENT",
            "chunk_type":            "table_fragment",
            "source_file":           source_file,
            "pdf_page_number":       page_number,
            "sequence_id":           self._sequence_counter,
            "raw_table_markdown":    frag.raw_markdown,
            "table_id":              frag.table_id,
            "table_group_id":        frag.table_group_id,
            "fragment_id":           frag.fragment_id,
            "table_name":            frag.table_name,
            "table_type":            frag.table_type,
            "header_signature":      frag.header_signature,
            "column_count":          frag.column_count,
            "row_count":             frag.row_count,
            "heading_h1":            frag.heading_h1,
            "heading_h2":            frag.heading_h2,
            "heading_h3":            frag.heading_h3,
            **sec,
        }
        chunks.append(frag_chunk)

        return frag, chunks

    def _chunk_table_as_whole(self, element: "dict") -> "list[dict]":
        """Fallback: store entire table as one chunk when parser fails."""
        source_file = element["source_file"]
        page_number = element["pdf_page_number"]
        table_md    = element["content"]
        sec         = self._section_snapshot()

        headers, data_rows = self._parse_markdown_table(table_md)
        table_type = self._detect_table_type(headers, sec.get("section_title", ""))
        table_name = self._get_table_name(table_type, sec.get("section_title", ""))

        embed_text = self._build_whole_table_embedding(
            headers=headers,
            data_rows=data_rows,
            table_name=table_name,
            table_type=table_type,
            section_title=sec.get("section_title", ""),
            page_number=page_number,
        )

        self._sequence_counter += 1
        return [{
            "content_for_embedding": embed_text,
            "content_for_display":   table_md,
            "asset_type":      "TABLE_ROW",
            "chunk_type":      "table_whole",
            "source_file":     source_file,
            "pdf_page_number": page_number,
            "sequence_id":     self._sequence_counter,
            "table_name":      table_name,
            "table_type":      table_type,
            "row_id":          f"table_p{page_number}",
            "row_label":       table_name,
            "headers":         headers,
            "row_index":       0,
            **sec,
        }]

    @staticmethod
    def _build_whole_table_embedding(
        headers: "list[str]",
        data_rows: "list[list[str]]",
        table_name: str,
        table_type: str,
        section_title: str,
        page_number: int,
    ) -> str:
        """
        Build a rich semantic embedding for an entire table.
        
        Instead of just listing headers and a sample, this creates a natural
        language description that captures WHAT the table is about, WHAT data
        it contains, and HOW it relates to the document context.
        """
        header_str = " | ".join(headers) if headers else "unknown columns"
        n_rows = len(data_rows)

        # Build a semantic summary
        parts = []

        # What is this table about?
        if table_type == "feature_flags":
            parts.append(
                f"This table lists feature flag configuration options for the {section_title} section. "
                f"It has {n_rows} rows with columns: {header_str}."
            )
            # Summarize key flags
            for row in data_rows[:6]:
                cells = [str(c).strip() for c in row if str(c).strip()]
                if cells:
                    parts.append(f"Row: {' — '.join(cells[:3])}")

        elif table_type == "conditional_matrix":
            parts.append(
                f"This is a widget configuration matrix showing how CEM Dashboard widgets "
                f"behave under different feature flag combinations. "
                f"Columns: {header_str}. Total {n_rows} configurations."
            )
            for row in data_rows[:5]:
                cells = [str(c).strip() for c in row]
                row_desc = " | ".join(f"{h}: {c}" for h, c in zip(headers, cells) if c.strip())
                if row_desc:
                    parts.append(f"Config: {row_desc}")

        else:
            # Generic table — describe content semantically
            parts.append(
                f"Table in section '{section_title}' on page {page_number}. "
                f"Columns: {header_str}. Contains {n_rows} data rows."
            )
            # Include ALL rows as natural language for complete semantic coverage
            for i, row in enumerate(data_rows):
                cells = [str(c).strip() for c in row]
                row_desc = " | ".join(
                    f"{h}: {c}" for h, c in zip(headers, cells) if c.strip()
                )
                if row_desc:
                    parts.append(f"Row {i+1}: {row_desc}")

        return "\n".join(parts)

    def _chunk_full_table(self, full: FullTable) -> "list[dict]":
        """Emit one TABLE_FULL chunk for a reconstructed FullTable."""
        # Build rich semantic embedding for the full reconstructed table
        embed_parts = []
        embed_parts.append(
            f"Complete table: {full.table_name}. "
            f"Type: {full.table_type}. "
            f"Section: {full.section_title}. "
            f"Columns: {' | '.join(full.headers)}. "
            f"Total rows: {len(full.rows)}. "
            f"Pages: {full.page_start}–{full.page_end}."
        )
        # Include all rows as natural language for full semantic coverage
        for i, row in enumerate(full.rows):
            cells = row.get("cells", [])
            row_desc = " | ".join(
                f"{h}: {c}" for h, c in zip(full.headers, cells)
                if str(c).strip()
            )
            if row_desc:
                embed_parts.append(f"Row {i+1}: {row_desc}")
        embed_text = "\n".join(embed_parts)

        display_text = full.full_table_markdown or embed_text

        self._sequence_counter += 1
        chunk: "dict" = {
            "content_for_embedding": embed_text,
            "content_for_display":   display_text,
            "asset_type":            "TABLE_FULL",
            "chunk_type":            "table_full",
            "source_file":           full.source_file,
            "pdf_page_number":       full.page_start,
            "page_start":            full.page_start,
            "page_end":              full.page_end,
            "sequence_id":           self._sequence_counter,
            "table_id":              full.table_id,
            "table_group_id":        full.table_group_id,
            "table_name":            full.table_name,
            "table_type":            full.table_type,
            "is_multi_page":         full.is_multi_page,
            "continuation_confidence": full.continuation_confidence,
            "continuation_reason":   full.continuation_reason,
            "full_table_markdown":   full.full_table_markdown,
            "full_table_json":       json.dumps(full.full_table_json) if full.full_table_json else "",
            "headers":               full.headers,
            "column_count":          len(full.headers),
            "row_count":             len(full.rows),
            "heading_h1":            full.heading_h1,
            "heading_h2":            full.heading_h2,
            "heading_h3":            full.heading_h3,
            "section_title":         full.section_title,
            "parent_path":           full.parent_path,
            "section_id":            "",
        }
        logger.debug(
            "[CHUNKER] TABLE_FULL seq=%d │ %s │ rows=%d │ multi=%s",
            self._sequence_counter, full.table_name, len(full.rows), full.is_multi_page,
        )
        return [chunk]

    # ──────────────────────────────────────────────────────────────────────────
    # TABLE path (legacy – still used for TABLE_ROW emission)
    # ──────────────────────────────────────────────────────────────────────────

    def _chunk_table(self, element: "dict") -> "list[dict]":
        source_file = element["source_file"]
        page_number = element["pdf_page_number"]
        table_md    = element["content"]

        sec = self._section_snapshot()   # inherit from nearest preceding heading

        headers, data_rows = self._parse_markdown_table(table_md)

        # ── Fallback: store whole table if parsing fails ──────────────────────
        if not headers or not data_rows:
            logger.warning(
                f"[CHUNKER] Could not parse table on page {page_number} of "
                f"'{source_file}' – storing as one fallback chunk."
            )
            self._sequence_counter += 1
            return [{
                "content_for_embedding": table_md,
                "content_for_display":   table_md,
                "asset_type":      "TABLE_ROW",
                "chunk_type":      "table_fallback",
                "source_file":     source_file,
                "pdf_page_number": page_number,
                "sequence_id":     self._sequence_counter,
                "table_name":      sec.get("section_title") or "Table",
                "table_type":      "generic_table",
                "row_id":          f"row_fallback_p{page_number}",
                "row_label":       f"Table on page {page_number}",
                "headers":         [],
                "row_index":       0,
                **sec,
            }]

        table_type = self._detect_table_type(headers, sec.get("section_title", ""))
        table_name = self._get_table_name(table_type, sec.get("section_title", ""))
        header_str = " | ".join(headers)
        chunks: "list[dict]" = []

        for row_idx, row in enumerate(data_rows):
            cells   = [str(c) for c in row]
            row_str = " | ".join(cells)

            content_for_display = (
                f"[Table Headers: {header_str}]\n"
                f"[Row {row_idx + 1}]: {row_str}"
            )
            row_meta = self._build_row_metadata(headers, cells, table_type, row_idx)
            content_for_embedding = self._build_embedding_text(
                headers, cells, table_type, table_name, row_meta
            )

            self._sequence_counter += 1
            chunk: "dict" = {
                "content_for_embedding": content_for_embedding,
                "content_for_display":   content_for_display,
                "asset_type":      "TABLE_ROW",
                "chunk_type":      "table_row",
                "source_file":     source_file,
                "pdf_page_number": page_number,
                "sequence_id":     self._sequence_counter,
                "table_name":      table_name,
                "table_type":      table_type,
                "row_index":       row_idx,
                "headers":         headers,
                **sec,
                **row_meta,
            }
            chunks.append(chunk)

        logger.debug(
            f"[CHUNKER] Table page {page_number} │ type={table_type} │ "
            f"{len(headers)} cols, {len(data_rows)} rows → {len(chunks)} chunks "
            f"(seq {chunks[0]['sequence_id']}–{chunks[-1]['sequence_id']})."
        )
        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    # Table helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_ui_screenshot_table(table_md: str) -> bool:
        """
        Return True if a Markdown table looks like OCR'd UI screenshot content
        (file browsers, navigation panels, dialog boxes) rather than real data.

        Heuristics:
        - Contains UI-specific keywords in headers/cells (e.g. "Browse", "Cancel",
          "Upload", "Drag and drop", "Provisioned", "Copy System Guid")
        - Very short cells that look like button labels
        - Contains file paths or folder navigation patterns
        """
        lines = table_md.strip().splitlines()
        if len(lines) < 2:
            return False

        all_text = table_md.lower()

        # Strong UI signals — if multiple present, likely a screenshot table
        ui_signals = [
            "browse dbfs", "upload data to dbfs", "copy system guid",
            "provisioning status", "drag and drop", "cancel",
            "select catalog", "workspace", "marketplace",
            "sql editor", "sql warehouses", "data engineering",
            "playground", "job runs", "compute",
            "items per page", "go to page",
        ]
        signal_count = sum(1 for s in ui_signals if s in all_text)
        if signal_count >= 2:
            return True

        # If table has columns that look like OS file browser
        file_browser_signals = [
            "file size", "date of upload", "provisioned",
            "copy path", "spark api format", "file api format",
        ]
        if sum(1 for s in file_browser_signals if s in all_text) >= 2:
            return True

        return False

    @staticmethod
    def _detect_table_type(headers: "list[str]", section_title: str) -> str:
        """Classify table as feature_flags, conditional_matrix, or generic_table."""
        h_low = " ".join(h.lower() for h in headers)
        s_low = section_title.lower()

        # Widget matrix – section title is the most reliable signal
        widget_section_kw = ["visualiz", "widget", "dashboard", "coexist",
                              "eo widget", "energy widget", "carbon widget"]
        if any(kw in s_low for kw in widget_section_kw):
            return "conditional_matrix"
        widget_header_kw = ["energy widget", "carbon widget", "eo widget",
                             "coexist", "energy default", "eo default"]
        if sum(1 for kw in widget_header_kw if kw in h_low) >= 2:
            return "conditional_matrix"

        # Feature flags
        flag_section_kw = ["feature flag", "flag option", "enable flag"]
        if any(kw in s_low for kw in flag_section_kw):
            return "feature_flags"
        flag_header_kw = ["flag", "applicable", "org level", "site level",
                          "user level", "product", "enablement"]
        if sum(1 for kw in flag_header_kw if kw in h_low) >= 2:
            return "feature_flags"

        return "generic_table"

    @staticmethod
    def _get_table_name(table_type: str, section_title: str) -> str:
        if table_type == "feature_flags":
            return "Feature Flag Options"
        if table_type == "conditional_matrix":
            return "CEM Widgets Visualization from Baseline App Data"
        return section_title or "Table"

    @staticmethod
    def _build_row_metadata(
        headers: "list[str]", cells: "list[str]",
        table_type: str, row_idx: int,
    ) -> "dict":
        """Return row_id, row_label, and structured fields for this row."""
        hc: "dict[str, str]" = {
            h.strip().lower(): (cells[i].strip() if i < len(cells) else "")
            for i, h in enumerate(headers)
        }
        if table_type == "feature_flags":
            return SemanticChunker._feature_flag_row_meta(headers, cells, hc, row_idx)
        if table_type == "conditional_matrix":
            return SemanticChunker._widget_matrix_row_meta(headers, cells, hc, row_idx)
        first_val = cells[0].strip() if cells else ""
        return {
            "row_id":    _slugify(first_val) if first_val else f"row_{row_idx + 1}",
            "row_label": first_val or f"Row {row_idx + 1}",
        }

    @staticmethod
    def _feature_flag_row_meta(
        headers: "list[str]", cells: "list[str]",
        hc: "dict[str, str]", row_idx: int,
    ) -> "dict":
        """Extract structured metadata from a feature-flag table row."""
        desc_kw = ["description", "feature", "functionality", "visibility",
                   "name", "baseline", "cem", "utility"]
        description = ""
        for kw in desc_kw:
            for hk, val in hc.items():
                if kw in hk and val:
                    description = val
                    break
            if description:
                break
        if not description and cells:
            description = cells[0].strip()

        camel_re = re.compile(r"\b[a-z][a-zA-Z]{7,}\b")
        feature_flags: "list[str]" = []
        enablement_levels: "dict[str, str]" = {}

        for i, (h, cell) in enumerate(zip(headers, cells)):
            h_low = h.strip().lower()
            cell  = cell.strip()
            if not cell or cell.lower() in ("yes", "no", "n/a", "-", ""):
                continue
            if any(kw in h_low for kw in ["flag name", "feature flag", "flag"]):
                level = ""
                for lv_kw in ["org level", "site level", "user level"]:
                    if lv_kw in h_low:
                        level = lv_kw
                        break
                if not level and i + 1 < len(headers):
                    next_h = headers[i + 1].strip().lower()
                    if "level" in next_h and i + 1 < len(cells):
                        level = cells[i + 1].strip()
                for part in re.split(r"[,\n]+", cell):
                    part = part.strip()
                    if camel_re.search(part):
                        feature_flags.append(part)
                        if level:
                            enablement_levels[part] = level

        if not feature_flags:
            for cell in cells:
                for m in camel_re.finditer(cell):
                    cand = m.group()
                    if any(kw in cand.lower() for kw in
                           ["enable", "allow", "show", "cems", "ems", "flag"]):
                        feature_flags.append(cand)

        product    = hc.get("product", "")
        applicable = next((v for k, v in hc.items()
                           if "applicable" in k or "apply" in k), "")
        return {
            "row_id":            _slugify(description) or f"flag_row_{row_idx + 1}",
            "row_label":         description or f"Row {row_idx + 1}",
            "description":       description,
            "product":           product,
            "applicable":        applicable,
            "feature_flags":     feature_flags,
            "enablement_levels": enablement_levels,
        }

    @staticmethod
    def _widget_matrix_row_meta(
        headers: "list[str]", cells: "list[str]",
        hc: "dict[str, str]", row_idx: int,
    ) -> "dict":
        """Extract structured metadata from a widget-matrix (conditional) row."""
        # Normalise broken OCR values like "Enabl ed" → "Enabled"
        # For flag columns (user/coexist): Enabled / Disabled
        # For baseline columns (energy/eo):  Yes / No
        def _norm_flag(val: str) -> str:
            v = val.strip()
            if re.fullmatch(r"ena\w*\s*\w*", v, re.I):  return "Enabled"
            if re.fullmatch(r"dis\w*\s*\w*", v, re.I):  return "Disabled"
            return v

        def _norm_baseline(val: str) -> str:
            v = val.strip()
            if re.fullmatch(r"ena\w*\s*\w*", v, re.I):  return "Yes"   # Enabled → Yes
            if re.fullmatch(r"dis\w*\s*\w*", v, re.I):  return "No"    # Disabled → No
            if v.lower() in ("yes", "no"):               return v.capitalize()
            return v

        def _norm_raw(val: str) -> str:
            return val.strip()

        def _find(norm_fn, *keywords: str) -> str:
            for kw in keywords:
                for h, v in hc.items():
                    if kw in h:
                        return norm_fn(v)
            return ""

        # Keywords match the PDF's actual (often truncated) column headers
        user_ff        = _find(_norm_flag,     "user feature flag", "user ff", "user flag",
                                               "cem user", "user widget")
        coexist_ff     = _find(_norm_flag,     "coexist feature flag", "coexist ff",
                                               "coexist flag", "coexist", "coex")
        energy_default = _find(_norm_baseline, "energy default baseline", "energy default", "ene")
        eo_default     = _find(_norm_baseline, "eo default baseline", "eo default")
        energy_w       = _find(_norm_raw, "energy widget")
        carbon_w       = _find(_norm_raw, "carbon widget")
        eo_all         = _find(_norm_raw, "eo widget - all", "eo widget all", "all utilities")
        eo_elec        = _find(_norm_raw, "eo widget - elec", "electricity")
        eo_gas         = _find(_norm_raw, "eo widget - gas", "gas")
        eo_hot         = _find(_norm_raw, "eo widget - hot", "hot water")

        parts: "list[str]" = []
        if user_ff:        parts.append(f"user_{_slugify(user_ff)}")
        if coexist_ff:     parts.append(f"coexist_{_slugify(coexist_ff)}")
        if energy_default: parts.append(f"energy_{_slugify(energy_default)}")
        if eo_default:     parts.append(f"eo_{_slugify(eo_default)}")
        row_id = "widget_" + "_".join(parts) if parts else f"widget_row_{row_idx + 1}"

        cond_parts: "list[str]" = []
        if user_ff:        cond_parts.append(f"User {user_ff}")
        if coexist_ff:     cond_parts.append(f"Coexist {coexist_ff}")
        if energy_default: cond_parts.append(f"Energy Default {energy_default}")
        if eo_default:     cond_parts.append(f"EO Default {eo_default}")
        row_label = " | ".join(cond_parts) if cond_parts else f"Widget Row {row_idx + 1}"

        return {
            "row_id":                  row_id,
            "row_label":               row_label,
            "user_feature_flag":       user_ff,
            "coexist_feature_flag":    coexist_ff,
            "energy_default_baseline": energy_default,
            "eo_default_baseline":     eo_default,
            "energy_widget":           energy_w,
            "carbon_widget":           carbon_w,
            "eo_widget_all_utilities": eo_all,
            "eo_widget_electricity":   eo_elec,
            "eo_widget_gas":           eo_gas,
            "eo_widget_hot_water":     eo_hot,
        }

    @staticmethod
    def _build_embedding_text(
        headers: "list[str]", cells: "list[str]",
        table_type: str, table_name: str,
        row_meta: "dict",
    ) -> str:
        """Build a natural-language sentence optimised for semantic embedding."""
        if table_type == "feature_flags":
            desc       = row_meta.get("description", "")
            flags      = row_meta.get("feature_flags", [])
            levels     = row_meta.get("enablement_levels", {})
            product    = row_meta.get("product", "")
            applicable = row_meta.get("applicable", "")
            parts = [f"Feature flag row for {desc}."] if desc else ["Feature flag row."]
            for flag in flags:
                lv = levels.get(flag, "")
                parts.append(
                    f"{flag} must be enabled at {lv}." if lv else f"{flag} feature flag."
                )
            if product:
                parts.append(f"Product {product}.")
            if applicable:
                parts.append(f"Applicable to {applicable}.")
            return " ".join(parts)

        if table_type == "conditional_matrix":
            parts = ["Widget matrix row."]
            for key, label in [
                ("user_feature_flag",       "User Feature Flag"),
                ("coexist_feature_flag",    "Coexist Feature Flag"),
                ("energy_default_baseline", "Energy Default Baseline"),
                ("eo_default_baseline",     "EO Default Baseline"),
                ("energy_widget",           "Energy Widget"),
                ("carbon_widget",           "Carbon Widget"),
                ("eo_widget_all_utilities", "EO Widget All Utilities"),
                ("eo_widget_electricity",   "EO Widget Electricity"),
                ("eo_widget_gas",           "EO Widget Gas"),
                ("eo_widget_hot_water",     "EO Widget Hot Water"),
            ]:
                val = row_meta.get(key, "")
                if val:
                    parts.append(f"{label} {val}.")
            return " ".join(parts)

        # Generic
        return f"[Table Headers: {' | '.join(headers)}]\n[Row]: {' | '.join(cells)}"

    # ──────────────────────────────────────────────────────────────────────────
    # Markdown table parser
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_markdown_table(
        table_md: str,
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Parses a Markdown table string into ``(headers, data_rows)``.

        Layout expected:
            | Col A | Col B | Col C |
            |-------|-------|-------|
            | val1  | val2  | val3  |
            | val4  | val5  | val6  |

        Returns ``([], [])`` if the string cannot be parsed as a table.
        Rows are normalised to the width of the header row (padded with
        empty strings or truncated as needed).
        """
        lines = [ln.strip() for ln in table_md.strip().splitlines() if ln.strip()]

        # Need at least: header row + separator row + 1 data row
        if len(lines) < 3:
            return [], []

        # Validate separator row (must contain only |, -, :, space)
        separator = lines[1]
        if not re.match(r"^\|[-:| ]+\|$", separator):
            return [], []

        def split_row(line: str) -> List[str]:
            # Strip outer pipes and split on remaining pipes
            return [cell.strip() for cell in line.strip().strip("|").split("|")]

        headers   = split_row(lines[0])
        col_count = len(headers)

        data_rows: List[List[str]] = []
        for line in lines[2:]:
            if not line.startswith("|"):
                continue
            row = split_row(line)
            # Normalise row width to match header
            if len(row) < col_count:
                row += [""] * (col_count - len(row))
            data_rows.append(row[:col_count])

        return headers, data_rows
