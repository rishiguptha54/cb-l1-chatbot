"""
document_ingestor.py
────────────────────
Stage 1 of the RAG pipeline.

Uses LlamaParse (LlamaIndex cloud API) to parse PDF files into an ordered
list of structured elements (TEXT, TABLE).  No local ML models are needed;
the PDF is uploaded to the LlamaParse API and returned as Markdown with
tables preserved as Markdown table blocks.

Changes from v1
───────────────
- Markdown image references are stripped from TEXT content.
- Section context (heading_h1/h2/h3, section_title, parent_path) is tracked
  and injected into every element dict.
- preceding_text / following_text snippets are added for table context.
- All new fields are additive – downstream code that reads only the original
  keys (content, element_type, pdf_page_number, source_file) continues to work.

Requires:
    LLAMA_CLOUD_API_KEY  – set in .env  (free key at cloud.llamaindex.ai)
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

# Complete Markdown table block:  header | separator | one-or-more data rows
_TABLE_RE = re.compile(
    r'(\|[^\n]+\|\n[ \t]*\|[-:| \t]+\|\n(?:[ \t]*\|[^\n]+\|\n?)+)',
    re.MULTILINE,
)

# Heading patterns (ATX-style):  ## Heading text
_H1_RE = re.compile(r'^#\s+(.+)$',   re.MULTILINE)
_H2_RE = re.compile(r'^##\s+(.+)$',  re.MULTILINE)
_H3_RE = re.compile(r'^###\s+(.+)$', re.MULTILINE)

# Snippet length for preceding / following context
_CONTEXT_SNIPPET_CHARS = 400


def _remove_image_refs(text: str) -> str:
    """Remove Markdown image tags from text, preserving any alt-text content.

    LlamaParse sometimes embeds nearby paragraph text inside the alt-text
    bracket of an image reference (e.g. ![Step 1. Do X…](url)).  Stripping
    the whole pattern loses that content.  This version extracts the alt-text
    and keeps it inline before removing the image syntax.
    """
    # Replace ![alt](url) with just the alt text (preserves embedded content)
    cleaned = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _last_n_chars(text: str, n: int) -> str:
    return text[-n:].strip() if text else ""


def _first_n_chars(text: str, n: int) -> str:
    return text[:n].strip() if text else ""


# ─────────────────────────────────────────────────────────────────────────────
# Section-context tracker
# ─────────────────────────────────────────────────────────────────────────────

class _SectionTracker:
    """Tracks the current heading context as we iterate through page elements."""

    def __init__(self) -> None:
        self.h1: str = ""
        self.h2: str = ""
        self.h3: str = ""

    def update(self, text: str) -> None:
        """Update heading state from any headings found in *text*."""
        for m in _H1_RE.finditer(text):
            self.h1 = m.group(1).strip()
            self.h2 = ""
            self.h3 = ""
        for m in _H2_RE.finditer(text):
            self.h2 = m.group(1).strip()
            self.h3 = ""
        for m in _H3_RE.finditer(text):
            self.h3 = m.group(1).strip()

    @property
    def section_title(self) -> str:
        return self.h3 or self.h2 or self.h1

    @property
    def parent_path(self) -> str:
        parts = [h for h in (self.h1, self.h2, self.h3) if h]
        return " > ".join(parts)

    def snapshot(self) -> Dict[str, str]:
        return {
            "heading_h1":    self.h1,
            "heading_h2":    self.h2,
            "heading_h3":    self.h3,
            "section_title": self.section_title,
            "parent_path":   self.parent_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DocumentIngestor
# ─────────────────────────────────────────────────────────────────────────────

class DocumentIngestor:
    """
    Parses PDF documents into ordered document elements via LlamaParse.

    Each element returned by ``parse_pdf`` is a ``dict`` with these keys:

    Always present (backward-compatible set)
    ─────────────────────────────────────────
    content                str  – raw text or Markdown table string
    element_type           str  – "TEXT" | "TABLE" | "IMAGE"
    pdf_page_number        int  – 1-based PDF page number
    llamaparse_page_number int  – same value (kept for traceability)
    source_file            str  – basename of the source PDF

    New fields (additive – old code can ignore them)
    ──────────────────────────────────────────────────
    heading_h1             str  – current H1 context at point of element
    heading_h2             str  – current H2 context
    heading_h3             str  – current H3 context
    section_title          str  – deepest non-empty heading
    parent_path            str  – "H1 > H2 > H3"
    preceding_text         str  – up to 400 chars of text before this element
    following_text         str  – up to 400 chars of text after this element
    raw_markdown           str  – original Markdown before any transformation
    extraction_method      str  – always "llamaparse" for LlamaParse-parsed elements

    For IMAGE elements only
    ───────────────────────
    image_refs             list – Markdown image references found on the page
    """

    def __init__(self) -> None:
        logger.info("[INGESTOR] ── Initialising DocumentIngestor with LlamaParse...")
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY", "")
        if not api_key:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY is not set. "
                "Add it to your .env file.  "
                "Get a free key at: https://cloud.llamaindex.ai"
            )

        # ── Corporate SSL bypass ──────────────────────────────────────────────
        # LlamaParse uses httpx internally and respects these env vars.
        # Must be set BEFORE the import so the httpx client picks them up.
        import ssl
        os.environ.setdefault("SSL_CERT_FILE", "")
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        # Monkey-patch ssl to skip verification for LlamaParse's httpx client
        _orig_create_default_context = ssl.create_default_context
        def _insecure_context(*args, **kwargs):
            ctx = _orig_create_default_context(*args, **kwargs)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        ssl.create_default_context = _insecure_context

        from llama_parse import LlamaParse  # lazy import – keeps startup fast
        self._parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            verbose=False,
            show_progress=False,
        )
        logger.info("[INGESTOR] ✓ DocumentIngestor ready.")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def parse_pdf(
        self,
        pdf_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse a single PDF and return an ordered list of element dicts.

        LlamaParse returns one Document per page; each page's Markdown is
        split into interleaved TEXT and TABLE elements preserving reading order.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"[INGESTOR] File not found: {pdf_path}")

        source_file = path.name
        logger.info("[INGESTOR] ── Parsing '%s' via LlamaParse API...", source_file)

        documents = self._parser.load_data(str(path))
        logger.info("[INGESTOR] ✓ LlamaParse returned %d page(s).", len(documents))

        # ── Step 1: Parse text/table elements from LlamaParse ─────────────────
        elements: List[Dict[str, Any]] = []
        section_tracker = _SectionTracker()

        for doc_idx, doc in enumerate(documents):
            page_no = int(doc.metadata.get("page_label", doc_idx + 1))
            page_elems = self._split_page(
                doc.text, page_no, source_file, section_tracker
            )
            elements.extend(page_elems)

        # ── Step 2: Add preceding/following context snippets ──────────────────
        self._attach_context_snippets(elements)

        # Sort by page then local order
        elements.sort(key=lambda e: (e["pdf_page_number"], e.get("_page_local_order", 9999)))
        for el in elements:
            el.pop("_page_local_order", None)

        text_count  = sum(1 for e in elements if e["element_type"] == "TEXT")
        table_count = sum(1 for e in elements if e["element_type"] == "TABLE")
        logger.info(
            "[INGESTOR] ✓ Parse complete │ '%s' → %d text + %d tables.",
            source_file, text_count, table_count,
        )
        return elements

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _split_page(
        self,
        markdown:        str,
        page_no:         int,
        source_file:     str,
        section_tracker: _SectionTracker,
    ) -> List[Dict[str, Any]]:
        """
        Split one page's Markdown into interleaved TEXT / TABLE / IMAGE elements
        preserving reading order.  Tables are identified by _TABLE_RE.

        Section context is updated as TEXT elements are processed.
        """
        elements: List[Dict[str, Any]] = []
        cursor    = 0
        local_ord = 0

        def _base(typ: str) -> Dict[str, Any]:
            d = {
                "element_type":           typ,
                "pdf_page_number":        page_no,
                "llamaparse_page_number": page_no,
                "source_file":            source_file,
                "extraction_method":      "llamaparse",
                "_page_local_order":      local_ord,
            }
            d.update(section_tracker.snapshot())
            return d

        for match in _TABLE_RE.finditer(markdown):
            # ── Text block before this table ──────────────────────────────────
            raw_before = markdown[cursor:match.start()]
            if raw_before.strip():
                section_tracker.update(raw_before)
                clean_before = _remove_image_refs(raw_before)
                if clean_before:
                    el = _base("TEXT")
                    el["content"]      = clean_before
                    el["raw_markdown"] = raw_before
                    elements.append(el)
                    local_ord += 1

            # ── Table block ────────────────────────────────────────────────────
            table_md = match.group(0).strip()
            el = _base("TABLE")
            el["content"]      = table_md
            el["raw_markdown"] = table_md
            elements.append(el)
            local_ord += 1
            cursor = match.end()

        # ── Trailing text after the last table on this page ───────────────────
        raw_after = markdown[cursor:]
        if raw_after.strip():
            section_tracker.update(raw_after)
            clean_after = _remove_image_refs(raw_after)
            if clean_after:
                el = _base("TEXT")
                el["content"]      = clean_after
                el["raw_markdown"] = raw_after
                elements.append(el)
                local_ord += 1

        return elements

    def _attach_context_snippets(self, elements: List[Dict[str, Any]]) -> None:
        """
        Add preceding_text and following_text to every element in-place.

        Only TEXT content is used for the snippets (tables and images are
        excluded from snippet bodies to keep context clean).
        """
        n = len(elements)
        for i, el in enumerate(elements):
            # Preceding: last N chars of the nearest TEXT element before this one
            prec = ""
            for j in range(i - 1, -1, -1):
                if elements[j]["element_type"] == "TEXT":
                    prec = _last_n_chars(elements[j]["content"], _CONTEXT_SNIPPET_CHARS)
                    break
            el["preceding_text"] = prec

            # Following: first N chars of the nearest TEXT element after this one
            foll = ""
            for j in range(i + 1, n):
                if elements[j]["element_type"] == "TEXT":
                    foll = _first_n_chars(elements[j]["content"], _CONTEXT_SNIPPET_CHARS)
                    break
            el["following_text"] = foll
