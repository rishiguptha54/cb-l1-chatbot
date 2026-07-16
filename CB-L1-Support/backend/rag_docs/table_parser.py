"""
table_parser.py
───────────────
Table parsing, continuation detection, and reconstruction for the RAG pipeline.

Components
──────────
TableParser              – Parses a raw Markdown table element into a TableFragment.
TableContinuationDetector – Scores pairs of TableFragments for likely continuation
                            across page boundaries using a weighted heuristic.
TableReconstructor       – Merges confirmed fragment groups into FullTable objects.

Lossless guarantee
──────────────────
- Raw Markdown for every fragment is always preserved.
- Row order is always preserved: global_row_index increments monotonically.
- Ambiguous pairs (confidence < threshold) are stored as separate fragments
  with linkage metadata — rows are NEVER dropped.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import FullTable, TableFragment, stable_id

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_header(h: str) -> str:
    """Normalise a header cell for stable comparison."""
    return re.sub(r"\s+", " ", h.strip().lower())


def _slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")[:max_len]


# ─────────────────────────────────────────────────────────────────────────────
# TableParser
# ─────────────────────────────────────────────────────────────────────────────

class TableParser:
    """
    Parses a Markdown table element (dict from DocumentIngestor) into a
    TableFragment with rich provenance metadata.

    Detects table type (feature_flags / conditional_matrix / generic_table),
    normalises headers, and computes a header signature used later for
    continuation detection.
    """

    # Separator row pattern:  | --- | :--: | ---: |
    _SEP_RE = re.compile(r"^\|[\s\-:]+\|[\s\-:|]*$")

    def parse(
        self,
        element: Dict[str, Any],
        *,
        fragment_index: int = 0,
        preceding_text: str = "",
        following_text: str = "",
    ) -> Optional[TableFragment]:
        """
        Parse a raw table element dict into a TableFragment.

        Returns None if the table cannot be parsed (e.g. pure separator tables).
        """
        raw_md  = element.get("content", "")
        page    = element.get("pdf_page_number", 0)
        source  = element.get("source_file", "")
        sec     = element.get("section_title", "")
        path    = element.get("parent_path", "")
        h1      = element.get("heading_h1", "")
        h2      = element.get("heading_h2", "")
        h3      = element.get("heading_h3", "")

        headers, raw_headers, rows = self._parse_markdown(raw_md)
        if not headers:
            logger.debug(
                "[TABLE_PARSER] Cannot parse headers for table page=%d file='%s'.",
                page, source,
            )
            return None

        table_type = self._detect_type(headers, sec)
        table_name = self._get_name(table_type, sec)

        # Deterministic IDs — stable across re-ingestion of identical content
        table_id  = stable_id(source, str(page), raw_md[:300])
        frag_id   = stable_id(source, str(page), str(fragment_index), raw_md[:300])
        group_id  = stable_id(source, raw_md[:300])   # content-only key

        frag = TableFragment(
            fragment_id=frag_id,
            table_id=table_id,
            table_group_id=group_id,
            raw_markdown=raw_md,
            headers=headers,
            raw_headers=raw_headers,
            rows=rows,
            page_number=page,
            source_file=source,
            section_title=sec,
            parent_path=path,
            preceding_text=preceding_text,
            following_text=following_text,
            column_count=len(headers),
            row_count=len(rows),
            table_type=table_type,
            table_name=table_name,
            fragment_index=fragment_index,
            heading_h1=h1,
            heading_h2=h2,
            heading_h3=h3,
        )
        frag.compute_header_signature()
        logger.debug(
            "[TABLE_PARSER] Parsed fragment page=%d type=%s cols=%d rows=%d.",
            page, table_type, len(headers), len(rows),
        )
        return frag

    # ─── Markdown parsing ────────────────────────────────────────────────────

    def _parse_markdown(
        self, table_md: str
    ) -> Tuple[List[str], List[str], List[List[str]]]:
        """Return (norm_headers, raw_headers, data_rows).  All empty on failure."""
        lines = [ln for ln in table_md.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return [], [], []

        def _split_row(line: str) -> List[str]:
            stripped = line.strip().strip("|")
            return [c.strip() for c in stripped.split("|")]

        raw_headers = _split_row(lines[0])
        if not raw_headers:
            return [], [], []

        # Locate separator row (usually line 1, but LlamaParse may insert blank lines)
        sep_idx: Optional[int] = None
        for i, ln in enumerate(lines[1:], start=1):
            if self._SEP_RE.match(ln.strip()):
                sep_idx = i
                break

        data_start = (sep_idx + 1) if sep_idx is not None else 1
        n_cols = len(raw_headers)
        data_rows: List[List[str]] = []

        for ln in lines[data_start:]:
            stripped = ln.strip()
            if not stripped or self._SEP_RE.match(stripped):
                continue
            cells = _split_row(ln)
            padded = (cells + [""] * n_cols)[:n_cols]
            data_rows.append(padded)

        norm_headers = [_norm_header(h) for h in raw_headers]
        return norm_headers, raw_headers, data_rows

    # ─── Table type & name ───────────────────────────────────────────────────

    @staticmethod
    def _detect_type(headers: List[str], section_title: str) -> str:
        """Classify as feature_flags, conditional_matrix, or generic_table."""
        h_low = " ".join(h.lower() for h in headers)
        s_low = section_title.lower()

        widget_sec_kw = [
            "visualiz", "widget", "dashboard", "coexist",
            "eo widget", "energy widget", "carbon widget",
        ]
        if any(kw in s_low for kw in widget_sec_kw):
            return "conditional_matrix"
        widget_hdr_kw = [
            "energy widget", "carbon widget", "eo widget",
            "coexist", "energy default", "eo default",
        ]
        if sum(1 for kw in widget_hdr_kw if kw in h_low) >= 2:
            return "conditional_matrix"

        flag_sec_kw = ["feature flag", "flag option", "enable flag"]
        if any(kw in s_low for kw in flag_sec_kw):
            return "feature_flags"
        flag_hdr_kw = [
            "flag", "applicable", "org level", "site level",
            "user level", "product", "enablement",
        ]
        if sum(1 for kw in flag_hdr_kw if kw in h_low) >= 2:
            return "feature_flags"

        return "generic_table"

    @staticmethod
    def _get_name(table_type: str, section_title: str) -> str:
        if table_type == "feature_flags":
            return "Feature Flag Options"
        if table_type == "conditional_matrix":
            return "CEM Widgets Visualization from Baseline App Data"
        return section_title or "Table"


# ─────────────────────────────────────────────────────────────────────────────
# TableContinuationDetector
# ─────────────────────────────────────────────────────────────────────────────

class TableContinuationDetector:
    """
    Scores pairs of TableFragments for likely continuation across page boundaries.

    Uses a weighted heuristic combining up to six signals.
    Returns a confidence score in [0, 1] and a human-readable reason string.
    Conservative by design: fragments are only merged when confidence ≥ threshold.
    """

    _CONTINUATION_RE = re.compile(
        r"\b(continued?|cont\.?|see\s+next|table\s+continues?)\b",
        re.IGNORECASE,
    )

    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold

    # ── Pair scoring ─────────────────────────────────────────────────────────

    def score_pair(
        self, a: TableFragment, b: TableFragment
    ) -> Tuple[float, str]:
        """
        Score how likely fragment *b* continues fragment *a*.

        Returns (confidence ∈ [0,1], reason_string).
        """
        weight_total = 0.0
        score_sum    = 0.0
        reasons: List[str] = []

        def _add(weight: float, value: float, reason: str) -> None:
            nonlocal weight_total, score_sum
            weight_total += weight
            score_sum    += weight * value
            if value > 0:
                reasons.append(reason)

        # Signal 1: page adjacency (weight 0.20)
        page_diff = b.page_number - a.page_number
        if page_diff == 1:
            _add(0.20, 1.0, "adjacent_pages")
        elif page_diff == 0:
            _add(0.20, 0.0, "")          # same page – no credit
        else:
            _add(0.20, -1.5, "")         # non-adjacent – strong penalty
            weight_total += 0.0          # already added in _add

        # Signal 2: header similarity (weight 0.35)
        if a.header_signature and b.header_signature:
            if a.header_signature == b.header_signature:
                _add(0.35, 1.0, "identical_headers")
            else:
                j = self._jaccard(a.headers, b.headers)
                _add(0.35, j, "similar_headers" if j >= 0.5 else "")

        # Signal 3: column count match (weight 0.15)
        if a.column_count > 0 and b.column_count > 0:
            if a.column_count == b.column_count:
                _add(0.15, 1.0, "same_column_count")
            else:
                _add(0.15, 0.0, "")

        # Signal 4: same source file (weight 0.10)
        if a.source_file == b.source_file:
            _add(0.10, 1.0, "same_source_file")

        # Signal 5: continuation cue in following_text of a (weight 0.15)
        if self._CONTINUATION_RE.search(a.following_text or ""):
            _add(0.15, 1.0, "continuation_cue_in_text")

        # Signal 6: same table type (weight 0.05)
        if a.table_type == b.table_type:
            _add(0.05, 1.0, "same_table_type")

        confidence = max(0.0, min(1.0, score_sum / weight_total)) if weight_total > 0 else 0.0
        reason_str = "; ".join(r for r in reasons if r) or "no_positive_signals"
        return confidence, reason_str

    # ── Group detection ──────────────────────────────────────────────────────

    def detect_groups(
        self, fragments: List[TableFragment]
    ) -> List[List[TableFragment]]:
        """
        Given a list of TableFragments (any order), group them into chains
        where consecutive fragments are likely continuations.

        Returns a list of groups.  Single-page tables → group of 1.
        Fragments from different source files are never merged.
        """
        if not fragments:
            return []

        sorted_frags = sorted(
            fragments, key=lambda f: (f.source_file, f.page_number)
        )

        groups: List[List[TableFragment]] = []
        current: List[TableFragment] = [sorted_frags[0]]

        for frag in sorted_frags[1:]:
            prev = current[-1]
            # Never merge across different source files
            if prev.source_file != frag.source_file:
                groups.append(current)
                current = [frag]
                continue

            confidence, reason = self.score_pair(prev, frag)
            if confidence >= self.threshold:
                current.append(frag)
                logger.debug(
                    "[DETECTOR] Merging page %d→%d │ conf=%.2f │ %s",
                    prev.page_number, frag.page_number, confidence, reason,
                )
            else:
                groups.append(current)
                current = [frag]

        groups.append(current)
        logger.info(
            "[DETECTOR] %d fragments → %d table group(s).",
            len(fragments), len(groups),
        )
        return groups

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        sa = {_norm_header(h) for h in a}
        sb = {_norm_header(h) for h in b}
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


# ─────────────────────────────────────────────────────────────────────────────
# TableReconstructor
# ─────────────────────────────────────────────────────────────────────────────

class TableReconstructor:
    """
    Merges fragment groups from TableContinuationDetector into FullTable objects.

    Lossless guarantees
    ───────────────────
    - Every row from every fragment is preserved in the merged rows list.
    - Row order = reading order (fragment order × row-within-fragment order).
    - Raw Markdown for each fragment is stored in full_table_json["raw_fragments"].
    - Each row carries its source page_number and fragment_index.
    """

    def reconstruct(
        self, fragment_groups: List[List[TableFragment]]
    ) -> List[FullTable]:
        """
        Build a FullTable for each group (including single-fragment groups).

        Returns a list of FullTable objects.
        """
        result: List[FullTable] = []
        for group in fragment_groups:
            try:
                result.append(self._build(group))
            except Exception as exc:
                logger.warning(
                    "[RECONSTRUCTOR] Failed for group (pages %s): %s",
                    [f.page_number for f in group], exc,
                )
        return result

    def _build(self, group: List[TableFragment]) -> FullTable:
        primary   = group[0]
        is_multi  = len(group) > 1

        # Canonical headers from the first fragment
        headers = primary.headers

        # Stable IDs for this reconstructed table
        group_id = stable_id(
            primary.source_file,
            *(f.fragment_id for f in group),
        )
        table_id = stable_id(group_id, primary.source_file)

        # Compute average continuation confidence across consecutive pairs
        if is_multi:
            detector = TableContinuationDetector()
            confs:   List[float] = []
            reasons: List[str]   = []
            for i in range(len(group) - 1):
                c, r = detector.score_pair(group[i], group[i + 1])
                confs.append(c)
                reasons.append(r)
            avg_conf       = sum(confs) / len(confs)
            combined_reason = " | ".join(reasons)
        else:
            avg_conf        = 1.0
            combined_reason = "single_fragment"

        # Merge rows with provenance
        all_rows: List[Dict[str, Any]] = []
        global_row_idx = 0
        for frag_idx, frag in enumerate(group):
            for local_idx, cells in enumerate(frag.rows):
                padded = (list(cells) + [""] * len(headers))[: len(headers)]
                all_rows.append({
                    "row_index":          global_row_idx,
                    "fragment_index":     frag_idx,
                    "fragment_row_index": local_idx,
                    "page_number":        frag.page_number,
                    "cells":              padded,
                })
                global_row_idx += 1

        page_start = min(f.page_number for f in group)
        page_end   = max(f.page_number for f in group)

        full = FullTable(
            table_id=table_id,
            table_group_id=group_id,
            table_name=primary.table_name,
            table_type=primary.table_type,
            headers=headers,
            rows=all_rows,
            fragments=group,
            page_start=page_start,
            page_end=page_end,
            source_file=primary.source_file,
            section_title=primary.section_title,
            parent_path=primary.parent_path,
            is_multi_page=is_multi,
            continuation_confidence=round(avg_conf, 4),
            continuation_reason=combined_reason,
            heading_h1=primary.heading_h1,
            heading_h2=primary.heading_h2,
            heading_h3=primary.heading_h3,
        )
        full.full_table_markdown = full.build_markdown()
        full.full_table_json     = full.build_json()

        logger.debug(
            "[RECONSTRUCTOR] FullTable %s │ pages %d–%d │ %d rows │ multi=%s conf=%.2f",
            table_id[:8], page_start, page_end, len(all_rows), is_multi, avg_conf,
        )
        return full
