"""Stage 5: build embeddable chunks from the knowledge base.

Each defect becomes up to five typed chunks (problem / root_cause / fix /
comment_insight / full_context). Splitting per aspect lets a fix-oriented query
match the fix chunk even when the problem text dominates the summary.

Run with:  python -m chatbot.build_chunks
"""

from __future__ import annotations

import json
import sys
from typing import Any

import config
from chatbot import utils
from chatbot.schemas import DefectChunk


def _meta(rec: dict) -> dict[str, Any]:
    """Common metadata copied onto every chunk for a defect."""
    return {
        "status": rec.get("status", ""),
        "resolution": rec.get("resolution", ""),
        "priority": rec.get("priority", ""),
        "summary": rec.get("summary", ""),
        "components": rec.get("components", []),
        "labels": rec.get("labels", []),
        "environment": rec.get("environment", ""),
        "product": rec.get("product", ""),
        "team": rec.get("team", ""),
        "fix_pattern": rec.get("fix_pattern", "Unknown"),
        "defect_type": rec.get("defect_type", "unknown"),
        "failure_area": rec.get("failure_area", ""),
        "root_cause_extracted": rec.get("root_cause_extracted", ""),
        "fix_applied_extracted": rec.get("fix_applied_extracted", ""),
        "workaround_extracted": rec.get("workaround_extracted", ""),
        "quality_score": rec.get("quality_score", 0.0),
        "is_fixed": rec.get("is_fixed", False),
        "is_cancelled": rec.get("is_cancelled", False),
        "created": rec.get("created", ""),
        "resolved": rec.get("resolved", ""),
    }


def _comments_blob(rec: dict, limit: int = 800) -> str:
    parts = [c.get("body", "") for c in rec.get("comments", []) if c.get("body")]
    return utils.truncate(" \n".join(parts), limit)


def _chunks_for(rec: dict) -> list[DefectChunk]:
    key = rec["issue_key"]
    meta = _meta(rec)
    out: list[DefectChunk] = []

    def add(chunk_type: str, text: str) -> None:
        text = (text or "").strip()
        if len(text) < 8:
            return
        out.append(DefectChunk(
            chunk_id=f"{key}::{chunk_type}",
            issue_key=key,
            chunk_type=chunk_type,
            text=utils.truncate(text, 1500),
            **meta,
        ))

    add("problem_chunk", utils._collapse_ws(
        " ".join([rec.get("summary", ""), rec.get("problem_text", ""),
                  rec.get("symptom_text", ""), rec.get("error_signature", "")])
    ))
    add("root_cause_chunk", " ".join([
        rec.get("root_cause_extracted", ""), rec.get("root_cause_raw", ""),
        rec.get("root_cause_category", ""), rec.get("key_findings", ""),
    ]))
    add("fix_chunk", " ".join([
        rec.get("fix_applied_extracted", ""), rec.get("workaround_extracted", ""),
        rec.get("resolution_description_raw", ""), rec.get("solution_text", ""),
    ]))
    add("comment_insight_chunk", " ".join([
        rec.get("key_findings", ""), _comments_blob(rec),
    ]))
    add("full_context_chunk", " ".join([
        f"[{key}]", rec.get("summary", ""), rec.get("problem_text", ""),
        f"Root cause: {rec.get('root_cause_extracted', '')}",
        f"Fix: {rec.get('fix_applied_extracted', '')}",
        f"Status: {rec.get('status', '')}", f"Priority: {rec.get('priority', '')}",
        f"Components: {', '.join(rec.get('components', []))}",
        f"Fix pattern: {rec.get('fix_pattern', '')}",
    ]))
    return out


def build_chunks() -> list[dict[str, Any]]:
    kb = utils.load_json(config.DEFECT_KB_PATH, default=[]) or []
    if not kb:
        raise FileNotFoundError(
            f"Knowledge base not found at {config.DEFECT_KB_PATH}. "
            "Run `python -m chatbot.build_knowledge_base` first."
        )

    all_chunks: list[dict[str, Any]] = []
    for rec in kb:
        for chunk in _chunks_for(rec):
            all_chunks.append(chunk.to_dict())

    import os
    os.makedirs(os.path.dirname(config.DEFECT_CHUNKS_PATH), exist_ok=True)
    with open(config.DEFECT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[chunks] {len(all_chunks)} chunks from {len(kb)} defects")
    print(f"[chunks] wrote {config.DEFECT_CHUNKS_PATH}")
    return all_chunks


def load_chunks() -> list[dict[str, Any]]:
    """Read the chunk JSONL back into memory."""
    out: list[dict[str, Any]] = []
    path = config.DEFECT_CHUNKS_PATH
    import os
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


if __name__ == "__main__":
    try:
        build_chunks()
    except Exception as exc:  # pragma: no cover
        print(f"[chunks] ERROR: {exc}", file=sys.stderr)
        raise
