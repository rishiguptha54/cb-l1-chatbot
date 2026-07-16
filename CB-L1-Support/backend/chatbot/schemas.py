"""Typed schemas and constant vocabularies for the chatbot pipeline.

These dataclasses document the shape of the records that flow between stages.
They are deliberately lightweight (``dataclasses`` + ``asdict``) so that the
JSON artifacts stay human-readable and tool-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ── Fix-pattern taxonomy (kept stable; retriever + fallback depend on it) ──
FIX_PATTERNS = [
    "Data Cleanup",
    "Null Handling",
    "Config-Environment",
    "Data Sync-Integration",
    "Logic Correction",
    "UI-Frontend",
    "Job-Workflow-Pipeline",
    "Performance",
    "Access-Authorization",
    "Workaround-Manual",
    "Deployment/Version",
    "External Dependency",
    "Unknown",
]

DEFECT_TYPES = [
    "functional bug",
    "data issue",
    "configuration issue",
    "environment issue",
    "integration issue",
    "performance issue",
    "access issue",
    "unknown",
]

# Chunk types produced per defect.
CHUNK_TYPES = [
    "problem_chunk",
    "root_cause_chunk",
    "fix_chunk",
    "comment_insight_chunk",
    "full_context_chunk",
]


@dataclass
class DefectRecord:
    """A single enriched defect in the chatbot knowledge base."""

    issue_key: str
    summary: str = ""
    description: str = ""
    status: str = ""
    resolution: str = ""
    priority: str = ""
    issue_type: str = ""
    components: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    environment: str = ""
    product: str = ""
    team: str = ""
    project: str = ""
    reporter: str = ""
    assignee: str = ""
    created: str = ""
    updated: str = ""
    resolved: str = ""
    root_cause_raw: str = ""
    root_cause_category: str = ""
    resolution_description_raw: str = ""
    steps_to_reproduce: str = ""
    linked_issues: list[str] = field(default_factory=list)
    comments: list[dict[str, Any]] = field(default_factory=list)

    # ── Enriched chatbot columns ──
    problem_text: str = ""
    symptom_text: str = ""
    error_signature: str = ""
    root_cause_extracted: str = ""
    fix_applied_extracted: str = ""
    workaround_extracted: str = ""
    key_findings: str = ""
    fix_pattern: str = "Unknown"
    defect_type: str = "unknown"
    failure_area: str = ""
    quality_score: float = 0.0
    is_fixed: bool = False
    is_cancelled: bool = False
    search_text: str = ""
    solution_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DefectChunk:
    """An embeddable chunk derived from a defect, plus retrieval metadata."""

    chunk_id: str
    issue_key: str
    chunk_type: str
    text: str
    status: str = ""
    resolution: str = ""
    priority: str = ""
    summary: str = ""
    components: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    environment: str = ""
    product: str = ""
    team: str = ""
    fix_pattern: str = "Unknown"
    defect_type: str = "unknown"
    failure_area: str = ""
    root_cause_extracted: str = ""
    fix_applied_extracted: str = ""
    workaround_extracted: str = ""
    quality_score: float = 0.0
    is_fixed: bool = False
    is_cancelled: bool = False
    created: str = ""
    resolved: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
