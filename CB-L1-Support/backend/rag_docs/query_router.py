"""
query_router.py
───────────────
Lightweight query classifier for the RAG pipeline.

Routes
──────
A. feature_flag_table
   Triggered by queries about feature flags, enablement levels, org/site level.

B. widget_matrix
   Triggered by queries about Energy Widget, Carbon Widget, EO Widget, or the
   four condition columns.

C. text_section
   Triggered by procedural/process questions (steps, how to, what should users do).

D. table_full
   Triggered by queries asking to see the full table / all rows / summarize table.

E. image_query
   Triggered by queries about images, diagrams, screenshots, charts, flowcharts.

F. hybrid (default)
   Fallback for everything else: search all asset_types.
"""

import re
from typing import Any, Dict, Optional


# ── Route constants ───────────────────────────────────────────────────────────
ROUTE_FEATURE_FLAG  = "feature_flag_table"
ROUTE_WIDGET_MATRIX = "widget_matrix"
ROUTE_TEXT_SECTION  = "text_section"
ROUTE_TABLE_FULL    = "table_full"
ROUTE_HYBRID        = "hybrid"

# ── Keyword patterns ──────────────────────────────────────────────────────────
_FLAG_TERMS = re.compile(
    r"feature\s+flag|flag\s+name|enable\s+at\s+(org|site|user)\s+level|"
    r"org\s+level|site\s+level|user\s+level|"
    r"baseline\s+app\s+vis|cem\s+navigation|utility\s+data\s+tab|"
    r"enablement|cemsEnable|cems\w+",
    re.IGNORECASE,
)

_WIDGET_TERMS = re.compile(
    r"energy\s+widget|carbon\s+widget|eo\s+widget|"
    r"user\s+feature\s+flag|coexist\s+(feature\s+)?flag|"
    r"energy\s+default\s+baseline|eo\s+default\s+baseline|"
    r"data\s+from\s+old|data\s+from\s+new\s+baseline|new\s+baseline\s+data|no\s+data",
    re.IGNORECASE,
)

_TEXT_TERMS = re.compile(
    r"what\s+are\s+the\s+(?:\w+\s+){0,6}steps|how\s+(can|do|to|should)|"
    r"explain|describe|what\s+should\s+(users?|i)|"
    r"procedure|process|upload|download|create|update|verify|configure|"
    r"cleansing|consumption\s+data|first\s+month|ingestion|"
    r"\bsteps\s+(include|for|to|in)\b|\bonboarding\s+steps\b",
    re.IGNORECASE,
)

_FULL_TABLE_TERMS = re.compile(
    r"full\s+table|entire\s+table|whole\s+table|complete\s+table|"
    r"show\s+(me\s+)?the\s+table|all\s+rows|every\s+row|"
    r"summarize\s+the\s+(matrix|table)|what\s+(columns?|fields?)|"
    r"list\s+(all|every)\s+.{0,30}(row|entry|option)",
    re.IGNORECASE,
)


# ── Condition extractors ──────────────────────────────────────────────────────
# Maps (pattern, field_name, normalised_value)
_CONDITION_PATTERNS: list[tuple] = [
    # User Feature Flag
    (re.compile(r"user\s+feature\s+flag\s+(is\s+|=\s*|:\s*)?(enabled?)", re.I),
     "user_feature_flag", "Enabled"),
    (re.compile(r"user\s+feature\s+flag\s+(is\s+|=\s*|:\s*)?(disabled?)", re.I),
     "user_feature_flag", "Disabled"),
    # Coexist Feature Flag
    (re.compile(r"coexist\s+(feature\s+)?flag\s+(is\s+|=\s*|:\s*)?(enabled?)", re.I),
     "coexist_feature_flag", "Enabled"),
    (re.compile(r"coexist\s+(feature\s+)?flag\s+(is\s+|=\s*|:\s*)?(disabled?)", re.I),
     "coexist_feature_flag", "Disabled"),
    # Energy Default Baseline
    (re.compile(r"energy\s+default\s+baseline\s+(is\s+|=\s*|:\s*)?(yes)", re.I),
     "energy_default_baseline", "Yes"),
    (re.compile(r"energy\s+default\s+baseline\s+(is\s+|=\s*|:\s*)?(no)", re.I),
     "energy_default_baseline", "No"),
    # EO Default Baseline
    (re.compile(r"eo\s+default\s+baseline\s+(is\s+|=\s*|:\s*)?(yes)", re.I),
     "eo_default_baseline", "Yes"),
    (re.compile(r"eo\s+default\s+baseline\s+(is\s+|=\s*|:\s*)?(no)", re.I),
     "eo_default_baseline", "No"),
]

_WIDGET_REQUIRED_CONDITIONS = {
    "user_feature_flag",
    "coexist_feature_flag",
    "energy_default_baseline",
    "eo_default_baseline",
}


class QueryRouter:
    """
    Classifies a natural-language query into one of four retrieval routes
    and extracts any explicit widget-matrix conditions.
    """

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Analyse *query* and return a routing decision dict:

        {
          "route":      str,              – one of the ROUTE_* constants
          "conditions": dict[str, str],   – widget conditions (may be empty)
          "missing_conditions": list[str] – widget conditions not found in query
        }

        Priority order:
          1. widget_matrix  (most specific – condition extraction)
          2. feature_flag_table
          3. table_full     (full-table retrieval intent)
          4. text_section   (procedural text)
          5. hybrid         (default)
        """
        # 1. Widget matrix check (specific condition extraction)
        conditions = self._extract_widget_conditions(query)
        if conditions or _WIDGET_TERMS.search(query):
            missing = sorted(_WIDGET_REQUIRED_CONDITIONS - set(conditions.keys()))
            return {
                "route":               ROUTE_WIDGET_MATRIX,
                "conditions":          conditions,
                "missing_conditions":  missing,
            }

        # 2. Feature flag
        if _FLAG_TERMS.search(query):
            return {
                "route":              ROUTE_FEATURE_FLAG,
                "conditions":         {},
                "missing_conditions": [],
            }

        # 3. Full-table retrieval intent
        if _FULL_TABLE_TERMS.search(query):
            return {
                "route":              ROUTE_TABLE_FULL,
                "conditions":         {},
                "missing_conditions": [],
            }

        # 4. Text / procedural
        if _TEXT_TERMS.search(query):
            return {
                "route":              ROUTE_TEXT_SECTION,
                "conditions":         {},
                "missing_conditions": [],
            }

        # 5. Default hybrid
        return {
            "route":              ROUTE_HYBRID,
            "conditions":         {},
            "missing_conditions": [],
        }

    @staticmethod
    def _extract_widget_conditions(query: str) -> Dict[str, str]:
        """
        Extract explicit widget-matrix conditions from *query*.

        Returns a dict such as:
            {"user_feature_flag": "Enabled",
             "energy_default_baseline": "No"}
        Only conditions that are explicitly stated in the query are included.
        """
        conditions: Dict[str, str] = {}
        for pattern, field, value in _CONDITION_PATTERNS:
            if pattern.search(query):
                conditions[field] = value
        return conditions
