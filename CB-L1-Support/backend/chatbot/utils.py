"""Shared helpers for the chatbot pipeline.

Includes robust JSON loaders, Jira-field compatibility shims, text cleaning and
PII masking, Jira-key extraction, status classification, and tokenization used
by the keyword scorer.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# ── Jira key pattern, e.g. HCBS-95693, BACS-44828 ──
JIRA_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9]+-\d{2,})\b")

# ── Status classification (mirrors process_data.py so the chatbot agrees with
#    the dashboard) ──
FIXED_STATUSES = {"fixed", "done", "closed", "resolved", "completed", "verified"}
CANCELLED_STATUSES = {
    "cancelled", "canceled", "won't do", "wont do", "won't fix", "wont fix",
    "rejected", "duplicate", "not a bug", "not a defect", "as designed",
}
FIXED_RESOLUTIONS = {"fixed", "done", "resolved", "completed"}
CANCELLED_RESOLUTIONS = {
    "won't do", "wont do", "won't fix", "wont fix", "rejected", "duplicate",
    "cannot reproduce", "not a bug", "as designed",
}

# ── PII masking patterns ──
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_GUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_URL_RE = re.compile(r"https?://[^\s)\]]+")
# Jira mention tokens like "User:712020:a0c4701e-..." leak account ids.
_JIRA_MENTION_RE = re.compile(r"User:[0-9]+:[0-9a-fA-F-]+")
# Secret-ish tokens (long opaque alnum strings / bearer tokens).
_TOKEN_RE = re.compile(r"\b(?:bearer\s+)?[A-Za-z0-9_\-]{32,}\b", re.IGNORECASE)
# Markdown image embeds: ![\"x.png\"](x.png) or !img.png|...!
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|!\[[^\]]*\]|![^!\n]+\.(?:png|jpg|jpeg|gif)[^!\n]*!", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_json(path: str, default: Any = None) -> Any:
    """Load JSON, returning ``default`` if the file is missing or unreadable."""
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Jira field compatibility (MCP keeps fields at the top level, REST nests them
#  under "fields"; values may be dicts, strings, or lists)
# ─────────────────────────────────────────────────────────────────────────────
def field_get(issue: dict, *keys: str, default: Any = None) -> Any:
    """Return the first present value among ``keys``, checking top level then
    ``fields``."""
    fields = issue.get("fields") if isinstance(issue.get("fields"), dict) else {}
    for key in keys:
        if key in issue and issue[key] is not None:
            return issue[key]
        if key in fields and fields[key] is not None:
            return fields[key]
    return default


def value_name(val: Any, default: str = "") -> str:
    """Extract a human name from a Jira field value (dict/str/list/scalar)."""
    if val is None:
        return default
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        for k in ("name", "value", "displayName", "display_name"):
            if val.get(k):
                inner = val[k]
                return value_name(inner, default) if not isinstance(inner, str) else inner.strip()
        return default
    if isinstance(val, list):
        names = [value_name(v, "") for v in val]
        return ", ".join(n for n in names if n) or default
    return str(val).strip()


def value_list(val: Any) -> list[str]:
    """Normalize a Jira field value into a list of clean strings."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, dict):
        name = value_name(val)
        return [name] if name else []
    if isinstance(val, list):
        out: list[str] = []
        for v in val:
            name = value_name(v) if not isinstance(v, str) else v.strip()
            if name:
                out.append(name)
        return out
    return [str(val)]


def project_from_key(issue_key: str) -> str:
    """Derive the Jira project from the key prefix (e.g. HCBS-123 -> HCBS)."""
    if issue_key and "-" in issue_key:
        return issue_key.split("-", 1)[0].strip().upper()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Status classification
# ─────────────────────────────────────────────────────────────────────────────
def classify_fixed(status: str, resolution: str) -> bool:
    s = (status or "").strip().lower()
    r = (resolution or "").strip().lower()
    return s in FIXED_STATUSES or r in FIXED_RESOLUTIONS


def classify_cancelled(status: str, resolution: str) -> bool:
    s = (status or "").strip().lower()
    r = (resolution or "").strip().lower()
    return s in CANCELLED_STATUSES or r in CANCELLED_RESOLUTIONS


# ─────────────────────────────────────────────────────────────────────────────
#  Text cleaning and masking
# ─────────────────────────────────────────────────────────────────────────────
def mask_pii(text: str, enabled: bool = True) -> str:
    """Redact emails, phones, tokens, IPs, GUIDs, URLs and Jira mentions while
    preserving technical content (error names, component names, defect keys)."""
    if not text:
        return ""
    text = _MD_IMAGE_RE.sub(" ", text)
    if not enabled:
        return _collapse_ws(text)
    # Protect Jira keys from the long-token regex by masking mentions/tokens first.
    text = _JIRA_MENTION_RE.sub("[user]", text)
    text = _EMAIL_RE.sub("[email]", text)
    text = _URL_RE.sub("[url]", text)
    text = _GUID_RE.sub("[id]", text)
    text = _IP_RE.sub("[ip]", text)
    text = _PHONE_RE.sub("[phone]", text)

    # Mask long opaque tokens, but never mask valid Jira keys.
    def _tok(m: re.Match) -> str:
        token = m.group(0)
        return token if JIRA_KEY_RE.fullmatch(token) else "[token]"

    text = _TOKEN_RE.sub(_tok, text)
    return _collapse_ws(text)


def clean_text(text: str, enabled_mask: bool = True) -> str:
    """Normalize Jira wiki markup noise and apply masking."""
    if not text:
        return ""
    # Strip common Jira/markdown control sequences but keep words.
    text = text.replace("\r", "\n")
    text = re.sub(r"[*_#>`~]{1,3}", " ", text)        # md emphasis / headings
    text = re.sub(r"\[([^\]|]+)\|[^\]]+\]", r"\1", text)  # [text|url] -> text
    text = re.sub(r"\{[a-z]+(:[^}]*)?\}", " ", text)  # {code}, {color:..}
    return mask_pii(text, enabled_mask)


def _collapse_ws(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


# ─────────────────────────────────────────────────────────────────────────────
#  Jira key extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_jira_keys(text: str) -> list[str]:
    """Return unique Jira keys found in text, preserving order."""
    if not text:
        return []
    seen: dict[str, None] = {}
    for m in JIRA_KEY_RE.finditer(text.upper()):
        seen.setdefault(m.group(1), None)
    return list(seen.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenization for keyword scoring
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "and", "but", "or", "if", "this", "that", "these", "those",
    "what", "which", "who", "how", "why", "when", "where", "it", "its", "i", "we",
    "you", "they", "them", "their", "our", "my", "me", "show", "tell", "about",
    "please", "need", "want", "get", "give", "there", "here", "any", "all",
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization with stopword removal."""
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9][a-z0-9_.-]*", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
