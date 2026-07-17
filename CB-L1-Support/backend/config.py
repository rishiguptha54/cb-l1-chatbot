"""
Configuration module for the Defect Dashboard project.
Loads credentials from .env and defines constants.
"""

import os
from dotenv import load_dotenv

# Load .env from the same directory as this file
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH, override=True)

# ── JIRA Connection ──
JIRA_URL = os.getenv("JIRA_URL", "https://honeywell.atlassian.net")
JIRA_USERNAME = os.getenv("JIRA_USERNAME", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
UVX_PATH = os.getenv("UVX_PATH", r"C:\Users\H675331\.local\bin\uvx.exe")

# ── Safety: READ ONLY MODE ──
READ_ONLY_MODE = True  # Hardcoded True - never allow writes

# ── Azure OpenAI Configuration ──
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# ── JQL Query for Defects ──
DEFECT_JQL = (
    'project IN ("BA Connected Strategy", "HCE Connected Buildings") '
    'AND created >= "2025-01-01" '
    'AND "Reported By[Dropdown]" IN ("Customer", "Internal Customer") '
    'AND issuetype = Defect '
    'AND "HCE-Environment" IN ("Production") '
    'AND "Reported By" NOT IN ("Security")'
)

# ── Fields to fetch from JIRA ──
# Standard fields we always request
STANDARD_FIELDS = (
    "summary,status,priority,components,assignee,reporter,"
    "created,updated,resolutiondate,resolution,labels,fixVersions,"
    "description,issuetype,environment,issuelinks,comment"
)

# ── Team custom field (used for the Teams analysis section) ──
# The JIRA custom field id that holds the team/squad value.
TEAM_FIELD_ID = os.getenv("TEAM_FIELD_ID", "customfield_12095")

# ── Customer Fix Priority (CFP) custom field ──
# Numeric JIRA field (0–1000) used to band defects into CFP1–CFP4.
CFP_FIELD_ID = os.getenv("CFP_FIELD_ID", "customfield_13184")
# CFP bands (numeric value → label). Higher CFP value = higher priority.
#   CFP1: value >= 400
#   CFP2: 150 <= value < 400
#   CFP3: 80  <= value < 150
#   CFP4: value < 80
CFP_BANDS = [("CFP1", 400), ("CFP2", 150), ("CFP3", 80), ("CFP4", 0)]

# ── MCP Worker Configuration ──
MCP_WORKER_COUNT = 5  # Number of parallel MCP workers for batch fetch
MCP_PAGE_SIZE = 50  # Max results per jira_search call

# ── SLA targets (calendar days to resolve, by priority label) ──
# An open defect older than its priority's target is counted as an SLA breach.
# Keys are matched case-insensitively; unmatched priorities use SLA_DEFAULT_DAYS.
SLA_TARGETS_DAYS = {
    "1 - critical": 3, "critical": 3, "blocker": 3, "urgent": 3,
    "2 - high": 7, "high": 7, "major": 7,
    "3 - normal": 14, "normal": 14, "medium": 14, "should have": 14,
    "4 - low": 30, "low": 30, "minor": 30,
}
SLA_DEFAULT_DAYS = 14

# ── SLA targets by CFP band (only CFP1 and CFP2 are tracked) ──
# Open defects in these bands older than their target are SLA breaches; CFP
# bands not listed here (CFP3, CFP4) are excluded from the SLA breach view.
CFP_SLA_TARGETS_DAYS = {"CFP1": 7, "CFP2": 14}

# ── Output Directories ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create dirs if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  CB L1 SUPPORT CHATBOT CONFIGURATION
#  (Separate pipeline; does not affect the dashboard pipeline above.)
# ═══════════════════════════════════════════════════════════════════

def _as_bool(value: str, default: bool = False) -> bool:
    """Parse an environment string into a boolean."""
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


# ── Chatbot directories ──
CHATBOT_DATA_DIR = os.path.join(DATA_DIR, "chatbot")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
os.makedirs(CHATBOT_DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ── Source data paths (reused from the dashboard pipeline) ──
RAW_DEFECTS_PATH = os.path.join(DATA_DIR, "raw_defects.json")
ISSUE_COMMENTS_PATH = os.path.join(DATA_DIR, "issue_comments.json")
LLM_ANALYSIS_PATH = os.path.join(DATA_DIR, "llm_analysis.json")

# ── Chatbot artifact paths ──
DEFECT_KB_PATH = os.path.join(CHATBOT_DATA_DIR, "defect_knowledge_base.json")
DEFECT_KB_CSV_PATH = os.path.join(CHATBOT_DATA_DIR, "defect_knowledge_base.csv")
DEFECT_CHUNKS_PATH = os.path.join(CHATBOT_DATA_DIR, "defect_chunks.jsonl")
EMBEDDING_METADATA_PATH = os.path.join(CHATBOT_DATA_DIR, "embedding_metadata.json")
SYNONYM_DICT_PATH = os.path.join(CHATBOT_DATA_DIR, "synonym_dictionary.json")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_defect_index.bin")
KEYWORD_INDEX_PATH = os.path.join(INDEX_DIR, "keyword_index.pkl")

# ── JIRA browse base URL (for clickable keys) ──
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", f"{JIRA_URL.rstrip('/')}/browse/")

# ── LLM toggles ──
# USE_LLM master switch; when false the chatbot uses deterministic fallbacks only.
USE_LLM = _as_bool(os.getenv("USE_LLM"), default=True)
# USE_AZURE_OPENAI selects the Azure OpenAI backend (the only chat backend wired here).
USE_AZURE_OPENAI = _as_bool(os.getenv("USE_AZURE_OPENAI"), default=True)

# Chat + embedding deployments. Chat deployment falls back to the dashboard's deployment.
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT", AZURE_OPENAI_DEPLOYMENT
)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")

# ── Local embedding fallback (sentence-transformers) ──
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Retrieval tuning ──
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "8"))
MAX_CONTEXT_DEFECTS = int(os.getenv("MAX_CONTEXT_DEFECTS", "8"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# Relevance filter for diagnostic retrieval: instead of always returning a full
# TOP_K, drop weak matches so "how to fix" answers are grounded only in
# genuinely similar defects. A defect is kept when its score clears an absolute
# floor AND stays within a relative margin of the best match. Tuned against the
# current embedding backend (cosine similarities); adjust via env if the
# embedding backend / score scale changes. Set RELEVANCE_MIN_SCORE=0 to disable.
RELEVANCE_MIN_SCORE = float(os.getenv("RELEVANCE_MIN_SCORE", "0.45"))
RELEVANCE_REL_MARGIN = float(os.getenv("RELEVANCE_REL_MARGIN", "0.75"))

# When true, cancelled (won't-fix/rejected) defects are excluded from the
# retrieval knowledge base + embeddings, so "how to fix" answers only surface
# real defects.
CHATBOT_EXCLUDE_CANCELLED = _as_bool(os.getenv("CHATBOT_EXCLUDE_CANCELLED"), default=True)

# ── Privacy ──
MASK_SENSITIVE_DATA = _as_bool(os.getenv("MASK_SENSITIVE_DATA"), default=True)

# ── API server ──
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
# Prefer Render's injected PORT for cloud deploys; keep SERVER_PORT/local default for dev.
SERVER_PORT = int(os.getenv("PORT", os.getenv("SERVER_PORT", "5100")))

# URL the dashboard's chatbot button points to (the running chatbot UI).
CHATBOT_URL = os.getenv("CHATBOT_URL", f"http://localhost:{SERVER_PORT}")

# ── External documentation RAG service ──
# A separate FastAPI RAG server that answers "how to fix / how to" questions from
# product documentation. The defect chatbot queries it as a second evidence
# source on diagnostic questions and clearly labels what came from documentation
# vs. from the historical-defect knowledge base. All failures degrade gracefully.
USE_RAG_DOCS = _as_bool(os.getenv("USE_RAG_DOCS"), default=True)
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")  # sent as X-API-Key only when set
RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT", "120"))  # seconds; answers can be slow

# On-demand PDF ingestion (header "upload documentation" button): max upload
# size, enforced before parsing so a huge file can't tie up the server.
RAG_INGEST_MAX_MB = int(os.getenv("RAG_INGEST_MAX_MB", "20"))

# LlamaParse cloud API key used by the on-demand PDF ingestion pipeline
# (document_ingestor.py). Free tier available at https://cloud.llamaindex.ai
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# ── Qdrant (shared cluster; the documentation RAG and the defect knowledge
# base use separate collections on it) ──
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Defect knowledge base: Qdrant collection name + vector dimension. Defect
# retrieval is Qdrant-only (see chatbot/retriever.py) — no local FAISS/BM25
# index. Dimension must match the embedding model actually used (3072 for
# text-embedding-3-large, the current provider — see data/chatbot/
# embedding_metadata.json's "dim").
DEFECT_QDRANT_COLLECTION = os.getenv("DEFECT_QDRANT_COLLECTION", "defect")
DEFECT_VECTOR_DIM = int(os.getenv("DEFECT_VECTOR_DIM", "3072"))

# On-demand defect ingestion via JQL (header "Add defect" button, bulk mode):
# safety cap on how many Jira issues one JQL query can pull in, so a broad
# query can't accidentally embed the entire Jira instance in one request.
JQL_ADD_MAX_RESULTS = int(os.getenv("JQL_ADD_MAX_RESULTS", "200"))


# ═══════════════════════════════════════════════════════════════════
#  PROVIDER-AGNOSTIC LLM ABSTRACTION
#  (Consumed by llm_provider.py. A single LLM_PROVIDER switch selects the
#   backend; everything below is shared catalog/stage configuration.)
# ═══════════════════════════════════════════════════════════════════

# ── Provider switch ──
# "azure"  → Azure OpenAI (uses AZURE_OPENAI_* keys/endpoint/deployments below).
# "github" → GitHub Models (token + OpenAI-compatible inference endpoint).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure").strip().lower()

# ── Embedding provider (separate from chat) ──
# The Copilot API only exposes 1536-dim embedding models, but the defect index
# and Qdrant collection are 3072-dim (text-embedding-3-large). So when chat runs
# on Copilot, embeddings must fall back to a backend that offers 3072-dim
# vectors. GitHub Models advertises text-embedding-3-large. Defaults to
# "github" whenever LLM_PROVIDER=copilot, otherwise mirrors LLM_PROVIDER.
EMBED_PROVIDER = os.getenv(
    "EMBED_PROVIDER", "github" if LLM_PROVIDER == "copilot" else LLM_PROVIDER
).strip().lower()

# ── GitHub Models backend ──
# Token resolution order (handled in llm_provider): GITHUB_TOKEN env → `gh auth token`.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
# OpenAI-compatible base. Chat:  {endpoint}/chat/completions
#                         Embed: {endpoint}/embeddings
#                         List:  {endpoint}/models
GITHUB_MODELS_ENDPOINT = os.getenv(
    "GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com"
).rstrip("/")

# ── GitHub Copilot API backend ──
# Publicly reachable, OpenAI-compatible endpoint that accepts a GitHub PAT
# DIRECTLY as a Bearer token (no token exchange) when the integration id is
# "copilot-developer-cli". This lets the deployed app reach an LLM from the
# public internet, bypassing the corporate-network-only Azure OpenAI resource.
#   Chat:  {endpoint}/chat/completions
#   Embed: {endpoint}/embeddings
#   List:  {endpoint}/models
COPILOT_API_ENDPOINT = os.getenv(
    "COPILOT_API_ENDPOINT", "https://api.githubcopilot.com"
).rstrip("/")
# Token: COPILOT_TOKEN env first, then reuse GITHUB_TOKEN. Must be a PAT with
# Copilot access (fine-grained PATs work; classic ghp_ tokens may not).
COPILOT_TOKEN = os.getenv("COPILOT_TOKEN", "") or GITHUB_TOKEN
COPILOT_INTEGRATION_ID = os.getenv("COPILOT_INTEGRATION_ID", "copilot-developer-cli")
COPILOT_EDITOR_VERSION = os.getenv("COPILOT_EDITOR_VERSION", "vscode/1.104.1")
COPILOT_USER_AGENT = os.getenv("COPILOT_USER_AGENT", "HermesAgent/1.0")
COPILOT_OPENAI_INTENT = os.getenv("COPILOT_OPENAI_INTENT", "conversation-edits")


def copilot_headers() -> dict:
    """Extra HTTP headers the GitHub Copilot API requires for PAT auth."""
    return {
        "Editor-Version": COPILOT_EDITOR_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        "User-Agent": COPILOT_USER_AGENT,
        "Openai-Intent": COPILOT_OPENAI_INTENT,
    }

# ── Document-RAG (rag_docs) backend selection ──
# "azure"   → original AzureChatOpenAI / AzureOpenAIEmbeddings (corporate net).
# "copilot" → GitHub Copilot API (publicly reachable; used on Render).
# Defaults to "copilot" when LLM_PROVIDER is copilot, else "azure".
RAG_LLM_PROVIDER = os.getenv(
    "RAG_LLM_PROVIDER", "copilot" if LLM_PROVIDER == "copilot" else "azure"
).strip().lower()
# Model ids used when RAG_LLM_PROVIDER=copilot (Copilot API model names).
RAG_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-4o")
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-large")
# RAG embedding backend (separate from chat). Copilot lacks 3072-dim embeddings,
# so default to GitHub Models when the RAG chat backend is copilot; otherwise
# mirror the RAG chat provider.
RAG_EMBED_PROVIDER = os.getenv(
    "RAG_EMBED_PROVIDER",
    "github" if RAG_LLM_PROVIDER == "copilot" else RAG_LLM_PROVIDER,
).strip().lower()

# ── Stage definitions ──
# Each LLM call is tagged with a "stage". The abstraction layer picks the first
# model from the stage's preference list that the active provider actually
# advertises (via /models discovery), falling back through the list and finally
# to the first entry. Order = most-preferred first.
#
#   STAGE       PURPOSE                                  USED BY (typical)
#   ─────────   ──────────────────────────────────────  ──────────────────────────
#   cheap       Fast/low-cost: intent classification,    intent_router, short
#               routing, yes/no & one-line replies.       deterministic confirmations
#   standard    Everyday grounded chat answers.          defect_qa default replies
#   complex     Deep reasoning / multi-defect analysis,  llm_analysis (buckets, root
#               root-cause synthesis, executive summary.  causes, exec summary)
#   embedding   Vector embeddings for RAG retrieval.     build_embeddings / retriever
#
# Model notes (GitHub Models ids; for Azure these map to your *deployment* names):
#
#   ── cheap / fast (classification, routing, short replies) ──
#   gpt-5-mini        Small GPT-5 tier; cheap, quick general tasks.
#   gpt-5.4-mini      Newer small GPT-5.4; improved quality at low cost.
#   gpt-5.4-nano      Smallest/cheapest GPT-5.4; ultra-fast trivial tasks.
#   gpt-4o-mini       Proven cheap multimodal-capable workhorse.
#   o3-mini           Small reasoning model; cheap structured/logic tasks.
#   o1-mini           Legacy small reasoning model; fallback.
#
#   ── standard chat ──
#   gpt-4o            Balanced flagship for general chat.
#   gpt-5.2           GPT-5.2 general chat tier.
#   gpt-5.4           GPT-5.4 general chat tier (higher quality).
#   gpt-3.5-turbo     Legacy cheap chat; broad availability fallback.
#   claude-haiku-4.5  Fast Anthropic chat model.
#   claude-3-haiku    Legacy fast Anthropic fallback.
#
#   ── complex reasoning / analysis ──
#   claude-sonnet-4.6 Top Anthropic reasoning (preferred for analysis).
#   claude-sonnet-4.5 Strong Anthropic reasoning.
#   claude-3.7-sonnet Extended-thinking Anthropic model.
#   claude-3.5-sonnet Reliable Anthropic reasoning fallback.
#   claude-3-opus     Legacy high-capability Anthropic model.
#   gpt-4-turbo       Strong OpenAI reasoning fallback.
#   gpt-4             Legacy high-capability OpenAI fallback.
#
#   ── embeddings / RAG retrieval ──
#   text-embedding-3-large  3072-dim, highest retrieval accuracy (preferred).
#   text-embedding-3-small  1536-dim, fast & cheap, good accuracy.
#   text-embedding-ada-002  1536-dim legacy embedding fallback.
#
# Excluded: gpt-5.3-codex / gpt-5.2-codex are code-completion models that do NOT
# support the /chat/completions API, so they are filtered out of every stage.
LLM_STAGE_MODELS: dict[str, list[str]] = {
    "cheap": [
        "gpt-4o-mini", "gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano",
        "o3-mini", "o1-mini",
    ],
    "standard": [
        "gpt-4o", "gpt-4.1", "gpt-3.5-turbo",
    ],
    "complex": [
        "gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-4",
    ],
    "embedding": [
        "text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002",
    ],
}

# Models that must never be selected (no /chat/completions support).
LLM_EXCLUDED_MODELS: set[str] = {"gpt-5.3-codex", "gpt-5.2-codex"}

# Optional explicit per-stage overrides. For GitHub Models set a model id; for
# Azure OpenAI set the *deployment* name. Empty → use automatic stage selection.
#   e.g.  LLM_MODEL_CHEAP=gpt-4o-mini   LLM_MODEL_COMPLEX=claude-sonnet-4.6
LLM_MODEL_OVERRIDES: dict[str, str] = {
    "cheap": os.getenv("LLM_MODEL_CHEAP", ""),
    "standard": os.getenv("LLM_MODEL_STANDARD", ""),
    "complex": os.getenv("LLM_MODEL_COMPLEX", ""),
    "embedding": os.getenv("LLM_MODEL_EMBEDDING", ""),
}
