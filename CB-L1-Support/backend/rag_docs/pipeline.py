"""
pipeline.py
───────────
In-process builder for the extracted document-RAG pipeline.

This mirrors the ORIGINAL project's ``api.build_query_pipeline()`` exactly —
same environment variables, same defaults, same construction order — but is
called directly from this project instead of over HTTP.  The heavy pipeline
(VectorStore + AnswerGenerator + QueryEnhancer + RAGAgent) is built ONCE and
cached, so only the first question pays the startup cost.

Public API
──────────
    answer_question(question, source_filter=None, conversation_history=None)
        -> (answer: str, timing: dict)
    pipeline_health() -> bool        # True when the pipeline is constructible
    get_agent()       -> RAGAgent    # the cached singleton (builds on first call)

Required environment variables (same names as the original RAG project):
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    QDRANT_URL
    QDRANT_API_KEY
Optional (defaults match the original project):
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT  (default "text-embedding-3-large")
    AZURE_OPENAI_API_VERSION           (default "2024-02-01")
    QDRANT_COLLECTION                  (default "rag_collection")
    AZURE_OPENAI_CHAT_DEPLOYMENT       (default "gpt-4o")
    QUERY_ENHANCER_DEPLOYMENT          (default = chat deployment)
    QUERY_N_VARIANTS                   (default "3")
    RAG_TOP_K                          (default "5")
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# CORPORATE SSL PROXY BYPASS  (must run before any network imports — copied
# verbatim from the original project's api.py so behaviour is identical behind
# the Zscaler/corporate inspection proxy).
# ══════════════════════════════════════════════════════════════════════════════
import os
import ssl
import warnings

ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import httpx as _httpx

_orig_client_init = _httpx.Client.__init__
def _no_verify_client_init(self, *args, **kwargs):          # type: ignore[misc]
    kwargs["verify"] = False
    _orig_client_init(self, *args, **kwargs)
_httpx.Client.__init__ = _no_verify_client_init             # type: ignore[method-assign]

_orig_async_client_init = _httpx.AsyncClient.__init__
def _no_verify_async_client_init(self, *args, **kwargs):    # type: ignore[misc]
    kwargs["verify"] = False
    _orig_async_client_init(self, *args, **kwargs)
_httpx.AsyncClient.__init__ = _no_verify_async_client_init  # type: ignore[method-assign]

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # urllib3 always present via requests, but stay safe
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Pipeline imports (safe after the SSL bypass above) ────────────────────────
import logging
from typing import Any, Dict, List, Optional, Tuple

from .answer_generator import AnswerGenerator
from .llm_factory import rag_provider
from .query_processor import QueryEnhancer
from .rag_agent import RAGAgent
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

_agent: Optional[RAGAgent] = None


def build_query_pipeline() -> RAGAgent:
    """Initialise VectorStore + AnswerGenerator + QueryEnhancer + RAGAgent.

    Identical construction to the original project's api.build_query_pipeline(),
    except the chat/embedding backend is selected by ``RAG_LLM_PROVIDER``
    ("azure" or "copilot").  In "copilot" mode the Azure OpenAI env vars are not
    required — only Qdrant + a Copilot token.
    """
    provider = rag_provider()
    if provider in ("copilot", "github"):
        import config
        from .llm_factory import rag_embed_provider

        required = ["QDRANT_URL", "QDRANT_API_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if provider == "copilot" and not config.COPILOT_TOKEN:
            missing.append("COPILOT_TOKEN (or GITHUB_TOKEN)")
        # GitHub Models (chat and/or 3072-dim embeddings) needs a GitHub token.
        if (provider == "github" or rag_embed_provider() == "github") and not config.GITHUB_TOKEN:
            missing.append("GITHUB_TOKEN")
    else:
        required = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "QDRANT_URL",
            "QDRANT_API_KEY",
        ]
        missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables for the document-RAG "
            f"pipeline (provider={provider}): {missing}. Add them to your "
            f"environment / .env file."
        )

    vector_store = VectorStore(
        qdrant_url=os.environ["QDRANT_URL"],
        qdrant_api_key=os.environ["QDRANT_API_KEY"],
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_embedding_deployment=os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
        ),
        azure_openai_api_version=os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-01"
        ),
        collection_name=os.getenv("QDRANT_COLLECTION", "rag_collection"),
    )

    answer_generator = AnswerGenerator(
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
        azure_openai_api_version=os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-01"
        ),
    )

    query_enhancer = QueryEnhancer(
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_chat_deployment=os.getenv(
            "QUERY_ENHANCER_DEPLOYMENT",
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
        ),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        n_variants=int(os.getenv("QUERY_N_VARIANTS", "3")),
    )

    return RAGAgent(
        vector_store=vector_store,
        answer_generator=answer_generator,
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        query_enhancer=query_enhancer,
    )


def get_agent() -> RAGAgent:
    """Return the cached RAGAgent, building it on first use."""
    global _agent
    if _agent is None:
        _agent = build_query_pipeline()
    return _agent


def get_vector_store() -> VectorStore:
    """Return the cached pipeline's VectorStore (builds the pipeline on first
    use). Used by the on-demand PDF ingestion endpoint to upsert new document
    chunks into the same Qdrant collection the query-time pipeline searches."""
    return get_agent()._vector_store


def answer_question(
    question: str,
    source_filter: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, float]]:
    """Run the full RAG pipeline for one question.

    Args:
        question: natural-language question.
        source_filter: optional PDF filename to restrict retrieval.
        conversation_history: optional list of ``{"role", "content"}`` turns.

    Returns:
        (answer_string, timing_dict) — same shape as RAGAgent.ask().
    """
    agent = get_agent()
    return agent.ask(
        question,
        source_filter=source_filter,
        conversation_history=conversation_history,
    )


def pipeline_health() -> bool:
    """Return True when the pipeline can be constructed (env + Qdrant reachable)."""
    try:
        get_agent()
        return True
    except Exception as exc:
        logger.warning("[RAG_DOCS] pipeline health check failed: %s", exc)
        return False


def pipeline_diagnostics(question: str = "What is this documentation about?") -> Dict[str, Any]:
    """Run a real end-to-end RAG query and report the outcome (or the exact
    error). Intended for a diagnostic endpoint so failures are visible instead
    of being swallowed into ``None``."""
    import traceback

    info: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "rag_provider": rag_provider(),
        "env_present": {
            k: bool(os.getenv(k))
            for k in (
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "QDRANT_URL",
                "QDRANT_API_KEY",
            )
        },
        "collection": os.getenv("QDRANT_COLLECTION", "rag_collection"),
        "embedding_deployment": os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
        ),
        "chat_deployment": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    }
    try:
        answer, timing = answer_question(question)
        info["ok"] = True
        info["answer_len"] = len((answer or "").strip())
        info["timing"] = timing
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["traceback"] = traceback.format_exc()[-2000:]
    return info
