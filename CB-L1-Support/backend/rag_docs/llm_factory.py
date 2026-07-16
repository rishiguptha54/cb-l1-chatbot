"""
llm_factory.py
──────────────
Backend selection for the document-RAG pipeline's chat + embedding clients.

The original pipeline hardcoded Azure OpenAI (``AzureChatOpenAI`` /
``AzureOpenAIEmbeddings``), which is only reachable from the corporate network.
This factory adds a provider switch so the SAME pipeline can run against the
publicly reachable GitHub Copilot API when deployed (e.g. on Render).

Selected via ``RAG_LLM_PROVIDER`` (config / env):
    "azure"   → AzureChatOpenAI / AzureOpenAIEmbeddings (corporate network).
    "copilot" → GitHub Copilot API (PAT-as-Bearer, OpenAI-compatible).

The returned objects are LangChain ``BaseChatModel`` / ``Embeddings`` instances,
so the rest of the pipeline (``invoke`` / ``stream`` / ``embed_query``) is
unchanged regardless of provider.
"""

from __future__ import annotations

import os

import httpx

import config


def rag_provider() -> str:
    """Active RAG chat backend id ("azure" or "copilot")."""
    return (
        os.getenv("RAG_LLM_PROVIDER")
        or getattr(config, "RAG_LLM_PROVIDER", "azure")
    ).strip().lower()


def rag_embed_provider() -> str:
    """Active RAG embedding backend id ("azure", "github" or "copilot").

    Embeddings may use a different backend than chat: the Copilot API has no
    3072-dim embedding model, so the 3072-dim Qdrant collection is served by
    GitHub Models when chat runs on Copilot.
    """
    return (
        os.getenv("RAG_EMBED_PROVIDER")
        or getattr(config, "RAG_EMBED_PROVIDER", "")
        or rag_provider()
    ).strip().lower()


def build_chat_llm(
    *,
    azure_deployment: str,
    azure_endpoint: str,
    azure_api_key: str,
    azure_api_version: str = "2024-02-01",
    max_tokens: int | None = None,
    timeout: int = 120,
):
    """Return a LangChain chat model for the active RAG provider."""
    if rag_provider() == "github":
        from langchain_openai import ChatOpenAI

        # GitHub Models is OpenAI-compatible and publicly reachable (works on
        # Render). Chat runs on RAG_CHAT_MODEL (e.g. gpt-4o) with GITHUB_TOKEN.
        kwargs: dict = dict(
            model=config.RAG_CHAT_MODEL,
            base_url=config.GITHUB_MODELS_ENDPOINT,
            api_key=config.GITHUB_TOKEN or "",
            http_client=httpx.Client(verify=False, timeout=timeout),
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    if rag_provider() == "copilot":
        from langchain_openai import ChatOpenAI

        kwargs: dict = dict(
            model=config.RAG_CHAT_MODEL,
            base_url=config.COPILOT_API_ENDPOINT,
            api_key=config.COPILOT_TOKEN or "",
            default_headers=config.copilot_headers(),
            http_client=httpx.Client(verify=False, timeout=timeout),
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    from langchain_openai import AzureChatOpenAI

    kwargs = dict(
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_version,
        http_client=httpx.Client(verify=False, timeout=timeout),
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return AzureChatOpenAI(**kwargs)


def build_embeddings(
    *,
    azure_deployment: str,
    azure_endpoint: str,
    azure_api_key: str,
    azure_api_version: str = "2024-02-01",
    timeout: int = 60,
):
    """Return a LangChain embeddings client for the active RAG embed provider."""
    provider = rag_embed_provider()

    if provider in ("copilot", "github"):
        from langchain_openai import OpenAIEmbeddings

        # GitHub Models hosts text-embedding-3-large (3072-dim), matching the
        # existing Qdrant collection; the Copilot API does not. Route either
        # OpenAI-compatible backend the same way, differing only in base_url +
        # token. check_embedding_ctx_length=False sends raw text (not tiktoken
        # token arrays), matching the working curl test.
        if provider == "github":
            base_url = config.GITHUB_MODELS_ENDPOINT
            api_key = config.GITHUB_TOKEN or ""
            default_headers = None
        else:  # copilot
            base_url = config.COPILOT_API_ENDPOINT
            api_key = config.COPILOT_TOKEN or ""
            default_headers = config.copilot_headers()

        kwargs: dict = dict(
            model=config.RAG_EMBED_MODEL,
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=False, timeout=timeout),
            check_embedding_ctx_length=False,
        )
        if default_headers:
            kwargs["default_headers"] = default_headers
        return OpenAIEmbeddings(**kwargs)

    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_version,
        http_client=httpx.Client(verify=False, timeout=timeout),
    )
