"""Provider-agnostic LLM abstraction layer.

A single ``LLM_PROVIDER`` setting (see ``config.py``) selects the backend:

* ``azure``  → Azure OpenAI, using the existing ``AZURE_OPENAI_*`` endpoint,
  key and *deployment* names. Model ids in the catalog map to deployments.
* ``github`` → GitHub Models, an OpenAI-compatible inference endpoint
  (``https://models.inference.ai.azure.com``) authenticated with a token from
  the ``GITHUB_TOKEN`` env var or, failing that, ``gh auth token``.

Public interface (identical regardless of provider)::

    from llm_provider import chat, embed, list_models, pick_model

    chat(messages, stage="standard")          -> str
    chat(messages, stage="cheap", stream=True) -> Iterator[str]
    embed(["text a", "text b"])                -> list[list[float]]
    list_models()                              -> list[str]   (discovered ∪ fallback)
    pick_model("complex")                      -> str         (best available id)

Each call is tagged with a *stage* (``cheap`` / ``standard`` / ``complex`` /
``embedding``). The layer auto-discovers the provider's catalogue from its
``/models`` endpoint (cached, with a hardcoded fallback) and selects the first
stage-preferred model the provider actually advertises. Code-completion models
that don't support ``/chat/completions`` are excluded centrally.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from typing import Iterable, Iterator

import config

# Stages that select a chat model vs. an embedding model.
_CHAT_STAGES = {"cheap", "standard", "complex"}
_EMBED_STAGES = {"embedding"}


# ─────────────────────────────────────────────────────────────────────────────
#  Provider plumbing
# ─────────────────────────────────────────────────────────────────────────────
def _provider() -> str:
    """Return the active provider id ("azure", "github" or "copilot")."""
    return (config.LLM_PROVIDER or "azure").strip().lower()


def _embed_provider() -> str:
    """Return the provider used for embeddings (may differ from chat).

    Copilot has no 3072-dim embedding model, so embeddings can be routed to a
    different backend (e.g. GitHub Models) via ``EMBED_PROVIDER``.
    """
    return (getattr(config, "EMBED_PROVIDER", "") or _provider()).strip().lower()


def _github_token() -> str:
    """Resolve a GitHub token: GITHUB_TOKEN env first, then ``gh auth token``."""
    if config.GITHUB_TOKEN:
        return config.GITHUB_TOKEN.strip()
    try:
        out = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        token = (out.stdout or "").strip()
        if token:
            return token
    except Exception:  # pragma: no cover - gh not installed / not logged in
        pass
    raise ValueError(
        "No GitHub token available. Set GITHUB_TOKEN in .env or run `gh auth login`."
    )


def _copilot_token() -> str:
    """Resolve the GitHub Copilot API token: COPILOT_TOKEN, else GITHUB_TOKEN."""
    if config.COPILOT_TOKEN:
        return config.COPILOT_TOKEN.strip()
    return _github_token()


@lru_cache(maxsize=2)
def _client(provider: str):
    """Build (and cache) the OpenAI-compatible client for a provider."""
    if provider == "copilot":
        import httpx
        from openai import OpenAI

        return OpenAI(
            base_url=config.COPILOT_API_ENDPOINT,
            api_key=_copilot_token(),
            default_headers=config.copilot_headers(),
            http_client=httpx.Client(verify=False, timeout=60.0),
            max_retries=3,
        )

    if provider == "github":
        import httpx
        from openai import OpenAI

        return OpenAI(
            base_url=config.GITHUB_MODELS_ENDPOINT,
            api_key=_github_token(),
            http_client=httpx.Client(verify=False, timeout=60.0),
            timeout=60.0,
            max_retries=3,
        )

    # Default: Azure OpenAI.
    from openai import AzureOpenAI

    if not (config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY):
        raise ValueError(
            "Azure OpenAI not configured. Set AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_API_KEY in .env (or switch LLM_PROVIDER=github)."
        )
    return AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        timeout=60.0,
        max_retries=3,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Model discovery + stage selection
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_catalog() -> set[str]:
    """Union of every model named in the configured stage lists."""
    catalog: set[str] = set()
    for ids in config.LLM_STAGE_MODELS.values():
        catalog.update(ids)
    return catalog - config.LLM_EXCLUDED_MODELS


@lru_cache(maxsize=2)
def _discover_models(provider: str) -> frozenset[str]:
    """Discover the model ids the provider *actually* advertises via ``/models``.

    Returns only the real, advertised ids (an empty set if discovery fails — no
    fallback union here, so selection can tell "advertised" apart from "guessed").
    Cached per-provider; call :func:`list_models` with ``refresh=True`` to rebuild.
    """
    ids: set[str] = set()
    try:
        if provider == "copilot":
            import httpx

            resp = httpx.get(
                f"{config.COPILOT_API_ENDPOINT}/models",
                headers={
                    "Authorization": f"Bearer {_copilot_token()}",
                    **config.copilot_headers(),
                },
                timeout=15.0,
                verify=False,
            )
            resp.raise_for_status()
            # Copilot /models returns rows with a usable API ``id`` (e.g.
            # "gpt-4o") plus a human ``name`` ("GPT-4o"). The id is what the
            # chat/embeddings endpoints accept, so parse ids directly here
            # (unlike GitHub Models, where ``name`` is the id).
            payload = resp.json()
            rows = payload.get("data", payload) if isinstance(payload, dict) else payload
            ids = {
                m.get("id") for m in rows
                if isinstance(m, dict) and m.get("id")
            }
        elif provider == "github":
            # GitHub Models' /models returns a bare JSON array, so query it
            # directly with httpx (bundled with the openai dependency) rather
            # than the OpenAI SDK paginator (which expects {"data": [...]}).
            import httpx

            resp = httpx.get(
                f"{config.GITHUB_MODELS_ENDPOINT}/models",
                headers={"Authorization": f"Bearer {_github_token()}"},
                timeout=15.0,
            )
            resp.raise_for_status()
            ids = _parse_model_ids(resp.json())
        else:  # azure
            # Azure exposes *deployments*, not an OpenAI-style /models list, so
            # the "advertised" set is whatever deployments are configured.
            ids = {
                d for d in (
                    config.AZURE_OPENAI_CHAT_DEPLOYMENT,
                    config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                    config.AZURE_OPENAI_DEPLOYMENT,
                ) if d
            }
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[llm] model discovery failed for {provider} ({exc}); using fallback.")

    return frozenset(ids - config.LLM_EXCLUDED_MODELS)


def _available_models(provider: str) -> frozenset[str]:
    """Advertised models if discovery succeeded, else the hardcoded fallback."""
    discovered = _discover_models(provider)
    if discovered:
        return discovered
    return frozenset(_fallback_catalog())


def _parse_model_ids(payload) -> set[str]:
    """Extract model ids from a /models response.

    Accepts an OpenAI SDK page (iterable of objects with ``.id``), a list, or a
    ``{"data": [...]}`` mapping. Azure ML registry paths such as
    ``azureml://registries/azure-openai/models/gpt-4o/versions/2`` are
    normalized to the short model name (``gpt-4o``).
    """
    rows = getattr(payload, "data", None)
    if rows is None:
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    ids: set[str] = set()
    try:
        iterator = list(rows)
    except TypeError:
        return ids
    for row in iterator:
        if isinstance(row, dict):
            # Prefer the short name; fall back to id.
            mid = row.get("name") or row.get("id")
        elif isinstance(row, str):
            mid = row
        else:
            mid = getattr(row, "name", None) or getattr(row, "id", None)
        if mid:
            ids.add(_normalize_model_id(str(mid)))
    return ids


def _normalize_model_id(model_id: str) -> str:
    """Reduce an Azure ML registry path to its bare model name."""
    if "azureml://" in model_id and "/models/" in model_id:
        tail = model_id.split("/models/", 1)[1]
        return tail.split("/", 1)[0]
    return model_id


def list_models(provider: str | None = None, refresh: bool = False) -> list[str]:
    """Return the sorted available model ids for a provider (advertised, else fallback)."""
    prov = (provider or _provider()).lower()
    if refresh:
        _discover_models.cache_clear()
    return sorted(_available_models(prov))


def pick_model(stage: str, provider: str | None = None) -> str:
    """Select the best usable model id for ``stage`` on the active provider.

    Resolution order:
      1. Explicit env override for the stage (``LLM_MODEL_OVERRIDES``).
      2. For Azure: the matching configured *deployment* (chat or embedding).
      3. First stage-preferred model the provider actually advertises.
      4. If none of the stage's preferred models are advertised, degrade to
         another advertised model of the right kind (chat vs embedding) so the
         call still succeeds instead of 400-ing on an unavailable id.
      5. If discovery failed entirely, trust the first stage preference.
    """
    prov = (provider or _provider()).lower()
    preferences = config.LLM_STAGE_MODELS.get(stage)
    if not preferences:
        raise ValueError(
            f"Unknown stage {stage!r}. Valid stages: "
            f"{', '.join(config.LLM_STAGE_MODELS)}."
        )

    override = config.LLM_MODEL_OVERRIDES.get(stage, "").strip()
    if override:
        return override

    if prov == "azure":
        if stage in _EMBED_STAGES:
            if config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
                return config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        elif config.AZURE_OPENAI_CHAT_DEPLOYMENT:
            return config.AZURE_OPENAI_CHAT_DEPLOYMENT

    advertised = _discover_models(prov)

    # Discovery failed → trust the curated preference order.
    if not advertised:
        for model in preferences:
            if model not in config.LLM_EXCLUDED_MODELS:
                return model
        raise ValueError(f"No selectable model for stage {stage!r}.")

    # Prefer a stage-preferred model that is actually advertised.
    for model in preferences:
        if model not in config.LLM_EXCLUDED_MODELS and model in advertised:
            return model

    # None of the preferred ids are advertised → degrade to an available model.
    return _degrade(stage, advertised)


def _degrade(stage: str, advertised: frozenset[str]) -> str:
    """Pick a sensible advertised model when no stage preference is available."""
    is_embed = stage in _EMBED_STAGES
    usable = sorted(
        m for m in advertised
        if m not in config.LLM_EXCLUDED_MODELS
        and (("embed" in m.lower()) == is_embed)
    )
    if not usable:
        usable = sorted(m for m in advertised if m not in config.LLM_EXCLUDED_MODELS)
    if not usable:
        raise ValueError(f"No usable model advertised for stage {stage!r}.")

    if is_embed:
        return usable[0]

    # For chat, borrow from another stage's preferences (in cost order) so we
    # land on a still-appropriate advertised model before any arbitrary one.
    borrow_order = {
        "cheap": ["cheap", "standard", "complex"],
        "standard": ["standard", "complex", "cheap"],
        "complex": ["complex", "standard", "cheap"],
    }.get(stage, ["standard", "complex", "cheap"])
    usable_set = set(usable)
    for other in borrow_order:
        for model in config.LLM_STAGE_MODELS.get(other, []):
            if model in usable_set:
                return model
    return usable[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Unified chat / embed interface
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  Provider fallback (e.g. GitHub Models rate-limited → Azure OpenAI)
# ─────────────────────────────────────────────────────────────────────────────
def _azure_keys() -> bool:
    """True when Azure OpenAI endpoint + key are configured."""
    return bool(config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY)


def _azure_chat_configured() -> bool:
    return _azure_keys() and bool(
        config.AZURE_OPENAI_CHAT_DEPLOYMENT or config.AZURE_OPENAI_DEPLOYMENT
    )


def _azure_embed_configured() -> bool:
    return _azure_keys() and bool(config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT)


def _provider_chain(primary: str, *, embedding: bool = False) -> list[str]:
    """Ordered providers to try: the active one, then Azure OpenAI as a fallback.

    Azure is appended only when it is configured (and isn't already primary), so
    a rate-limited / unavailable primary provider degrades to Azure instead of
    failing straight to the deterministic template.
    """
    chain = [primary]
    azure_ok = _azure_embed_configured() if embedding else _azure_chat_configured()
    if primary != "azure" and azure_ok:
        chain.append("azure")
    return chain


def _should_fallback(exc: Exception) -> bool:
    """True for rate limits and transient/availability errors worth retrying on
    another provider."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    name = exc.__class__.__name__.lower()
    return any(
        k in name
        for k in ("ratelimit", "apiconnection", "apitimeout", "internalserver")
    )


def _short(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    return text if len(text) <= 160 else text[:157] + "..."


def _params_for(provider: str, messages: list[dict], model: str | None,
                stage: str, temperature: float, max_tokens: int, kwargs: dict) -> dict:
    """Build chat params, re-selecting the model for the given provider."""
    chosen = model or pick_model(stage, provider)
    return dict(
        model=chosen,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    stage: str = "standard",
    temperature: float = 0.2,
    max_tokens: int = 1024,
    stream: bool = False,
    **kwargs,
):
    """Chat completion against the active provider, with Azure OpenAI fallback.

    ``model`` overrides automatic stage selection. With ``stream=False`` returns
    the assistant text; with ``stream=True`` returns an iterator of text chunks.
    If the active provider is rate-limited (HTTP 429) or hits a transient error,
    the call automatically retries against Azure OpenAI when it is configured.
    """
    if stream:
        return _chat_stream_resilient(
            messages, model, stage, temperature, max_tokens, kwargs
        )
    return _chat_once_resilient(
        messages, model, stage, temperature, max_tokens, kwargs
    )


def _chat_once_resilient(messages, model, stage, temperature, max_tokens, kwargs) -> str:
    primary = _provider()
    chain = _provider_chain(primary)
    for idx, provider in enumerate(chain):
        # Only force the caller's explicit model on the primary provider; let a
        # fallback provider choose its own appropriate model/deployment.
        mdl = model if provider == primary else None
        try:
            client = _client(provider)
            params = _params_for(
                provider, messages, mdl, stage, temperature, max_tokens, kwargs
            )
            return _chat_once(client, params)
        except Exception as exc:
            if idx < len(chain) - 1 and _should_fallback(exc):
                print(f"[llm] {provider} chat failed ({_short(exc)}); "
                      f"falling back to {chain[idx + 1]}.")
                continue
            raise


def _chat_stream_resilient(messages, model, stage, temperature, max_tokens, kwargs):
    """Generator that streams from the primary provider, switching to Azure if
    the stream fails to start (e.g. 429). Once tokens begin, it never switches
    mid-stream to avoid duplicated output."""
    primary = _provider()
    chain = _provider_chain(primary)
    for idx, provider in enumerate(chain):
        mdl = model if provider == primary else None
        try:
            client = _client(provider)
            params = dict(
                _params_for(provider, messages, mdl, stage, temperature, max_tokens, kwargs),
                stream=True,
            )
            stream = client.chat.completions.create(**_sanitize(client, params))
        except Exception as exc:
            if idx < len(chain) - 1 and _should_fallback(exc):
                print(f"[llm] {provider} stream failed ({_short(exc)}); "
                      f"falling back to {chain[idx + 1]}.")
                continue
            raise
        yield from _iter_stream(stream)
        return


def _chat_once(client, params: dict) -> str:
    resp = client.chat.completions.create(**_sanitize(client, params))
    return (resp.choices[0].message.content or "").strip()


def _iter_stream(stream) -> Iterator[str]:
    for event in stream:
        if not getattr(event, "choices", None):
            continue
        piece = getattr(event.choices[0].delta, "content", None)
        if piece:
            yield piece


def _sanitize(client, params: dict) -> dict:
    """Adapt params for reasoning models that reject temperature/max_tokens.

    o1/o3/gpt-5 reasoning models use ``max_completion_tokens`` and a fixed
    temperature. Detect them by id and adjust so calls don't 400.
    """
    params = dict(params)
    model_id = str(params.get("model", "")).lower()
    is_reasoning = model_id.startswith(("o1", "o3", "o4")) or model_id.startswith(
        ("gpt-5",)
    )
    if is_reasoning:
        params.pop("temperature", None)
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
    return params


def embed(
    texts: Iterable[str],
    *,
    model: str | None = None,
    stage: str = "embedding",
) -> list[list[float]]:
    """Return embedding vectors for ``texts``, with Azure OpenAI fallback.

    If the active provider is rate-limited or unavailable, retries against Azure
    OpenAI when an embedding deployment is configured.
    """
    items = [t if (t and t.strip()) else " " for t in texts]
    if not items:
        return []

    primary = _embed_provider()
    chain = _provider_chain(primary, embedding=True)
    for idx, provider in enumerate(chain):
        mdl = model if provider == primary else None
        try:
            client = _client(provider)
            chosen = mdl or pick_model(stage, provider)
            resp = client.embeddings.create(model=chosen, input=items)
            return [d.embedding for d in resp.data]
        except Exception as exc:
            if idx < len(chain) - 1 and _should_fallback(exc):
                print(f"[llm] {provider} embed failed ({_short(exc)}); "
                      f"falling back to {chain[idx + 1]}.")
                continue
            raise


def provider_info() -> dict:
    """Diagnostic snapshot of the active provider and per-stage model choices."""
    prov = _provider()
    info: dict = {"provider": prov, "stages": {}}
    for stage in config.LLM_STAGE_MODELS:
        try:
            info["stages"][stage] = pick_model(stage)
        except Exception as exc:  # pragma: no cover
            info["stages"][stage] = f"<error: {exc}>"
    return info
