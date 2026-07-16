"""API routes: ``/health``, ``POST /api/ask`` and ``POST /api/ask/stream`` (SSE).

The HTML/SPA serving lives in ``api/app.py``; this module is the JSON/stream API
layer consumed by the React frontend (and any other client).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from chatbot import defect_qa
from chatbot.retriever import get_retriever

router = APIRouter()

MAX_QUESTION_LEN = 2000


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LEN)
    mode: str | None = Field(
        default=None,
        description="Optional routing hint: 'defect', or None for auto.",
    )
    history: list[dict] | None = Field(
        default=None,
        description="Optional recent turns [{question, answer}]; only the last 4 are used.",
    )
    sections: list[str] | None = Field(
        default=None,
        description=(
            "Selected defect answer sections (any of 'root_cause', 'resolve', "
            "'similar'). When omitted for a defect question, the API returns an "
            "options picker instead of a full answer."
        ),
    )


@router.get("/health")
@router.head("/health")
def health() -> dict:
    """Lightweight liveness/readiness endpoint.

    Always returns HTTP 200 with at least ``{status, service, timestamp}`` so
    external uptime monitors (e.g. UptimeRobot pinging this on a schedule to
    keep a free-tier Render instance from spinning down) never see a false
    "down" reading. FastAPI/Starlette automatically serve HEAD requests for a
    GET route, so no separate handler is needed for HEAD.

    The richer diagnostic fields (subsystem readiness, LLM provider, etc.)
    used by the frontend's connection badge are added on a best-effort basis:
    if any subsystem lookup raises, it's simply omitted rather than causing
    this endpoint to error, per requirement that it work even when other
    services are unavailable.
    """
    payload: dict = {
        "status": "ok",
        "service": "cb-l1-support",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        retr = get_retriever()

        # Reflect the ACTIVE LLM provider, not just Azure. The chatbot supports
        # azure / github / copilot; the badge should be "on" when the selected
        # provider has its credentials configured.
        provider = (config.LLM_PROVIDER or "azure").strip().lower()
        if provider == "github":
            llm_ready = bool(config.GITHUB_TOKEN)
        elif provider == "copilot":
            llm_ready = bool(config.COPILOT_TOKEN or config.GITHUB_TOKEN)
        else:  # azure
            llm_ready = bool(
                config.USE_AZURE_OPENAI
                and config.AZURE_OPENAI_ENDPOINT
                and config.AZURE_OPENAI_API_KEY
            )

        payload.update(
            {
                "knowledge_base_loaded": bool(retr.kb),
                "vector_store_loaded": retr.ready,
                "llm_enabled": bool(config.USE_LLM and llm_ready),
                "llm_provider": provider,
                "rag_docs_enabled": bool(config.USE_RAG_DOCS),
                "top_k": config.TOP_K_RESULTS,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive; must never 500
        payload["diagnostics_error"] = f"{type(exc).__name__}: {exc}"

    return payload


@router.get("/health/rag")
def health_rag() -> dict:
    """Diagnostic: run a real document-RAG query and surface the exact error.

    Useful for debugging deployments where the documentation section silently
    fails to appear (the live answer path swallows RAG errors into ``None``).
    """
    if not config.USE_RAG_DOCS:
        return {"ok": False, "error": "USE_RAG_DOCS is disabled"}
    try:
        from rag_docs import pipeline_diagnostics
        return pipeline_diagnostics()
    except Exception as exc:  # import error etc.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _validate(question: str) -> str:
    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if len(question) > MAX_QUESTION_LEN:
        raise HTTPException(status_code=400, detail="Question is too long (max 2000 chars).")
    return question


@router.post("/api/ask")
def ask(req: AskRequest) -> dict:
    """Non-streaming answer: returns the full ``{intent, answer, similar_defects}``."""
    question = _validate(req.question)
    try:
        return defect_qa.ask(
            question, mode=req.mode, history=req.history, sections=req.sections
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@router.post("/api/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Server-Sent Events stream of the answer.

    Emits ``meta`` (intent), repeated ``token`` events, a ``sources`` event, then
    ``done``. On error an ``error`` event is sent before the stream closes.
    """
    question = _validate(req.question)

    def event_source():
        try:
            for event in defect_qa.ask_stream(
                question, mode=req.mode, history=req.history, sections=req.sections
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # pragma: no cover - defensive
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/rag/ingest")
async def ingest_pdf(file: UploadFile = File(...)) -> dict:
    """On-demand documentation ingestion (header upload button).

    Parses the uploaded PDF into text chunks, embeds them, and upserts them
    into the same Qdrant collection the documentation RAG searches — so the
    new document is answerable immediately, no rebuild/restart required.
    """
    filename, data = await _validate_ingest_upload(file)

    try:
        from rag_docs import ingest_pdf as run_ingest

        result = run_ingest(filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - env dependent
        raise HTTPException(
            status_code=500, detail=f"Ingestion failed: {exc}"
        ) from exc

    return {"status": "ok", **result}


async def _validate_ingest_upload(file: UploadFile) -> tuple[str, bytes]:
    """Shared pre-flight checks for both the sync and streaming ingest routes.

    Returns ``(filename, file_bytes)`` or raises ``HTTPException`` with a clear
    status/detail for the first failing check.
    """
    if not config.USE_RAG_DOCS:
        raise HTTPException(
            status_code=400, detail="Documentation RAG is disabled on this server."
        )

    if not config.LLAMA_CLOUD_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "Documentation ingestion is not configured on this server "
                "(missing LLAMA_CLOUD_API_KEY)."
            ),
        )

    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_bytes = config.RAG_INGEST_MAX_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {config.RAG_INGEST_MAX_MB} MB upload limit.",
        )

    return filename, data


@router.post("/api/rag/ingest/stream")
async def ingest_pdf_stream(file: UploadFile = File(...)) -> StreamingResponse:
    """Streaming variant of ``/api/rag/ingest`` (Server-Sent Events).

    Emits a progress event before each pipeline stage — parsing (LlamaParse),
    chunking, embedding/upserting — so the UI can show exactly what the
    backend is doing, then a final ``{"stage": "done", "result": {...}}``
    event. On error, a single ``{"stage": "error", "message": ...}`` event is
    sent before the stream closes.
    """
    filename, data = await _validate_ingest_upload(file)

    def event_source():
        try:
            from rag_docs import ingest_pdf_stages

            for event in ingest_pdf_stages(filename, data):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(exc)})}\n\n"
        except Exception as exc:  # pragma: no cover - env dependent
            yield (
                "data: "
                + json.dumps({"stage": "error", "message": f"Ingestion failed: {exc}"})
                + "\n\n"
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class AddDefectRequest(BaseModel):
    issue_key: str = Field(..., min_length=3, max_length=40)


def _validate_add_defect_config() -> None:
    """Shared pre-flight check for both the sync and streaming add-defect
    routes: Jira must be configured on this server."""
    if not (config.JIRA_URL and config.JIRA_USERNAME and config.JIRA_API_TOKEN):
        raise HTTPException(
            status_code=500,
            detail=(
                "Jira is not configured on this server (missing JIRA_URL / "
                "JIRA_USERNAME / JIRA_API_TOKEN)."
            ),
        )


@router.post("/api/defects/add")
def add_defect(req: AddDefectRequest) -> dict:
    """On-demand defect ingestion (header 'Add defect' button).

    Fetches the given Jira key live, runs it through the same normalization +
    chunking pipeline used for the bulk knowledge base, embeds it, and
    upserts it into the same Qdrant "defect" collection the retriever
    searches — so it's answerable immediately, no rebuild/restart required.
    """
    _validate_add_defect_config()
    try:
        from chatbot.defect_ingest import add_defect as run_add_defect

        result = run_add_defect(req.issue_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - env dependent
        raise HTTPException(status_code=500, detail=f"Add defect failed: {exc}") from exc

    return {"status": "ok", **result}


@router.post("/api/defects/add/stream")
def add_defect_stream(req: AddDefectRequest) -> StreamingResponse:
    """Streaming variant of ``/api/defects/add`` (Server-Sent Events).

    Emits a progress event before each pipeline stage — fetching (Jira),
    normalizing, chunking, embedding/upserting — then a final
    ``{"stage": "done", "result": {...}}`` event. On error, a single
    ``{"stage": "error", "message": ...}`` event is sent before the stream
    closes.
    """
    _validate_add_defect_config()

    def event_source():
        try:
            from chatbot.defect_ingest import add_defect_stages

            for event in add_defect_stages(req.issue_key):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(exc)})}\n\n"
        except Exception as exc:  # pragma: no cover - env dependent
            yield (
                "data: "
                + json.dumps({"stage": "error", "message": f"Add defect failed: {exc}"})
                + "\n\n"
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class AddDefectsByJqlRequest(BaseModel):
    jql: str = Field(..., min_length=3, max_length=2000)


@router.post("/api/defects/add-jql")
def add_defects_by_jql(req: AddDefectsByJqlRequest) -> dict:
    """Bulk on-demand defect ingestion by JQL query (header 'Add defect'
    button, bulk mode).

    Runs the JQL query against Jira, then for EVERY matching issue: derives
    the same enriched columns as the bulk knowledge base (root cause, fix
    applied, fix pattern, defect type, quality score, is_fixed/is_cancelled),
    chunks it, embeds it, and upserts it into the same Qdrant "defect"
    collection the retriever searches — all matched defects become
    answerable immediately, no rebuild/restart required. Capped at
    ``config.JQL_ADD_MAX_RESULTS`` defects per request.
    """
    _validate_add_defect_config()
    try:
        from chatbot.defect_ingest import add_defects_by_jql as run_add_defects_by_jql

        result = run_add_defects_by_jql(req.jql)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - env dependent
        raise HTTPException(status_code=500, detail=f"Bulk add failed: {exc}") from exc

    return {"status": "ok", **result}


@router.post("/api/defects/add-jql/stream")
def add_defects_by_jql_stream(req: AddDefectsByJqlRequest) -> StreamingResponse:
    """Streaming variant of ``/api/defects/add-jql`` (Server-Sent Events).

    Emits a progress event before each pipeline stage — searching Jira,
    normalizing each matched defect (batched every 10), embedding/upserting —
    then a final ``{"stage": "done", "result": {...}}`` event summarizing how
    many defects/chunks were added. On error, a single ``{"stage": "error",
    "message": ...}`` event is sent before the stream closes.
    """
    _validate_add_defect_config()

    def event_source():
        try:
            from chatbot.defect_ingest import add_defects_by_jql_stages

            for event in add_defects_by_jql_stages(req.jql):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(exc)})}\n\n"
        except Exception as exc:  # pragma: no cover - env dependent
            yield (
                "data: "
                + json.dumps({"stage": "error", "message": f"Bulk add failed: {exc}"})
                + "\n\n"
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


