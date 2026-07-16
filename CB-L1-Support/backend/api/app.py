"""FastAPI application factory for the CB L1 Support Chatbot.

Loads the knowledge base, FAISS index and embedding metadata once at startup
(not per request) so the first user query is fast.

Run with either:
    python run_chatbot.py --serve
    uvicorn api.app:app --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import config
from api.routes import router
from chatbot.retriever import get_retriever
from chatbot import rag_client

# Directory that holds the built React app (produced by `npm run build`).
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the singletons so artifacts are loaded once at boot.
    retr = get_retriever()
    # Non-blocking reachability check for the external documentation RAG service.
    if config.USE_RAG_DOCS:
        rag_status = "reachable" if rag_client.rag_health() else "unreachable"
    else:
        rag_status = "off"
    print(
        f"[api] kb={len(retr.kb)} "
        f"vector_store={'yes' if retr.ready else 'no'} "
        f"llm={'on' if (config.USE_LLM and config.USE_AZURE_OPENAI) else 'off'} "
        f"rag={rag_status}"
    )
    yield


app = FastAPI(title="CB L1 Support Chatbot", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API + health endpoints.
app.include_router(router)

# Serve the built React app's hashed assets (if a build exists).
_assets_dir = os.path.join(STATIC_DIR, "assets")
if os.path.isdir(_assets_dir):
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    """SPA catch-all: serve any built static file, else fall back to index.html.

    API routes are registered before this handler, so they take precedence.
    """
    # The HTML shell must never be cached, so UI changes show on a normal reload.
    no_cache = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    # Direct file hit (e.g. favicon, vite.svg, manifest).
    candidate = os.path.normpath(os.path.join(STATIC_DIR, full_path))
    if (
        full_path
        and candidate.startswith(STATIC_DIR)
        and os.path.isfile(candidate)
    ):
        return FileResponse(candidate)

    if os.path.isfile(INDEX_HTML):
        return FileResponse(INDEX_HTML, headers=no_cache)

    return HTMLResponse(
        "<h1>CB L1 Support Chatbot</h1>"
        "<p>Frontend not built yet. Run <code>npm install &amp;&amp; npm run build</code> "
        "in <code>frontend/</code>, or POST to <code>/api/ask</code>.</p>",
        status_code=200,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
