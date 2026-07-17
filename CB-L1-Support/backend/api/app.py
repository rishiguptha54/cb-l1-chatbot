"""FastAPI application factory for the CB L1 Support Chatbot.

Starts quickly so cloud platforms can detect an open port. Heavy retriever and
RAG client initialization is lazy (on first request) rather than blocking app
startup.

Run with either:
    python run_chatbot.py --serve
    uvicorn api.app:app --host 0.0.0.0 --port 5100
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

# Directory that holds the built React app (produced by `npm run build`).
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep startup non-blocking on Render; expensive subsystem checks happen lazily.
    print("[api] startup complete (lazy init enabled)")
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
