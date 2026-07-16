# ─────────────────────────────────────────────────────────────────────────────
# CB L1 Support Chatbot — single-service Docker image.
#
# NOTE: render.yaml no longer uses this file — it deploys via Render's native
# Python runtime instead, because Render's Docker runtime requires a verified
# payment method even for Free instances, while native runtimes do not (see
# render.yaml's header comment). This Dockerfile is kept as an optional path
# if you move to a paid Render plan (or any other Docker host) and want the
# frontend build handled automatically instead of a manual `npm run build` +
# commit step.
#
# Stage 1 builds the React/Vite SPA (Node). Its output is written straight
# into backend/api/static (see frontend/vite.config.ts `build.outDir`).
# Stage 2 is a slim Python runtime that serves both the API and the built SPA
# from the same FastAPI app/port (see backend/api/app.py), so the deployed
# service is a single Web Service with no CORS/proxy split needed.
#
# LLM provider: this image ships no hardcoded provider — it's selected entirely
# by the .env file mounted/injected at runtime (LLM_PROVIDER / RAG_LLM_PROVIDER
# / EMBED_PROVIDER / RAG_EMBED_PROVIDER). Production is currently configured
# for "azure" everywhere (see backend/.env). IMPORTANT: the Azure OpenAI
# resource used here is on a corporate network — confirm the host running
# this container can actually reach AZURE_OPENAI_ENDPOINT before deploying to
# a public cloud host; if not reachable, switch those four vars to "github"
# (publicly reachable GitHub Models) instead.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: build the frontend ──
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ──
FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Slim, production-only dependency set (no torch/sentence-transformers — the
# chatbot embeds queries via the remote LLM provider, see chatbot/build_embeddings.py).
COPY backend/requirements-deploy.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY backend/ ./
# Overlay the built SPA produced in stage 1 (outDir already targets this path;
# this COPY guarantees it's present even if a local build wasn't run first).
COPY --from=frontend-build /app/backend/api/static ./api/static

EXPOSE 10000

# Render injects $PORT at runtime; default 10000 lets `docker run` work locally.
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
