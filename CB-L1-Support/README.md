# CB-L1-Support

CB L1 Support Chatbot — an AI assistant that answers defect root-cause and fix
questions from historical Jira defects and product documentation.

## Structure

- `backend/` — FastAPI service, chatbot/RAG pipeline, retrieval indexes and data.
- `frontend/` — React + TypeScript + Vite chat application.

## Backend

```powershell
cd backend
python -m venv ../.venv
../.venv/Scripts/pip install -r requirements.txt
../.venv/Scripts/python run_chatbot.py --serve
```

Copy `.env.example` (create your own `.env`, never commit it) with the required
API keys: Azure OpenAI, Jira, Qdrant, and GitHub/Copilot tokens.

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

The dev server proxies `/api` and `/health` to the backend on port 5100.

## Deploy to Render (free tier, no card required)

`render.yaml` provisions a single Web Service on Render's **native Python
runtime** — not Docker. Render's Docker runtime requires a verified payment
method even for free instances; native runtimes don't.

Because the native runtime has no Node/npm, the frontend is built **locally**
and its output (`backend/api/static/`) is committed to the repo, so the one
Python service can serve both the API and the SPA from a single origin.

1. Before deploying (and whenever the UI changes):
   ```powershell
   cd frontend
   npm run build
   git add ../backend/api/static
   git commit -m "Rebuild frontend"
   ```
2. In the Render dashboard: **New +** → **Blueprint** → select this repo.
3. Render reads `render.yaml` and provisions one Free-tier web service
   (`pip install -r requirements-deploy.txt`, then `uvicorn api.app:app`).
4. Fill in the secrets it prompts for (`GITHUB_TOKEN`, `QDRANT_URL`,
   `QDRANT_API_KEY`, `QDRANT_COLLECTION`; Azure OpenAI vars are optional).
5. Deploy — the health check hits `/health`.

See `backend/.env.example` for the full list of environment variables.

A `Dockerfile` is also included as an optional alternative if you later move
to a paid Render plan (it builds the frontend automatically instead of
requiring a manual `npm run build` + commit step) — it is not used by the
current `render.yaml`.

