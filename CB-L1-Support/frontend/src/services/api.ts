import type {
  AddDefectsByJqlResult,
  HealthStatus,
  HistoryTurn,
  IngestResult,
  StreamEvent,
} from "@/types";

/**
 * Same-origin API client. In development, Vite proxies `/api` and `/health` to
 * the FastAPI server on :5100 (see `vite.config.ts`); in production the compiled
 * SPA is served by FastAPI itself, so relative URLs resolve to the same origin.
 */
const BASE = "";

/** Fetch backend readiness. Returns `null` if the server is unreachable. */
export async function fetchHealth(signal?: AbortSignal): Promise<HealthStatus | null> {
  try {
    const res = await fetch(`${BASE}/health`, { signal });
    if (!res.ok) return null;
    return (await res.json()) as HealthStatus;
  } catch {
    return null;
  }
}

/**
 * Consume a Server-Sent Events response body, calling `onEvent` for each
 * parsed `data:` frame. Shared by `askStream()` and `ingestPdfStream()`.
 */
async function consumeSSE<T>(res: Response, onEvent: (event: T) => void): Promise<void> {
  if (!res.body) return;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      try {
        onEvent(JSON.parse(payload) as T);
      } catch {
        /* ignore malformed frame */
      }
    }
  }
}

export interface AskStreamParams {
  question: string;
  history: HistoryTurn[];
  sections?: string[];
  mode?: string;
}

export interface AskStreamHandlers {
  onEvent: (event: StreamEvent) => void;
  onClose: () => void;
}

/**
 * Stream an answer via Server-Sent Events.
 *
 * `history` carries recent {question, answer} turns so the backend can resolve
 * follow-ups; `sections` carries the user's option-picker selection for
 * defect-family questions. Returns an abort function to cancel the in-flight
 * request.
 */
export function askStream(
  params: AskStreamParams,
  { onEvent, onClose }: AskStreamHandlers,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/api/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: params.question,
          history: params.history,
          sections: params.sections,
          mode: params.mode,
        }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        onEvent({ type: "error", message: await safeDetail(res) });
        onClose();
        return;
      }

      await consumeSSE<StreamEvent>(res, onEvent);
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onEvent({ type: "error", message: (err as Error).message });
      }
    } finally {
      onClose();
    }
  })();

  return () => controller.abort();
}

async function safeDetail(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: string };
    return data.detail || `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status})`;
  }
}

/**
 * Upload a PDF for on-demand documentation ingestion. The backend parses,
 * embeds, and upserts it into the same Qdrant collection the documentation
 * RAG searches, so it's answerable immediately — no rebuild/restart needed.
 * Throws with a user-facing message on failure.
 */
export async function ingestPdf(file: File, signal?: AbortSignal): Promise<IngestResult> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/api/rag/ingest`, {
    method: "POST",
    body: form,
    signal,
  });

  if (!res.ok) {
    throw new Error(await safeDetail(res));
  }
  return (await res.json()) as IngestResult;
}

/** Discrete progress events emitted by `POST /api/rag/ingest/stream`. */
export type IngestStreamEvent =
  | { stage: "parsing" | "chunking" | "embedding"; message: string }
  | { stage: "done"; result: IngestResult }
  | { stage: "error"; message: string };

export interface IngestStreamHandlers {
  onEvent: (event: IngestStreamEvent) => void;
  onClose: () => void;
}

/**
 * Upload a PDF and stream backend progress labels (parsing → chunking →
 * embedding → done) via Server-Sent Events, so the UI can show exactly what
 * the ingestion pipeline is doing instead of a single opaque spinner. Returns
 * an abort function to cancel the in-flight request.
 */
export function ingestPdfStream(
  file: File,
  { onEvent, onClose }: IngestStreamHandlers,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${BASE}/api/rag/ingest/stream`, {
        method: "POST",
        body: form,
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        onEvent({ stage: "error", message: await safeDetail(res) });
        onClose();
        return;
      }

      await consumeSSE<IngestStreamEvent>(res, onEvent);
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onEvent({ stage: "error", message: (err as Error).message });
      }
    } finally {
      onClose();
    }
  })();

  return () => controller.abort();
}

/** Discrete progress events emitted by `POST /api/defects/add-jql/stream`. */
export type AddDefectsByJqlStreamEvent =
  | { stage: "searching" | "found" | "normalizing" | "embedding"; message: string }
  | { stage: "done"; result: AddDefectsByJqlResult }
  | { stage: "error"; message: string };

export interface AddDefectsByJqlStreamHandlers {
  onEvent: (event: AddDefectsByJqlStreamEvent) => void;
  onClose: () => void;
}

/**
 * Bulk-add every defect matching a JQL query and stream backend progress
 * labels (searching Jira → found N → normalizing each → embedding → done)
 * via Server-Sent Events. Returns an abort function to cancel the in-flight
 * request.
 */
export function addDefectsByJqlStream(
  jql: string,
  { onEvent, onClose }: AddDefectsByJqlStreamHandlers,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/api/defects/add-jql/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jql }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        onEvent({ stage: "error", message: await safeDetail(res) });
        onClose();
        return;
      }

      await consumeSSE<AddDefectsByJqlStreamEvent>(res, onEvent);
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onEvent({ stage: "error", message: (err as Error).message });
      }
    } finally {
      onClose();
    }
  })();

  return () => controller.abort();
}

