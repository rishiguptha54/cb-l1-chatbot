/**
 * Shared domain types for the CB L1 Support Chatbot frontend.
 *
 * These mirror the FastAPI backend contract exactly (see `api/routes.py` and
 * `chatbot/defect_qa.py`) so streaming, the options picker and source panels
 * behave identically to the original pipeline.
 */

export type Role = "user" | "assistant";

/** A historical defect surfaced as evidence for an answer. */
export interface SimilarDefect {
  issue_key: string;
  summary: string;
  status: string;
  resolution: string;
  priority: string;
  root_cause: string;
  fix_applied: string;
  relevance_score: number;
  quality_score: number;
}

/** A selectable section for a defect-family question (root_cause / resolve / similar). */
export interface DefectOption {
  id: string;
  label: string;
  hint?: string;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
  intent?: string;
  sources?: SimilarDefect[];
  streaming?: boolean;
  error?: boolean;
  /** When set, this assistant turn is an options picker for a defect question. */
  options?: DefectOption[];
  /** The original question to resend once the user picks options. */
  optionsQuestion?: string;
  /** True once the user has submitted a selection (locks the picker). */
  optionsSubmitted?: boolean;
  /** The selection the user submitted from the options picker (for display/replay). */
  selectedSections?: string[];
  /** Transient note shown while a documentation/RAG lookup runs. */
  docStatus?: string;
  /** The question that produced this assistant answer (enables exact regenerate). */
  sourceQuestion?: string;
  /** The sections used to produce this assistant answer (enables exact regenerate). */
  sections?: string[];
}

/** `/health` response describing readiness of each backend subsystem. */
export interface HealthStatus {
  status: string;
  service?: string;
  knowledge_base_loaded: boolean;
  vector_store_loaded: boolean;
  llm_enabled: boolean;
  llm_provider?: string;
  rag_docs_enabled?: boolean;
  top_k: number;
}

/** `POST /api/rag/ingest` response after a PDF is parsed, embedded, and stored. */
export interface IngestResult {
  status: string;
  source_file: string;
  collection?: string;
  pages: number;
  chunks: number;
  text_chunks?: number;
  table_row_chunks?: number;
  table_fragment_chunks?: number;
  table_full_chunks?: number;
}

/** `POST /api/defects/add-jql` response after a JQL query is fetched, normalized, and stored in bulk. */
export interface AddDefectsByJqlResult {
  status: string;
  jql: string;
  collection?: string;
  matched: number;
  processed: number;
  chunks: number;
  issue_keys: string[];
}


/** Discrete events emitted by the SSE stream (`POST /api/ask/stream`). */
export type StreamEvent =
  | { type: "meta"; intent: string }
  | { type: "options"; intent: string; options: DefectOption[]; question: string }
  | { type: "token"; text: string }
  | { type: "status"; text: string }
  | { type: "sources"; similar_defects: SimilarDefect[] }
  | { type: "done" }
  | { type: "error"; message: string };

/** A single {question, answer} turn used as backend history context. */
export interface HistoryTurn {
  question: string;
  answer: string;
}

/** A persisted conversation thread (stored client-side in localStorage). */
export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  history: HistoryTurn[];
  createdAt: number;
  updatedAt: number;
  pinned?: boolean;
}

export type ThemePreference = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";
