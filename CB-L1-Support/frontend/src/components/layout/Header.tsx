import { Bug, Menu, Moon, PanelLeftOpen, Sun, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { useHealth } from "@/hooks/useHealth";
import { useTheme } from "@/contexts/ThemeContext";
import { cn } from "@/utils/cn";
import { Logo } from "@/components/ui/Logo";
import { Spinner } from "@/components/ui/Spinner";
import { Tooltip } from "@/components/ui/Tooltip";
import { APP_NAME } from "@/config/app";
import { addDefectsByJqlStream, ingestPdfStream } from "@/services/api";

interface HeaderProps {
  onOpenMobileSidebar: () => void;
  sidebarCollapsed: boolean;
  onExpandSidebar: () => void;
}

export function Header({
  onOpenMobileSidebar,
  sidebarCollapsed,
  onExpandSidebar,
}: HeaderProps) {
  const { data: health, isLoading } = useHealth();
  const { theme, toggle } = useTheme();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Shared progress/result UI for both "upload docs" and "add defect" — only
  // one runs at a time, so one pair of state slots covers both.
  const [busy, setBusy] = useState(false);
  const [taskStage, setTaskStage] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<{ ok: boolean; text: string } | null>(null);

  const [addDefectOpen, setAddDefectOpen] = useState(false);
  const [jqlInput, setJqlInput] = useState("");

  const connected = Boolean(health && health.status === "ok");
  const statusLabel = isLoading
    ? "Connecting…"
    : connected
      ? "Connected"
      : "Offline";
  const provider = health?.llm_provider;

  function showStatus(ok: boolean, text: string) {
    setTaskStatus({ ok, text });
    window.setTimeout(() => setTaskStatus(null), 6000);
  }

  function handleFileSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file later
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      showStatus(false, "Only PDF files are supported.");
      return;
    }

    setBusy(true);
    setTaskStage(`📤 Uploading "${file.name}"…`);
    setTaskStatus(null);

    ingestPdfStream(file, {
      onEvent: (event) => {
        if (event.stage === "done") {
          const tableRows = event.result.table_row_chunks ?? 0;
          const tableNote = tableRows > 0 ? ` (including ${tableRows} table rows)` : "";
          const pages = event.result.pages;
          showStatus(
            true,
            `Ingested "${event.result.source_file}" — ${event.result.chunks} chunks${tableNote} from ${pages} page${pages === 1 ? "" : "s"}.`,
          );
          setTaskStage(null);
        } else if (event.stage === "error") {
          showStatus(false, event.message || "Ingestion failed.");
          setTaskStage(null);
        } else {
          setTaskStage(event.message);
        }
      },
      onClose: () => setBusy(false),
    });
  }

  function submitAddDefectsByJql() {
    const jql = jqlInput.trim();
    if (!jql) return;

    setAddDefectOpen(false);
    setJqlInput("");
    setBusy(true);
    setTaskStage(`🔎 Searching Jira…`);
    setTaskStatus(null);

    addDefectsByJqlStream(jql, {
      onEvent: (event) => {
        if (event.stage === "done") {
          const { processed, matched, chunks } = event.result;
          const skipped = matched - processed;
          const skippedNote = skipped > 0 ? ` (${skipped} skipped — too little content)` : "";
          showStatus(
            true,
            `Added ${processed} defect${processed === 1 ? "" : "s"}${skippedNote} — ${chunks} chunks total.`,
          );
          setTaskStage(null);
        } else if (event.stage === "error") {
          showStatus(false, event.message || "Bulk add failed.");
          setTaskStage(null);
        } else {
          setTaskStage(event.message);
        }
      },
      onClose: () => setBusy(false),
    });
  }

  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-2 border-b border-border bg-background/80 px-3 backdrop-blur-xl sm:px-4">
      {/* Mobile menu */}
      <button
        onClick={onOpenMobileSidebar}
        className="grid h-9 w-9 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground md:hidden"
        aria-label="Open sidebar"
      >
        <Menu className="h-5 w-5" />
      </button>

      {/* Desktop expand (only when collapsed) */}
      {sidebarCollapsed && (
        <button
          onClick={onExpandSidebar}
          className="hidden h-9 w-9 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground md:grid"
          aria-label="Expand sidebar"
        >
          <PanelLeftOpen className="h-5 w-5" />
        </button>
      )}

      <div className="flex items-center gap-2.5">
        <Logo size={30} className="md:hidden" />
        <div className="flex flex-col leading-none">
          <span className="whitespace-nowrap text-sm font-semibold text-foreground">{APP_NAME}</span>
          <span className="mt-0.5 hidden text-[0.7rem] text-muted-foreground sm:block">
            Grounded in real defect history
          </span>
        </div>
      </div>

      <div className="ml-auto flex items-center gap-1.5">
        {/* Upload documentation PDF for RAG ingestion */}
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          onChange={handleFileSelected}
        />
        <Tooltip label="Upload a documentation PDF — parsed, chunked, and embedded into the RAG knowledge base" side="bottom">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={busy}
            className="grid h-9 w-9 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
            aria-label="Upload documentation PDF"
          >
            {busy ? <Spinner className="h-[1.15rem] w-[1.15rem]" /> : <Upload className="h-[1.15rem] w-[1.15rem]" />}
          </button>
        </Tooltip>

        {/* Add defect(s) from Jira into the defect knowledge base */}
        <div className="relative">
          <Tooltip label="Add defects from Jira into the knowledge base" side="bottom">
            <button
              onClick={() => setAddDefectOpen((v) => !v)}
              disabled={busy}
              className="grid h-9 w-9 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
              aria-label="Add defect from Jira"
            >
              <Bug className="h-[1.15rem] w-[1.15rem]" />
            </button>
          </Tooltip>

          {addDefectOpen && (
            <div className="absolute right-0 top-11 z-40 w-80 rounded-lg border border-border bg-card p-3 shadow-soft">
              <div className="mb-2">
                <h3 className="text-sm font-semibold text-foreground">Add Defects to Knowledge Base</h3>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Fetches from Jira, extracts root cause / fix / classification, and
                  embeds into the same search index the assistant uses — searchable
                  immediately, no rebuild needed.
                </p>
              </div>

              <label className="mb-1 block text-xs font-medium text-muted-foreground">
                JQL query
              </label>
              <p className="mb-1.5 text-[0.7rem] text-muted-foreground">
                Bulk-fetch and index every defect matching a JQL query
                (capped at 200 per run).
              </p>
              <textarea
                autoFocus
                value={jqlInput}
                onChange={(e) => setJqlInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submitAddDefectsByJql();
                  if (e.key === "Escape") setAddDefectOpen(false);
                }}
                placeholder='e.g. project = HCBS AND status = Done AND created >= -30d'
                rows={4}
                className="w-full resize-none rounded-md border border-border bg-background px-2.5 py-1.5 text-sm outline-none focus:ring-2 focus:ring-ring"
              />
              <p className="mt-1 text-[0.65rem] text-muted-foreground">Ctrl/Cmd + Enter to submit</p>

              <div className="mt-2 flex justify-end gap-2">
                <button
                  onClick={() => setAddDefectOpen(false)}
                  className="rounded-md px-2.5 py-1 text-xs text-muted-foreground hover:bg-accent"
                >
                  Cancel
                </button>
                <button
                  onClick={submitAddDefectsByJql}
                  disabled={!jqlInput.trim()}
                  className="rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:pointer-events-none disabled:opacity-50"
                >
                  Add
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Connection status */}
        <Tooltip
          label={
            connected
              ? `Backend online${provider ? ` · LLM: ${provider}` : ""}`
              : "Backend unreachable"
          }
          side="bottom"
        >
          <span
            className={cn(
              "hidden items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium sm:inline-flex",
              connected
                ? "border-success/30 bg-success/10 text-success"
                : "border-destructive/30 bg-destructive/10 text-destructive",
            )}
          >
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                connected ? "bg-success" : "bg-destructive",
                isLoading && "animate-pulse",
              )}
            />
            {statusLabel}
          </span>
        </Tooltip>

        {/* Theme toggle */}
        <Tooltip label={theme === "dark" ? "Light mode" : "Dark mode"} side="bottom">
          <button
            onClick={toggle}
            className="grid h-9 w-9 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Sun className="h-[1.15rem] w-[1.15rem]" /> : <Moon className="h-[1.15rem] w-[1.15rem]" />}
          </button>
        </Tooltip>

        {/* Profile */}
        <button
          className="ml-0.5 grid h-9 w-9 place-items-center rounded-full bg-brand-gradient text-xs font-semibold text-white shadow-sm transition-transform hover:scale-105"
          aria-label="Profile"
        >
          OP
        </button>
      </div>

      {/* Live backend progress while an upload/add-defect task is running */}
      {taskStage && (
        <div
          className="fixed right-4 top-16 z-50 flex max-w-sm items-center gap-2 rounded-lg border border-border bg-card px-3.5 py-2.5 text-xs font-medium text-foreground shadow-soft"
          role="status"
        >
          <Spinner className="h-3.5 w-3.5 shrink-0" />
          <span>{taskStage}</span>
        </div>
      )}

      {/* Transient result toast (success/error) */}
      {taskStatus && (
        <div
          className={cn(
            "fixed right-4 top-16 z-50 max-w-sm rounded-lg border px-3.5 py-2.5 text-xs font-medium shadow-soft",
            taskStatus.ok
              ? "border-success/30 bg-success/10 text-success"
              : "border-destructive/30 bg-destructive/10 text-destructive",
          )}
          role="status"
        >
          {taskStatus.text}
        </div>
      )}
    </header>
  );
}
