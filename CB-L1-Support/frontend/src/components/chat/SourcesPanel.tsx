import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, ExternalLink, FileSearch, Gauge } from "lucide-react";
import type { SimilarDefect } from "@/types";
import { JIRA_BROWSE_BASE } from "@/config/app";
import { cn } from "@/utils/cn";

function statusTone(status: string): string {
  const s = status.toLowerCase();
  if (/(done|closed|resolved|fixed)/.test(s))
    return "bg-success/15 text-success ring-success/30";
  if (/(progress|review|open|reopen)/.test(s))
    return "bg-primary/15 text-primary ring-primary/30";
  if (/(cancel|reject|won)/.test(s))
    return "bg-destructive/15 text-destructive ring-destructive/30";
  return "bg-muted text-muted-foreground ring-border";
}

function DefectRow({ defect }: { defect: SimilarDefect }) {
  const [open, setOpen] = useState(false);
  const pct = Math.round((defect.relevance_score || 0) * 100);

  return (
    <li className="overflow-hidden rounded-lg border border-border bg-background/60">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-3 px-3 py-2.5 text-left transition-colors hover:bg-accent/50"
        aria-expanded={open}
      >
        <a
          href={`${JIRA_BROWSE_BASE}${defect.issue_key}`}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="inline-flex shrink-0 items-center gap-1 rounded-md bg-primary/10 px-2 py-1 font-mono text-xs font-semibold text-primary ring-1 ring-primary/25 transition-colors hover:bg-primary/20"
        >
          {defect.issue_key}
          <ExternalLink className="h-3 w-3" />
        </a>
        <span className="min-w-0 flex-1">
          <span className="line-clamp-2 text-sm text-foreground">{defect.summary}</span>
          <span className="mt-1 flex flex-wrap items-center gap-1.5">
            <span
              className={cn(
                "inline-flex items-center rounded px-1.5 py-0.5 text-[0.65rem] font-medium ring-1",
                statusTone(defect.status),
              )}
            >
              {defect.status || "—"}
            </span>
            <span className="inline-flex items-center gap-1 rounded bg-muted px-1.5 py-0.5 text-[0.65rem] font-medium text-muted-foreground ring-1 ring-border">
              <Gauge className="h-3 w-3" />
              {pct}% match
            </span>
          </span>
        </span>
        <ChevronDown
          className={cn(
            "mt-1 h-4 w-4 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <dl className="space-y-2 border-t border-border px-3 py-2.5 text-xs">
              {defect.root_cause && (
                <div>
                  <dt className="font-semibold text-foreground">Root cause</dt>
                  <dd className="mt-0.5 text-muted-foreground">{defect.root_cause}</dd>
                </div>
              )}
              {defect.fix_applied && (
                <div>
                  <dt className="font-semibold text-foreground">Fix applied</dt>
                  <dd className="mt-0.5 text-muted-foreground">{defect.fix_applied}</dd>
                </div>
              )}
              {defect.resolution && (
                <div className="flex gap-2">
                  <dt className="font-semibold text-foreground">Resolution:</dt>
                  <dd className="text-muted-foreground">{defect.resolution}</dd>
                </div>
              )}
            </dl>
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  );
}

/** Collapsible evidence panel listing the historical defects behind an answer. */
export function SourcesPanel({ sources }: { sources: SimilarDefect[] }) {
  const [expanded, setExpanded] = useState(false);
  if (!sources.length) return null;

  return (
    <section className="mt-3 rounded-xl border border-border bg-card/60 p-1">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm font-medium text-foreground transition-colors hover:bg-accent/50"
        aria-expanded={expanded}
      >
        <FileSearch className="h-4 w-4 text-primary" />
        <span>Evidence</span>
        <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-semibold text-primary">
          {sources.length}
        </span>
        <ChevronDown
          className={cn(
            "ml-auto h-4 w-4 text-muted-foreground transition-transform",
            expanded && "rotate-180",
          )}
        />
      </button>
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.ul
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="space-y-2 overflow-hidden p-2"
          >
            {sources.map((d) => (
              <DefectRow key={d.issue_key} defect={d} />
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </section>
  );
}
