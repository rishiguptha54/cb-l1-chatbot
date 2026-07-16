import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { TypingIndicator } from "@/components/chat/TypingIndicator";
import { Logo } from "@/components/ui/Logo";
import { APP_NAME } from "@/config/app";

/** A stylized, non-interactive preview of the chat product for the hero. */
export function HeroPreview() {
  return (
    <div className="relative mx-auto w-full max-w-md">
      <div className="absolute -inset-3 -z-10 rounded-[28px] bg-brand-gradient opacity-20 blur-2xl" aria-hidden="true" />
      <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-elevated">
        {/* Window chrome */}
        <div className="flex items-center gap-2 border-b border-border bg-muted/40 px-4 py-3">
          <span className="flex gap-1.5" aria-hidden="true">
            <span className="h-3 w-3 rounded-full bg-destructive/60" />
            <span className="h-3 w-3 rounded-full bg-amber-400/70" />
            <span className="h-3 w-3 rounded-full bg-success/60" />
          </span>
          <span className="ml-2 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5 text-primary" /> {APP_NAME}
          </span>
        </div>

        {/* Transcript */}
        <div className="space-y-4 p-4">
          <div className="flex justify-end">
            <div className="max-w-[80%] rounded-2xl rounded-tr-md bg-primary px-3.5 py-2 text-sm text-primary-foreground">
              How do I fix the EOM publish failure?
            </div>
          </div>

          <div className="flex gap-2.5">
            <Logo size={28} />
            <div className="min-w-0 flex-1 space-y-2">
              <div className="rounded-2xl rounded-tl-md border border-border bg-background px-3.5 py-2.5">
                <p className="text-sm text-foreground">
                  The EOM publish failure is most often caused by a stale mapping cache. Based on{" "}
                  <span className="font-mono text-xs font-semibold text-primary">HCBS-95506</span>:
                </p>
                <div className="mt-2 space-y-1.5">
                  {[100, 82, 90].map((w, i) => (
                    <motion.div
                      key={i}
                      className="h-2 rounded-full bg-muted"
                      style={{ width: `${w}%` }}
                      initial={{ opacity: 0.4 }}
                      animate={{ opacity: [0.4, 0.8, 0.4] }}
                      transition={{ duration: 2, repeat: Infinity, delay: i * 0.2 }}
                    />
                  ))}
                </div>
              </div>
              <div className="inline-flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-1.5">
                <TypingIndicator label="Fetching from documentation…" />
              </div>
            </div>
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-border p-3">
          <div className="flex items-center gap-2 rounded-xl border border-border bg-background px-3 py-2.5">
            <span className="text-sm text-muted-foreground">Ask any question about CB…</span>
            <span className="ml-auto grid h-7 w-7 place-items-center rounded-lg bg-primary text-primary-foreground">
              <Sparkles className="h-4 w-4" />
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
