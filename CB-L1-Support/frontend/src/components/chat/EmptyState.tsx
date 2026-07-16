import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import { SUGGESTED_CATEGORIES } from "@/config/app";

/** Shown when a conversation has no messages yet. */
export function EmptyState({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="mx-auto flex min-h-full max-w-3xl flex-col items-center justify-center px-4 py-10 xl:max-w-4xl 2xl:max-w-5xl">
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <Logo size={56} />
      </motion.div>

      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05, duration: 0.4 }}
        className="mt-6 text-center text-2xl font-bold tracking-tight text-foreground sm:text-3xl"
      >
        How can I help you today?
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1, duration: 0.4 }}
        className="mt-2 max-w-md text-center text-sm text-muted-foreground"
      >
        Ask any question about CB and get a root-cause and fix — grounded in your
        historical defect data and product documentation.
      </motion.p>

      <div className="mt-9 grid w-full gap-3 sm:grid-cols-2">
        {SUGGESTED_CATEGORIES.map((cat, ci) => (
          <motion.div
            key={cat.title}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 + ci * 0.06, duration: 0.4 }}
            className="rounded-2xl border border-border bg-card p-4 shadow-soft"
          >
            <div className="mb-2 flex items-center gap-2.5 px-1">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary ring-1 ring-primary/15">
                <cat.icon className="h-4 w-4" />
              </span>
              <h2 className="text-sm font-semibold text-foreground">{cat.title}</h2>
            </div>
            <ul className="space-y-0.5">
              {cat.questions.map((q) => (
                <li key={q}>
                  <button
                    type="button"
                    onClick={() => onPick(q)}
                    className="group flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[0.8125rem] leading-snug text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground"
                  >
                    <span className="flex-1">{q}</span>
                    <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground/40 opacity-0 transition-all group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-primary group-hover:opacity-100" />
                  </button>
                </li>
              ))}
            </ul>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
