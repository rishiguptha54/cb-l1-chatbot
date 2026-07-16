import { useState } from "react";
import { motion } from "framer-motion";
import { Check, Crosshair, Layers, Sparkles, Wrench } from "lucide-react";
import type { DefectOption } from "@/types";
import { cn } from "@/utils/cn";

const OPTION_ICON: Record<string, typeof Crosshair> = {
  root_cause: Crosshair,
  resolve: Wrench,
  similar: Layers,
};

interface OptionsCardProps {
  options: DefectOption[];
  submitted?: boolean;
  selectedSections?: string[];
  disabled?: boolean;
  onSubmit: (sections: string[]) => void;
}

/** Interactive "pick what you want" card for defect-family questions. */
export function OptionsCard({
  options,
  submitted,
  selectedSections,
  disabled,
  onSubmit,
}: OptionsCardProps) {
  const [selected, setSelected] = useState<Set<string>>(
    () => new Set(selectedSections ?? options.map((o) => o.id)),
  );

  const locked = Boolean(submitted);
  const shown = locked ? new Set(selectedSections ?? []) : selected;

  const toggle = (id: string) => {
    if (locked || disabled) return;
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  return (
    <div className="mt-3 rounded-xl border border-primary/20 bg-primary/[0.03] p-3">
      <div className="mb-2.5 flex items-center gap-2 text-sm font-semibold text-foreground">
        <Sparkles className="h-4 w-4 text-primary" />
        {locked ? "You selected" : "Choose what to include"}
      </div>
      <div className="grid gap-2 sm:grid-cols-3">
        {options.map((opt) => {
          const Icon = OPTION_ICON[opt.id] ?? Sparkles;
          const active = shown.has(opt.id);
          return (
            <button
              key={opt.id}
              type="button"
              onClick={() => toggle(opt.id)}
              disabled={locked || disabled}
              aria-pressed={active}
              className={cn(
                "group relative flex flex-col gap-1.5 rounded-xl border p-3 text-left transition-all duration-200",
                active
                  ? "border-primary/60 bg-primary/10 shadow-sm ring-1 ring-primary/20"
                  : "border-border bg-background hover:border-primary/30 hover:bg-accent/50",
                (locked || disabled) && "cursor-default",
              )}
            >
              <span className="flex items-center justify-between">
                <Icon
                  className={cn("h-4 w-4", active ? "text-primary" : "text-muted-foreground")}
                />
                <span
                  className={cn(
                    "grid h-5 w-5 place-items-center rounded-full border-2 transition-all duration-200",
                    active
                      ? "scale-100 border-primary bg-primary text-primary-foreground"
                      : "scale-90 border-muted-foreground/25",
                  )}
                >
                  {active && <Check className="h-3 w-3" strokeWidth={3.5} />}
                </span>
              </span>
              <span className="text-sm font-medium text-foreground">{opt.label}</span>
              {opt.hint && (
                <span className="text-xs leading-snug text-muted-foreground">{opt.hint}</span>
              )}
            </button>
          );
        })}
      </div>

      {!locked && (
        <motion.button
          type="button"
          whileTap={{ scale: 0.98 }}
          disabled={disabled || selected.size === 0}
          onClick={() => onSubmit([...selected])}
          className={cn(
            "mt-3 inline-flex h-10 w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90",
            "disabled:pointer-events-none disabled:opacity-50",
          )}
        >
          <Sparkles className="h-4 w-4" />
          Get answer
          {selected.size > 0 && (
            <span className="rounded-full bg-primary-foreground/20 px-1.5 text-xs">
              {selected.size}
            </span>
          )}
        </motion.button>
      )}
    </div>
  );
}
