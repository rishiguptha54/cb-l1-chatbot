import { cn } from "@/utils/cn";

/** Three-dot "assistant is typing" animation. */
export function TypingIndicator({ label, className }: { label?: string; className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)} aria-live="polite">
      <span className="flex items-center gap-1" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1.5 w-1.5 rounded-full bg-current opacity-70 animate-blink"
            style={{ animationDelay: `${i * 0.18}s` }}
          />
        ))}
      </span>
      {label && <span className="text-xs font-medium text-muted-foreground">{label}</span>}
    </span>
  );
}
