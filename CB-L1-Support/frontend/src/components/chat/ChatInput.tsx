import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowUp, Square } from "lucide-react";
import { useChat } from "@/features/chat/ChatProvider";
import { cn } from "@/utils/cn";

const MAX_LEN = 2000;

export function ChatInput() {
  const { sendMessage, isStreaming, stop } = useChat();
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize the textarea up to a max height.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [value]);

  // Focus on mount for immediate typing.
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const trimmed = value.trim();
  const canSend = trimmed.length > 0 && !isStreaming;
  const nearLimit = value.length > MAX_LEN * 0.85;

  const submit = () => {
    if (!canSend) return;
    sendMessage(trimmed);
    setValue("");
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-border bg-background/80 backdrop-blur-xl">
      <div className="mx-auto w-full max-w-3xl px-4 pb-4 pt-3 sm:px-6 xl:max-w-4xl 2xl:max-w-5xl">
        <div
          className={cn(
            "relative rounded-[1.75rem] border border-border bg-card shadow-soft transition-all",
            "focus-within:border-primary/40 focus-within:shadow-glow focus-within:ring-4 focus-within:ring-primary/10",
          )}
        >
          <label htmlFor="chat-input" className="sr-only">
            Message the assistant
          </label>
          <textarea
            id="chat-input"
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value.slice(0, MAX_LEN))}
            onKeyDown={onKeyDown}
            rows={1}
            placeholder="Ask any question about CB…"
            className="scrollbar-thin block max-h-[220px] min-h-[3.25rem] w-full resize-none bg-transparent px-4 pt-3.5 pb-1.5 text-[0.9375rem] leading-6 text-foreground outline-none placeholder:text-muted-foreground"
          />

          <div className="flex items-center justify-end gap-2 px-2.5 pb-2.5 pt-0.5">
            <div className="flex items-center gap-2.5">
              {value.length > 0 && (
                <span
                  className={cn(
                    "text-[0.7rem] tabular-nums transition-colors",
                    nearLimit ? "text-destructive" : "text-muted-foreground/60",
                  )}
                >
                  {value.length}/{MAX_LEN}
                </span>
              )}

              {isStreaming ? (
                <motion.button
                  type="button"
                  onClick={stop}
                  whileTap={{ scale: 0.92 }}
                  className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-foreground text-background transition-colors hover:bg-foreground/90"
                  aria-label="Stop generating"
                >
                  <Square className="h-3.5 w-3.5 fill-current" />
                </motion.button>
              ) : (
                <motion.button
                  type="button"
                  onClick={submit}
                  disabled={!canSend}
                  whileTap={{ scale: 0.92 }}
                  className={cn(
                    "grid h-9 w-9 shrink-0 place-items-center rounded-full transition-all",
                    canSend
                      ? "bg-brand-gradient text-white shadow-glow hover:opacity-90"
                      : "bg-muted text-muted-foreground/50",
                  )}
                  aria-label="Send message"
                >
                  <ArrowUp className="h-[1.15rem] w-[1.15rem]" strokeWidth={2.5} />
                </motion.button>
              )}
            </div>
          </div>
        </div>

        <p className="mt-2 text-center text-[0.7rem] text-muted-foreground/70">
          <kbd className="rounded border border-border bg-muted px-1 py-0.5 font-sans text-[0.65rem]">
            Enter
          </kbd>{" "}
          to send ·{" "}
          <kbd className="rounded border border-border bg-muted px-1 py-0.5 font-sans text-[0.65rem]">
            Shift + Enter
          </kbd>{" "}
          for a new line
        </p>
      </div>
    </div>
  );
}
