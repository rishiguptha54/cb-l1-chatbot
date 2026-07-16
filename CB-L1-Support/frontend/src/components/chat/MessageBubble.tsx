import { motion } from "framer-motion";
import { Check, Copy, RefreshCw, User } from "lucide-react";
import type { ChatMessage } from "@/types";
import { useChat } from "@/features/chat/ChatProvider";
import { useCopyToClipboard } from "@/hooks/useCopyToClipboard";
import { clockTime } from "@/utils/format";
import { cn } from "@/utils/cn";
import { Logo } from "@/components/ui/Logo";
import { Tooltip } from "@/components/ui/Tooltip";
import { APP_NAME } from "@/config/app";
import { Markdown } from "./Markdown";
import { SourcesPanel } from "./SourcesPanel";
import { OptionsCard } from "./OptionsCard";
import { TypingIndicator } from "./TypingIndicator";

const INTENT_LABELS: Record<string, string> = {
  DEFECT_BY_KEY: "Defect lookup",
  DEFECT_DIAGNOSTIC: "Diagnostics",
  SIMILAR_DEFECTS: "Similar defects",
  GENERAL_HELP: "Assistant",
};

function UserMessage({ message }: { message: ChatMessage }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="flex justify-end gap-3"
    >
      <div className="flex max-w-[85%] flex-col items-end sm:max-w-[75%]">
        <div className="whitespace-pre-wrap break-words rounded-2xl rounded-tr-md bg-primary px-4 py-2.5 text-[0.9375rem] leading-7 text-primary-foreground shadow-sm">
          {message.content}
        </div>
        <span className="mt-1 px-1 text-[0.7rem] text-muted-foreground">
          {clockTime(message.createdAt)}
        </span>
      </div>
      <span
        className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-secondary text-secondary-foreground ring-1 ring-border"
        aria-hidden="true"
      >
        <User className="h-4 w-4" />
      </span>
    </motion.div>
  );
}

function AssistantMessage({ message }: { message: ChatMessage }) {
  const { regenerate, submitOptions, isStreaming } = useChat();
  const [copied, copy] = useCopyToClipboard();

  const showTyping = message.streaming && !message.content;
  const intentLabel = message.intent ? INTENT_LABELS[message.intent] : undefined;
  const hasActions = !message.streaming && !message.options && Boolean(message.content);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="flex gap-3"
    >
      <span className="mt-0.5 shrink-0">
        <Logo size={32} />
      </span>

      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-center gap-2">
          <span className="text-sm font-semibold text-foreground">{APP_NAME}</span>
          {intentLabel && (
            <span className="rounded-full bg-accent px-2 py-0.5 text-[0.65rem] font-medium text-accent-foreground ring-1 ring-border">
              {intentLabel}
            </span>
          )}
        </div>

        <div
          className={cn(
            "group/msg rounded-2xl rounded-tl-md border bg-card px-4 py-3 shadow-soft",
            message.error ? "border-destructive/40 bg-destructive/5" : "border-border",
          )}
        >
          {showTyping ? (
            <TypingIndicator label="Thinking…" />
          ) : (
            <Markdown content={message.content} />
          )}

          {message.docStatus && (
            <div className="mt-3 flex items-center gap-2 rounded-lg bg-muted/60 px-3 py-2 text-xs text-muted-foreground">
              <TypingIndicator />
              <span>{message.docStatus}</span>
            </div>
          )}

          {message.options && (
            <OptionsCard
              options={message.options}
              submitted={message.optionsSubmitted}
              selectedSections={message.selectedSections}
              disabled={isStreaming}
              onSubmit={(sections) => submitOptions(message.id, sections)}
            />
          )}

          {message.sources && message.sources.length > 0 && (
            <SourcesPanel sources={message.sources} />
          )}
        </div>

        {hasActions && (
          <div className="mt-1.5 flex items-center gap-1 opacity-0 transition-opacity duration-200 focus-within:opacity-100 group-hover/msg:opacity-100 md:opacity-60 md:hover:opacity-100">
            <Tooltip label={copied ? "Copied" : "Copy"}>
              <button
                type="button"
                onClick={() => copy(message.content)}
                className="inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                aria-label="Copy message"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-success" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            </Tooltip>
            <Tooltip label="Regenerate">
              <button
                type="button"
                onClick={() => regenerate(message.id)}
                disabled={isStreaming}
                className="inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-40"
                aria-label="Regenerate response"
              >
                <RefreshCw className="h-3.5 w-3.5" />
              </button>
            </Tooltip>
            <span className="ml-1 text-[0.7rem] text-muted-foreground">
              {clockTime(message.createdAt)}
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  return message.role === "user" ? (
    <UserMessage message={message} />
  ) : (
    <AssistantMessage message={message} />
  );
}
