import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { askStream } from "@/services/api";
import type {
  ChatMessage,
  Conversation,
  DefectOption,
  HistoryTurn,
  StreamEvent,
} from "@/types";
import { deriveTitle } from "@/utils/format";
import { uid } from "@/utils/id";
import { readJSON, readString, STORAGE_KEYS, writeJSON, writeString } from "@/utils/storage";

const OPTIONS_PROMPT =
  "This looks like a **defect question**. Pick what you'd like me to pull together and I'll fetch just those parts.";

interface ChatContextValue {
  conversations: Conversation[];
  activeId: string | null;
  activeConversation: Conversation | null;
  isStreaming: boolean;

  newConversation: () => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  togglePin: (id: string) => void;
  clearAllConversations: () => void;
  clearActiveConversation: () => void;

  sendMessage: (text: string) => void;
  submitOptions: (messageId: string, sections: string[]) => void;
  regenerate: (messageId: string) => void;
  stop: () => void;
}

const ChatContext = createContext<ChatContextValue | null>(null);

function loadConversations(): Conversation[] {
  const list = readJSON<Conversation[]>(STORAGE_KEYS.conversations, []);
  return Array.isArray(list) ? list : [];
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(
    () => readString(STORAGE_KEYS.activeConversation) || null,
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<null | (() => void)>(null);

  // ── Persistence (debounced) ──
  useEffect(() => {
    const t = window.setTimeout(() => writeJSON(STORAGE_KEYS.conversations, conversations), 250);
    return () => window.clearTimeout(t);
  }, [conversations]);

  useEffect(() => {
    if (activeId) writeString(STORAGE_KEYS.activeConversation, activeId);
  }, [activeId]);

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? null,
    [conversations, activeId],
  );

  // ── Low-level state helpers (always functional to avoid stale closures) ──
  const patchConversation = useCallback(
    (convId: string, patch: Partial<Conversation>) => {
      setConversations((prev) =>
        prev.map((c) => (c.id === convId ? { ...c, ...patch, updatedAt: Date.now() } : c)),
      );
    },
    [],
  );

  const patchMessage = useCallback(
    (convId: string, msgId: string, patch: Partial<ChatMessage> | ((m: ChatMessage) => ChatMessage)) => {
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          return {
            ...c,
            updatedAt: Date.now(),
            messages: c.messages.map((m) => {
              if (m.id !== msgId) return m;
              return typeof patch === "function" ? patch(m) : { ...m, ...patch };
            }),
          };
        }),
      );
    },
    [],
  );

  const appendMessages = useCallback((convId: string, msgs: ChatMessage[]) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? { ...c, updatedAt: Date.now(), messages: [...c.messages, ...msgs] }
          : c,
      ),
    );
  }, []);

  const pushHistory = useCallback((convId: string, turn: HistoryTurn) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId ? { ...c, history: [...c.history, turn].slice(-8) } : c,
      ),
    );
  }, []);

  // ── Conversation CRUD ──
  const newConversation = useCallback((): string => {
    const now = Date.now();
    const conv: Conversation = {
      id: uid("conv"),
      title: "New chat",
      messages: [],
      history: [],
      createdAt: now,
      updatedAt: now,
    };
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    return conv.id;
  }, []);

  const selectConversation = useCallback((id: string) => setActiveId(id), []);

  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id);
        setActiveId((cur) => (cur === id ? next[0]?.id ?? null : cur));
        return next;
      });
    },
    [],
  );

  const renameConversation = useCallback(
    (id: string, title: string) => {
      const clean = title.trim();
      if (clean) patchConversation(id, { title: clean });
    },
    [patchConversation],
  );

  const togglePin = useCallback((id: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, pinned: !c.pinned } : c)),
    );
  }, []);

  const clearAllConversations = useCallback(() => {
    abortRef.current?.();
    setConversations([]);
    setActiveId(null);
  }, []);

  const clearActiveConversation = useCallback(() => {
    abortRef.current?.();
    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeId
          ? { ...c, title: "New chat", messages: [], history: [], updatedAt: Date.now() }
          : c,
      ),
    );
    setIsStreaming(false);
  }, [activeId]);

  // ── Streaming engine ──
  const runStream = useCallback(
    (convId: string, question: string, sections: string[] | undefined, assistantId: string) => {
      let buffer = "";
      let intent = "";
      let sawOptions = false;

      setIsStreaming(true);

      const handleEvent = (event: StreamEvent) => {
        switch (event.type) {
          case "meta":
            intent = event.intent;
            patchMessage(convId, assistantId, { intent });
            break;
          case "options": {
            sawOptions = true;
            const options: DefectOption[] = event.options ?? [];
            patchMessage(convId, assistantId, {
              content: OPTIONS_PROMPT,
              intent: event.intent,
              options,
              optionsQuestion: event.question || question,
              streaming: false,
            });
            break;
          }
          case "token":
            buffer += event.text;
            patchMessage(convId, assistantId, { content: buffer });
            break;
          case "status":
            patchMessage(convId, assistantId, { docStatus: event.text });
            break;
          case "sources":
            patchMessage(convId, assistantId, { sources: event.similar_defects ?? [] });
            break;
          case "error":
            buffer = event.message || "Something went wrong while generating the answer.";
            patchMessage(convId, assistantId, {
              content: buffer,
              error: true,
              streaming: false,
              docStatus: undefined,
            });
            break;
          case "done":
            break;
        }
      };

      const handleClose = () => {
        setIsStreaming(false);
        abortRef.current = null;
        patchMessage(convId, assistantId, (m) => ({
          ...m,
          streaming: false,
          docStatus: undefined,
          sourceQuestion: question,
          sections,
        }));
        // Record a history turn only for a real (non-options) answer.
        if (!sawOptions && buffer.trim()) {
          pushHistory(convId, { question, answer: buffer });
        }
      };

      abortRef.current = askStream(
        { question, history: [], sections },
        { onEvent: handleEvent, onClose: handleClose },
      );
    },
    [patchMessage, pushHistory],
  );

  const startAssistantTurn = useCallback((convId: string): string => {
    const assistantId = uid("msg");
    appendMessages(convId, [
      {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
        streaming: true,
      },
    ]);
    return assistantId;
  }, [appendMessages]);

  // ── Public actions ──
  const sendMessage = useCallback(
    (text: string) => {
      const clean = text.trim();
      if (!clean || isStreaming) return;

      let convId = activeId;
      if (!convId || !conversations.some((c) => c.id === convId)) {
        convId = newConversation();
      }

      const userMsg: ChatMessage = {
        id: uid("msg"),
        role: "user",
        content: clean,
        createdAt: Date.now(),
      };
      const assistantId = uid("msg");
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: Date.now() + 1,
        streaming: true,
      };

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          const isFirst = c.messages.length === 0;
          return {
            ...c,
            title: isFirst ? deriveTitle(clean) : c.title,
            updatedAt: Date.now(),
            messages: [...c.messages, userMsg, assistantMsg],
          };
        }),
      );

      runStream(convId, clean, undefined, assistantId);
    },
    [activeId, conversations, isStreaming, newConversation, runStream],
  );

  const submitOptions = useCallback(
    (messageId: string, sections: string[]) => {
      if (isStreaming || sections.length === 0) return;
      const convId = activeId;
      if (!convId) return;
      const conv = conversations.find((c) => c.id === convId);
      const picker = conv?.messages.find((m) => m.id === messageId);
      if (!picker?.optionsQuestion) return;

      patchMessage(convId, messageId, {
        optionsSubmitted: true,
        selectedSections: sections,
      });
      const assistantId = startAssistantTurn(convId);
      runStream(convId, picker.optionsQuestion, sections, assistantId);
    },
    [activeId, conversations, isStreaming, patchMessage, runStream, startAssistantTurn],
  );

  const regenerate = useCallback(
    (messageId: string) => {
      if (isStreaming) return;
      const convId = activeId;
      if (!convId) return;
      const conv = conversations.find((c) => c.id === convId);
      if (!conv) return;
      const target = conv.messages.find((m) => m.id === messageId);
      if (!target || target.role !== "assistant") return;

      // Determine the question that produced this answer.
      let question = target.sourceQuestion;
      if (!question) {
        const idx = conv.messages.findIndex((m) => m.id === messageId);
        for (let i = idx - 1; i >= 0; i--) {
          if (conv.messages[i].role === "user") {
            question = conv.messages[i].content;
            break;
          }
        }
      }
      if (!question) return;

      patchMessage(convId, messageId, {
        content: "",
        error: false,
        streaming: true,
        sources: undefined,
        docStatus: undefined,
        options: undefined,
      });
      runStream(convId, question, target.sections, messageId);
    },
    [activeId, conversations, isStreaming, patchMessage, runStream],
  );

  const stop = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
    setIsStreaming(false);
    if (activeId) {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === activeId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.streaming ? { ...m, streaming: false, docStatus: undefined } : m,
                ),
              }
            : c,
        ),
      );
    }
  }, [activeId]);

  const value = useMemo<ChatContextValue>(
    () => ({
      conversations,
      activeId,
      activeConversation,
      isStreaming,
      newConversation,
      selectConversation,
      deleteConversation,
      renameConversation,
      togglePin,
      clearAllConversations,
      clearActiveConversation,
      sendMessage,
      submitOptions,
      regenerate,
      stop,
    }),
    [
      conversations,
      activeId,
      activeConversation,
      isStreaming,
      newConversation,
      selectConversation,
      deleteConversation,
      renameConversation,
      togglePin,
      clearAllConversations,
      clearActiveConversation,
      sendMessage,
      submitOptions,
      regenerate,
      stop,
    ],
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useChat(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used within a ChatProvider");
  return ctx;
}
