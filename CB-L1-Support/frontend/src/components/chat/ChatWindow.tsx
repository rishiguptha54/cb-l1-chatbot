import { useChat } from "@/features/chat/ChatProvider";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { EmptyState } from "./EmptyState";

export function ChatWindow() {
  const { activeConversation, sendMessage } = useChat();
  const messages = activeConversation?.messages ?? [];

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {messages.length === 0 ? (
        <div className="flex-1 overflow-y-auto scrollbar-thin">
          <EmptyState onPick={sendMessage} />
        </div>
      ) : (
        <MessageList messages={messages} />
      )}
      <ChatInput />
    </div>
  );
}
