import { useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  MessageSquarePlus,
  MoreHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Pin,
  PinOff,
  Search,
  Trash2,
  X,
  type LucideIcon,
} from "lucide-react";
import type { Conversation } from "@/types";
import { useChat } from "@/features/chat/ChatProvider";
import { dateBucket, relativeTime } from "@/utils/format";
import { cn } from "@/utils/cn";
import { Logo } from "@/components/ui/Logo";
import { Tooltip } from "@/components/ui/Tooltip";

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  onExpand: () => void;
  onNavigate?: () => void;
  onCloseMobile?: () => void;
  isMobile?: boolean;
}

export function Sidebar({
  collapsed,
  onToggleCollapse,
  onExpand,
  onNavigate,
  onCloseMobile,
  isMobile = false,
}: SidebarProps) {
  const {
    conversations,
    activeId,
    newConversation,
    selectConversation,
    deleteConversation,
    renameConversation,
    togglePin,
    clearAllConversations,
  } = useChat();

  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const rail = collapsed && !isMobile;

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return conversations;
    return conversations.filter((c) => c.title.toLowerCase().includes(q));
  }, [conversations, query]);

  const pinned = filtered.filter((c) => c.pinned);
  const unpinned = filtered.filter((c) => !c.pinned);

  const groups = useMemo(() => {
    const map = new Map<string, Conversation[]>();
    for (const c of unpinned) {
      const key = dateBucket(c.updatedAt);
      const arr = map.get(key) ?? [];
      arr.push(c);
      map.set(key, arr);
    }
    return Array.from(map.entries());
  }, [unpinned]);

  const handleNew = () => {
    newConversation();
    onNavigate?.();
  };

  const handleSelect = (id: string) => {
    selectConversation(id);
    onNavigate?.();
  };

  // ── Collapsed desktop rail ──
  if (rail) {
    return (
      <nav
        aria-label="Conversation navigation"
        className="flex h-full w-[68px] flex-col items-center gap-2 border-r border-sidebar-border bg-sidebar py-3 text-sidebar-foreground"
      >
        <div className="mb-1 grid h-10 w-10 place-items-center">
          <Logo size={34} />
        </div>
        <Tooltip label="Expand sidebar" side="right">
          <button
            onClick={onToggleCollapse}
            className="grid h-10 w-10 place-items-center rounded-lg text-sidebar-foreground/70 transition-colors hover:bg-white/5 hover:text-white"
            aria-label="Expand sidebar"
          >
            <PanelLeftOpen className="h-5 w-5" />
          </button>
        </Tooltip>
        <Tooltip label="New chat" side="right">
          <button
            onClick={handleNew}
            className="grid h-10 w-10 place-items-center rounded-lg bg-white/5 text-white transition-colors hover:bg-primary hover:text-primary-foreground"
            aria-label="New chat"
          >
            <MessageSquarePlus className="h-5 w-5" />
          </button>
        </Tooltip>
        <Tooltip label="Search chats" side="right">
          <button
            onClick={onExpand}
            className="grid h-10 w-10 place-items-center rounded-lg text-sidebar-foreground/70 transition-colors hover:bg-white/5 hover:text-white"
            aria-label="Search chats"
          >
            <Search className="h-5 w-5" />
          </button>
        </Tooltip>
      </nav>
    );
  }

  // ── Expanded sidebar (desktop + mobile drawer) ──
  return (
    <nav
      aria-label="Conversation navigation"
      className="flex h-full w-72 flex-col bg-sidebar text-sidebar-foreground"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3">
        <Logo size={34} withWordmark subtitle="Grounded in real history" className="[&_span]:text-white" />
        <div className="flex items-center gap-1">
          {isMobile ? (
            <button
              onClick={onCloseMobile}
              className="grid h-9 w-9 place-items-center rounded-lg text-sidebar-foreground/70 transition-colors hover:bg-white/5 hover:text-white"
              aria-label="Close sidebar"
            >
              <X className="h-5 w-5" />
            </button>
          ) : (
            <Tooltip label="Collapse sidebar" side="right">
              <button
                onClick={onToggleCollapse}
                className="grid h-9 w-9 place-items-center rounded-lg text-sidebar-foreground/70 transition-colors hover:bg-white/5 hover:text-white"
                aria-label="Collapse sidebar"
              >
                <PanelLeftClose className="h-5 w-5" />
              </button>
            </Tooltip>
          )}
        </div>
      </div>

      {/* New chat */}
      <div className="px-3">
        <button
          onClick={handleNew}
          className="flex w-full items-center gap-2.5 rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 text-sm font-medium text-white transition-all hover:border-primary/50 hover:bg-primary/20"
        >
          <MessageSquarePlus className="h-[1.15rem] w-[1.15rem]" />
          New chat
        </button>
      </div>

      {/* Search */}
      <div className="px-3 pb-1 pt-3">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-sidebar-foreground/50" />
          <input
            ref={searchRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search conversations"
            aria-label="Search conversations"
            className="w-full rounded-lg border border-white/10 bg-black/20 py-2 pl-9 pr-8 text-sm text-white placeholder:text-sidebar-foreground/50 outline-none transition-colors focus:border-primary/50"
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              className="absolute right-2 top-1/2 grid h-6 w-6 -translate-y-1/2 place-items-center rounded text-sidebar-foreground/60 hover:text-white"
              aria-label="Clear search"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Conversation list */}
      <div className="scrollbar-thin flex-1 overflow-y-auto px-2 py-2">
        {filtered.length === 0 && (
          <p className="px-3 py-8 text-center text-xs text-sidebar-foreground/50">
            {query ? "No conversations match your search." : "No conversations yet."}
          </p>
        )}

        {pinned.length > 0 && (
          <Section label="Pinned">
            {pinned.map((c) => (
              <ConversationItem
                key={c.id}
                conversation={c}
                active={c.id === activeId}
                onSelect={() => handleSelect(c.id)}
                onDelete={() => deleteConversation(c.id)}
                onRename={(t) => renameConversation(c.id, t)}
                onTogglePin={() => togglePin(c.id)}
              />
            ))}
          </Section>
        )}

        {groups.map(([label, items]) => (
          <Section key={label} label={label}>
            {items.map((c) => (
              <ConversationItem
                key={c.id}
                conversation={c}
                active={c.id === activeId}
                onSelect={() => handleSelect(c.id)}
                onDelete={() => deleteConversation(c.id)}
                onRename={(t) => renameConversation(c.id, t)}
                onTogglePin={() => togglePin(c.id)}
              />
            ))}
          </Section>
        ))}
      </div>

      {/* Footer */}
      <div className="border-t border-sidebar-border p-2">
        {conversations.length > 0 && (
          <FooterButton onClick={clearAllConversations} icon={Trash2} label="Clear conversations" danger />
        )}
      </div>
    </nav>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-2">
      <p className="px-3 py-1.5 text-[0.7rem] font-semibold uppercase tracking-wider text-sidebar-foreground/45">
        {label}
      </p>
      <ul className="space-y-0.5">{children}</ul>
    </div>
  );
}

function FooterButton({
  onClick,
  icon: Icon,
  label,
  danger,
}: {
  onClick: () => void;
  icon: LucideIcon;
  label: string;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors",
        danger
          ? "text-sidebar-foreground/80 hover:bg-destructive/20 hover:text-white"
          : "text-sidebar-foreground/80 hover:bg-white/5 hover:text-white",
      )}
    >
      <Icon className="h-[1.15rem] w-[1.15rem]" />
      {label}
    </button>
  );
}

interface ItemProps {
  conversation: Conversation;
  active: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onRename: (title: string) => void;
  onTogglePin: () => void;
}

function ConversationItem({
  conversation,
  active,
  onSelect,
  onDelete,
  onRename,
  onTogglePin,
}: ItemProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(conversation.title);

  const commit = () => {
    onRename(draft);
    setEditing(false);
  };

  if (editing) {
    return (
      <li>
        <div className="flex items-center gap-1 rounded-lg bg-white/5 px-2 py-1">
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commit();
              if (e.key === "Escape") setEditing(false);
            }}
            className="min-w-0 flex-1 bg-transparent px-1 py-1 text-sm text-white outline-none"
            aria-label="Rename conversation"
          />
          <button
            onClick={commit}
            className="grid h-6 w-6 place-items-center rounded text-success hover:bg-white/10"
            aria-label="Save name"
          >
            <Check className="h-4 w-4" />
          </button>
          <button
            onClick={() => setEditing(false)}
            className="grid h-6 w-6 place-items-center rounded text-sidebar-foreground/60 hover:bg-white/10 hover:text-white"
            aria-label="Cancel rename"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </li>
    );
  }

  return (
    <li
      className="group/item relative"
      onMouseLeave={() => setMenuOpen(false)}
    >
      <button
        onClick={onSelect}
        className={cn(
          "flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors",
          active
            ? "bg-white/10 text-white"
            : "text-sidebar-foreground/85 hover:bg-white/5 hover:text-white",
        )}
      >
        {conversation.pinned && <Pin className="h-3 w-3 shrink-0 text-primary" />}
        <span className="min-w-0 flex-1 truncate">{conversation.title}</span>
        <span className="shrink-0 text-[0.65rem] text-sidebar-foreground/40 group-hover/item:hidden">
          {relativeTime(conversation.updatedAt)}
        </span>
      </button>

      <button
        onClick={() => setMenuOpen((v) => !v)}
        className={cn(
          "absolute right-1.5 top-1/2 grid h-7 w-7 -translate-y-1/2 place-items-center rounded-md text-sidebar-foreground/70 transition-all hover:bg-white/10 hover:text-white",
          menuOpen ? "opacity-100" : "opacity-0 group-hover/item:opacity-100",
        )}
        aria-label="Conversation options"
        aria-haspopup="menu"
        aria-expanded={menuOpen}
      >
        <MoreHorizontal className="h-4 w-4" />
      </button>

      <AnimatePresence>
        {menuOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -4 }}
            transition={{ duration: 0.12 }}
            role="menu"
            className="absolute right-1 top-9 z-20 w-40 overflow-hidden rounded-lg border border-border bg-popover p-1 text-popover-foreground shadow-elevated"
          >
            <MenuItem
              icon={conversation.pinned ? PinOff : Pin}
              label={conversation.pinned ? "Unpin" : "Pin"}
              onClick={() => {
                onTogglePin();
                setMenuOpen(false);
              }}
            />
            <MenuItem
              icon={Pencil}
              label="Rename"
              onClick={() => {
                setDraft(conversation.title);
                setEditing(true);
                setMenuOpen(false);
              }}
            />
            <MenuItem
              icon={Trash2}
              label="Delete"
              danger
              onClick={() => {
                onDelete();
                setMenuOpen(false);
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  );
}

function MenuItem({
  icon: Icon,
  label,
  onClick,
  danger,
}: {
  icon: typeof Pin;
  label: string;
  onClick: () => void;
  danger?: boolean;
}) {
  return (
    <button
      role="menuitem"
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors",
        danger
          ? "text-destructive hover:bg-destructive/10"
          : "text-foreground hover:bg-accent",
      )}
    >
      <Icon className="h-4 w-4" />
      {label}
    </button>
  );
}
