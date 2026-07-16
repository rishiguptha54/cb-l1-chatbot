/** Safe, typed wrappers around localStorage with graceful failure. */

export function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function writeJSON<T>(key: string, value: T): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage full or unavailable — degrade silently */
  }
}

export function readString(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeString(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}

export const STORAGE_KEYS = {
  conversations: "ai-assistant.conversations.v1",
  activeConversation: "ai-assistant.active.v1",
  theme: "ai-assistant.theme.v1",
  sidebarCollapsed: "ai-assistant.sidebar.collapsed.v1",
} as const;
