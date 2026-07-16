import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { ResolvedTheme, ThemePreference } from "@/types";
import { readString, STORAGE_KEYS, writeString } from "@/utils/storage";

interface ThemeContextValue {
  /** The user's stored preference (may be "system"). */
  preference: ThemePreference;
  /** The concrete theme currently applied ("light" | "dark"). */
  theme: ResolvedTheme;
  setPreference: (pref: ThemePreference) => void;
  toggle: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function systemTheme(): ResolvedTheme {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function resolve(pref: ThemePreference): ResolvedTheme {
  return pref === "system" ? systemTheme() : pref;
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [preference, setPreferenceState] = useState<ThemePreference>(() => {
    const stored = readString(STORAGE_KEYS.theme) as ThemePreference | null;
    return stored ?? "system";
  });
  const [theme, setTheme] = useState<ResolvedTheme>(() => resolve(preference));

  // Apply the resolved theme to <html> and keep <meta theme-color> in sync.
  useEffect(() => {
    const resolved = resolve(preference);
    setTheme(resolved);
    const root = document.documentElement;
    root.classList.toggle("dark", resolved === "dark");
    root.style.colorScheme = resolved;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", resolved === "dark" ? "#0b1120" : "#4f46e5");
  }, [preference]);

  // React to OS-level changes while the user is on "system".
  useEffect(() => {
    if (preference !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setPreferenceState("system");
    const handler = () => onChange();
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [preference]);

  const setPreference = useCallback((pref: ThemePreference) => {
    setPreferenceState(pref);
    writeString(STORAGE_KEYS.theme, pref);
  }, []);

  const toggle = useCallback(() => {
    setPreferenceState((prev) => {
      const next: ThemePreference = resolve(prev) === "dark" ? "light" : "dark";
      writeString(STORAGE_KEYS.theme, next);
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({ preference, theme, setPreference, toggle }),
    [preference, theme, setPreference, toggle],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within a ThemeProvider");
  return ctx;
}
