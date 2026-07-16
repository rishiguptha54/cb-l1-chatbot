import { useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { readJSON, STORAGE_KEYS, writeJSON } from "@/utils/storage";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";

export function ChatLayout({ children }: { children: ReactNode }) {
  const isDesktop = useMediaQuery("(min-width: 768px)");
  const [collapsed, setCollapsed] = useState<boolean>(() =>
    readJSON<boolean>(STORAGE_KEYS.sidebarCollapsed, false),
  );
  const [mobileOpen, setMobileOpen] = useState(false);

  const setCollapsedPersist = (value: boolean) => {
    setCollapsed(value);
    writeJSON(STORAGE_KEYS.sidebarCollapsed, value);
  };

  return (
    <div className="flex h-full w-full overflow-hidden bg-background">
      {/* Desktop sidebar */}
      <aside className="hidden shrink-0 border-r border-sidebar-border md:block">
        <Sidebar
          collapsed={collapsed}
          onToggleCollapse={() => setCollapsedPersist(!collapsed)}
          onExpand={() => setCollapsedPersist(false)}
        />
      </aside>

      {/* Mobile sidebar drawer */}
      <AnimatePresence>
        {mobileOpen && !isDesktop && (
          <div className="fixed inset-0 z-40 md:hidden">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileOpen(false)}
              className="absolute inset-0 bg-black/50 backdrop-blur-sm"
              aria-hidden="true"
            />
            <motion.div
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", damping: 30, stiffness: 300 }}
              className="absolute left-0 top-0 h-full border-r border-sidebar-border shadow-elevated"
            >
              <Sidebar
                collapsed={false}
                isMobile
                onToggleCollapse={() => setMobileOpen(false)}
                onExpand={() => setMobileOpen(false)}
                onCloseMobile={() => setMobileOpen(false)}
                onNavigate={() => setMobileOpen(false)}
              />
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <Header
          onOpenMobileSidebar={() => setMobileOpen(true)}
          sidebarCollapsed={collapsed}
          onExpandSidebar={() => setCollapsedPersist(false)}
        />
        <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
      </div>
    </div>
  );
}
