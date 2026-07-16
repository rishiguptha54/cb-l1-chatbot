import { lazy, Suspense } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import { Spinner } from "@/components/ui/Spinner";

const LandingPage = lazy(() => import("@/pages/LandingPage"));
const ChatPage = lazy(() => import("@/pages/ChatPage"));

function RouteFallback() {
  return (
    <div className="grid h-full place-items-center bg-background text-primary">
      <Spinner className="h-6 w-6" />
    </div>
  );
}

export function AppRoutes() {
  const location = useLocation();
  return (
    <Suspense fallback={<RouteFallback />}>
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<LandingPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AnimatePresence>
    </Suspense>
  );
}
