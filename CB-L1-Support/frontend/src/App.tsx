import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { queryClient } from "@/services/queryClient";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { ChatProvider } from "@/features/chat/ChatProvider";
import { AppRoutes } from "@/routes/AppRoutes";

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>
          <ChatProvider>
            <AppRoutes />
          </ChatProvider>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
