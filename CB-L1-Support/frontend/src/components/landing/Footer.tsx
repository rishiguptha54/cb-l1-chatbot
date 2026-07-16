import { Link } from "react-router-dom";
import { APP_NAME, APP_VERSION } from "@/config/app";
import { Logo } from "@/components/ui/Logo";

export function Footer() {
  return (
    <footer className="border-t border-border bg-muted/30">
      <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6">
        <div>
          <Logo size={36} withWordmark subtitle="Grounded in real history" />
          <p className="mt-4 max-w-md text-sm leading-relaxed text-muted-foreground">
            Reads your defect history and product docs to answer root-cause and fix
            questions — every answer cited by its Jira key.
          </p>
        </div>

        <div className="mt-12 grid gap-6 border-t border-border pt-8 sm:grid-cols-2">
          <div>
            <h4 className="text-[0.7rem] font-semibold uppercase tracking-[0.16em] text-primary">
              Powered by
            </h4>
            <p className="mt-2 text-sm text-muted-foreground">
              Jira <span className="text-muted-foreground/70">(MCP, read-only)</span> · FAISS + BM25 · Qdrant · GPT-4o
            </p>
          </div>
          <div className="sm:text-right">
            <h4 className="text-[0.7rem] font-semibold uppercase tracking-[0.16em] text-primary">
              Built with
            </h4>
            <p className="mt-2 text-sm text-muted-foreground">
              Python · FastAPI · React · LangChain · Docker · Render
            </p>
          </div>
        </div>

        <div className="mt-10 flex flex-col items-center justify-between gap-4 border-t border-border pt-6 sm:flex-row">
          <p className="text-xs text-muted-foreground">
            © {new Date().getFullYear()} {APP_NAME} · v{APP_VERSION} · Honeywell
          </p>
          <Link
            to="/chat"
            className="text-xs font-medium text-primary transition-colors hover:text-primary/80"
          >
            Open the Chatbot →
          </Link>
        </div>
      </div>
    </footer>
  );
}
