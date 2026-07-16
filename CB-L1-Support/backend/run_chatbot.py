"""Build and serve the CB L1 Support Chatbot.

Usage:
    python run_chatbot.py --build      # build KB -> chunks -> embeddings -> Qdrant
    python run_chatbot.py --serve      # start the FastAPI server
    python run_chatbot.py --ask "How to fix EOM publish failure?"   # one-off CLI query
    python run_chatbot.py --build --serve
"""

from __future__ import annotations

import argparse
import sys


def do_build() -> None:
    from chatbot.build_knowledge_base import build_knowledge_base
    from chatbot.build_chunks import build_chunks
    from chatbot.build_embeddings import build_embeddings
    from chatbot.migrate_defects_qdrant import migrate

    print("=" * 60)
    print(" Building CB L1 Support Chatbot artifacts")
    print("=" * 60)
    build_knowledge_base()
    build_chunks()
    build_embeddings()
    migrate()
    print("\n✓ Build complete.\n")


def do_ask(question: str) -> None:
    from chatbot import defect_qa

    result = defect_qa.ask(question)
    print(f"\n[intent] {result['intent']}\n")
    print(result["answer"])
    if result["similar_defects"]:
        print("\n[similar_defects]")
        for d in result["similar_defects"]:
            print(f"  - {d['issue_key']} ({d['relevance_score']}) {d['summary'][:70]}")


def do_serve() -> None:
    import uvicorn

    import config

    uvicorn.run(
        "api.app:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CB L1 Support Chatbot")
    parser.add_argument("--build", action="store_true", help="Build chatbot artifacts")
    parser.add_argument("--serve", action="store_true", help="Run the API server")
    parser.add_argument("--ask", type=str, default=None, help="Ask a single question and exit")
    args = parser.parse_args()

    if not (args.build or args.serve or args.ask):
        parser.print_help()
        sys.exit(0)

    if args.build:
        do_build()
    if args.ask:
        do_ask(args.ask)
    if args.serve:
        do_serve()


if __name__ == "__main__":
    main()
