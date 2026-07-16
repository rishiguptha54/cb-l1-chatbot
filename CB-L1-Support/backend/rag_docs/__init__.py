# rag_docs package
# ─────────────────
# Self-contained QUERY-TIME slice of the external "final_model_1" RAG project,
# extracted verbatim so the defect chatbot can query the same Qdrant document
# knowledge base IN-PROCESS (no separate HTTP RAG server required).
#
# Extracted modules (logic unchanged from the source project):
#   • vector_store.py     – Qdrant connection + all retrieval methods
#   • query_router.py     – route classification + widget-condition extraction
#   • query_processor.py  – QueryNormalizer / QueryEnhancer / FallbackHandler
#   • answer_generator.py – Azure GPT answer synthesis with citations
#   • rag_agent.py        – orchestrates the full ask() pipeline
#
# pipeline.py (new) wires them together the same way the original api.py did.

from .pipeline import (
    answer_question,
    pipeline_health,
    pipeline_diagnostics,
    get_agent,
)
from .ingest import ingest_pdf, ingest_pdf_stages

__all__ = [
    "answer_question",
    "pipeline_health",
    "pipeline_diagnostics",
    "get_agent",
    "ingest_pdf",
    "ingest_pdf_stages",
]
