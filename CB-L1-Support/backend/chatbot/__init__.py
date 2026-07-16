"""CB L1 Support Chatbot package.

A retrieval-augmented pipeline that answers defect diagnostic / fix questions
over the existing Jira defect data: questions are answered from a
hybrid-retrieved set of historical defects, with optional LLM synthesis and a
deterministic fallback, augmented by the product-documentation RAG.
"""

__all__ = [
    "build_knowledge_base",
    "build_chunks",
    "build_embeddings",
    "build_indexes",
    "retriever",
    "intent_router",
    "defect_qa",
    "answer_generator",
    "prompts",
    "schemas",
    "utils",
]
