"""Stage 6: compute embeddings for every chunk.

Provides a single :class:`EmbeddingProvider` used both at build time and at query
time, so vectors are always produced by the same model. Prefers an Azure OpenAI
embedding deployment; otherwise falls back to a local sentence-transformers
model. Vectors are L2-normalized so a FAISS inner-product index yields cosine
similarity.

Run with:  python -m chatbot.build_embeddings
"""

from __future__ import annotations

import os
import sys
from typing import Any

import config
from chatbot import utils
from chatbot.build_chunks import load_chunks

EMBEDDINGS_NPY_PATH = os.path.join(config.INDEX_DIR, "embeddings.npy")


class EmbeddingProvider:
    """Unified text-embedding interface with Azure-first, local-fallback logic."""

    def __init__(self) -> None:
        self.backend = "none"
        self._provider_embed = None
        self._st_model = None
        self.model_id = ""
        self.dim = 0
        self._init_backend()

    def _init_backend(self) -> None:
        # Prefer the provider-agnostic embedding API (Azure or GitHub Models,
        # selected by LLM_PROVIDER). Falls back to a local sentence-transformers
        # model when no remote embedding provider is usable.
        if config.USE_LLM:
            try:
                import llm_provider

                # Probe once so config/auth errors surface here, not mid-build.
                probe = llm_provider.embed([" "], stage="embedding")
                self._provider_embed = llm_provider.embed
                self.dim = len(probe[0]) if probe else 0
                self.backend = "remote"
                self.model_id = llm_provider.pick_model("embedding")
                return
            except Exception as exc:  # pragma: no cover - env dependent
                print(f"[embed] remote embedding init failed ({exc}); using local model.")

        # Fall back to sentence-transformers.
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
            self.dim = self._st_model.get_sentence_embedding_dimension()
            self.backend = "local"
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "No embedding backend available. Configure an embedding "
                "provider (LLM_PROVIDER) or install sentence-transformers."
            ) from exc

    # ── Embedding ──
    def embed(self, texts: list[str]) -> "Any":
        import numpy as np

        if not texts:
            return np.zeros((0, max(self.dim, 1)), dtype="float32")

        if self.backend == "remote":
            vectors = self._embed_remote(texts)
        else:
            vectors = self._st_model.encode(
                texts, batch_size=config.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False, convert_to_numpy=True,
            )
        vectors = np.asarray(vectors, dtype="float32")
        self.dim = vectors.shape[1]
        return _l2_normalize(vectors)

    def embed_query(self, text: str) -> "Any":
        return self.embed([text or ""])[0]

    def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        batch = max(1, config.EMBEDDING_BATCH_SIZE)
        total = (len(texts) + batch - 1) // batch
        for n, i in enumerate(range(0, len(texts), batch), start=1):
            chunk = [t if t.strip() else " " for t in texts[i : i + batch]]
            out.extend(self._provider_embed(chunk, stage="embedding"))
            print(f"[embed]   batch {n}/{total} ({len(out)}/{len(texts)})", flush=True)
        return out


def _l2_normalize(vectors):
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


def build_embeddings() -> int:
    import numpy as np

    chunks = load_chunks()
    if not chunks:
        raise FileNotFoundError(
            f"No chunks found at {config.DEFECT_CHUNKS_PATH}. "
            "Run `python -m chatbot.build_chunks` first."
        )

    provider = EmbeddingProvider()
    texts = [c.get("text", "") for c in chunks]
    print(f"[embed] backend={provider.backend} | embedding {len(texts)} chunks…")
    vectors = provider.embed(texts)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    np.save(EMBEDDINGS_NPY_PATH, vectors)

    metadata = {
        "backend": provider.backend,
        "model": (
            config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            if provider.backend == "azure"
            else config.LOCAL_EMBEDDING_MODEL
        ),
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "chunks": chunks,  # row-aligned with the embedding matrix
    }
    utils.save_json(config.EMBEDDING_METADATA_PATH, metadata)

    print(f"[embed] wrote {EMBEDDINGS_NPY_PATH} ({vectors.shape})")
    print(f"[embed] wrote {config.EMBEDDING_METADATA_PATH}")
    return int(vectors.shape[0])


if __name__ == "__main__":
    try:
        build_embeddings()
    except Exception as exc:  # pragma: no cover
        print(f"[embed] ERROR: {exc}", file=sys.stderr)
        raise
