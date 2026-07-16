"""One-time migration: upsert the already-built defect chunks + embeddings
(``embedding_metadata.json`` + ``embeddings.npy``) into the Qdrant "defect"
collection (see ``config.DEFECT_QDRANT_COLLECTION``).

Reuses the embeddings already computed by ``build_embeddings.py`` — no
re-embedding, so this is fast and free. Run once after the local knowledge
base has been built (``build_knowledge_base`` -> ``build_chunks`` ->
``build_embeddings``), or again any time to re-sync.

Run with:  python -m chatbot.migrate_defects_qdrant
"""

from __future__ import annotations

import os
import sys

import config
from chatbot import utils
from chatbot.build_embeddings import EMBEDDINGS_NPY_PATH
from chatbot.defect_vector_store import get_defect_vector_store

_BATCH_SIZE = 200


def migrate() -> int:
    meta = utils.load_json(config.EMBEDDING_METADATA_PATH, default={}) or {}
    chunks: list[dict] = meta.get("chunks", [])
    if not chunks:
        raise FileNotFoundError(
            f"No chunks found in {config.EMBEDDING_METADATA_PATH}. "
            "Run `python -m chatbot.build_embeddings` first."
        )
    if not os.path.exists(EMBEDDINGS_NPY_PATH):
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS_NPY_PATH}. "
            "Run `python -m chatbot.build_embeddings` first."
        )

    import numpy as np

    vectors = np.load(EMBEDDINGS_NPY_PATH).astype("float32")
    if vectors.shape[0] != len(chunks):
        raise RuntimeError(
            f"Chunk/vector count mismatch: {len(chunks)} chunks vs "
            f"{vectors.shape[0]} vectors. Rebuild embeddings first."
        )

    store = get_defect_vector_store()
    print(
        f"[migrate] Upserting {len(chunks)} defect chunks into Qdrant "
        f"collection '{store.collection_name}' (dim={vectors.shape[1]})..."
    )

    total = len(chunks)
    for i in range(0, total, _BATCH_SIZE):
        batch_chunks = chunks[i : i + _BATCH_SIZE]
        batch_vectors = vectors[i : i + _BATCH_SIZE]
        store.upsert_chunks(batch_chunks, batch_vectors)
        print(f"[migrate]   {min(i + _BATCH_SIZE, total)}/{total}")

    print(f"[migrate] Done — {total} chunks stored in '{store.collection_name}'.")
    return total


if __name__ == "__main__":
    try:
        migrate()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"[migrate] ERROR: {exc}", file=sys.stderr)
        raise
