"""Stage 7: build the FAISS vector index and the keyword/metadata index.

- FAISS ``IndexFlatIP`` over L2-normalized vectors == cosine similarity. If FAISS
  is unavailable the retriever transparently falls back to a NumPy dot product
  over ``embeddings.npy``.
- The keyword index is a compact BM25-style structure (document frequencies +
  per-chunk tokens) plus an ``issue_key -> chunk rows`` map for aggregation.

Run with:  python -m chatbot.build_indexes
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from collections import Counter, defaultdict

import config
from chatbot import utils
from chatbot.build_embeddings import EMBEDDINGS_NPY_PATH


def build_faiss_index() -> bool:
    import numpy as np

    if not os.path.exists(EMBEDDINGS_NPY_PATH):
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS_NPY_PATH}. "
            "Run `python -m chatbot.build_embeddings` first."
        )
    vectors = np.load(EMBEDDINGS_NPY_PATH).astype("float32")

    try:
        import faiss
    except ImportError:
        print("[index] faiss not installed; NumPy fallback will be used at query time.")
        return False

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    print(f"[index] wrote FAISS index {config.FAISS_INDEX_PATH} ({index.ntotal} vectors)")
    return True


def build_keyword_index() -> None:
    meta = utils.load_json(config.EMBEDDING_METADATA_PATH, default={}) or {}
    chunks = meta.get("chunks", [])
    if not chunks:
        raise FileNotFoundError(
            "Embedding metadata missing chunks. Run `python -m chatbot.build_embeddings` first."
        )

    doc_tokens: list[list[str]] = []
    doc_freq: Counter = Counter()
    issue_to_rows: dict[str, list[int]] = defaultdict(list)
    doc_len: list[int] = []

    for row, chunk in enumerate(chunks):
        # Index searchable text plus high-signal metadata fields.
        blob = " ".join([
            chunk.get("text", ""), chunk.get("summary", ""),
            " ".join(chunk.get("components", [])), " ".join(chunk.get("labels", [])),
            chunk.get("environment", ""), chunk.get("product", ""),
            chunk.get("fix_pattern", ""), chunk.get("defect_type", ""),
        ])
        tokens = utils.tokenize(blob)
        doc_tokens.append(tokens)
        doc_len.append(len(tokens))
        for t in set(tokens):
            doc_freq[t] += 1
        issue_to_rows[chunk.get("issue_key", "")].append(row)

    n = len(chunks)
    avgdl = (sum(doc_len) / n) if n else 0.0
    idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    payload = {
        "n": n,
        "avgdl": avgdl,
        "idf": idf,
        "doc_tokens": doc_tokens,
        "doc_len": doc_len,
        "issue_to_rows": dict(issue_to_rows),
    }
    os.makedirs(os.path.dirname(config.KEYWORD_INDEX_PATH), exist_ok=True)
    with open(config.KEYWORD_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"[index] wrote keyword index {config.KEYWORD_INDEX_PATH} ({n} docs)")


def build_indexes() -> None:
    build_faiss_index()
    build_keyword_index()


if __name__ == "__main__":
    try:
        build_indexes()
    except Exception as exc:  # pragma: no cover
        print(f"[index] ERROR: {exc}", file=sys.stderr)
        raise
