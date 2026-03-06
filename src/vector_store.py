"""FAISS-backed vector index wrapper with metadata-aware similarity search."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin FAISS wrapper with metadata persistence for local semantic retrieval."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: pd.DataFrame | None = None

    def build(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
        dim = embeddings.shape[1]
        # Embeddings are unit-normalized; inner product is equivalent to cosine similarity.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        self.metadata = metadata.reset_index(drop=True)
        logger.info("Built FAISS index with %s vectors (dim=%s)", embeddings.shape[0], dim)

    def save(self, index_path: Path, metadata_path: Path) -> None:
        if self.index is None or self.metadata is None:
            raise ValueError("Vector store not initialized.")
        faiss.write_index(self.index, str(index_path))
        self.metadata.to_pickle(metadata_path)

    def load(self, index_path: Path, metadata_path: Path) -> None:
        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_pickle(metadata_path)
        logger.info("Loaded FAISS index and metadata from disk")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.index is None or self.metadata is None:
            raise ValueError("Vector store not initialized.")

        if query_embedding.ndim == 1:
            # FAISS expects 2D arrays: (num_queries, dim)
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.metadata.iloc[int(idx)]
            results.append(
                {
                    "doc_id": int(row["doc_id"]),
                    "text": str(row["text"]),
                    "score": float(score),
                    "target_name": str(row.get("target_name", "")),
                }
            )

        return results
