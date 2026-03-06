from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CacheEntry:
    """One semantic cache record tied to a predicted cluster."""
    query: str
    embedding: np.ndarray
    result: str
    cluster_id: int


class SemanticCache:
    """
    Cluster-aware in-memory semantic cache.

    Storage shape:
    cluster_id -> [CacheEntry, CacheEntry, ...]
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold
        self._store: Dict[int, List[CacheEntry]] = {}
        self.hit_count = 0
        self.miss_count = 0

    @property
    def total_entries(self) -> int:
        return sum(len(entries) for entries in self._store.values())

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # Embeddings are normalized at creation time, so dot product equals cosine.
        return float(np.dot(a, b))

    def get(
        self,
        query_embedding: np.ndarray,
        cluster_id: int,
    ) -> tuple[bool, Optional[CacheEntry], float]:
        """
        Return best matching entry within the same cluster.
        A hit is declared only when best cosine >= configured threshold.
        """
        entries = self._store.get(cluster_id, [])
        best_entry: Optional[CacheEntry] = None
        best_score = -1.0

        for entry in entries:
            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.similarity_threshold:
            self.hit_count += 1
            return True, best_entry, best_score

        self.miss_count += 1
        return False, None, best_score

    def put(self, query: str, embedding: np.ndarray, result: str, cluster_id: int) -> None:
        """Insert query-response pair in cluster-local bucket."""
        entry = CacheEntry(
            query=query,
            embedding=embedding,
            result=result,
            cluster_id=cluster_id,
        )
        self._store.setdefault(cluster_id, []).append(entry)

    def clear(self) -> None:
        """Flush cache entries and reset observability counters."""
        self._store.clear()
        self.hit_count = 0
        self.miss_count = 0

    def stats(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
        }
