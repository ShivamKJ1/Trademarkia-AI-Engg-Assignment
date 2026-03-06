from __future__ import annotations

import numpy as np

from src.semantic_cache import SemanticCache


def test_semantic_cache_hit_and_miss_metrics() -> None:
    cache = SemanticCache(similarity_threshold=0.85)

    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)

    cache.put("query-a", a, "result-a", cluster_id=1)

    hit, entry, score = cache.get(a, cluster_id=1)
    assert hit is True
    assert entry is not None
    assert entry.query == "query-a"
    assert score >= 0.99

    hit, entry, _ = cache.get(b, cluster_id=1)
    assert hit is False
    assert entry is None

    stats = cache.stats()
    assert stats["total_entries"] == 1
    assert stats["hit_count"] == 1
    assert stats["miss_count"] == 1
