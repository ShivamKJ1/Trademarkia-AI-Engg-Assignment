from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Input payload for semantic search."""
    query: str = Field(..., min_length=1, description="Natural language query")


class QueryResponse(BaseModel):
    """Response payload returned by /query."""
    query: str
    cache_hit: bool
    matched_query: str | None = None
    similarity_score: float | None = None
    result: str
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    """Observability payload for cache behavior."""
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
