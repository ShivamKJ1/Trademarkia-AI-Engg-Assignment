from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.models import CacheStatsResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest, request: Request) -> QueryResponse:
    """Main semantic search endpoint with cache-aware retrieval."""
    engine = getattr(request.app.state, "search_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready.")

    result = engine.query(payload.query)
    return QueryResponse(
        query=result.query,
        cache_hit=result.cache_hit,
        matched_query=result.matched_query,
        similarity_score=result.similarity_score,
        result=result.result,
        dominant_cluster=result.dominant_cluster,
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats(request: Request) -> CacheStatsResponse:
    """Return cache observability metrics (entries, hits, misses, hit rate)."""
    engine = getattr(request.app.state, "search_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready.")

    stats = engine.cache.stats()
    return CacheStatsResponse(**stats)


@router.delete("/cache")
def clear_cache(request: Request) -> dict[str, str]:
    """Flush in-memory semantic cache and reset counters."""
    engine = getattr(request.app.state, "search_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready.")

    engine.cache.clear()
    return {"message": "Cache cleared and metrics reset."}
