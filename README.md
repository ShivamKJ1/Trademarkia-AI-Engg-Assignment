# Trademarkia Semantic Retrieval System

Production-style AI/ML + backend service for semantic search over the 20 Newsgroups corpus with:
- dense embeddings (`all-MiniLM-L6-v2`)
- FAISS vector retrieval (cosine similarity)
- soft clustering (Gaussian Mixture Model)
- custom semantic cache implemented from scratch
- FastAPI serving layer with startup orchestration and cache analytics

This repository is designed to demonstrate strong ML system design, API engineering, and code organization expected in production teams.

## Highlights

- End-to-end pipeline from ingestion to serving
- GPU acceleration (`cuda`) with automatic CPU fallback
- Artifact persistence for fast reboots (embeddings, index, cluster model)
- Cluster-aware cache that reduces repeated vector searches
- Clean, modular codebase with type hints, docstrings, and logging

## System Architecture

```text
                        +----------------------------+
                        |  FastAPI (api/main.py)     |
                        |  /query /cache/stats /cache|
                        +-------------+--------------+
                                      |
                                      v
                        +----------------------------+
                        | SearchEngine Orchestrator  |
                        +-------------+--------------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
          +----------------------+         +----------------------+
          | EmbeddingService     |         | SoftClusterer (GMM)  |
          | all-MiniLM-L6-v2     |         | dominant cluster     |
          +----------+-----------+         +----------+-----------+
                     |                                 |
                     +----------------+----------------+
                                      |
                                      v
                           +-----------------------+
                           | SemanticCache         |
                           | cluster -> entries[]  |
                           +-----------+-----------+
                                       |
                          hit          | miss
                          +------------+------------+
                          |                         |
                          v                         v
                 return cached result      +------------------+
                                           | VectorStore      |
                                           | FAISS IndexFlatIP|
                                           +--------+---------+
                                                    |
                                                    v
                                          top-k docs + format
```

## Project Layout

```text
project_root/
  README.md
  requirements.txt
  Dockerfile
  docker-compose.yml
  .env
  data/
    artifacts/                 # persisted embeddings/index/clustering artifacts
  src/
    config.py                  # environment-driven settings
    data_loader.py             # 20 Newsgroups ingestion
    preprocessing.py           # normalization and text cleaning
    embeddings.py              # model loading + batch encoding
    vector_store.py            # FAISS index build/load/search
    clustering.py              # GMM soft clustering + model selection
    semantic_cache.py          # custom cache + metrics
    search_engine.py           # query pipeline orchestration
    models.py                  # request/response schemas
    utils.py                   # logging and filesystem helpers
  api/
    main.py                    # app factory + startup lifecycle
    routes.py                  # HTTP endpoints
  tests/
    test_semantic_cache.py
  notebooks/
```

## Technical Decisions

1. Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- strong quality-speed tradeoff for laptop inference
- normalized vectors enable cosine similarity via dot product

2. Vector DB: FAISS `IndexFlatIP`
- simple, local, and high-performance for medium-scale retrieval
- no external service dependency

3. Soft Clustering: `GaussianMixture`
- returns full probability distribution per document
- supports assignment uncertainty, unlike hard `k-means`

4. Cache Strategy: cluster-local semantic matching
- narrows comparisons to one cluster
- improves lookup cost and semantic precision
- configurable similarity threshold (default `0.85`)

## Query Flow

1. Receive query at `POST /query`
2. Encode query into normalized embedding
3. Predict dominant cluster with GMM
4. Check semantic cache within that cluster only
5. On hit: return cached response with similarity and matched query
6. On miss: run FAISS top-k search, format response, insert into cache

## Startup Lifecycle

Executed once during FastAPI startup:

1. Load and preprocess dataset (`remove=("headers", "footers", "quotes")`)
2. Load embeddings from disk or compute and save
3. Load FAISS index + metadata or build and save
4. Load GMM + probabilities or train and save
5. Initialize empty in-memory cache and metrics

This design keeps first run heavier and subsequent runs fast.

## Setup (Windows)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn api.main:app --reload --port 8010
```

Open:
- Swagger: `http://127.0.0.1:8010/docs`
- ReDoc: `http://127.0.0.1:8010/redoc`

## API Usage

### Show that API is reachable in Terminal 2

```bash 
Invoke-RestMethod -Uri "http://127.0.0.1:8010/cache/stats" -Method Get
```

### 1) Query

```bash 
$body1 = @{ query = "How can I install Linux on my PC?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8010/query" -Method Post -ContentType "application/json" -Body $body1

```

Sample response:

```json
{
  "query": "How do I install Linux on my PC?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "1. [doc_id=123, score=0.8123, topic=comp.os.ms-windows.misc] ...",
  "dominant_cluster": 7
}
```

### 2) Cache Stats

```bash
curl "http://127.0.0.1:8010/cache/stats"
```

### 3) Flush Cache

```bash
curl -X DELETE "http://127.0.0.1:8010/cache"
```

## Demo Script (Quick Walkthrough)

Use this sequence during interviews or demos to show the full behavior:

```bash
# 1) Start API
uvicorn api.main:app --reload --port 8010
```

In a second terminal:

```bash
#2. Show API is reachable (Terminal 2)
Invoke-RestMethod -Uri "http://127.0.0.1:8010/cache/stats" -Method Get

#3. First query (expected cache miss)
$body1 = @{ query = "How can I install Linux on my PC?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8010/query" -Method Post -ContentType "application/json" -Body $body1

#4. Similar query (show semantic cache behavior)
$body2 = @{ query = "How do I install Linux on my computer?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8010/query" -Method Post -ContentType "application/json" -Body $body2

#5. Show cache stats
Invoke-RestMethod -Uri "http://127.0.0.1:8010/cache/stats" -Method Get

#6. Clear cache
Invoke-RestMethod -Uri "http://127.0.0.1:8010/cache" -Method Delete

#7. Verify reset
Invoke-RestMethod -Uri "http://127.0.0.1:8010/cache/stats" -Method Get
```

## Docker

```bash
docker compose up --build
```

Service is exposed on port `8000` by default in container config.

## Configuration

Tune behavior from `.env`:
- `BATCH_SIZE`
- `TOP_K`
- `CACHE_SIMILARITY_THRESHOLD`
- `MIN_CLUSTERS`
- `MAX_CLUSTERS`
- `CLUSTER_SAMPLE_SIZE`

## Recruiter Notes

This implementation intentionally emphasizes:
- production-grade modularity over notebook-style scripting
- deterministic and explainable data/ML flow
- practical serving concerns (cold start vs warm start, persistence, logging)
- measurable performance behavior (cache hit/miss/hit-rate)
