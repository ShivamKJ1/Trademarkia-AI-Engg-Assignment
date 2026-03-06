from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.clustering import SoftClusterer
from src.config import Settings
from src.data_loader import load_newsgroups_dataset
from src.embeddings import EmbeddingService
from src.semantic_cache import SemanticCache
from src.utils import ensure_dir
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class QueryExecution:
    """Structured output returned by the query pipeline and then mapped to API schema."""
    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float | None
    result: str
    dominant_cluster: int


class SearchEngine:
    """
    Coordinates all ML/runtime components used by the API:
    embeddings, vector search, clustering, and semantic cache.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_service = EmbeddingService(
            model_name=settings.embedding_model,
            batch_size=settings.batch_size,
        )
        self.vector_store = VectorStore()
        self.clusterer = SoftClusterer(
            min_clusters=settings.min_clusters,
            max_clusters=settings.max_clusters,
            random_state=settings.cluster_random_state,
            sample_size=settings.cluster_sample_size,
        )
        self.cache = SemanticCache(settings.cache_similarity_threshold)
        self.documents: pd.DataFrame | None = None

    def initialize(self) -> None:
        """
        Startup orchestration:
        - always load raw dataset (source of truth)
        - reuse persisted artifacts when available
        - train/build only on first run or when artifacts are missing
        """
        logger.info("Initializing search engine pipeline")
        ensure_dir(self.settings.data_dir)
        ensure_dir(self.settings.artifact_dir)

        self.documents = load_newsgroups_dataset()

        embeddings = self._load_or_create_embeddings(self.documents["text"].tolist())
        self._load_or_create_vector_store(embeddings, self.documents)
        self._load_or_create_clusters(embeddings)

        logger.info("Initialization complete")

    def _load_or_create_embeddings(self, texts: list[str]):
        if self.settings.embeddings_path.exists():
            logger.info("Loading embeddings from %s", self.settings.embeddings_path)
            return self.embedding_service.load(self.settings.embeddings_path)

        # Batch encoding is handled by EmbeddingService to avoid memory spikes.
        logger.info("Computing embeddings for %s documents", len(texts))
        embeddings = self.embedding_service.encode(texts)
        self.embedding_service.save(self.settings.embeddings_path, embeddings)
        return embeddings

    def _load_or_create_vector_store(self, embeddings, documents: pd.DataFrame) -> None:
        if self.settings.faiss_index_path.exists() and self.settings.metadata_path.exists():
            logger.info("Loading FAISS artifacts from disk")
            self.vector_store.load(self.settings.faiss_index_path, self.settings.metadata_path)
            return

        # Build once and persist; warm restarts can skip index construction.
        logger.info("Building FAISS index")
        self.vector_store.build(embeddings=embeddings, metadata=documents)
        self.vector_store.save(self.settings.faiss_index_path, self.settings.metadata_path)

    def _load_or_create_clusters(self, embeddings) -> None:
        if (
            self.settings.cluster_model_path.exists()
            and self.settings.cluster_probabilities_path.exists()
            and self.settings.cluster_labels_path.exists()
        ):
            logger.info("Loading clustering artifacts from disk")
            self.clusterer.load(
                self.settings.cluster_model_path,
                self.settings.cluster_probabilities_path,
                self.settings.cluster_labels_path,
            )
            return

        # GMM fitting and silhouette evaluation are the most expensive CPU steps.
        logger.info("Training clustering model")
        self.clusterer.fit(embeddings)
        self.clusterer.save(
            self.settings.cluster_model_path,
            self.settings.cluster_probabilities_path,
            self.settings.cluster_labels_path,
        )

    @staticmethod
    def _format_results(results: list[dict]) -> str:
        if not results:
            return "No documents found."

        lines = []
        for i, item in enumerate(results, start=1):
            snippet = item["text"][:280].replace("\n", " ")
            lines.append(
                f"{i}. [doc_id={item['doc_id']}, score={item['score']:.4f}, topic={item['target_name']}] {snippet}"
            )
        return "\n".join(lines)

    def query(self, query_text: str, top_k: int | None = None) -> QueryExecution:
        """Execute query -> embed -> cluster -> cache -> FAISS fallback pipeline."""
        top_k = top_k or self.settings.top_k

        query_embedding = self.embedding_service.encode([query_text])[0]
        cluster_id = self.clusterer.predict_cluster(query_embedding)

        # Cache lookup is restricted to the predicted cluster for faster matching.
        hit, cache_entry, similarity = self.cache.get(query_embedding, cluster_id)
        if hit and cache_entry is not None:
            return QueryExecution(
                query=query_text,
                cache_hit=True,
                matched_query=cache_entry.query,
                similarity_score=similarity,
                result=cache_entry.result,
                dominant_cluster=cluster_id,
            )

        results = self.vector_store.search(query_embedding, top_k=top_k)
        formatted = self._format_results(results)
        # Store the formatted response as-is, which keeps cache reads constant-time.
        self.cache.put(query_text, query_embedding, formatted, cluster_id)

        return QueryExecution(
            query=query_text,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result=formatted,
            dominant_cluster=cluster_id,
        )
