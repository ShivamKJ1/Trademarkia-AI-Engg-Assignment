from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Trademarkia Semantic Search API")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    artifact_dir: Path = Path(os.getenv("ARTIFACT_DIR", "data/artifacts"))

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    cache_similarity_threshold: float = float(
        os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85")
    )

    min_clusters: int = int(os.getenv("MIN_CLUSTERS", "5"))
    max_clusters: int = int(os.getenv("MAX_CLUSTERS", "30"))
    cluster_random_state: int = int(os.getenv("CLUSTER_RANDOM_STATE", "42"))
    cluster_sample_size: int = int(os.getenv("CLUSTER_SAMPLE_SIZE", "3000"))

    @property
    def embeddings_path(self) -> Path:
        return self.artifact_dir / "embeddings.npy"

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / "documents.pkl"

    @property
    def faiss_index_path(self) -> Path:
        return self.artifact_dir / "faiss.index"

    @property
    def cluster_model_path(self) -> Path:
        return self.artifact_dir / "gmm.pkl"

    @property
    def cluster_probabilities_path(self) -> Path:
        return self.artifact_dir / "cluster_probabilities.npy"

    @property
    def cluster_labels_path(self) -> Path:
        return self.artifact_dir / "cluster_labels.npy"


settings = Settings()
