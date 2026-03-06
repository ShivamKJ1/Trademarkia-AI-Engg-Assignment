from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info("Embedding model loaded: %s on device=%s", model_name, self.device)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    @staticmethod
    def save(path: Path, embeddings: np.ndarray) -> None:
        np.save(path, embeddings)

    @staticmethod
    def load(path: Path) -> np.ndarray:
        return np.load(path).astype(np.float32)
