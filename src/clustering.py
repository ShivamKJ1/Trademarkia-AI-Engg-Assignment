"""Soft clustering component using Gaussian Mixture Models and silhouette selection."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class SoftClusterer:
    """
    Soft clustering wrapper around sklearn GaussianMixture.

    Why GMM:
    - provides probabilistic cluster memberships per document
    - directly supports "dominant cluster" and uncertainty-aware logic
    """

    def __init__(
        self,
        min_clusters: int = 5,
        max_clusters: int = 30,
        random_state: int = 42,
        sample_size: int = 3000,
    ) -> None:
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.sample_size = sample_size
        self.model: GaussianMixture | None = None
        self.cluster_probabilities: np.ndarray | None = None
        self.cluster_labels: np.ndarray | None = None

    def _select_k(self, embeddings: np.ndarray) -> int:
        """Pick cluster count using silhouette score in the configured search range."""
        n = len(embeddings)
        if n < self.min_clusters:
            return max(2, min(n - 1, self.min_clusters))

        eval_idx = np.arange(n)
        if n > self.sample_size:
            # Subsample to keep model selection tractable on local hardware.
            rng = np.random.default_rng(self.random_state)
            eval_idx = rng.choice(n, size=self.sample_size, replace=False)

        eval_vectors = embeddings[eval_idx]

        best_k = self.min_clusters
        best_score = -1.0

        for k in range(self.min_clusters, self.max_clusters + 1):
            if k >= len(eval_vectors):
                break

            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=self.random_state,
                reg_covar=1e-6,
            )
            labels = gmm.fit_predict(eval_vectors)

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                # Silhouette is undefined for degenerate single-cluster predictions.
                continue

            score = silhouette_score(eval_vectors, labels, metric="cosine")
            logger.info("Silhouette for k=%s: %.4f", k, score)

            if score > best_score:
                best_score = score
                best_k = k

        logger.info("Selected %s clusters with silhouette %.4f", best_k, best_score)
        return best_k

    def fit(self, embeddings: np.ndarray) -> None:
        """Train final GMM and cache both hard labels and soft probabilities."""
        best_k = self._select_k(embeddings)
        self.model = GaussianMixture(
            n_components=best_k,
            covariance_type="diag",
            random_state=self.random_state,
            reg_covar=1e-6,
        )
        self.cluster_labels = self.model.fit_predict(embeddings)
        self.cluster_probabilities = self.model.predict_proba(embeddings)
        logger.info("Fitted GMM soft clustering for %s documents", len(embeddings))

    def predict_proba(self, embedding: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Clusterer not initialized.")
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        return self.model.predict_proba(embedding)[0]

    def predict_cluster(self, embedding: np.ndarray) -> int:
        """Return dominant cluster id (argmax over cluster probabilities)."""
        probs = self.predict_proba(embedding)
        return int(np.argmax(probs))

    def get_dominant_cluster(self, doc_id: int) -> int:
        """Helper required by assignment: dominant cluster for a document id."""
        if self.cluster_probabilities is None:
            raise ValueError("Cluster probabilities not initialized.")
        return int(np.argmax(self.cluster_probabilities[doc_id]))

    def save(self, model_path: Path, prob_path: Path, labels_path: Path) -> None:
        if self.model is None or self.cluster_probabilities is None or self.cluster_labels is None:
            raise ValueError("Clusterer not initialized.")
        with model_path.open("wb") as f:
            pickle.dump(self.model, f)
        np.save(prob_path, self.cluster_probabilities.astype(np.float32))
        np.save(labels_path, self.cluster_labels.astype(np.int32))

    def load(self, model_path: Path, prob_path: Path, labels_path: Path) -> None:
        with model_path.open("rb") as f:
            self.model = pickle.load(f)
        self.cluster_probabilities = np.load(prob_path)
        self.cluster_labels = np.load(labels_path)
        logger.info("Loaded clustering model and probabilities from disk")
