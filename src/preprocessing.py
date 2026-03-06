"""Text normalization helpers used before embedding generation."""

from __future__ import annotations

import re
from typing import Iterable, List


_whitespace_re = re.compile(r"\s+")
_non_word_re = re.compile(r"[^a-z0-9\s.,!?;:'\"()\-]")


def basic_clean_text(text: str) -> str:
    """
    Perform lightweight normalization for semantic retrieval.

    Design choice:
    - Keep punctuation that can preserve sentence intent.
    - Remove noisy symbols and collapse whitespace for cleaner embeddings.
    """
    text = text.lower().strip()
    text = _non_word_re.sub(" ", text)
    text = _whitespace_re.sub(" ", text)
    return text


def preprocess_texts(texts: Iterable[str]) -> List[str]:
    """Apply basic cleaner to each document in ingestion order."""
    return [basic_clean_text(t) for t in texts]
