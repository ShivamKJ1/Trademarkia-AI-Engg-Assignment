from __future__ import annotations

import logging

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from src.preprocessing import preprocess_texts

logger = logging.getLogger(__name__)


def load_newsgroups_dataset() -> pd.DataFrame:
    """
    Load 20 Newsgroups and apply canonical cleanup.

    Design choice:
    `remove=("headers", "footers", "quotes")` reduces metadata leakage,
    signatures, and reply chains so embeddings focus on semantic content.
    """
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )

    # Text normalization is kept intentionally lightweight to preserve meaning.
    cleaned_text = preprocess_texts(dataset.data)
    df = pd.DataFrame(
        {
            "doc_id": list(range(len(cleaned_text))),
            "text": cleaned_text,
            "target": dataset.target,
            "target_name": [dataset.target_names[i] for i in dataset.target],
        }
    )

    # Drop fully empty documents after cleanup and reindex ids to stay contiguous.
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["doc_id"] = range(len(df))

    logger.info("Loaded %s cleaned documents from 20 Newsgroups", len(df))
    return df
