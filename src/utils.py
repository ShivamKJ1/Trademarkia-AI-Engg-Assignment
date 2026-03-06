from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    """Initialize root logging once for consistent service logs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    """Create directory tree if missing (safe on repeated calls)."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Utility helper for writing small JSON artifacts."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """Utility helper for loading small JSON artifacts."""
    return json.loads(path.read_text(encoding="utf-8"))
