"""Persistent disk cache for pipeline input data.

Saves successful live API fetches to ``data_cache/`` as pickle files so the
pipeline can fall back to stale-but-valid data when APIs are unreachable.

Usage::

    from hyperspace.data import cache

    # After a successful live fetch:
    cache.save("ohlcv", dataframe)

    # Before raising RuntimeError on total failure:
    cached = cache.load("ohlcv")
    if cached is not None:
        return cached, "Cached: OHLCV (offline)"
    raise RuntimeError(...)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ``data_cache/`` lives at the repository root (three levels up from this file:
#   hyperspace/data/cache.py → hyperspace/data/ → hyperspace/ → repo root)
CACHE_DIR: Path = Path(__file__).parent.parent.parent / "data_cache"


def save(key: str, data: Any) -> None:
    """Persist *data* to disk under *key*.  Silently skips on I/O errors."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{key}.pkl"
        with path.open("wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.debug("Cache saved: %s", path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Cache write failed for %r: %s", key, exc)


def load(key: str) -> Any | None:
    """Load cached data for *key*.  Returns ``None`` if not found or corrupt."""
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            data = pickle.load(fh)
        log.debug("Cache hit: %s", path)
        return data
    except Exception as exc:  # noqa: BLE001
        log.warning("Cache read failed for %r: %s", key, exc)
        return None
