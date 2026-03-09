"""Shared utility functions for the UKT framework."""
from __future__ import annotations

import numpy as np


def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate a 1D array to target length."""
    arr = arr.flatten()
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)))
