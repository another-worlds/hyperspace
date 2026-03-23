"""Intelligent caching utilities for expensive operations.

Provides deterministic caching for:
- SAE training results
- SVD decompositions
- Stability estimation
- Model weights
- Derived dataframes (pivot tables, correlation matrices, centrality)
- Plotly chart objects
- Narrative text (kernel, canvas, concept, reality regression)

Cache keys are built from data hashes + hyperparameters to ensure
deterministic, auditable, and reproducible results.
"""
from __future__ import annotations

import hashlib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any, Callable


def hash_ndarray(arr: np.ndarray, prefix: str = "") -> str:
    """Generate a SHA-256 hash of a numpy array for cache keying.

    Args:
        arr: Input array
        prefix: Optional prefix for composite keys

    Returns:
        Hex hash string
    """
    h = hashlib.sha256()
    if prefix:
        h.update(prefix.encode())
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()[:16]  # Use first 16 chars for brevity


def hash_params(params: dict) -> str:
    """Generate hash from a parameters dict.

    Args:
        params: Dictionary of parameters

    Returns:
        Hex hash string
    """
    h = hashlib.sha256()
    for key in sorted(params.keys()):
        h.update(f"{key}={params[key]}".encode())
    return h.hexdigest()[:16]


def hash_dataframe(df: pd.DataFrame, prefix: str = "") -> str:
    """Generate a SHA-256 hash of a pandas DataFrame for cache keying.

    Hashes shape, dtypes, column names, and first 100 rows for speed.
    """
    h = hashlib.sha256()
    if prefix:
        h.update(prefix.encode())
    h.update(str(df.shape).encode())
    h.update(str(list(df.dtypes)).encode())
    h.update(str(list(df.columns)).encode())
    sample = df.head(100)
    h.update(pd.util.hash_pandas_object(sample).values.tobytes())
    return h.hexdigest()[:16]


def hash_list(lst: list, prefix: str = "") -> str:
    """Generate a SHA-256 hash of a list for cache keying."""
    h = hashlib.sha256()
    if prefix:
        h.update(prefix.encode())
    h.update(pickle.dumps(lst))
    return h.hexdigest()[:16]


def get_or_compute_dataframe(
    cache_key: str,
    compute_fn: Callable[[], Any],
    force_recompute: bool = False,
) -> Any:
    """Get cached derived dataframe or compute it.

    Args:
        cache_key: Pre-built cache key (use hash_dataframe / hash_params)
        compute_fn: Zero-arg callable that produces the result
        force_recompute: If True, ignore cache
    """
    full_key = f"cache_df_{cache_key}"
    if force_recompute:
        st.session_state.pop(full_key, None)
    if full_key in st.session_state:
        return st.session_state[full_key]
    result = compute_fn()
    st.session_state[full_key] = result
    return result


def get_or_compute_figure(
    cache_key: str,
    compute_fn: Callable[[], Any],
    force_recompute: bool = False,
) -> Any:
    """Get cached Plotly figure or compute it.

    Args:
        cache_key: Pre-built cache key
        compute_fn: Zero-arg callable that returns a go.Figure
        force_recompute: If True, ignore cache
    """
    full_key = f"cache_fig_{cache_key}"
    if force_recompute:
        st.session_state.pop(full_key, None)
    if full_key in st.session_state:
        return st.session_state[full_key]
    result = compute_fn()
    st.session_state[full_key] = result
    return result


def get_or_compute_narrative(
    cache_key: str,
    compute_fn: Callable[[], str | None],
    force_recompute: bool = False,
) -> str | None:
    """Get cached narrative text or compute it.

    Args:
        cache_key: Pre-built cache key (should include policy_language_mode)
        compute_fn: Zero-arg callable that returns narrative string
        force_recompute: If True, ignore cache
    """
    full_key = f"cache_narr_{cache_key}"
    if force_recompute:
        st.session_state.pop(full_key, None)
    if full_key in st.session_state:
        return st.session_state[full_key]
    result = compute_fn()
    if result is not None:
        st.session_state[full_key] = result
    return result


def make_sae_cache_key(
    matrix: np.ndarray,
    hidden_dim: int = 16,
    epochs: int = 100,
    lr: float = 0.005,
    prefix: str = "sae",
) -> str:
    """Create a cache key for SAE training.

    Combines matrix hash + hyperparameter hash.
    """
    matrix_hash = hash_ndarray(matrix)
    param_hash = hash_params({
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
    })
    return f"{prefix}_{matrix_hash}_{param_hash}"


def make_svd_cache_key(matrix: np.ndarray, prefix: str = "svd") -> str:
    """Create a cache key for SVD decomposition."""
    return f"{prefix}_{hash_ndarray(matrix)}"


def make_stability_cache_key(
    matrix: np.ndarray,
    n_runs: int = 8,
    noise_std: float = 0.01,
    seed: int | None = None,
    prefix: str = "stability",
) -> str:
    """Create a cache key for stability estimation.

    If seed is provided, the result is deterministic.
    If seed is None, each run is unique (stochastic mode).
    """
    matrix_hash = hash_ndarray(matrix)
    param_hash = hash_params({
        "n_runs": n_runs,
        "noise_std": noise_std,
        "seed": seed if seed is not None else "none",
    })
    return f"{prefix}_{matrix_hash}_{param_hash}"


def get_or_compute_sae(
    matrix: np.ndarray,
    hidden_dim: int = 16,
    epochs: int = 100,
    lr: float = 0.005,
    compute_fn: Any = None,
    force_retrain: bool = False,
) -> Any:
    """Get cached SAE result or compute it.

    Args:
        matrix: Input feature matrix
        hidden_dim: SAE hidden dimension
        epochs: Training epochs
        lr: Learning rate (must match train_sparse_ae default: 0.005)
        compute_fn: Function to call if cache miss: compute_fn(matrix, hidden_dim, epochs, lr)
        force_retrain: If True, ignore cache and retrain

    Returns:
        SAE result from cache or fresh computation
    """
    if compute_fn is None:
        raise ValueError("compute_fn must be provided")

    cache_key = make_sae_cache_key(matrix, hidden_dim, epochs, lr)

    # Check if force_retrain button was clicked
    if force_retrain:
        st.session_state.pop(f"cache_{cache_key}", None)
        return compute_fn(matrix, hidden_dim, epochs, lr)

    # Check session state cache
    cached_key = f"cache_{cache_key}"
    if cached_key in st.session_state:
        st.toast(f"✓ Using cached SAE result (same matrix & hyperparameters)")
        return st.session_state[cached_key]

    # Compute and cache
    result = compute_fn(matrix, hidden_dim, epochs, lr)
    st.session_state[cached_key] = result
    return result


def get_or_compute_svd(
    matrix: np.ndarray,
    compute_fn: Any = None,
    force_recompute: bool = False,
) -> Any:
    """Get cached SVD result or compute it.

    Args:
        matrix: Input feature matrix
        compute_fn: Function to call if cache miss: compute_fn(matrix)
        force_recompute: If True, ignore cache and recompute

    Returns:
        SVD result from cache or fresh computation
    """
    if compute_fn is None:
        raise ValueError("compute_fn must be provided")

    cache_key = make_svd_cache_key(matrix)

    if force_recompute:
        st.session_state.pop(f"cache_{cache_key}", None)
        return compute_fn(matrix)

    cached_key = f"cache_{cache_key}"
    if cached_key in st.session_state:
        return st.session_state[cached_key]

    result = compute_fn(matrix)
    st.session_state[cached_key] = result
    return result


def get_or_compute_stability(
    matrix: np.ndarray,
    n_runs: int = 8,
    noise_std: float = 0.01,
    seed: int | None = 42,
    compute_fn: Any = None,
    force_recompute: bool = False,
) -> Any:
    """Get cached stability estimation or compute it.

    Args:
        matrix: Input feature matrix
        n_runs: Number of noise runs
        noise_std: Noise standard deviation
        seed: Random seed for determinism (None for stochastic)
        compute_fn: Function to call if cache miss
        force_recompute: If True, ignore cache

    Returns:
        Stability result from cache or fresh computation
    """
    if compute_fn is None:
        raise ValueError("compute_fn must be provided")

    cache_key = make_stability_cache_key(matrix, n_runs, noise_std, seed)

    if force_recompute:
        st.session_state.pop(f"cache_{cache_key}", None)
        return compute_fn(matrix, n_runs, noise_std, seed)

    cached_key = f"cache_{cache_key}"
    if cached_key in st.session_state:
        return st.session_state[cached_key]

    result = compute_fn(matrix, n_runs, noise_std, seed)
    st.session_state[cached_key] = result
    return result


def clear_sae_cache() -> None:
    """Clear all SAE-related caches from session state."""
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith("cache_sae_")]
    for k in keys_to_remove:
        st.session_state.pop(k, None)
    st.toast("✓ Cleared SAE cache")


def clear_stability_cache() -> None:
    """Clear all stability-related caches from session state."""
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith("cache_stability_")]
    for k in keys_to_remove:
        st.session_state.pop(k, None)
    st.toast("✓ Cleared stability cache")


def clear_all_caches() -> None:
    """Clear all computation caches from session state."""
    prefixes = [
        "cache_sae_", "cache_svd_", "cache_stability_",
        "cache_df_", "cache_fig_", "cache_narr_",
    ]
    for prefix in prefixes:
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith(prefix)]
        for k in keys_to_remove:
            st.session_state.pop(k, None)
    st.toast("✓ Cleared all caches")


def get_cache_stats() -> dict:
    """Get summary statistics of cached items."""
    stats = {
        "sae": len([k for k in st.session_state.keys() if k.startswith("cache_sae_")]),
        "svd": len([k for k in st.session_state.keys() if k.startswith("cache_svd_")]),
        "stability": len([k for k in st.session_state.keys() if k.startswith("cache_stability_")]),
        "dataframe": len([k for k in st.session_state.keys() if k.startswith("cache_df_")]),
        "figure": len([k for k in st.session_state.keys() if k.startswith("cache_fig_")]),
        "narrative": len([k for k in st.session_state.keys() if k.startswith("cache_narr_")]),
    }
    stats["total"] = sum(stats.values())
    return stats
