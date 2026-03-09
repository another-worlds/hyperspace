"""Stability analysis for UKT reality regression.

Tests whether the discovered kernel structure is robust to small perturbations
in the input data. A stable reality regression means conclusions are not
artifacts of noise — they reflect genuine structure.
"""
from __future__ import annotations

import numpy as np


def estimate_regression_stability(
    matrix: np.ndarray,
    n_runs: int = 8,
    noise_std: float = 0.01,
    seed: int = 42,
) -> dict:
    """Estimate cross-run stability of reality regression under perturbations.

    Adds small Gaussian noise to the UKT matrix repeatedly and checks if the
    leading reality-regression direction remains stable (high cosine similarity).

    Args:
        matrix: (n_blocks, feature_dim) UKT feature matrix.
        n_runs: Number of noisy perturbation runs.
        noise_std: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        Dict with n_runs, mean_cosine, min_cosine, std_cosine.
        High mean_cosine (>= 0.75) indicates stable regression.
    """
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {"n_runs": 0, "mean_cosine": 0.0, "min_cosine": 0.0, "std_cosine": 0.0}

    rng = np.random.default_rng(seed)
    regressions = []
    for _ in range(n_runs):
        noisy = matrix + rng.normal(0.0, noise_std, size=matrix.shape)
        U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
        importance = S / (S.sum() + 1e-8)
        rr = importance @ Vt[:len(S), :]
        rr = rr / (np.linalg.norm(rr) + 1e-8)
        regressions.append(rr)

    cosines = []
    for i in range(len(regressions)):
        for j in range(i + 1, len(regressions)):
            cosines.append(float(np.dot(regressions[i], regressions[j])))

    if not cosines:
        return {"n_runs": n_runs, "mean_cosine": 0.0, "min_cosine": 0.0, "std_cosine": 0.0}

    arr = np.array(cosines, dtype=float)
    return {
        "n_runs": n_runs,
        "mean_cosine": float(arr.mean()),
        "min_cosine": float(arr.min()),
        "std_cosine": float(arr.std()),
    }
