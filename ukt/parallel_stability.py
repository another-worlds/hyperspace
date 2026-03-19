"""Parallel stability estimation for reality regression robustness.

Parallelizes bootstrap noise runs to measure the stability of discovered
kernel structure under perturbations. Expected speedup: 4-6× for 8+ runs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


def estimate_regression_stability_parallel(
    matrix: np.ndarray,
    n_runs: int = 8,
    noise_std: float = 0.01,
    seed: int | None = 42,
    max_workers: int = 4,
) -> dict:
    """Estimate stability of reality regression in parallel.

    Adds small Gaussian noise to the UKT matrix in parallel and checks if the
    leading reality-regression direction remains stable (high cosine similarity).

    Args:
        matrix: (n_blocks, feature_dim) UKT feature matrix.
        n_runs: Number of noisy perturbation runs.
        noise_std: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility (None for stochastic).
        max_workers: Number of parallel workers.

    Returns:
        Dict with n_runs, mean_cosine, min_cosine, std_cosine.
        High mean_cosine (>= 0.75) indicates stable regression.
    """
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {"n_runs": 0, "mean_cosine": 0.0, "min_cosine": 0.0, "std_cosine": 0.0}

    regressions = [None] * n_runs

    def compute_noisy_regression(run_idx: int) -> tuple[int, np.ndarray]:
        """Compute one noisy regression and return (run_idx, result)."""
        if seed is not None:
            rng = np.random.default_rng(seed + run_idx)
        else:
            rng = np.random.default_rng()

        noisy = matrix + rng.normal(0.0, noise_std, size=matrix.shape)
        try:
            U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback on degenerate matrix
            return run_idx, np.zeros(matrix.shape[1])

        importance = S / (S.sum() + 1e-8)
        rr = importance @ Vt[:len(S), :]
        rr = rr / (np.linalg.norm(rr) + 1e-8)
        return run_idx, rr

    # Execute noise perturbations in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_noisy_regression, i): i
            for i in range(n_runs)
        }

        for future in as_completed(futures):
            try:
                run_idx, rr = future.result()
                regressions[run_idx] = rr
            except Exception:
                regressions[futures[future]] = np.zeros(matrix.shape[1])

    # Compute cosine similarities between all pairs
    cosines = []
    for i in range(len(regressions)):
        if regressions[i] is None:
            continue
        for j in range(i + 1, len(regressions)):
            if regressions[j] is None:
                continue
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


def estimate_joint_stability_parallel(
    matrix: np.ndarray,
    kernels_baseline: np.ndarray,
    n_runs: int = 8,
    noise_std: float = 0.01,
    seed: int | None = 42,
    max_workers: int = 4,
) -> dict:
    """Estimate stability of kernel orientation against baseline.

    Measures how much each noisy perturbation changes the relative importance
    and orientation of kernels.

    Args:
        matrix: (n_blocks, feature_dim) UKT feature matrix.
        kernels_baseline: Baseline kernel orientations (V^T from SVD).
        n_runs: Number of perturbation runs.
        noise_std: Noise standard deviation.
        seed: Random seed (None for stochastic).
        max_workers: Parallel workers.

    Returns:
        Dict with per-kernel stability estimates.
    """
    results = []

    def compute_kernel_stability(run_idx: int) -> tuple[int, np.ndarray | None]:
        """Compute kernel orientations for one noisy sample."""
        if seed is not None:
            rng = np.random.default_rng(seed + run_idx)
        else:
            rng = np.random.default_rng()

        noisy = matrix + rng.normal(0.0, noise_std, size=matrix.shape)
        try:
            U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
        except np.linalg.LinAlgError:
            return run_idx, None

        return run_idx, Vt

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_kernel_stability, i): i
            for i in range(n_runs)
        }

        for future in as_completed(futures):
            try:
                run_idx, vt = future.result()
                if vt is not None:
                    results.append(vt)
            except Exception:
                pass

    if not results:
        return {"n_runs": 0, "kernel_stabilities": []}

    # Compute per-kernel stability (cosine similarity to baseline)
    n_kernels = min(kernels_baseline.shape[0], min(v.shape[0] for v in results))
    kernel_sims = {i: [] for i in range(n_kernels)}

    for run_vt in results:
        for k_idx in range(n_kernels):
            baseline_k = kernels_baseline[k_idx, :]
            run_k = run_vt[k_idx, :]
            cosine = float(np.dot(baseline_k, run_k) / (np.linalg.norm(baseline_k) * np.linalg.norm(run_k) + 1e-8))
            kernel_sims[k_idx].append(cosine)

    # Summarize per kernel
    kernel_stabilities = []
    for k_idx in range(n_kernels):
        sims = np.array(kernel_sims[k_idx])
        kernel_stabilities.append({
            "kernel_idx": k_idx,
            "mean_cosine": float(sims.mean()),
            "min_cosine": float(sims.min()),
            "std_cosine": float(sims.std()),
        })

    return {
        "n_runs": len(results),
        "kernel_stabilities": kernel_stabilities,
    }
