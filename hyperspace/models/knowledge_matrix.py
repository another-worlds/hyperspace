"""Universal Knowledge Tensor (UKT): incremental SVD, reality regression, semantic interpretability.

The UKT grows after each pipeline block. At every step it:
  1. Accepts a feature vector from the block
  2. Appends it as a new row (blocks-so-far x feature_dim)
  3. Decomposes via SVD to get current kernel structure
  4. Computes "universal reality regression" (weighted kernel basis)
  5. Generates semantic labels for active kernels
"""
from __future__ import annotations

import numpy as np

from hyperspace.config import UKT_FEATURE_DIM


def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate a 1D array to target length."""
    arr = arr.flatten()
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)))


# Feature provenance labels: indices 0-15 = temporal, 16-31 = embedding,
# 32-47 = structural, 48-63 = agent/dynamic
FEATURE_REGION_LABELS = {
    (0, 16): "temporal-pattern",
    (16, 32): "semantic-embedding",
    (32, 48): "structural-centrality",
    (48, 64): "dynamic-agent",
}


def _label_kernel(k_idx: int, vt_row: np.ndarray, u_col: np.ndarray,
                  block_names: list[str], importance: float) -> dict:
    """Generate a semantic label for a single kernel."""
    # Find which feature region dominates
    region_scores = {}
    for (lo, hi), label in FEATURE_REGION_LABELS.items():
        region_scores[label] = float(np.abs(vt_row[lo:hi]).sum())
    dominant_region = max(region_scores, key=region_scores.get)

    # Find which block dominates
    dominant_block_idx = int(np.argmax(np.abs(u_col)))
    dominant_block = block_names[dominant_block_idx] if dominant_block_idx < len(block_names) else "unknown"

    # Top 5 feature indices
    top_features = np.argsort(np.abs(vt_row))[-5:][::-1].tolist()

    return dict(
        kernel_id=f"K{k_idx}",
        dominant_block=dominant_block,
        dominant_region=dominant_region,
        importance=float(importance),
        top_feature_indices=top_features,
        label=f"K{k_idx}: {dominant_block}/{dominant_region} ({importance:.1%} var)",
    )


class UniversalKnowledgeTensor:
    """Incrementally built cross-block knowledge tensor with SVD decomposition."""

    def __init__(self, feature_dim: int = UKT_FEATURE_DIM):
        self.feature_dim = feature_dim
        self.block_names: list[str] = []
        self.rows: list[np.ndarray] = []
        self.snapshots: list[dict] = []

    def add_block(self, name: str, features: np.ndarray) -> dict:
        """Add a block's feature vector, decompose, interpret.

        Args:
            name: Human-readable block name (e.g. "Finance", "Clusters").
            features: Raw feature array from the block (any shape, will be flattened).

        Returns:
            Snapshot dict with SVD results, kernel labels, reality regression.
        """
        self.block_names.append(name)
        self.rows.append(_pad_or_truncate(features, self.feature_dim))

        # Build current matrix (blocks_so_far x feature_dim)
        matrix = np.stack(self.rows)
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        n_kernels = len(S)

        # Importance: normalized singular values
        total = S.sum() + 1e-8
        importance = S / total

        # Kernel activation matrix (blocks x kernels)
        kernel_activation = U * S[np.newaxis, :]

        # Universal reality regression: importance-weighted combination of
        # right singular vectors = the "best" single vector summarizing all
        # blocks' contributions in feature space
        reality_regression = importance @ Vt[:n_kernels, :]

        # Semantic interpretation for each kernel
        kernel_labels = []
        for k in range(n_kernels):
            label = _label_kernel(
                k, Vt[k], U[:, k], self.block_names, importance[k],
            )
            kernel_labels.append(label)

        # Reconstruction error (should be ~0 when n_kernels == n_rows)
        recon = U[:, :n_kernels] @ np.diag(S[:n_kernels]) @ Vt[:n_kernels, :]
        recon_error = float(np.linalg.norm(matrix - recon))

        # Interpretability report: natural-language summary
        report_lines = [
            f"Step {len(self.rows)}: Added '{name}' block",
            f"  Active kernels: {n_kernels}",
        ]
        for kl in kernel_labels:
            report_lines.append(f"  {kl['label']}")
        report_lines.append(
            f"  Reality regression norm: {np.linalg.norm(reality_regression):.4f}"
        )
        report_lines.append(f"  Reconstruction error: {recon_error:.6f}")

        snapshot = dict(
            step=len(self.rows),
            block_name=name,
            matrix=matrix.copy(),
            U=U.copy(),
            S=S.copy(),
            Vt=Vt.copy(),
            n_kernels=n_kernels,
            importance=importance.copy(),
            kernel_activation=kernel_activation.copy(),
            reality_regression=reality_regression.copy(),
            kernel_labels=kernel_labels,
            reconstruction_error=recon_error,
            report="\n".join(report_lines),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_final_matrix(self) -> np.ndarray | None:
        """Return the current UKT matrix or None if empty."""
        if not self.rows:
            return None
        return np.stack(self.rows)

    def get_latest_snapshot(self) -> dict | None:
        """Return the most recent snapshot or None."""
        return self.snapshots[-1] if self.snapshots else None
