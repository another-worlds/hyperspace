"""Universal Knowledge Tensor (UKT): incremental SVD, reality regression, semantic interpretability.

The UKT grows after each pipeline block. At every step it:
  1. Accepts a feature vector from the block
  2. Normalizes it to [0, 1] per-region for cross-block comparability
  3. Appends it as a new row (blocks-so-far x feature_dim)
  4. Decomposes via SVD to get current kernel structure
  5. Computes "universal reality regression" (weighted kernel basis)
  6. Generates rich, human-readable semantic labels for active kernels
"""
from __future__ import annotations

import numpy as np

from hyperspace.config import (
    FEATURE_NAMES,
    REGION_DESCRIPTIONS,
    UKT_FEATURE_DIM,
)


def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate a 1D array to target length."""
    arr = arr.flatten()
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)))


# Feature provenance labels: indices 0-15 = temporal, 16-31 = embedding,
# 32-47 = structural, 48-63 = agent/dynamic
FEATURE_REGION_LABELS: dict[tuple[int, int], str] = {
    (0, 16): "temporal-pattern",
    (16, 32): "semantic-embedding",
    (32, 48): "structural-centrality",
    (48, 64): "dynamic-agent",
}


def _normalize_features(arr: np.ndarray) -> np.ndarray:
    """Normalize each region of a feature vector to [0, 1] range.

    This ensures cross-block comparability so that SVD kernels are not
    dominated by whichever block happens to have the largest raw values.
    """
    out = arr.copy()
    for (lo, hi) in FEATURE_REGION_LABELS:
        region = out[lo:hi]
        rng = region.max() - region.min()
        if rng > 1e-8:
            out[lo:hi] = (region - region.min()) / rng
    return out


def _feature_name(idx: int) -> str:
    """Return the human-readable name for a feature index."""
    if idx < len(FEATURE_NAMES):
        return FEATURE_NAMES[idx]
    return f"feature_{idx}"


def _region_for_index(idx: int) -> str:
    """Return the region label for a feature index."""
    for (lo, hi), label in FEATURE_REGION_LABELS.items():
        if lo <= idx < hi:
            return label
    return "unknown"


def _describe_top_features(vt_row: np.ndarray, top_n: int = 5) -> list[dict]:
    """Return human-readable descriptions for the top-N loaded features."""
    indices = np.argsort(np.abs(vt_row))[-top_n:][::-1]
    descriptions = []
    for idx in indices:
        descriptions.append(dict(
            index=int(idx),
            name=_feature_name(idx),
            region=_region_for_index(idx),
            loading=float(vt_row[idx]),
            abs_loading=float(abs(vt_row[idx])),
        ))
    return descriptions


def _generate_kernel_narrative(
    k_idx: int,
    dominant_block: str,
    dominant_region: str,
    importance: float,
    top_features: list[dict],
    block_names: list[str],
    u_col: np.ndarray,
) -> str:
    """Generate a rich, human-readable narrative for a kernel."""
    # Block contribution breakdown
    block_contributions = []
    for i, bn in enumerate(block_names):
        contrib = abs(float(u_col[i]))
        if contrib > 0.1:
            block_contributions.append(f"{bn} ({contrib:.2f})")

    # Build narrative
    region_desc = REGION_DESCRIPTIONS.get(dominant_region, dominant_region)
    feature_strs = [
        f"'{f['name']}' ({f['loading']:+.3f})" for f in top_features[:3]
    ]

    lines = [
        f"Kernel K{k_idx} explains {importance:.1%} of total variance.",
        f"It is primarily driven by the {dominant_block} block and captures "
        f"{dominant_region.replace('-', ' ')} dynamics.",
        f"",
        f"Top feature loadings: {', '.join(feature_strs)}.",
        f"",
        f"Block contributions: {', '.join(block_contributions)}.",
        f"",
        f"Region context: {region_desc}",
    ]
    return "\n".join(lines)


def _label_kernel(k_idx: int, vt_row: np.ndarray, u_col: np.ndarray,
                  block_names: list[str], importance: float) -> dict:
    """Generate a rich semantic label for a single kernel."""
    # Find which feature region dominates
    region_scores = {}
    for (lo, hi), label in FEATURE_REGION_LABELS.items():
        region_scores[label] = float(np.abs(vt_row[lo:hi]).sum())
    dominant_region = max(region_scores, key=region_scores.get)

    # Find which block dominates
    dominant_block_idx = int(np.argmax(np.abs(u_col)))
    dominant_block = (block_names[dominant_block_idx]
                      if dominant_block_idx < len(block_names) else "unknown")

    # Top 5 features with human-readable descriptions
    top_features = _describe_top_features(vt_row, top_n=5)

    # Short label
    top_feat_name = top_features[0]["name"] if top_features else "?"
    short_label = (
        f"K{k_idx}: {dominant_block} — {dominant_region.replace('-', ' ')} "
        f"({importance:.1%} var, lead: {top_feat_name})"
    )

    # Full narrative
    narrative = _generate_kernel_narrative(
        k_idx, dominant_block, dominant_region, importance,
        top_features, block_names, u_col,
    )

    return dict(
        kernel_id=f"K{k_idx}",
        dominant_block=dominant_block,
        dominant_region=dominant_region,
        importance=float(importance),
        top_features=top_features,
        top_feature_indices=[f["index"] for f in top_features],
        label=short_label,
        narrative=narrative,
        region_scores={k: round(v, 4) for k, v in region_scores.items()},
    )


class UniversalKnowledgeTensor:
    """Incrementally built cross-block knowledge tensor with SVD decomposition."""

    def __init__(self, feature_dim: int = UKT_FEATURE_DIM):
        self.feature_dim = feature_dim
        self.block_names: list[str] = []
        self.rows: list[np.ndarray] = []
        self.snapshots: list[dict] = []

    def add_block(self, name: str, features: np.ndarray) -> dict:
        """Add a block's feature vector, normalize, decompose, interpret.

        Args:
            name: Human-readable block name (e.g. "Finance", "Clusters").
            features: Raw feature array from the block (any shape, will be flattened).

        Returns:
            Snapshot dict with SVD results, kernel labels, reality regression.
        """
        self.block_names.append(name)
        raw = _pad_or_truncate(features, self.feature_dim)
        normalized = _normalize_features(raw)
        self.rows.append(normalized)

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

        # ---- Rich interpretability report ----
        report_lines = [
            f"=== Step {len(self.rows)}: Added '{name}' block ===",
            f"Active kernels: {n_kernels}",
            f"Reconstruction error: {recon_error:.6f}",
            f"Reality regression norm: {np.linalg.norm(reality_regression):.4f}",
            "",
        ]
        for kl in kernel_labels:
            report_lines.append(f"--- {kl['label']} ---")
            report_lines.append(kl["narrative"])
            report_lines.append("")

        # Cross-block coherence summary
        if len(self.rows) > 1:
            report_lines.append("--- Cross-Block Coherence ---")
            for k in range(n_kernels):
                contribs = []
                for i, bn in enumerate(self.block_names):
                    val = float(kernel_activation[i, k])
                    if abs(val) > 0.01:
                        contribs.append(f"{bn}={val:+.3f}")
                report_lines.append(
                    f"K{k} activation across blocks: {', '.join(contribs)}"
                )
            report_lines.append("")

        # Reality regression highlights
        rr_top = np.argsort(np.abs(reality_regression))[-5:][::-1]
        report_lines.append("--- Reality Regression (top features) ---")
        for idx in rr_top:
            report_lines.append(
                f"  {_feature_name(idx)} [{_region_for_index(idx)}]: "
                f"{reality_regression[idx]:+.4f}"
            )

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
