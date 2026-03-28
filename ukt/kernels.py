"""SVD kernel decomposition, labeling, and narrative generation.

Kernels are the emergent cross-block patterns discovered by SVD on the UKT
feature matrix. They are not pre-defined — they emerge automatically from
the structure in the data. Each kernel explains a portion of total variance
and can be traced back to specific feature blocks and data sources.

This module is network-agnostic: it operates on the (blocks x features) matrix
regardless of which neural network produced the features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ukt.registry import FeatureRegionRegistry


def _resolve_ranges(
    block_feature_ranges: FeatureRegionRegistry | dict[str, tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    """Convert a FeatureRegionRegistry to a plain dict if needed."""
    if isinstance(block_feature_ranges, FeatureRegionRegistry):
        return block_feature_ranges.region_bounds()
    return block_feature_ranges


@dataclass
class KernelDecomposition:
    """Result of SVD decomposition on the UKT matrix."""
    U: np.ndarray             # (n_blocks, n_kernels) — block loadings
    S: np.ndarray             # (n_kernels,) — singular values
    Vt: np.ndarray            # (n_kernels, feature_dim) — feature loadings
    n_kernels: int
    importance: np.ndarray    # (n_kernels,) — normalized singular values
    kernel_activation: np.ndarray  # (n_blocks, n_kernels) — U * S
    reality_regression: np.ndarray  # (feature_dim,) — weighted kernel basis
    reconstruction_error: float


def decompose_svd(matrix: np.ndarray) -> KernelDecomposition:
    """Decompose a UKT feature matrix via SVD.

    Args:
        matrix: (n_blocks, feature_dim) feature matrix.

    Returns:
        KernelDecomposition with all SVD components and derived quantities.
    """
    try:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        n = matrix.shape[0]
        U = np.eye(n)
        S = np.ones(n) * 0.01
        Vt = np.zeros((n, matrix.shape[1]))
        for i in range(min(n, matrix.shape[1])):
            Vt[i, i] = 1.0

    n_kernels = len(S)
    total = S.sum() + 1e-8
    importance = S / total
    kernel_activation = U * S[np.newaxis, :]
    reality_regression = importance @ Vt[:n_kernels, :]

    recon = U[:, :n_kernels] @ np.diag(S[:n_kernels]) @ Vt[:n_kernels, :]
    recon_error = float(np.linalg.norm(matrix - recon))

    return KernelDecomposition(
        U=U, S=S, Vt=Vt,
        n_kernels=n_kernels,
        importance=importance,
        kernel_activation=kernel_activation,
        reality_regression=reality_regression,
        reconstruction_error=recon_error,
    )


def describe_top_features(
    vt_row: np.ndarray,
    block_feature_ranges: FeatureRegionRegistry | dict[str, tuple[int, int]],
    feature_names: list[str],
    feature_meta: dict[int, dict] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Describe the top-N loaded features in a kernel (Vt row).

    Args:
        vt_row: (feature_dim,) — one row of Vt (feature loadings for a kernel).
        block_feature_ranges: Mapping of block name to (start, end) feature indices,
            or a FeatureRegionRegistry.
        feature_names: List of feature names by index.
        feature_meta: Optional per-feature metadata from the data pipeline.
        top_n: How many top features to return.

    Returns:
        List of dicts with index, name, source_block, loading, and metadata.
    """
    ranges = _resolve_ranges(block_feature_ranges)
    indices = np.argsort(np.abs(vt_row))[-top_n:][::-1]
    descriptions = []

    # Build reverse mapping: index -> block_name
    idx_to_block = {}
    for block_name, (start, end) in ranges.items():
        for idx in range(start, end):
            idx_to_block[idx] = block_name

    for idx in indices:
        idx = int(idx)
        meta = feature_meta.get(idx, {}) if feature_meta else {}
        source_block = idx_to_block.get(idx, "unknown")

        # Use feature_names list for feature name
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"

        descriptions.append(dict(
            index=idx,
            name=name,
            source_block=source_block,
            loading=float(vt_row[idx]),
            abs_loading=float(abs(vt_row[idx])),
            entity=meta.get("entity"),
            metric=meta.get("metric"),
            source=meta.get("source"),
            time_scope=meta.get("time_scope"),
        ))
    return descriptions


def compute_block_scores(
    vt_row: np.ndarray,
    block_feature_ranges: FeatureRegionRegistry | dict[str, tuple[int, int]],
) -> dict[str, float]:
    """Compute per-block absolute loading scores for a kernel.

    Args:
        vt_row: (feature_dim,) — feature loadings for one kernel.
        block_feature_ranges: Mapping of block name to (start, end) feature indices,
            or a FeatureRegionRegistry.

    Returns:
        {block_name: total_absolute_loading}
    """
    ranges = _resolve_ranges(block_feature_ranges)
    scores = {}
    for block_name, (start, end) in ranges.items():
        scores[block_name] = float(np.abs(vt_row[start:end]).sum())
    return scores


def label_kernel(
    k_idx: int,
    decomposition: KernelDecomposition,
    block_feature_ranges: FeatureRegionRegistry | dict[str, tuple[int, int]],
    feature_names: list[str],
    block_names: list[str],
    feature_meta: dict[int, dict] | None = None,
    timeframe_context: dict | None = None,
) -> dict:
    """Generate a semantic label for a single kernel.

    In a shared projection space, kernels genuinely span multiple blocks.
    The label reflects this: when two or more blocks each contribute >15%
    of total loading, the label names the cross-block coupling pattern.

    Args:
        k_idx: Kernel index.
        decomposition: SVD decomposition result.
        block_feature_ranges: Mapping of block name to (start, end) feature indices.
        feature_names: List of feature names by index.
        block_names: Names of blocks (rows) in the UKT matrix.
        feature_meta: Optional per-feature metadata.
        timeframe_context: Optional temporal context for narrative.

    Returns:
        Dict with kernel_id, label, narrative, importance, block_scores, etc.
    """
    vt_row = decomposition.Vt[k_idx]
    u_col = decomposition.U[:, k_idx]
    importance = float(decomposition.importance[k_idx])

    block_scores = compute_block_scores(vt_row, block_feature_ranges)
    total_score = sum(block_scores.values()) + 1e-8

    sorted_blocks = sorted(block_scores.items(), key=lambda x: -x[1])
    dominant_block = sorted_blocks[0][0] if sorted_blocks else "unknown"

    # Identify contributing blocks (>15% of total loading)
    contributing_blocks = [
        (name, score / total_score)
        for name, score in sorted_blocks
        if score / total_score > 0.15
    ]

    # Block contributions from U column
    block_contribs = []
    for i, bn in enumerate(block_names):
        if i < len(u_col) and abs(float(u_col[i])) > 0.1:
            block_contribs.append((bn, abs(float(u_col[i]))))
    block_contribs.sort(key=lambda x: -x[1])

    dominant_block_idx = int(np.argmax(np.abs(u_col)))
    dominant_block_from_u = (block_names[dominant_block_idx]
                      if dominant_block_idx < len(block_names) else "unknown")

    top_features = describe_top_features(vt_row, block_feature_ranges, feature_names, feature_meta, top_n=5)
    top_feat_name = top_features[0]["name"] if top_features else "?"

    # Build the label: show cross-block coupling when present
    _short = lambda name: name.replace("-", " ").replace("_", " ").split()[0]
    if len(contributing_blocks) >= 3:
        block_tag_label = " × ".join(_short(b) for b, _ in contributing_blocks[:3])
    elif len(contributing_blocks) == 2:
        block_tag_label = f"{_short(contributing_blocks[0][0])} × {_short(contributing_blocks[1][0])}"
    else:
        block_tag_label = dominant_block.replace("-", " ").replace("_", " ")

    block_tag = "+".join(bn for bn, _ in block_contribs[:2]) if block_contribs else dominant_block_from_u
    short_label = f"K{k_idx}: {block_tag} — {block_tag_label} ({importance:.1%} var, lead: {top_feat_name})"

    narrative = generate_kernel_narrative(
        k_idx, dominant_block_from_u, dominant_block, importance,
        top_features, block_names, u_col, block_feature_ranges, timeframe_context,
    )

    return dict(
        kernel_id=f"K{k_idx}",
        dominant_block=dominant_block_from_u,
        dominant_feature_block=dominant_block,
        contributing_blocks=contributing_blocks,
        block_contributions=[(bn, round(v, 4)) for bn, v in block_contribs],
        importance=importance,
        top_features=top_features,
        top_feature_indices=[f["index"] for f in top_features],
        label=short_label,
        narrative=narrative,
        block_scores={k: round(v, 4) for k, v in block_scores.items()},
    )


def generate_kernel_narrative(
    k_idx: int,
    dominant_block: str,
    dominant_feature_block: str,
    importance: float,
    top_features: list[dict],
    block_names: list[str],
    u_col: np.ndarray,
    block_feature_ranges: FeatureRegionRegistry | dict[str, tuple[int, int]],
    timeframe_context: dict | None = None,
) -> str:
    """Generate a data-grounded narrative for a kernel.

    Narrative structure adapts to kernel complexity:
    - Single-block kernels: brief, factual — one block dominates.
    - Two-block coupling: highlight the cross-block link and what it means.
    - Three+ blocks: emphasize the emergent multi-block pattern.
    """
    # Block contributions from U column
    block_contributions = []
    for i, bn in enumerate(block_names):
        if i < len(u_col):
            contrib = abs(float(u_col[i]))
            if contrib > 0.1:
                block_contributions.append((bn, contrib))
    block_contributions.sort(key=lambda x: -x[1])
    block_strs = [f"{bn} ({c:.2f})" for bn, c in block_contributions]

    # Feature evidence grouped by source_block
    block_groups: dict[str, list[dict]] = {}
    for f in top_features:
        block_groups.setdefault(f["source_block"], []).append(f)

    n_blocks = len(block_groups)
    feature_strs = [f"{f['name']} ({f['loading']:+.3f})" for f in top_features[:3]]

    timeframe_line = ""
    if timeframe_context:
        start = timeframe_context.get("start_date")
        end = timeframe_context.get("end_date")
        if start and end:
            timeframe_line = f"Time alignment window: {start} to {end}."

    lines = [f"Kernel K{k_idx} explains {importance:.1%} of total variance."]

    if n_blocks >= 3:
        # Multi-block emergent pattern — the most interesting case
        block_names_list = list(block_groups.keys())
        lines.append(
            f"Emergent cross-block pattern spanning {n_blocks} blocks: "
            f"{', '.join(b.replace('-', ' ').replace('_', ' ') for b in block_names_list)}."
        )
        lines.append(
            f"Contributing blocks: {', '.join(block_strs) if block_strs else 'mixed'}."
        )
        lines.append(f"Key evidence features: {', '.join(feature_strs)}.")
        # Detail each block's contribution
        for bname, feats in block_groups.items():
            feat_detail = ", ".join(f"{f['name']}={f['loading']:+.3f}" for f in feats[:2])
            lines.append(
                f"  {bname.replace('-', ' ').replace('_', ' ')}: {feat_detail}"
            )
    elif n_blocks == 2:
        # Two-block coupling — highlight the cross-block link
        b1, b2 = list(block_groups.keys())
        lines.append(
            f"Cross-block coupling between {b1.replace('-', ' ').replace('_', ' ')} and "
            f"{b2.replace('-', ' ').replace('_', ' ')}."
        )
        lines.append(
            f"Primary driver: {block_contributions[0][0] if block_contributions else dominant_block} block."
        )
        lines.append(f"Evidence features: {', '.join(feature_strs)}.")
        f1 = block_groups[b1][0]
        f2 = block_groups[b2][0]
        lines.append(
            f"Coupling signature: {f1['name']} ({f1['loading']:+.3f}) "
            f"co-varies with {f2['name']} ({f2['loading']:+.3f})."
        )
    else:
        # Single-block dominant — brief and factual
        lines.append(
            f"Single-block pattern: {dominant_block} block, "
            f"{dominant_feature_block.replace('-', ' ').replace('_', ' ')} feature block."
        )
        lines.append(f"Top features: {', '.join(feature_strs)}.")

    if block_contributions and n_blocks < 3:
        lines.append(f"Block contributions: {', '.join(block_strs)}.")

    if timeframe_line:
        lines.append(timeframe_line)

    return "\n".join(lines)
