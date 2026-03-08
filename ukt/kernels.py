"""SVD kernel decomposition, labeling, and narrative generation.

Kernels are the emergent cross-domain patterns discovered by SVD on the UKT
feature matrix. They are not pre-defined — they emerge automatically from
the structure in the data. Each kernel explains a portion of total variance
and can be traced back to specific feature regions and data sources.

This module is network-agnostic: it operates on the (blocks x features) matrix
regardless of which neural network produced the features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ukt.registry import FeatureRegionRegistry


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
    registry: FeatureRegionRegistry,
    feature_meta: dict[int, dict] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Describe the top-N loaded features in a kernel (Vt row).

    Args:
        vt_row: (feature_dim,) — one row of Vt (feature loadings for a kernel).
        registry: Feature region registry for name/region lookup.
        feature_meta: Optional per-feature metadata from the data pipeline.
        top_n: How many top features to return.

    Returns:
        List of dicts with index, name, region, loading, and metadata.
    """
    indices = np.argsort(np.abs(vt_row))[-top_n:][::-1]
    descriptions = []
    for idx in indices:
        idx = int(idx)
        meta = feature_meta.get(idx, {}) if feature_meta else {}
        region = registry.region_for_index(idx)
        region_name = region.name if region else "unknown"

        # Prefer metadata label, fall back to registry name
        name = meta.get("label") or registry.feature_name(idx)

        descriptions.append(dict(
            index=idx,
            name=name,
            region=region_name,
            loading=float(vt_row[idx]),
            abs_loading=float(abs(vt_row[idx])),
            entity=meta.get("entity"),
            metric=meta.get("metric"),
            source=meta.get("source"),
            time_scope=meta.get("time_scope"),
        ))
    return descriptions


def compute_region_scores(
    vt_row: np.ndarray,
    registry: FeatureRegionRegistry,
) -> dict[str, float]:
    """Compute per-region absolute loading scores for a kernel.

    Args:
        vt_row: (feature_dim,) — feature loadings for one kernel.
        registry: Feature region registry.

    Returns:
        {region_name: total_absolute_loading}
    """
    scores = {}
    for name, region in registry.regions.items():
        scores[name] = float(np.abs(vt_row[region.start:region.end]).sum())
    return scores


def label_kernel(
    k_idx: int,
    decomposition: KernelDecomposition,
    registry: FeatureRegionRegistry,
    block_names: list[str],
    feature_meta: dict[int, dict] | None = None,
    timeframe_context: dict | None = None,
) -> dict:
    """Generate a semantic label for a single kernel.

    Args:
        k_idx: Kernel index.
        decomposition: SVD decomposition result.
        registry: Feature region registry.
        block_names: Names of blocks (rows) in the UKT matrix.
        feature_meta: Optional per-feature metadata.
        timeframe_context: Optional temporal context for narrative.

    Returns:
        Dict with kernel_id, label, narrative, importance, region_scores, etc.
    """
    vt_row = decomposition.Vt[k_idx]
    u_col = decomposition.U[:, k_idx]
    importance = float(decomposition.importance[k_idx])

    region_scores = compute_region_scores(vt_row, registry)
    dominant_region = max(region_scores, key=region_scores.get)

    dominant_block_idx = int(np.argmax(np.abs(u_col)))
    dominant_block = (block_names[dominant_block_idx]
                      if dominant_block_idx < len(block_names) else "unknown")

    top_features = describe_top_features(vt_row, registry, feature_meta, top_n=5)
    top_feat_name = top_features[0]["name"] if top_features else "?"

    short_label = (
        f"K{k_idx}: {dominant_block} — {dominant_region.replace('-', ' ')} "
        f"({importance:.1%} var, lead: {top_feat_name})"
    )

    narrative = generate_kernel_narrative(
        k_idx, dominant_block, dominant_region, importance,
        top_features, block_names, u_col, registry, timeframe_context,
    )

    return dict(
        kernel_id=f"K{k_idx}",
        dominant_block=dominant_block,
        dominant_region=dominant_region,
        importance=importance,
        top_features=top_features,
        top_feature_indices=[f["index"] for f in top_features],
        label=short_label,
        narrative=narrative,
        region_scores={k: round(v, 4) for k, v in region_scores.items()},
    )


def generate_kernel_narrative(
    k_idx: int,
    dominant_block: str,
    dominant_region: str,
    importance: float,
    top_features: list[dict],
    block_names: list[str],
    u_col: np.ndarray,
    registry: FeatureRegionRegistry,
    timeframe_context: dict | None = None,
) -> str:
    """Generate a data-grounded narrative for a kernel.

    This is the algorithmic narrative — purely data-driven, no LLM required.
    """
    block_contributions = []
    for i, bn in enumerate(block_names):
        if i < len(u_col):
            contrib = abs(float(u_col[i]))
            if contrib > 0.1:
                block_contributions.append(f"{bn} ({contrib:.2f})")

    region = registry.regions.get(dominant_region)
    region_desc = region.description if region else dominant_region
    feature_strs = [f"{f['name']} ({f['loading']:+.3f})" for f in top_features[:3]]

    # Group features by region for cross-region coupling analysis
    region_groups: dict[str, list[dict]] = {}
    for f in top_features:
        region_groups.setdefault(f["region"], []).append(f)

    evidence_lines = []
    for rname, feats in region_groups.items():
        if rname != dominant_region:
            evidence_lines.append(
                f"Cross-region coupling with {rname}: " + ", ".join(
                    f"{f['name']}={f['loading']:+.3f}" for f in feats[:2]
                )
            )

    # Check for multi-region coupling (the most interesting finding)
    if len(region_groups) >= 2:
        regions_involved = list(region_groups.keys())
        evidence_lines.append(
            f"Emergent cross-domain pattern: {regions_involved[0]} <-> "
            f"{regions_involved[1]} coupling detected in kernel structure."
        )

    timeframe_line = ""
    if timeframe_context:
        start = timeframe_context.get("start_date")
        end = timeframe_context.get("end_date")
        if start and end:
            timeframe_line = f"Time alignment window: {start} to {end}."

    lines = [
        f"Kernel K{k_idx} explains {importance:.1%} of total variance.",
        f"Primary driver: {dominant_block} block; dominant region: {dominant_region.replace('-', ' ')}.",
        f"Top evidence features: {', '.join(feature_strs)}.",
        f"Block contributions: {', '.join(block_contributions) if block_contributions else 'weak/mixed'}.",
        f"Region context: {region_desc}",
    ]
    if timeframe_line:
        lines.append(timeframe_line)
    lines.extend(evidence_lines)
    return "\n".join(lines)
