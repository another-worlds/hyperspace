"""Universal Knowledge Tensor: the core incremental feature matrix.

The UKT grows row-by-row as each data source / neural network layer / pipeline
block contributes its features. At every step, it normalizes, decomposes via
SVD, and produces interpretable kernel labels.

This module is the network-agnostic heart of the framework. It does not know
or care whether features came from a transformer, CNN, GNN, or hand-crafted
pipeline. It only requires:
  1. A FeatureRegionRegistry defining the feature space
  2. Feature vectors (numpy arrays) from each source

The UKT treats neural networks as what they are: sophisticated feature
generators. It standardizes and interprets whatever features they produce.
"""
from __future__ import annotations

import numpy as np

from ukt.registry import FeatureRegionRegistry
from ukt.kernels import (
    KernelDecomposition,
    decompose_svd,
    label_kernel,
    describe_top_features,
)
from ukt.utils import _pad_or_truncate


def _normalize_features(
    arr: np.ndarray,
    registry: FeatureRegionRegistry,
) -> np.ndarray:
    """Normalize each region of a feature vector to [0, 1] range.

    Per-region normalization ensures cross-region comparability regardless
    of the scale of each network's raw activations.
    """
    out = arr.copy()
    for name, region in registry.regions.items():
        lo, hi = region.start, region.end
        segment = out[lo:hi]
        rng = segment.max() - segment.min()
        if rng > 1e-8:
            out[lo:hi] = (segment - segment.min()) / rng
    return out


class UniversalKnowledgeTensor:
    """Incrementally built cross-source feature tensor with SVD decomposition.

    The UKT is network-agnostic. It accepts normalized feature vectors from
    any source, decomposes them into kernels, and produces interpretable labels.

    Args:
        registry: Feature region registry defining the feature space.
                  If None, a default registry with a single region is created.

    Example:
        registry = FeatureRegionRegistry()
        registry.register("layer_1", 0, 64, "First hidden layer")
        registry.register("layer_2", 64, 128, "Second hidden layer")

        ukt = UniversalKnowledgeTensor(registry)
        snapshot = ukt.add_block("encoder", encoder_features)
        snapshot = ukt.add_block("decoder", decoder_features)
        print(snapshot["kernel_labels"])
    """

    def __init__(self, registry: FeatureRegionRegistry | None = None) -> None:
        if registry is None:
            registry = FeatureRegionRegistry()
            registry.register("default", 0, 80, "Default feature region")
        self.registry = registry
        self.feature_dim = registry.total_dim
        self.block_names: list[str] = []
        self.rows: list[np.ndarray] = []
        self.snapshots: list[dict] = []
        self.global_feature_meta: dict[int, dict] = {}

    def add_block(
        self,
        name: str,
        features: np.ndarray,
        feature_meta: dict[int, dict] | None = None,
        timeframe_context: dict | None = None,
    ) -> dict:
        """Add a source's feature vector, normalize, decompose, interpret.

        Args:
            name: Human-readable name for this source/block/layer.
            features: Raw feature vector from the source.
            feature_meta: Optional per-feature metadata (label, entity, source, etc.).
            timeframe_context: Optional temporal context (start_date, end_date).

        Returns:
            Snapshot dict with decomposition results, kernel labels, and report.
        """
        self.block_names.append(name)
        if feature_meta:
            self.global_feature_meta.update(feature_meta)

        raw = _pad_or_truncate(features, self.feature_dim)
        normalized = _normalize_features(raw, self.registry)
        self.rows.append(normalized)

        matrix = np.stack(self.rows)
        decomposition = decompose_svd(matrix)

        # Label each kernel
        kernel_labels = []
        for k in range(decomposition.n_kernels):
            kl = label_kernel(
                k, decomposition, self.registry, self.block_names,
                self.global_feature_meta, timeframe_context,
            )
            kernel_labels.append(kl)

        # Build report
        report = self._build_report(
            name, decomposition, kernel_labels, matrix,
        )

        snapshot = dict(
            step=len(self.rows),
            block_name=name,
            matrix=matrix.copy(),
            U=decomposition.U.copy(),
            S=decomposition.S.copy(),
            Vt=decomposition.Vt.copy(),
            n_kernels=decomposition.n_kernels,
            importance=decomposition.importance.copy(),
            kernel_activation=decomposition.kernel_activation.copy(),
            reality_regression=decomposition.reality_regression.copy(),
            kernel_labels=kernel_labels,
            reconstruction_error=decomposition.reconstruction_error,
            report=report,
            feature_meta=self.global_feature_meta.copy(),
            timeframe_context=timeframe_context or {},
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_final_matrix(self) -> np.ndarray | None:
        """Return the full (n_blocks, feature_dim) matrix, or None if empty."""
        if not self.rows:
            return None
        return np.stack(self.rows)

    def get_latest_snapshot(self) -> dict | None:
        """Return the most recent snapshot, or None."""
        return self.snapshots[-1] if self.snapshots else None

    def _build_report(
        self,
        name: str,
        decomposition: KernelDecomposition,
        kernel_labels: list[dict],
        matrix: np.ndarray,
    ) -> str:
        """Build a human-readable interpretability report."""
        lines = [
            f"=== Step {len(self.rows)}: Added '{name}' block ===",
            f"Active kernels: {decomposition.n_kernels}",
            f"Reconstruction error: {decomposition.reconstruction_error:.6f}",
            f"Reality regression norm: {np.linalg.norm(decomposition.reality_regression):.4f}",
            "",
        ]

        for kl in kernel_labels:
            lines.append(f"--- {kl['label']} ---")
            lines.append(kl["narrative"])
            lines.append("")

        if len(self.rows) > 1:
            lines.append("--- Cross-Block Coherence ---")
            for k in range(decomposition.n_kernels):
                contribs = []
                for i, bn in enumerate(self.block_names):
                    val = float(decomposition.kernel_activation[i, k])
                    if abs(val) > 0.01:
                        contribs.append(f"{bn}={val:+.3f}")
                lines.append(f"K{k} activation across blocks: {', '.join(contribs)}")
            lines.append("")

        rr = decomposition.reality_regression
        rr_top = np.argsort(np.abs(rr))[-5:][::-1]
        lines.append("--- Reality Regression (top features) ---")
        for idx in rr_top:
            fname = self.registry.feature_name(int(idx))
            region = self.registry.region_for_index(int(idx))
            rname = region.name if region else "unknown"
            lines.append(f"  {fname} [{rname}]: {rr[int(idx)]:+.4f}")

        return "\n".join(lines)
