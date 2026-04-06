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

Extension points (constructor keyword-only arguments):
  - ``projection``: Optional :class:`~ukt.projection.SharedProjection` that
    mixes features across blocks before SVD.  When set, all previously added
    blocks are re-projected through the updated coupling matrix each time a new
    block arrives, so SVD discovers genuine cross-block patterns.
  - ``normalizer``: Callable ``(arr, registry) -> arr`` that replaces the
    default per-region [0, 1] normalization.  Useful when wrappers need a
    different scaling strategy (e.g. symmetric [-1, 1]).
  - ``on_snapshot``: Callable ``(snapshot_dict) -> snapshot_dict | None``
    called after every snapshot is assembled.  Wrappers use this hook to
    inject domain-specific enrichments (narratives, provenance fields, etc.)
    without duplicating ``add_block`` logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from ukt.registry import FeatureRegionRegistry
from ukt.kernels import (
    KernelDecomposition,
    decompose_svd,
    label_kernel,
    describe_top_features,
)
from ukt.utils import _pad_or_truncate

if TYPE_CHECKING:
    from ukt.projection import SharedProjection


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

    The UKT is network-agnostic. It accepts feature vectors from any source,
    normalizes them, optionally mixes them via a shared projection, decomposes
    via SVD, and produces interpretable kernel labels.

    Args:
        registry: Feature region registry defining the feature space.
                  If None, a default registry with a single 80-dim region is created.
        projection: Optional :class:`~ukt.projection.SharedProjection` for
                    adaptive cross-block feature mixing through a learned
                    coupling matrix.  When provided, ``add_block`` calls
                    ``projection.observe`` for each new block and re-projects
                    all stored rows through the updated matrix.
        normalizer: Optional callable ``(arr: ndarray, registry) -> ndarray``
                    that replaces the default per-region [0, 1] normalization.
                    Receives the padded raw feature vector and the registry.
        on_snapshot: Optional callable ``(snapshot: dict) -> dict | None``
                     invoked after each snapshot is assembled.  May mutate the
                     snapshot in-place and/or return an enriched copy.  Return
                     value replaces the snapshot when not None.

    Example:
        registry = FeatureRegionRegistry()
        registry.register("layer_1", 0, 64, "First hidden layer")
        registry.register("layer_2", 64, 128, "Second hidden layer")

        ukt = UniversalKnowledgeTensor(registry)
        snapshot = ukt.add_block("encoder", encoder_features)
        snapshot = ukt.add_block("decoder", decoder_features)
        print(snapshot["kernel_labels"])

    Example with cross-block projection:
        from ukt import SharedProjection

        projection = SharedProjection(block_names=["encoder", "decoder"])
        ukt = UniversalKnowledgeTensor(registry, projection=projection)
        ukt.add_block("encoder", encoder_features)
        ukt.add_block("decoder", decoder_features)
    """

    def __init__(
        self,
        registry: FeatureRegionRegistry | None = None,
        *,
        feature_dim: int | None = None,
        projection: "SharedProjection | None" = None,
        normalizer: Callable[[np.ndarray, FeatureRegionRegistry], np.ndarray] | None = None,
        on_snapshot: Callable[[dict], "dict | None"] | None = None,
    ) -> None:
        if registry is None:
            if feature_dim is not None and feature_dim > 0:
                registry = FeatureRegionRegistry()
                registry.register("default", 0, feature_dim, "Legacy fixed-size region")
            else:
                registry = FeatureRegionRegistry()  # Empty — regions self-register as blocks arrive
        self.registry = registry
        self.feature_dim = registry.total_dim  # 0 for empty emergent registry
        self.block_names: list[str] = []
        self.rows: list[np.ndarray] = []
        self.snapshots: list[dict] = []
        self.global_feature_meta: dict[int, dict] = {}
        # Extension points
        self.projection = projection
        self._normalizer = normalizer if normalizer is not None else _normalize_features
        self._on_snapshot = on_snapshot
        # Stores per-block normalized-but-unprojected rows so the projection
        # can re-mix all previous rows whenever a new block updates the coupling.
        self._raw_normalized_rows: list[np.ndarray] = []

    def add_block(
        self,
        name: str,
        features: np.ndarray,
        feature_meta: dict[int, dict] | None = None,
        timeframe_context: dict | None = None,
    ) -> dict:
        """Add a source's feature vector, normalize, (optionally project), decompose, interpret.

        Args:
            name: Human-readable name for this source/block/layer.
            features: Raw feature vector from the source.
            feature_meta: Optional per-feature metadata (label, entity, source, etc.).
            timeframe_context: Optional temporal context (start_date, end_date).

        Returns:
            Snapshot dict with decomposition results, kernel labels, and report.
            When ``on_snapshot`` is set, the returned snapshot is the enriched version.
        """
        self.block_names.append(name)
        if feature_meta:
            self.global_feature_meta.update(feature_meta)

        # Auto-register this block as a new feature region if the registry
        # doesn't know it yet (emergent / regionless mode).
        self._auto_register_block(name, features)

        raw = self._embed_at_region(name, features)
        normalized = self._normalizer(raw, self.registry)

        if self.projection is not None:
            # Store normalized (pre-projection) row so we can re-project all
            # rows through the updated coupling matrix after observe() returns.
            self._raw_normalized_rows.append(normalized)
            self.projection.observe(name, normalized)
            self.rows = [
                self.projection.project(r) for r in self._raw_normalized_rows
            ]
        else:
            self.rows.append(normalized)

        matrix = np.stack(self.rows)
        decomposition = decompose_svd(matrix)

        # Label each kernel
        feature_names = [
            self.registry.feature_name(i)
            for i in range(self.feature_dim)
        ]
        kernel_labels = []
        for k in range(decomposition.n_kernels):
            kl = label_kernel(
                k, decomposition, self.registry, feature_names,
                self.block_names, self.global_feature_meta, timeframe_context,
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
            _registry=self.registry,  # Emergent registry reference for downstream consumers
        )

        # Allow wrappers to inject domain-specific fields without overriding add_block.
        if self._on_snapshot is not None:
            enriched = self._on_snapshot(snapshot)
            if enriched is not None:
                snapshot = enriched

        self.snapshots.append(snapshot)
        return snapshot

    def _auto_register_block(
        self, name: str, features: np.ndarray
    ) -> None:
        """Register *name* as a new feature region if not already present.

        In emergent mode (registry starts empty) each block call allocates the
        next contiguous slice of the feature vector.  All previously stored rows
        are zero-padded on the right so the matrix stays rectangular.
        """
        if name in self.registry:
            return  # Already registered (pre-configured or previously seen)
        feat = np.asarray(features, dtype=float).flatten()
        start = self.feature_dim
        end = start + len(feat)
        self.registry.register(name, start, end)
        new_dim = end
        # Expand all previously accumulated rows
        if new_dim > self.feature_dim:
            self.rows = [
                np.pad(r, (0, new_dim - len(r))) for r in self.rows
            ]
            self._raw_normalized_rows = [
                np.pad(r, (0, new_dim - len(r))) for r in self._raw_normalized_rows
            ]
            self.feature_dim = new_dim

    def _embed_at_region(self, name: str, features: np.ndarray) -> np.ndarray:
        """Place *features* at their registered region offset inside the full vector.

        For pre-configured registries (fixed-dim), callers that already provide
        a full-dim vector have their values placed at [region.start:region.end]
        which is identical to passing the raw slice — backward-compatible.
        For emergent registries each block occupies its own contiguous slice.
        """
        region = self.registry.regions.get(name)
        vec = np.zeros(self.feature_dim, dtype=float)
        if region is None:
            # Fallback: best-effort front-fill (unknown region)
            flat = np.asarray(features, dtype=float).flatten()
            length = min(len(flat), self.feature_dim)
            vec[:length] = flat[:length]
            return vec
        flat = np.asarray(features, dtype=float).flatten()
        length = min(len(flat), region.dim)
        vec[region.start : region.start + length] = flat[:length]
        return vec

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
