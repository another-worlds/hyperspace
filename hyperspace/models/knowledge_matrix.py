"""Universal Knowledge Tensor (UKT): Hyperspace-specific wrapper around the
standalone ukt framework.

This module preserves the existing Hyperspace API while delegating core logic
to the standalone ``ukt`` package. The standalone package is network-agnostic;
this wrapper adds Hyperspace-specific feature regions, block-region mappings,
and Semantic Canvas integration.
"""
from __future__ import annotations

import numpy as np

from hyperspace.config import (
    FEATURE_NAMES,
    KERNEL_BLOCK_CONTRIBUTION_MIN,
    KERNEL_CONTRIBUTING_REGION_THRESHOLD,
    KERNEL_NARRATOR_IMPORTANCE_MIN,
    REGION_DESCRIPTIONS,
    UKT_FEATURE_DIM,
)

# Re-export from standalone ukt framework
from ukt.registry import FeatureRegionRegistry
from ukt.kernels import (
    decompose_svd,
    describe_top_features,
    label_kernel as _standalone_label_kernel,
)
from ukt.projection import SharedProjection
from ukt.stability import estimate_regression_stability
from ukt.utils import _pad_or_truncate

# Hyperspace-specific semantic canvas
from hyperspace.models.semantic_canvas import (
    SemanticCanvas,
    train_stage_sae,
)


# --------------------------------------------------------------------------- #
# Hyperspace feature region registry (pre-configured for the 5-block pipeline)#
# --------------------------------------------------------------------------- #

def _build_hyperspace_registry() -> FeatureRegionRegistry:
    """Build the default Hyperspace feature region registry (80-dim, 5 regions)."""
    registry = FeatureRegionRegistry()
    registry.register(
        "temporal-pattern", 0, 16,
        description=REGION_DESCRIPTIONS.get("temporal-pattern", ""),
        feature_names=FEATURE_NAMES[0:16],
    )
    registry.register(
        "semantic-embedding", 16, 32,
        description=REGION_DESCRIPTIONS.get("semantic-embedding", ""),
        feature_names=FEATURE_NAMES[16:32],
    )
    registry.register(
        "structural-centrality", 32, 48,
        description=REGION_DESCRIPTIONS.get("structural-centrality", ""),
        feature_names=FEATURE_NAMES[32:48],
    )
    registry.register(
        "dynamic-agent", 48, 64,
        description=REGION_DESCRIPTIONS.get("dynamic-agent", ""),
        feature_names=FEATURE_NAMES[48:64],
    )
    registry.register(
        "geospatial-kernel", 64, 80,
        description=REGION_DESCRIPTIONS.get("geospatial-kernel", ""),
        feature_names=FEATURE_NAMES[64:80],
    )
    return registry


HYPERSPACE_REGISTRY = _build_hyperspace_registry()

# Feature provenance labels (backward-compatible dict form)
FEATURE_REGION_LABELS: dict[tuple[int, int], str] = {
    (r.start, r.end): r.name
    for r in HYPERSPACE_REGISTRY.ordered_regions
}

# Block-to-region mapping: block name → (region_name, start, end)
BLOCK_REGION_MAP: dict[str, tuple[str, int, int]] = {
    "Finance": ("temporal-pattern", 0, 16),
    "Clusters": ("semantic-embedding", 16, 32),
    "Graph": ("structural-centrality", 32, 48),
    "Agents": ("dynamic-agent", 48, 64),
    "Spatial": ("geospatial-kernel", 64, 80),
}


def _normalize_features(arr: np.ndarray) -> np.ndarray:
    """Normalize feature vector to [0, 1] range using min-max scaling.

    In the monolithic feature space, normalization is global (no region boundaries).
    Uses min-max scaling across the entire vector for comparable block scales.
    """
    out = arr.copy()
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    rng = arr_max - arr_min
    if rng > 1e-8:
        out = (arr - arr_min) / rng  # Scale to [0, 1]
    return np.clip(out, 0.0, 1.0)


def estimate_reality_regression_stability(
    matrix: np.ndarray,
    n_runs: int = 8,
    noise_std: float = 0.01,
    seed: int = 42,
) -> dict:
    """Delegate to standalone ukt framework."""
    return estimate_regression_stability(matrix, n_runs, noise_std, seed)


def _feature_name(idx: int, feature_meta: dict[int, dict] | None = None) -> str:
    """Return the name for a feature index, preferring metadata-derived labels."""
    if feature_meta and idx in feature_meta and feature_meta[idx].get("label"):
        return str(feature_meta[idx]["label"])
    return HYPERSPACE_REGISTRY.feature_name(idx)


def _region_for_index(idx: int) -> str:
    """Return the region label for a feature index."""
    region = HYPERSPACE_REGISTRY.region_for_index(idx)
    return region.name if region else "unknown"


# Block-to-feature-range mapping — hard-coded defaults, can be overridden per instance
_BLOCK_TO_REGION: dict[str, str] = {
    "Finance":  "temporal-pattern",
    "Clusters": "semantic-embedding",
    "Graph":    "structural-centrality",
    "Agents":   "dynamic-agent",
    "Spatial":  "geospatial-kernel",
}

# Build default block_feature_ranges from the registry
_DEFAULT_BLOCK_FEATURE_RANGES: dict[str, tuple[int, int]] = {}
for _block_name, _region_name in _BLOCK_TO_REGION.items():
    if _region_name in HYPERSPACE_REGISTRY.regions:
        _r = HYPERSPACE_REGISTRY.regions[_region_name]
        _DEFAULT_BLOCK_FEATURE_RANGES[_block_name] = (_r.start, _r.end)


class UniversalKnowledgeTensor:
    """Hyperspace-specific UKT with Semantic Canvas integration.

    Wraps the standalone ukt framework with Hyperspace's N-block pipeline,
    semantic canvas subsystem, and Tiny-LLM narratives.

    The feature dimension is derived from the registry — not hardcoded.
    New blocks can be added by registering regions in the registry.
    """

    def __init__(self, feature_dim: int | None = None):
        self.feature_dim = feature_dim or HYPERSPACE_REGISTRY.total_dim
        self.block_names: list[str] = []
        self.rows: list[np.ndarray] = []
        self._raw_features: list[np.ndarray] = []  # Unprojected, for re-projection
        self.snapshots: list[dict] = []
        self.global_feature_meta: dict[int, dict] = {}
        self.canvas = SemanticCanvas()
        # Block feature ranges: maps block name to (start, end) indices in 80-dim space
        self._block_feature_ranges: dict[str, tuple[int, int]] = _DEFAULT_BLOCK_FEATURE_RANGES.copy()
        # Initialize projection without registry — only block names needed for coupling topology
        self.projection = SharedProjection(
            block_names=list(self._block_feature_ranges.keys())
        )

    def add_block(
        self,
        name: str,
        features: np.ndarray,
        feature_meta: dict[int, dict] | None = None,
        timeframe_context: dict | None = None,
    ) -> dict:
        """Add a block's feature vector, normalize, decompose, interpret."""
        self.block_names.append(name)
        if feature_meta:
            self.global_feature_meta.update(feature_meta)
        raw = _pad_or_truncate(features, self.feature_dim)
        self._raw_features.append(raw)

        # Normalize BEFORE projection: bring each feature to [-1, 1] so blocks
        # enter the shared space on comparable scales. Then project — the
        # cross-block structure P creates is preserved for SVD.
        normalized_raw = _normalize_features(raw)

        # Feed full 80-dim normalized vector to the adaptive projection
        # The projection couples blocks via their full vectors, not sliced regions
        if name in self._block_feature_ranges:
            self.projection.observe(name, normalized_raw)

        # Project all raw features through the current projection matrix
        self.rows = [
            self.projection.project(_normalize_features(r))
            for r in self._raw_features
        ]

        matrix = np.stack(self.rows)
        decomposition = decompose_svd(matrix)
        n_kernels = decomposition.n_kernels

        kernel_labels = []
        for k in range(n_kernels):
            label = _standalone_label_kernel(
                k,
                decomposition,
                self._block_feature_ranges,
                FEATURE_NAMES,
                self.block_names,
                feature_meta=self.global_feature_meta,
                timeframe_context=timeframe_context,
            )
            # Enrich with dominant_region (feature-space region name) derived
            # from the block-level dominant_feature_block via _BLOCK_TO_REGION.
            dfb = label.get("dominant_feature_block", "")
            label["dominant_region"] = _BLOCK_TO_REGION.get(dfb, "unknown")
            kernel_labels.append(label)

        # Semantic Canvas: reset and replay ALL blocks with the current
        # projection, so every block's canvas coordinates are computed in
        # the same projection space.  No per-block SAE — canvas coordinates
        # are data-driven via feature distribution (entropy/concentration).
        stage_sae_result = None
        self.canvas = SemanticCanvas()
        canvas_entry = None
        for step_idx, (bn, row) in enumerate(zip(self.block_names, self.rows)):
            # Get block feature range from _block_feature_ranges
            if bn not in self._block_feature_ranges:
                continue
            lo, hi = self._block_feature_ranges[bn]
            region_features = row[lo:hi]
            # Get region name from registry for canvas (used as region_name parameter)
            region = HYPERSPACE_REGISTRY.region_for_index(lo)
            rname = region.name if region else bn

            entry = self.canvas.project_block(
                block_name=bn,
                step=step_idx + 1,
                region_name=rname,
                features=region_features,
                sae_result=None,
            )
            # Attach feature provenance evidence
            top_local = np.argsort(np.abs(region_features))[-3:][::-1]
            canvas_dim_keys = [
                self.canvas.dimensions[ci].key
                for ci, _ in self.canvas.region_mapping.get(rname, [])
                if ci < self.canvas.n_dims
            ]
            evidence = []
            for local_idx in top_local:
                global_idx = lo + int(local_idx)
                meta = self.global_feature_meta.get(global_idx, {})
                evidence.append({
                    "index": global_idx,
                    "name": _feature_name(global_idx, self.global_feature_meta),
                    "loading": round(float(row[global_idx]), 4),
                    "source": meta.get("source", "synthetic"),
                    "region": rname,
                    "canvas_dims": canvas_dim_keys,
                })
            entry.feature_evidence = evidence
            if bn == name:
                canvas_entry = entry

        # Tiny-LLM semantic translator: translates machine neuron clusters
        # (kernels, canvas coordinates) into human-readable narratives.
        # The LLM is the primary path; TemplateNarrator is the internal fallback.
        # Wrapped in try/except to allow pipeline to complete even if narrator
        # is slow (LLM generation on CPU can be significant per-kernel).
        layer_narrative = None
        try:
            from hyperspace.models.semantic_narrator import (
                narrate_layer, narrate_kernel,
            )
            if canvas_entry is not None:
                layer_narrative = narrate_layer(canvas_entry, self.canvas)
            for kl in kernel_labels:
                if kl["importance"] > KERNEL_NARRATOR_IMPORTANCE_MIN:
                    k_narr = narrate_kernel(kl, self.canvas)
                    if k_narr:
                        kl["semantic_narrative"] = k_narr
        except Exception:
            pass  # Narrator degrades gracefully; pipeline never fails

        # Build report
        report_lines = [
            f"=== Step {len(self.rows)}: Added '{name}' block ===",
            f"Active kernels: {n_kernels}",
            f"Reconstruction error: {decomposition.reconstruction_error:.6f}",
            f"Reality regression norm: {np.linalg.norm(decomposition.reality_regression):.4f}",
            "",
        ]

        if canvas_entry is not None:
            report_lines.append("--- Semantic Canvas ---")
            report_lines.append(canvas_entry.interpretation)
            if layer_narrative:
                report_lines.append(f"Narrative: {layer_narrative}")
            report_lines.append("")

        for kl in kernel_labels:
            report_lines.append(f"--- {kl['label']} ---")
            report_lines.append(kl["narrative"])
            if kl.get("semantic_narrative"):
                report_lines.append(f"Semantic: {kl['semantic_narrative']}")
            report_lines.append("")

        if len(self.rows) > 1:
            report_lines.append("--- Cross-Block Coherence ---")
            for k in range(n_kernels):
                contribs = []
                for i, bn in enumerate(self.block_names):
                    val = float(decomposition.kernel_activation[i, k])
                    if abs(val) > 0.01:
                        contribs.append(f"{bn}={val:+.3f}")
                report_lines.append(f"K{k} activation across blocks: {', '.join(contribs)}")
            report_lines.append("")

        rr_top = np.argsort(np.abs(decomposition.reality_regression))[-5:][::-1]
        report_lines.append("--- Reality Regression (top features) ---")
        for idx in rr_top:
            report_lines.append(
                f"  {_feature_name(int(idx), self.global_feature_meta)} [{_region_for_index(int(idx))}]: "
                f"{decomposition.reality_regression[int(idx)]:+.4f}"
            )

        snapshot = dict(
            step=len(self.rows),
            block_name=name,
            raw_features=[r.copy() for r in self._raw_features],
            matrix=matrix.copy(),
            U=decomposition.U.copy(),
            S=decomposition.S.copy(),
            Vt=decomposition.Vt.copy(),
            n_kernels=n_kernels,
            importance=decomposition.importance.copy(),
            kernel_activation=decomposition.kernel_activation.copy(),
            reality_regression=decomposition.reality_regression.copy(),
            kernel_labels=kernel_labels,
            reconstruction_error=decomposition.reconstruction_error,
            report="\n".join(report_lines),
            feature_meta=self.global_feature_meta.copy(),
            timeframe_context=timeframe_context or {},
            stage_sae_result=stage_sae_result,
            canvas_entry=canvas_entry,
            layer_narrative=layer_narrative,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_final_matrix(self) -> np.ndarray | None:
        if not self.rows:
            return None
        return np.stack(self.rows)

    def get_latest_snapshot(self) -> dict | None:
        return self.snapshots[-1] if self.snapshots else None

    def export_latent_units(self) -> dict[str, object]:
        """Return a machine-readable export of the latest latent units."""
        latest = self.get_latest_snapshot()
        if latest is None:
            return {"kernel_count": 0, "kernels": [], "canvas": None}

        canvas_state = self.canvas.get_accumulated_state()
        kernels = [
            {
                "kernel_id": k.get("kernel_id"),
                "importance": float(k.get("importance", 0.0)),
                "dominant_block": k.get("dominant_block"),
                "dominant_region": k.get("dominant_region"),
                "top_feature_indices": k.get("top_feature_indices", []),
            }
            for k in latest.get("kernel_labels", [])
        ]
        return {
            "kernel_count": len(kernels),
            "kernels": kernels,
            "canvas": {
                "coordinates": np.asarray(canvas_state.get("coordinates", [])).tolist(),
                "dominant_narrative": canvas_state.get("dominant_narrative", []),
            },
        }

    def export_feature_attributions(
        self,
        input_batch: object | None = None,
    ) -> dict[str, object]:
        """Return top feature attributions from reality regression."""
        del input_batch  # UKT attribution is snapshot-level, not per-input.
        latest = self.get_latest_snapshot()
        if latest is None:
            return {"attributions": []}

        rr = latest.get("reality_regression")
        if rr is None:
            return {"attributions": []}

        rr = np.asarray(rr)
        top_idx = np.argsort(np.abs(rr))[-10:][::-1]
        feature_meta = latest.get("feature_meta", {})
        attributions = []
        for idx in top_idx:
            i = int(idx)
            attributions.append({
                "index": i,
                "name": _feature_name(i, feature_meta),
                "region": _region_for_index(i),
                "value": float(rr[i]),
                "abs_value": float(abs(rr[i])),
            })
        return {"attributions": attributions}

    def export_alignment_report(
        self,
        reference_modalities: list[str] | None = None,
    ) -> dict[str, object]:
        """Return a modality alignment summary from latest kernel structure."""
        latest = self.get_latest_snapshot()
        if latest is None:
            return {"modalities": [], "kernel_block_coverage": {}, "n_kernels": 0}

        labels = latest.get("kernel_labels", [])
        modalities = sorted({k.get("dominant_block", "unknown") for k in labels})
        if reference_modalities:
            allowed = set(reference_modalities)
            modalities = [m for m in modalities if m in allowed]

        block_coverage: dict[str, float] = {}
        for k in labels:
            block = k.get("dominant_feature_block", "unknown")
            block_coverage[block] = block_coverage.get(block, 0.0) + float(
                k.get("importance", 0.0),
            )

        return {
            "modalities": modalities,
            "kernel_block_coverage": block_coverage,
            "n_kernels": int(latest.get("n_kernels", 0)),
        }

    def explain_prediction(
        self,
        context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Return structured explanations for the latest UKT state."""
        latest = self.get_latest_snapshot()
        if latest is None:
            return {"summary": "No UKT snapshots available.", "context": context or {}}

        labels = latest.get("kernel_labels", [])
        top_kernel = max(labels, key=lambda x: x.get("importance", 0.0)) if labels else None
        return {
            "summary": latest.get("report", ""),
            "top_kernel": {
                "kernel_id": top_kernel.get("kernel_id") if top_kernel else None,
                "label": top_kernel.get("label") if top_kernel else None,
                "narrative": top_kernel.get("narrative") if top_kernel else None,
            },
            "context": context or {},
        }
