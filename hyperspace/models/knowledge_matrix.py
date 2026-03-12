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
    REGION_DESCRIPTIONS,
    UKT_FEATURE_DIM,
)

# Re-export from standalone ukt framework
from ukt.registry import FeatureRegionRegistry
from ukt.kernels import (
    decompose_svd,
    describe_top_features,
    generate_kernel_narrative as _standalone_generate_kernel_narrative,
)
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


def _normalize_features(arr: np.ndarray) -> np.ndarray:
    """Normalize each region of a feature vector to [0, 1] range."""
    out = arr.copy()
    for (lo, hi) in FEATURE_REGION_LABELS.keys():
        region = out[lo:hi]
        rng = region.max() - region.min()
        if rng > 1e-8:
            out[lo:hi] = (region - region.min()) / rng
    return out


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


def _describe_top_features(
    vt_row: np.ndarray,
    feature_meta: dict[int, dict] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Return descriptions for the top-N loaded features."""
    return describe_top_features(vt_row, HYPERSPACE_REGISTRY, feature_meta, top_n)


def _generate_kernel_narrative(
    k_idx: int,
    dominant_block: str,
    dominant_region: str,
    importance: float,
    top_features: list[dict],
    block_names: list[str],
    u_col: np.ndarray,
    timeframe_context: dict | None,
) -> str:
    """Generate a data-grounded narrative for a kernel."""
    return _standalone_generate_kernel_narrative(
        k_idx, dominant_block, dominant_region, importance,
        top_features, block_names, u_col, HYPERSPACE_REGISTRY,
        timeframe_context,
    )


def _label_kernel(
    k_idx: int,
    vt_row: np.ndarray,
    u_col: np.ndarray,
    block_names: list[str],
    importance: float,
    feature_meta: dict[int, dict] | None,
    timeframe_context: dict | None,
) -> dict:
    """Generate a semantic label for a single kernel."""
    region_scores = {}
    for (lo, hi), label in FEATURE_REGION_LABELS.items():
        region_scores[label] = float(np.abs(vt_row[lo:hi]).sum())
    dominant_region = max(region_scores, key=region_scores.get)

    dominant_block_idx = int(np.argmax(np.abs(u_col)))
    dominant_block = (block_names[dominant_block_idx]
                      if dominant_block_idx < len(block_names) else "unknown")

    top_features = _describe_top_features(vt_row, feature_meta=feature_meta, top_n=5)
    top_feat_name = top_features[0]["name"] if top_features else "?"
    short_label = (
        f"K{k_idx}: {dominant_block} — {dominant_region.replace('-', ' ')} "
        f"({importance:.1%} var, lead: {top_feat_name})"
    )

    narrative = _generate_kernel_narrative(
        k_idx, dominant_block, dominant_region, importance,
        top_features, block_names, u_col, timeframe_context,
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


# Map block names to their UKT region keys for canvas projection
BLOCK_REGION_MAP: dict[str, tuple[str, int, int]] = {
    "Finance":  ("temporal-pattern",      0,  16),
    "Clusters": ("semantic-embedding",    16, 32),
    "Graph":    ("structural-centrality", 32, 48),
    "Agents":   ("dynamic-agent",         48, 64),
    "Spatial":  ("geospatial-kernel",     64, 80),
}


class UniversalKnowledgeTensor:
    """Hyperspace-specific UKT with Semantic Canvas integration.

    Wraps the standalone ukt framework with Hyperspace's 5-block pipeline,
    semantic canvas subsystem, and Tiny-LLM narratives.
    """

    def __init__(self, feature_dim: int = UKT_FEATURE_DIM):
        self.feature_dim = feature_dim
        self.block_names: list[str] = []
        self.rows: list[np.ndarray] = []
        self.snapshots: list[dict] = []
        self.global_feature_meta: dict[int, dict] = {}
        self.canvas = SemanticCanvas()

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
        normalized = _normalize_features(raw)
        self.rows.append(normalized)

        matrix = np.stack(self.rows)
        decomposition = decompose_svd(matrix)
        n_kernels = decomposition.n_kernels

        kernel_labels = []
        for k in range(n_kernels):
            label = _label_kernel(
                k,
                decomposition.Vt[k],
                decomposition.U[:, k],
                self.block_names,
                float(decomposition.importance[k]),
                self.global_feature_meta,
                timeframe_context,
            )
            kernel_labels.append(label)

        # Semantic Canvas subsystem: per-stage SAE + canvas projection
        stage_sae_result = None
        canvas_entry = None
        region_info = BLOCK_REGION_MAP.get(name)
        if region_info is not None:
            region_name, lo, hi = region_info
            region_features = normalized[lo:hi]
            stage_sae_result = train_stage_sae(region_features, concept_dim=8, epochs=60)
            canvas_entry = self.canvas.project_block(
                block_name=name,
                step=len(self.rows),
                region_name=region_name,
                features=region_features,
                sae_result=stage_sae_result,
            )

            # Attach feature provenance evidence — links canvas narrative back to
            # concrete feature indices, names, and data sources for governance audit.
            top_local = np.argsort(np.abs(region_features))[-3:][::-1]
            canvas_dim_keys = [
                self.canvas.dimensions[ci].key
                for ci, _ in self.canvas.region_mapping.get(region_name, [])
                if ci < self.canvas.n_dims
            ]
            evidence = []
            for local_idx in top_local:
                global_idx = lo + int(local_idx)
                meta = self.global_feature_meta.get(global_idx, {})
                evidence.append({
                    "index": global_idx,
                    "name": _feature_name(global_idx, self.global_feature_meta),
                    "loading": round(float(normalized[global_idx]), 4),
                    "source": meta.get("source", "synthetic"),
                    "region": region_name,
                    "canvas_dims": canvas_dim_keys,
                })
            canvas_entry.feature_evidence = evidence

        # Generate Tiny-LLM narratives (graceful degradation)
        layer_narrative = None
        try:
            from hyperspace.models.semantic_narrator import (
                narrate_layer, narrate_kernel,
            )
            if canvas_entry is not None:
                layer_narrative = narrate_layer(canvas_entry, self.canvas)
            for kl in kernel_labels:
                if kl["importance"] > 0.15:
                    k_narr = narrate_kernel(kl, self.canvas)
                    if k_narr:
                        kl["semantic_narrative"] = k_narr
        except Exception:
            pass

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
            return {"modalities": [], "kernel_region_coverage": {}, "n_kernels": 0}

        labels = latest.get("kernel_labels", [])
        modalities = sorted({k.get("dominant_block", "unknown") for k in labels})
        if reference_modalities:
            allowed = set(reference_modalities)
            modalities = [m for m in modalities if m in allowed]

        region_coverage: dict[str, float] = {}
        for k in labels:
            region = k.get("dominant_region", "unknown")
            region_coverage[region] = region_coverage.get(region, 0.0) + float(
                k.get("importance", 0.0),
            )

        return {
            "modalities": modalities,
            "kernel_region_coverage": region_coverage,
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
