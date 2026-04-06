"""Universal Knowledge Tensor (UKT): Hyperspace-specific wrapper around the
standalone ukt framework.

This module preserves the existing Hyperspace API while delegating core logic
to the standalone ``ukt`` package. The standalone package is network-agnostic;
this wrapper adds Semantic Canvas integration and Hyperspace-specific snapshot
enrichments (narratives, contrastive alignment, dominant_region labelling).

Design: ``UniversalKnowledgeTensor`` *subclasses* the framework's class.
The registry is **emergent** — blocks self-register as they are added, so no
dimensionality or region layout is pre-configured. Any neural network can plug
in by simply calling ``add_block(name, features)``.

The constructor injects:
  - ``projection`` — :class:`~ukt.projection.SharedProjection` (dim=0, grows
    automatically as blocks arrive so cross-block coupling is data-driven from
    the first pair)
  - ``normalizer`` — symmetric [-1, 1] global normalizer (richer coupling)
  - ``on_snapshot`` — ``_enrich_snapshot`` hook that adds Hyperspace-specific
    fields (dominant_region, LLM narratives, projection_matrix, …) without
    duplicating ``add_block`` logic
"""
from __future__ import annotations

import numpy as np

from hyperspace.config import (
    KERNEL_BLOCK_CONTRIBUTION_MIN,
    KERNEL_CONTRIBUTING_REGION_THRESHOLD,
    KERNEL_NARRATOR_IMPORTANCE_MIN,
)

# Re-export from standalone ukt framework
from ukt.registry import FeatureRegionRegistry
from ukt.projection import SharedProjection
from ukt.stability import estimate_regression_stability
from ukt.tensor import (
    UniversalKnowledgeTensor as _UKTFramework,
    _normalize_features,
)
from ukt.utils import _pad_or_truncate

# Disable preconfigured region labels in emergent mode; kept for legacy APIs
FEATURE_REGION_LABELS: dict[str, tuple[int, int]] = {}

# Hyperspace-specific semantic canvas
from hyperspace.models.semantic_canvas import (
    SemanticCanvas,
)


def _symmetric_normalize(
    arr: np.ndarray,
    registry: FeatureRegionRegistry,
) -> np.ndarray:
    """Normalize a feature vector to [-1, 1] using global min-max scaling.

    Global (not per-region) normalization is used in the Hyperspace pipeline so
    that negative loadings create richer coupling structure in the SharedProjection
    rank-1 outer products.

    Only indices belonging to registered regions are included in the global
    min/max computation and transformed.  Unoccupied (zero-padded) indices
    remain 0.0 so they do not inflate projection energy or distort coupling
    direction vectors.
    """
    out = np.zeros_like(arr)
    regions = registry.ordered_regions
    if not regions:
        # No regions registered yet — normalize the entire vector as-is
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
        rng = arr_max - arr_min
        if rng > 1e-8:
            out = 2.0 * (arr - arr_min) / rng - 1.0
        return np.clip(out, -1.0, 1.0)

    # Collect all occupied values to compute a single global min/max
    occupied_vals = np.concatenate(
        [arr[r.start:r.end] for r in regions if r.start < len(arr)]
    )
    if len(occupied_vals) == 0:
        return out
    arr_min = float(np.min(occupied_vals))
    arr_max = float(np.max(occupied_vals))
    rng = arr_max - arr_min

    # Apply symmetric [-1, 1] scaling only to occupied indices
    for r in regions:
        lo = r.start
        hi = min(r.end, len(arr))
        if lo >= len(arr):
            continue
        if rng > 1e-8:
            out[lo:hi] = np.clip(2.0 * (arr[lo:hi] - arr_min) / rng - 1.0, -1.0, 1.0)
        else:
            out[lo:hi] = 0.0  # Constant region → zero (no signal)
    return out


def estimate_reality_regression_stability(
    matrix: np.ndarray,
    n_runs: int = 8,
    noise_std: float = 0.01,
    seed: int = 42,
) -> dict:
    """Delegate to standalone ukt framework."""
    return estimate_regression_stability(matrix, n_runs, noise_std, seed)


def _feature_name(
    idx: int,
    registry: FeatureRegionRegistry,
    feature_meta: dict[int, dict] | None = None,
) -> str:
    """Return the name for a feature index, preferring metadata-derived labels."""
    if feature_meta and idx in feature_meta and feature_meta[idx].get("label"):
        return str(feature_meta[idx]["label"])
    return registry.feature_name(idx)


def _region_for_index(idx: int, registry: FeatureRegionRegistry) -> str:
    """Return the region label for a feature index."""
    region = registry.region_for_index(idx)
    return region.name if region else "unknown"


# --------------------------------------------------------------------------- #
# Hyperspace-specific UKT wrapper                                             #
# --------------------------------------------------------------------------- #

class UniversalKnowledgeTensor(_UKTFramework):
    """Hyperspace-specific UKT with Semantic Canvas integration.

    Subclasses the standalone :class:`~ukt.tensor.UniversalKnowledgeTensor`
    and injects:

    - Symmetric [-1, 1] global normalization (``normalizer``)
    - Adaptive cross-block SharedProjection (``projection``) that starts with
      zero dimensions and self-expands as blocks arrive — no topology prior
    - ``_enrich_snapshot`` hook (``on_snapshot``) that adds downstream
      Hyperspace fields to every snapshot: ``dominant_region`` per kernel,
      Tiny-LLM semantic narratives, ``projection_matrix``, etc.

    Downstream code calls ``add_block()`` and ``get_final_matrix()`` exactly as
    before.  The registry, feature_dim, and projection all grow automatically.
    """

    def __init__(
        self,
        feature_dim: int | None = None,
    ) -> None:
        # Projection starts dimensionless; self-expands as blocks arrive
        _projection = SharedProjection()

        super().__init__(
            registry=None,  # Empty emergent registry — self-registers per block
            feature_dim=feature_dim,
            projection=_projection,
            normalizer=_symmetric_normalize,
            on_snapshot=self._enrich_snapshot,
        )

        # Hyperspace-specific state
        self.canvas = SemanticCanvas()
        # Tracks raw (pre-normalization) feature vectors for snapshot compat
        self._hs_raw_features: list[np.ndarray] = []

    # ---------------------------------------------------------------------- #
    # Override add_block to also capture raw (un-normalized) features        #
    # ---------------------------------------------------------------------- #

    def add_block(
        self,
        name: str,
        features: np.ndarray,
        feature_meta: dict[int, dict] | None = None,
        timeframe_context: dict | None = None,
    ) -> dict:
        """Add a block, capturing raw features before delegating to the framework."""
        self._hs_raw_features.append(np.asarray(features, dtype=float).flatten())
        return super().add_block(name, features, feature_meta, timeframe_context)

    # ---------------------------------------------------------------------- #
    # on_snapshot hook: Hyperspace-specific snapshot enrichment              #
    # ---------------------------------------------------------------------- #

    def _enrich_snapshot(self, snapshot: dict) -> dict:
        """Inject Hyperspace-specific fields into the just-assembled snapshot.

        Called by the framework's ``add_block`` after the base snapshot dict is
        fully built.  Mutates and returns it — framework replaces its snapshot
        with the return value.
        """
        # 1. Add dominant_region to each kernel label from the emergent registry
        for kl in snapshot.get("kernel_labels", []):
            dfb = kl.get("dominant_feature_block", "")
            region = self.registry.region_for_index(
                # dominant_feature_block is a block name; look up its region start
                self.registry.regions[dfb].start if dfb in self.registry.regions else 0
            )
            kl["dominant_region"] = region.name if region else dfb or "unknown"

        # 2. Add backward-compatible extra fields
        snapshot["raw_features"] = [r.copy() for r in self._hs_raw_features]
        if self.projection is not None:
            snapshot["projection_matrix"] = self.projection.projection_matrix.copy()
        snapshot.setdefault("stage_sae_result", None)
        snapshot.setdefault("canvas_entry", None)
        snapshot.setdefault("layer_narrative", None)
        snapshot.setdefault("contrastive_alignment_score", None)

        # 3. Tiny-LLM semantic translator: generates per-kernel narratives.
        #    The canvas is empty until the global SAE runs after all 5 blocks
        #    are added, so narrate_kernel receives an empty canvas here — this
        #    is the same as the previous behaviour.  Non-fatal.
        try:
            from hyperspace.models.semantic_narrator import narrate_kernel
            from hyperspace.core.caching import get_or_compute_narrative, hash_params
            import streamlit as _st
            _policy = _st.session_state.get("policy_language_mode", False)
            for kl in snapshot.get("kernel_labels", []):
                if kl.get("importance", 0.0) > KERNEL_NARRATOR_IMPORTANCE_MIN:
                    _kk_params = {
                        "id": kl.get("id", ""),
                        "imp": round(kl["importance"], 6),
                        "region": kl.get("dominant_region", ""),
                        "policy": _policy,
                    }
                    _kk = f"kernel_{hash_params(_kk_params)}"
                    k_narr = get_or_compute_narrative(
                        _kk, lambda _kl=kl: narrate_kernel(_kl, self.canvas),
                    )
                    if k_narr:
                        kl["semantic_narrative"] = k_narr
        except Exception:
            pass  # Narrator degrades gracefully; pipeline never fails

        # 4. Optional contrastive alignment score (SPEC-4, non-fatal)
        try:
            from hyperspace.config import ENABLE_CONTRASTIVE_ALIGNMENT, CONTRASTIVE_WEIGHT
            if (
                ENABLE_CONTRASTIVE_ALIGNMENT
                and CONTRASTIVE_WEIGHT > 0.0
                and len(self.rows) >= 2
            ):
                import streamlit as _st
                from hyperspace.models.contrastive_encoder import ContrastiveEncoderBank
                encoder_bank = _st.session_state.get("contrastive_encoder_bank")
                if encoder_bank is None:
                    encoder_bank = ContrastiveEncoderBank()
                    _st.session_state["contrastive_encoder_bank"] = encoder_bank
                encoder_bank.train_step(self.rows[-1])
                snapshot["contrastive_alignment_score"] = encoder_bank.alignment_score(
                    self.rows[-1]
                )
        except Exception:
            pass  # Contrastive path is non-fatal

        return snapshot

    # ---------------------------------------------------------------------- #
    # Emergent canvas (called after global SAE completes)                    #
    # ---------------------------------------------------------------------- #

    def build_emergent_canvas(self, sae_result: dict) -> SemanticCanvas:
        """Build emergent canvas from global SAE results and store as self.canvas.

        Called by the pipeline after the global SAE runs on the full matrix so
        that canvas dimensions are 100% data-driven SAE concepts, not hardcoded.
        """
        from semantic_interpreter.canvas import (
            build_emergent_canvas as _build_ec,
        )
        self.canvas = _build_ec(sae_result, self.block_names)
        return self.canvas

    # ---------------------------------------------------------------------- #
    # Interpretability contract methods (SPEC-3)                             #
    # ---------------------------------------------------------------------- #

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
                "name": _feature_name(i, self.registry, feature_meta),
                "region": _region_for_index(i, self.registry),
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


# --------------------------------------------------------------------------- #
# Global Registry Access (Vision Compliance - Invariant 8)                   #
# --------------------------------------------------------------------------- #

# Create a module-level UKT instance that serves as the canonical registry source
# This follows Vision Invariant 8: "Centralized thresholds" - all magic numbers  
# and structural references must be defined in a single configuration source.
# Rather than having multiple UKT instances with different registries, this  
# provides a global registry that counterfactual analysis and other components
# can reliably reference.
_canonical_ukt_instance = UniversalKnowledgeTensor()

# Export the registry from the canonical instance for global access
# This registry will be populated as blocks are added to any UKT instance
# that follows the emergent architecture pattern.
HYPERSPACE_REGISTRY = _canonical_ukt_instance.registry
