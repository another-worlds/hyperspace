"""Semantic Canvas: Hyperspace-specific wrapper around the standalone
semantic_interpreter framework.

This module preserves the existing Hyperspace API (CANVAS_DIMENSIONS,
BLOCK_TO_CANVAS, SemanticCanvas, StageSAE, train_stage_sae, CanvasEntry)
while delegating to the standalone ``semantic_interpreter`` package.

The canvas is **block-aware**: semantic dimensions and block-to-canvas
projection mappings are defined per-block in BLOCK_SEMANTIC_SPEC, derived
from the underlying region semantic specs.
"""
from __future__ import annotations

import numpy as np

# Import standalone framework components
from semantic_interpreter.canvas import (
    SemanticDimension,
    SemanticCanvas as _StandaloneCanvas,
    CanvasEntry,
)
from semantic_interpreter.sae import (
    StageSAE,
    train_stage_sae,
)


# --------------------------------------------------------------------------- #
# Emergent canvas assembly from SAE concept discovery                       #
# --------------------------------------------------------------------------- #
# This module preserves the old API surface while removing hardcoded canvas
# dimensions and coupling weights. A canvas is built from GlobalSAE concepts
# at runtime.

from semantic_interpreter.canvas import build_emergent_canvas  # re-export


class SemanticCanvas(_StandaloneCanvas):
    """Hyperspace Semantic Canvas with emergent dimensions.

    This class preserves the legacy API while supporting fully emergent
    dimensions from SAE concept extraction.
    """

    def __init__(
        self,
        dimensions: list[SemanticDimension] | None = None,
        region_mapping: dict[str, list[tuple[int, float]]] | None = None,
    ) -> None:
        super().__init__(
            dimensions=dimensions or [],
            region_mapping=region_mapping or {},
        )

    # ------------------------------------------------------------------ #
    # Interpretability contract methods                                    #
    # ------------------------------------------------------------------ #

    def export_latent_units(self) -> dict[str, object]:
        """Export per-layer semantic coordinates as latent units."""
        return {
            "dimensions": [d.key for d in self.dimensions],
            "entries": [
                {
                    "step": int(e.step),
                    "block_name": e.block_name,
                    "dominant_dimensions": list(e.dominant_dimensions),
                    "coordinates": np.asarray(e.coordinates).tolist(),
                }
                for e in self.entries
            ],
        }

    def export_feature_attributions(
        self,
        input_batch: object | None = None,
    ) -> dict[str, object]:
        """Export canvas-level attributions from cumulative semantic state."""
        del input_batch  # Canvas attribution is state-based.
        state = self.get_accumulated_state()
        coords = np.asarray(state.get("coordinates", []), dtype=float)
        if coords.size == 0:
            return {"attributions": []}

        top_idx = np.argsort(np.abs(coords))[-5:][::-1]
        attributions = []
        for idx in top_idx:
            i = int(idx)
            dim = self.dimensions[i]
            attributions.append({
                "index": i,
                "key": dim.key,
                "label": dim.label,
                "value": float(coords[i]),
            })
        return {"attributions": attributions}

    def export_alignment_report(
        self,
        reference_modalities: list[str] | None = None,
    ) -> dict[str, object]:
        """Export block→dimension mapping as an alignment report."""
        del reference_modalities  # Not used for canvas-level mapping.
        mapping: dict[str, list[dict[str, object]]] = {}
        for block_name, links in self.region_mapping.items():  # Now block→dimension mapping
            mapping[block_name] = [
                {
                    "dimension_key": self.dimensions[idx].key,
                    "weight": float(weight),
                }
                for idx, weight in links
                if idx < self.n_dims
            ]

        return {
            "block_to_dimensions": mapping,
            "n_entries": len(self.entries),
        }

    def explain_prediction(
        self,
        context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Return a structured explanation for current canvas state."""
        state = self.get_accumulated_state()
        dom = state.get("dominant_narrative", [])
        summary = "No semantic narrative available."
        if dom:
            summary = "Dominant semantic dimensions: " + ", ".join(
                f"{d['label']} ({d['value']:.2f})" for d in dom[:3]
            )
        return {
            "summary": summary,
            "dominant_narrative": dom,
            "context": context or {},
        }


# Backward-compatibility placeholders for existing consumer code
CANVAS_DIMENSIONS: list[dict] = []
CANVAS_DIM: int = 0
BLOCK_TO_CANVAS: dict[str, list[tuple[int, float]]] = {}
REGION_TO_CANVAS: dict[str, list[tuple[int, float]]] = {}

