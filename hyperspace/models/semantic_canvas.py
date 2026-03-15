"""Semantic Canvas: Hyperspace-specific wrapper around the standalone
semantic_interpreter framework.

This module preserves the existing Hyperspace API (CANVAS_DIMENSIONS,
REGION_TO_CANVAS, SemanticCanvas, StageSAE, train_stage_sae, CanvasEntry)
while delegating to the standalone ``semantic_interpreter`` package.

The canvas is **registry-aware**: semantic dimensions and region-to-canvas
projection mappings are defined per-region in REGION_SEMANTIC_SPEC, so adding
a new region to the UKT feature registry automatically adds its canvas
dimensions and projection. No hardcoded dimension count or region list.
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
# Per-region semantic specification                                            #
#                                                                              #
# Each region declares:                                                        #
#   - "dimensions": list of (key, label, desc) for canvas axes it introduces   #
#   - "primary":    list of own dimension keys at weight 1.0                   #
#   - "coupling":   list of (dim_key, weight) for cross-domain signals         #
#                                                                              #
# The overall canvas is assembled by iterating all regions in registry order.  #
# --------------------------------------------------------------------------- #

REGION_SEMANTIC_SPEC: dict[str, dict] = {
    "temporal-pattern": {
        "dimensions": [
            ("market_momentum", "Market Momentum",
             "Strength and direction of short-term financial momentum signals."),
            ("temporal_memory", "Temporal Memory Depth",
             "How far back the system looks — short memory vs long historical patterns."),
            ("volatility_regime", "Volatility Regime",
             "Market stability vs turbulence; regime shift signals."),
        ],
        "primary": ["market_momentum", "temporal_memory"],
        "coupling": [("volatility_regime", 0.8), ("systemic_stress", 0.3)],
    },
    "semantic-embedding": {
        "dimensions": [
            ("information_focus", "Information Focus",
             "Whether the information landscape is dominated by a single narrative or fragmented."),
            ("narrative_diversity", "Narrative Diversity",
             "Breadth of distinct informational themes in the discourse environment."),
        ],
        "primary": ["information_focus", "narrative_diversity"],
        "coupling": [("power_concentration", 0.4)],
    },
    "structural-centrality": {
        "dimensions": [
            ("alliance_polarity", "Alliance Polarity",
             "Unipolar (one dominant bloc) vs multipolar (competing blocs) structure."),
            ("network_cohesion", "Network Cohesion",
             "Density and clustering of the geopolitical relationship graph."),
            ("power_concentration", "Power Concentration",
             "How concentrated resources and influence are among actors."),
        ],
        "primary": ["alliance_polarity", "network_cohesion"],
        "coupling": [("power_concentration", 0.5), ("market_momentum", 0.3),
                      ("systemic_stress", 0.4)],
    },
    "dynamic-agent": {
        "dimensions": [
            ("cooperation_signal", "Cooperation Signal",
             "Net positive alignment and cooperative dynamics between agents."),
            ("competition_signal", "Competition Signal",
             "Net negative alignment, rivalry, and zero-sum dynamics."),
        ],
        "primary": ["cooperation_signal", "competition_signal"],
        "coupling": [("power_concentration", 0.5), ("narrative_diversity", 0.3)],
    },
    "geospatial-kernel": {
        "dimensions": [
            ("geographic_coupling", "Geographic Coupling",
             "Co-variance of physical and socioeconomic factors across geopolitical nodes."),
            ("systemic_stress", "Systemic Stress",
             "Aggregate pressure across conflict, economic, and political dimensions."),
        ],
        "primary": ["geographic_coupling", "systemic_stress"],
        "coupling": [("network_cohesion", 0.3)],
    },
}


def _build_canvas_from_spec(
    region_order: list[str] | None = None,
):
    """Build CANVAS_DIMENSIONS and REGION_TO_CANVAS dynamically from the
    region semantic spec.

    Args:
        region_order: Ordered list of region names. If None, uses the
            canonical order from REGION_SEMANTIC_SPEC keys.

    Returns:
        (dimensions_list, region_to_canvas_dict, canvas_dim_count)
    """
    if region_order is None:
        region_order = list(REGION_SEMANTIC_SPEC.keys())

    # 1. Collect all unique dimensions in region order
    seen_keys: set[str] = set()
    dimensions: list[dict[str, str]] = []
    key_to_index: dict[str, int] = {}

    for region_name in region_order:
        spec = REGION_SEMANTIC_SPEC.get(region_name)
        if spec is None:
            continue
        for key, label, desc in spec["dimensions"]:
            if key not in seen_keys:
                key_to_index[key] = len(dimensions)
                dimensions.append({"key": key, "label": label, "desc": desc})
                seen_keys.add(key)

    # 2. Build region-to-canvas projection mapping
    region_to_canvas: dict[str, list[tuple[int, float]]] = {}
    for region_name in region_order:
        spec = REGION_SEMANTIC_SPEC.get(region_name)
        if spec is None:
            continue
        links: list[tuple[int, float]] = []
        # Primary dims at weight 1.0
        for dim_key in spec["primary"]:
            if dim_key in key_to_index:
                links.append((key_to_index[dim_key], 1.0))
        # Cross-domain coupling
        for dim_key, weight in spec["coupling"]:
            if dim_key in key_to_index:
                links.append((key_to_index[dim_key], weight))
        region_to_canvas[region_name] = links

    return dimensions, region_to_canvas, len(dimensions)


# Build at import time using the canonical spec order.
# When the registry is available, SemanticCanvas.__init__ will rebuild
# using the actual registry order (which may differ if regions are added).
CANVAS_DIMENSIONS, REGION_TO_CANVAS, CANVAS_DIM = _build_canvas_from_spec()


def _build_hyperspace_dimensions() -> list[SemanticDimension]:
    """Convert Hyperspace dimension dicts to standalone SemanticDimension objects."""
    return [
        SemanticDimension(key=d["key"], label=d["label"], description=d["desc"])
        for d in CANVAS_DIMENSIONS
    ]


class SemanticCanvas(_StandaloneCanvas):
    """Hyperspace-configured Semantic Canvas with registry-derived dimensions.

    Dimensions and region-to-canvas projection are auto-built from
    REGION_SEMANTIC_SPEC. When the UKT registry is available, the canvas
    uses registry order to ensure consistency. Adding a new region with
    a semantic spec entry automatically extends the canvas.
    """

    def __init__(self) -> None:
        # Try to use registry order if available (avoids circular import
        # by deferring the import to instantiation time)
        dims = CANVAS_DIMENSIONS
        mapping = REGION_TO_CANVAS
        try:
            from hyperspace.models.knowledge_matrix import HYPERSPACE_REGISTRY
            region_order = [r.name for r in HYPERSPACE_REGISTRY.ordered_regions]
            dims, mapping, _ = _build_canvas_from_spec(region_order)
        except ImportError:
            pass  # Standalone usage without knowledge_matrix
        super().__init__(
            dimensions=[
                SemanticDimension(key=d["key"], label=d["label"], description=d["desc"])
                for d in dims
            ],
            region_mapping=mapping,
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
        """Export modality→dimension mapping as an alignment report."""
        del reference_modalities  # Not used for canvas-level mapping.
        mapping: dict[str, list[dict[str, object]]] = {}
        for region, links in self.region_mapping.items():
            mapping[region] = [
                {
                    "dimension_key": self.dimensions[idx].key,
                    "weight": float(weight),
                }
                for idx, weight in links
                if idx < self.n_dims
            ]

        return {
            "region_to_dimensions": mapping,
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
