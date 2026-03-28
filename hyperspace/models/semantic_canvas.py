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
# Block-to-region mapping and block semantic specification                     #
# --------------------------------------------------------------------------- #
# Each block owns one region; semantic specs are region-keyed but mapped to blocks
BLOCK_TO_REGION: dict[str, str] = {
    "Finance": "temporal-pattern",
    "Clusters": "semantic-embedding",
    "Graph": "structural-centrality",
    "Agents": "dynamic-agent",
    "Spatial": "geospatial-kernel",
}

def _build_canvas_from_blocks(
    block_order: list[str] | None = None,
):
    """Build CANVAS_DIMENSIONS and BLOCK_TO_CANVAS dynamically from blocks.

    Args:
        block_order: Ordered list of block names. If None, uses the
            canonical order from BLOCK_TO_REGION keys.

    Returns:
        (dimensions_list, block_to_canvas_dict, canvas_dim_count)
    """
    if block_order is None:
        block_order = list(BLOCK_TO_REGION.keys())

    # 1. Collect all unique dimensions in block order (via their regions)
    seen_keys: set[str] = set()
    dimensions: list[dict[str, str]] = []
    key_to_index: dict[str, int] = {}

    for block_name in block_order:
        region_name = BLOCK_TO_REGION.get(block_name)
        if region_name is None:
            continue
        spec = REGION_SEMANTIC_SPEC.get(region_name)
        if spec is None:
            continue
        for key, label, desc in spec["dimensions"]:
            if key not in seen_keys:
                key_to_index[key] = len(dimensions)
                dimensions.append({"key": key, "label": label, "desc": desc})
                seen_keys.add(key)

    # 2. Build block-to-canvas projection mapping (same as region-to-canvas)
    block_to_canvas: dict[str, list[tuple[int, float]]] = {}
    for block_name in block_order:
        region_name = BLOCK_TO_REGION.get(block_name)
        if region_name is None:
            continue
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
        block_to_canvas[block_name] = links

    return dimensions, block_to_canvas, len(dimensions)

REGION_SEMANTIC_SPEC: dict[str, dict] = {  # Keyed by region name (for backward compat)
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

# Build at import time using the canonical block order
CANVAS_DIMENSIONS, BLOCK_TO_CANVAS, CANVAS_DIM = _build_canvas_from_blocks()

# Backward compatibility: REGION_TO_CANVAS maps regions to dimensions
# (used by knowledge_matrix.py in canvas replay)
REGION_TO_CANVAS: dict[str, list[tuple[int, float]]] = {}
for block_name, region_name in BLOCK_TO_REGION.items():
    if block_name in BLOCK_TO_CANVAS:
        REGION_TO_CANVAS[region_name] = BLOCK_TO_CANVAS[block_name]


def _build_hyperspace_dimensions() -> list[SemanticDimension]:
    """Convert Hyperspace dimension dicts to standalone SemanticDimension objects."""
    return [
        SemanticDimension(key=d["key"], label=d["label"], description=d["desc"])
        for d in CANVAS_DIMENSIONS
    ]


class SemanticCanvas(_StandaloneCanvas):
    """Hyperspace-configured Semantic Canvas with block-derived dimensions.

    Dimensions and block-to-canvas projection are auto-built from
    BLOCK_SEMANTIC_SPEC (derived from regional specs). Canvas maps each block
    to its semantic dimensions via BLOCK_TO_CANVAS.
    """

    def __init__(self) -> None:
        # Use block order from BLOCK_TO_REGION
        dims = CANVAS_DIMENSIONS
        # Include both block-name and region-name keys so project_block()
        # works whether called with block name or region name.
        mapping = dict(BLOCK_TO_CANVAS)
        mapping.update(REGION_TO_CANVAS)
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
