"""Semantic Canvas: Hyperspace-specific wrapper around the standalone
semantic_interpreter framework.

This module preserves the existing Hyperspace API (CANVAS_DIMENSIONS,
REGION_TO_CANVAS, SemanticCanvas, StageSAE, train_stage_sae, CanvasEntry)
while delegating to the standalone ``semantic_interpreter`` package.

The standalone package is domain-agnostic; this wrapper configures it with
Hyperspace's 12 geopolitical semantic dimensions and 5-region mapping.
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
# Hyperspace-specific semantic dimensions (12 geopolitical axes)               #
# --------------------------------------------------------------------------- #

CANVAS_DIMENSIONS: list[dict[str, str]] = [
    {"key": "market_momentum",       "label": "Market Momentum",
     "desc": "Strength and direction of short-term financial momentum signals."},
    {"key": "temporal_memory",       "label": "Temporal Memory Depth",
     "desc": "How far back the system looks — short memory vs long historical patterns."},
    {"key": "volatility_regime",     "label": "Volatility Regime",
     "desc": "Market stability vs turbulence; regime shift signals."},
    {"key": "information_focus",     "label": "Information Focus",
     "desc": "Whether the information landscape is dominated by a single narrative or fragmented."},
    {"key": "narrative_diversity",   "label": "Narrative Diversity",
     "desc": "Breadth of distinct informational themes in the discourse environment."},
    {"key": "alliance_polarity",     "label": "Alliance Polarity",
     "desc": "Unipolar (one dominant bloc) vs multipolar (competing blocs) structure."},
    {"key": "network_cohesion",      "label": "Network Cohesion",
     "desc": "Density and clustering of the geopolitical relationship graph."},
    {"key": "power_concentration",   "label": "Power Concentration",
     "desc": "How concentrated resources and influence are among actors."},
    {"key": "geographic_coupling",   "label": "Geographic Coupling",
     "desc": "Co-variance of physical and socioeconomic factors across geopolitical nodes."},
    {"key": "systemic_stress",       "label": "Systemic Stress",
     "desc": "Aggregate pressure across conflict, economic, and political dimensions."},
    {"key": "cooperation_signal",    "label": "Cooperation Signal",
     "desc": "Net positive alignment and cooperative dynamics between agents."},
    {"key": "competition_signal",    "label": "Competition Signal",
     "desc": "Net negative alignment, rivalry, and zero-sum dynamics."},
    # Cross-domain dimensions — receive contributions from multiple regions
    {"key": "finance_geopolitical_coupling", "label": "Finance–Geopolitical Coupling",
     "desc": "Co-activation of market momentum and geopolitical network structure — "
             "indicates whether financial stress and political instability move together."},
    {"key": "information_power_dynamics",    "label": "Information–Power Dynamics",
     "desc": "Interaction between narrative diversity and agent power concentration — "
             "reveals whether information fragmentation tracks with power shifts."},
    {"key": "spatial_systemic_risk",         "label": "Spatial–Systemic Risk",
     "desc": "Joint signal from geospatial stress indicators and structural network "
             "centrality — captures geographically-grounded systemic risk."},
]

CANVAS_DIM = len(CANVAS_DIMENSIONS)

# Map: which UKT feature regions project onto which canvas dimensions
REGION_TO_CANVAS: dict[str, list[tuple[int, float]]] = {
    "temporal-pattern": [
        (0, 1.0),   # market_momentum
        (1, 1.0),   # temporal_memory
        (2, 0.8),   # volatility_regime
        (12, 0.7),  # finance_geopolitical_coupling (cross-domain)
    ],
    "semantic-embedding": [
        (3, 1.0),   # information_focus
        (4, 1.0),   # narrative_diversity
        (13, 0.8),  # information_power_dynamics (cross-domain)
    ],
    "structural-centrality": [
        (5, 1.0),   # alliance_polarity
        (6, 1.0),   # network_cohesion
        (7, 0.5),   # power_concentration
        (12, 0.7),  # finance_geopolitical_coupling (cross-domain)
        (14, 0.5),  # spatial_systemic_risk (cross-domain)
    ],
    "dynamic-agent": [
        (7, 0.5),   # power_concentration (shared with structural)
        (10, 1.0),  # cooperation_signal
        (11, 1.0),  # competition_signal
        (13, 0.6),  # information_power_dynamics (cross-domain)
    ],
    "geospatial-kernel": [
        (8, 1.0),   # geographic_coupling
        (9, 1.0),   # systemic_stress
        (14, 0.9),  # spatial_systemic_risk (cross-domain)
    ],
}


def _build_hyperspace_dimensions() -> list[SemanticDimension]:
    """Convert Hyperspace dimension dicts to standalone SemanticDimension objects."""
    return [
        SemanticDimension(key=d["key"], label=d["label"], description=d["desc"])
        for d in CANVAS_DIMENSIONS
    ]


class SemanticCanvas(_StandaloneCanvas):
    """Hyperspace-configured Semantic Canvas with 12 geopolitical dimensions.

    Pre-configured with Hyperspace's CANVAS_DIMENSIONS and REGION_TO_CANVAS
    mapping. Drop-in replacement for the original SemanticCanvas.
    """

    def __init__(self) -> None:
        super().__init__(
            dimensions=_build_hyperspace_dimensions(),
            region_mapping=REGION_TO_CANVAS,
        )
