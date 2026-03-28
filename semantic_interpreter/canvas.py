"""Semantic Canvas: a unified interpretive coordinate system for neural network internals.

The canvas provides named semantic axes — human-readable dimensions like
"momentum", "cohesion", "stress" — onto which hidden-layer activations are
projected. Instead of staring at 768-dimensional activation vectors, you get
coordinates in a space you can reason about.

The canvas is:
- **Configurable**: define your own semantic dimensions for your domain
- **Accumulative**: builds a trajectory as layers/sources are added
- **Network-agnostic**: works with any feature source that maps to regions

Architecture:
    Hidden Layer → SAE → Sparse Concepts → Canvas Projection → Named Coordinates
    Canvas accumulates across depth → Narrator reads coordinates → Narrative
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SemanticDimension:
    """A named axis in the semantic canvas."""
    key: str        # Machine-readable key (e.g., "market_momentum")
    label: str      # Human-readable label (e.g., "Market Momentum")
    description: str  # What this dimension captures

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "desc": self.description}


@dataclass
class CanvasEntry:
    """One layer/source's contribution to the semantic canvas."""
    block_name: str
    step: int
    coordinates: np.ndarray          # (n_dims,) — this layer's semantic position
    concept_activations: np.ndarray  # raw concept activations from stage SAE
    active_concepts: int
    dominant_dimensions: list[str]   # top-2 canvas dimension keys
    interpretation: str              # short algorithmic interpretation
    feature_evidence: list[dict] = field(default_factory=list)  # provenance trace


@dataclass
class SemanticCanvas:
    """Unified interpretive space accumulating semantic contributions across depth.

    Each source/layer projects its discovered concepts onto named semantic
    dimensions. The canvas grows as layers are added, building a trajectory
    of semantic meaning through the network.

    Args:
        dimensions: List of SemanticDimension defining the canvas axes.
                    If None, a minimal default set is used.
        region_mapping: Dict mapping UKT region names to lists of
                       (canvas_dim_index, weight) tuples. Defines how each
                       feature region projects onto the canvas.
    """
    dimensions: list[SemanticDimension] = field(default_factory=list)
    region_mapping: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    entries: list[CanvasEntry] = field(default_factory=list)
    cumulative: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.dimensions:
            self.dimensions = _default_dimensions()
        if not self.region_mapping:
            self.region_mapping = _default_region_mapping()
        if self.cumulative is None:
            self.cumulative = np.zeros(len(self.dimensions))

    @property
    def n_dims(self) -> int:
        return len(self.dimensions)

    def project_block(
        self,
        block_name: str,
        step: int,
        region_name: str,
        features: np.ndarray,
        sae_result: dict | None,
    ) -> CanvasEntry:
        """Project a block's features onto the canvas semantic dimensions.

        The canvas is a fixed interpretive lens — named semantic axes defined
        by domain knowledge.  Coordinates are data-driven: they depend on the
        actual feature DISTRIBUTION (entropy and concentration), not just total
        energy, so different feature patterns produce different canvas positions.

        Args:
            block_name: Name of the source/block/layer.
            step: Processing step number.
            region_name: Feature region key (must be in region_mapping).
            features: The block's feature vector (after projection into shared space).
            sae_result: Output from train_stage_sae() (used for heatmap only).

        Returns:
            CanvasEntry with semantic coordinates.
        """
        coords = np.zeros(self.n_dims)
        canvas_targets = self.region_mapping.get(region_name, [])

        # Data-driven coordinates: use feature distribution, not just total energy.
        # - concentration: how peaked the feature vector is (max / mean)
        # - entropy: how spread out the features are (higher = more diverse)
        # - energy: total activation strength
        # These three properties modulate the canvas coordinates differently.
        abs_feat = np.abs(features)
        energy = float(abs_feat.sum()) + 1e-8
        dim = max(len(features), 1)
        mean_act = energy / dim
        max_act = float(abs_feat.max()) + 1e-8
        concentration = max_act / (mean_act + 1e-8)  # Peaked → high
        # Normalized entropy: 0 = all mass on one feature, 1 = uniform
        probs = abs_feat / energy
        log_probs = np.log(probs + 1e-10)
        entropy = float(-np.sum(probs * log_probs)) / (np.log(dim) + 1e-8)

        for i, (canvas_idx, weight) in enumerate(canvas_targets):
            if canvas_idx >= self.n_dims:
                continue
            # Primary dimensions (weight=1.0): driven by energy * concentration
            # Coupling dimensions (weight<1.0): driven by energy * entropy
            # This means: primary axes respond to strong, focused signals;
            # coupling axes respond to broad, distributed patterns.
            if weight >= 0.99:
                coords[canvas_idx] += weight * mean_act * concentration
            else:
                coords[canvas_idx] += weight * mean_act * (1.0 + entropy)

        # Normalize to [0, 1]
        max_val = coords.max()
        if max_val > 1e-8:
            coords = coords / max_val

        # Dominant dimensions
        top_indices = np.argsort(coords)[-2:][::-1]
        dominant_dims = [self.dimensions[i].key
                         for i in top_indices if coords[i] > 0.01]

        interpretation = self._interpret_coordinates(
            block_name, region_name, coords, features, sae_result,
        )

        active = sae_result["active_count"] if sae_result else 0
        concept_act = sae_result["activations"] if sae_result else np.zeros((1, 8))

        entry = CanvasEntry(
            block_name=block_name,
            step=step,
            coordinates=coords,
            concept_activations=concept_act,
            active_concepts=active,
            dominant_dimensions=dominant_dims,
            interpretation=interpretation,
        )
        self.entries.append(entry)
        self.cumulative = self.cumulative + coords
        return entry

    def _interpret_coordinates(
        self,
        block_name: str,
        region_name: str,
        coords: np.ndarray,
        features: np.ndarray,
        sae_result: dict | None,
    ) -> str:
        """Generate a short algorithmic interpretation of coordinates."""
        parts = [f"[{block_name}]"]

        for i, dim in enumerate(self.dimensions):
            if i < len(coords) and coords[i] > 0.3:
                strength = "strong" if coords[i] > 0.7 else "moderate"
                parts.append(f"{strength} {dim.label} signal ({coords[i]:.2f})")

        if sae_result:
            active = sae_result["active_count"]
            total = len(sae_result["active_mask"])
            parts.append(f"{active}/{total} concepts active")

        return "; ".join(parts)

    def get_accumulated_state(self) -> dict:
        """Return the full canvas state for narrative generation."""
        if not self.entries:
            return {
                "dimensions": [d.to_dict() for d in self.dimensions],
                "coordinates": np.zeros(self.n_dims),
                "entries": [],
                "trajectory": [],
            }

        total = self.cumulative.copy()
        max_val = total.max()
        if max_val > 1e-8:
            total = total / max_val

        trajectory = []
        running = np.zeros(self.n_dims)
        for entry in self.entries:
            running = running + entry.coordinates
            r_norm = running.copy()
            r_max = r_norm.max()
            if r_max > 1e-8:
                r_norm = r_norm / r_max
            trajectory.append({
                "step": entry.step,
                "block": entry.block_name,
                "state": r_norm.copy(),
            })

        top_dims = np.argsort(total)[-3:][::-1]
        dominant_narrative = [
            {"dimension": self.dimensions[i].key,
             "label": self.dimensions[i].label,
             "value": float(total[i]),
             "desc": self.dimensions[i].description}
            for i in top_dims if total[i] > 0.05
        ]

        return {
            "dimensions": [d.to_dict() for d in self.dimensions],
            "coordinates": total,
            "entries": self.entries,
            "trajectory": trajectory,
            "dominant_narrative": dominant_narrative,
        }

    def format_for_narrator(self) -> str:
        """Format the canvas state as structured text for a narrator backend.

        Returns a rich context string that an LLM or template engine can process.
        """
        state = self.get_accumulated_state()
        coords = state["coordinates"]
        entries = state["entries"]

        lines = [
            "=== Semantic Canvas: Cross-Domain Analysis Report ===",
            "",
            "The system analyzed data across multiple sources and layers.",
            "The following semantic coordinates emerged from the analysis:",
            "",
        ]

        for i, dim in enumerate(self.dimensions):
            if i < len(coords):
                val = coords[i]
                if val > 0.05:
                    if val > 0.7:
                        strength = "STRONG"
                    elif val > 0.3:
                        strength = "MODERATE"
                    else:
                        strength = "WEAK"
                    lines.append(f"  {dim.label}: {strength} ({val:.2f}) — {dim.description}")

        lines.append("")
        lines.append("Layer-by-layer findings:")
        lines.append("")

        for entry in entries:
            lines.append(f"  Step {entry.step} ({entry.block_name}): {entry.interpretation}")

        lines.append("")

        dominant = state.get("dominant_narrative", [])
        if dominant:
            dim_names = [d["label"] for d in dominant]
            lines.append(f"The dominant patterns are: {', '.join(dim_names)}.")
            lines.append("")

        lines.append(
            "Based on these cross-domain semantic coordinates, the key finding is that"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Default semantic dimensions (can be overridden for any domain)               #
# --------------------------------------------------------------------------- #

def _default_dimensions() -> list[SemanticDimension]:
    """Return a minimal set of generic semantic dimensions."""
    return [
        SemanticDimension("feature_strength", "Feature Strength",
                          "Overall activation intensity of discovered features."),
        SemanticDimension("feature_diversity", "Feature Diversity",
                          "How many distinct feature patterns are active."),
        SemanticDimension("cross_layer_coupling", "Cross-Layer Coupling",
                          "Degree of feature correlation across network layers."),
        SemanticDimension("sparsity_level", "Sparsity Level",
                          "How concentrated the feature activations are."),
    ]


def _default_region_mapping() -> dict[str, list[tuple[int, float]]]:
    """Return a default region-to-canvas mapping for common region names.

    Maps standard Hyperspace region names to the 4 default canvas dimensions
    so that a bare SemanticCanvas() can project blocks meaningfully.
    """
    return {
        "temporal-pattern": [(0, 1.0), (2, 0.5)],
        "semantic-embedding": [(1, 1.0), (2, 0.6)],
        "structural-centrality": [(2, 1.0), (3, 0.4)],
        "dynamic-agent": [(0, 0.5), (1, 0.7), (3, 1.0)],
        "geospatial-kernel": [(2, 0.6), (3, 1.0)],
    }
