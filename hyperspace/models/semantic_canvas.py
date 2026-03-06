"""Semantic Canvas: a unified interpretive space for cross-layer semantic coordinates.

The Semantic Canvas accumulates structured semantic contributions from each pipeline
stage. Instead of leaving interpretations buried in raw hidden-layer activations,
each stage's SAE projects its discovered concepts onto named semantic dimensions.

Architecture:
    Pipeline Stage → Hidden Activations → Stage SAE → Sparse Concepts → Canvas Projection
    Canvas accumulates across depth → Tiny-LLM reads structured canvas → Narrative

The canvas has named semantic axes (e.g. "market_momentum", "information_focus",
"alliance_polarity") that provide a human-interpretable coordinate system for
the model's internal representations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Semantic dimensions: the named axes of the canvas                           #
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
]

CANVAS_DIM = len(CANVAS_DIMENSIONS)

# Map: which UKT feature regions project onto which canvas dimensions
# Each entry: (canvas_dim_index, [(ukt_feature_start, ukt_feature_end, weight)])
REGION_TO_CANVAS: dict[str, list[tuple[int, float]]] = {
    # temporal-pattern [0:16] → market_momentum, temporal_memory, volatility_regime
    "temporal-pattern": [
        (0, 1.0),   # market_momentum
        (1, 1.0),   # temporal_memory
        (2, 0.8),   # volatility_regime
    ],
    # semantic-embedding [16:32] → information_focus, narrative_diversity
    "semantic-embedding": [
        (3, 1.0),   # information_focus
        (4, 1.0),   # narrative_diversity
    ],
    # structural-centrality [32:48] → alliance_polarity, network_cohesion
    "structural-centrality": [
        (5, 1.0),   # alliance_polarity
        (6, 1.0),   # network_cohesion
        (7, 0.5),   # power_concentration
    ],
    # dynamic-agent [48:64] → power_concentration, cooperation, competition
    "dynamic-agent": [
        (7, 0.5),   # power_concentration (shared with structural)
        (10, 1.0),  # cooperation_signal
        (11, 1.0),  # competition_signal
    ],
    # geospatial-kernel [64:80] → geographic_coupling, systemic_stress
    "geospatial-kernel": [
        (8, 1.0),   # geographic_coupling
        (9, 1.0),   # systemic_stress
    ],
}


# --------------------------------------------------------------------------- #
# Per-stage Sparse Autoencoder: lightweight SAE for each pipeline block       #
# --------------------------------------------------------------------------- #

class StageSAE(nn.Module):
    """Lightweight SAE attached to a single pipeline stage.

    Observes the stage's feature vector (16-dim) and learns sparse concepts
    that are then projected onto the semantic canvas.
    """

    def __init__(self, input_dim: int = 16, concept_dim: int = 8,
                 sparsity_weight: float = 0.02):
        super().__init__()
        self.encoder = nn.Linear(input_dim, concept_dim)
        self.decoder = nn.Linear(concept_dim, input_dim)
        self.sparsity_weight = sparsity_weight
        self.concept_dim = concept_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.encoder(x))
        x_hat = self.decoder(h)
        return x_hat, h

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x_hat, h = self.forward(x)
        recon = nn.functional.mse_loss(x_hat, x)
        sparsity = self.sparsity_weight * h.abs().mean()
        total = recon + sparsity
        return total, {"recon": recon.item(), "sparsity": sparsity.item(),
                       "total": total.item()}


def train_stage_sae(
    features: np.ndarray,
    concept_dim: int = 8,
    epochs: int = 60,
    lr: float = 0.008,
) -> dict | None:
    """Train a per-stage SAE on a single block's feature region.

    Args:
        features: (1, region_dim) or (n, region_dim) feature array from one block.
        concept_dim: Number of sparse concepts to discover per stage.
        epochs: Training epochs.
        lr: Learning rate.

    Returns:
        Dict with concept_vectors, activations, active flags, etc.
    """
    try:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        input_dim = features.shape[1]
        model = StageSAE(input_dim, concept_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X = torch.tensor(features, dtype=torch.float32)
        # Augment with noise for richer training signal
        noise_scales = [0.03, 0.06, 0.1]
        augmented = [X]
        for s in noise_scales:
            augmented.append(X + torch.randn_like(X) * s)
        X_aug = torch.cat(augmented, dim=0)

        for _ in range(epochs):
            loss, _ = model.compute_loss(X_aug)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, activations = model.forward(X)
            concept_vectors = model.encoder.weight.detach().cpu().numpy()
            act_np = activations.cpu().numpy()

        mean_act = act_np.mean(axis=0)
        active_mask = mean_act >= mean_act.mean()

        return {
            "concept_vectors": concept_vectors,   # (concept_dim, input_dim)
            "activations": act_np,                 # (n_samples, concept_dim)
            "mean_activation": mean_act,
            "active_mask": active_mask,
            "active_count": int(active_mask.sum()),
            "model": model,
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Semantic Canvas                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class CanvasEntry:
    """One layer's contribution to the semantic canvas."""
    block_name: str
    step: int
    coordinates: np.ndarray          # (CANVAS_DIM,) — this layer's semantic position
    concept_activations: np.ndarray  # raw concept activations from stage SAE
    active_concepts: int
    dominant_dimensions: list[str]   # top-2 canvas dimension keys for this layer
    interpretation: str              # short algorithmic interpretation before LLM


@dataclass
class SemanticCanvas:
    """Unified interpretive space accumulating semantic contributions across depth.

    Each pipeline stage projects its discovered concepts onto named semantic
    dimensions. The canvas grows as layers are added, building a trajectory
    of semantic meaning through the pipeline.
    """
    entries: list[CanvasEntry] = field(default_factory=list)
    cumulative: np.ndarray = field(default_factory=lambda: np.zeros(CANVAS_DIM))

    def project_block(
        self,
        block_name: str,
        step: int,
        region_name: str,
        features: np.ndarray,
        sae_result: dict | None,
    ) -> CanvasEntry:
        """Project a block's features and SAE concepts onto the canvas.

        Args:
            block_name: Name of the pipeline block (e.g. "Finance").
            step: Pipeline step number.
            region_name: UKT region key (e.g. "temporal-pattern").
            features: The block's raw feature vector (16-dim).
            sae_result: Output from train_stage_sae() for this block.

        Returns:
            CanvasEntry with the block's semantic coordinates.
        """
        coords = np.zeros(CANVAS_DIM)
        canvas_targets = REGION_TO_CANVAS.get(region_name, [])

        if sae_result is not None:
            # Use SAE concept activations to weight the projection
            mean_act = sae_result["mean_activation"]
            concept_energy = float(mean_act.sum()) + 1e-8
            # Normalize activations to get concept importance distribution
            concept_weights = mean_act / concept_energy

            # Project: each concept's energy is distributed to canvas dims
            # weighted by how strongly this region maps to each canvas axis
            feature_energy = float(np.abs(features).sum()) + 1e-8
            for canvas_idx, weight in canvas_targets:
                coords[canvas_idx] += weight * concept_energy / feature_energy
        else:
            # Fallback: direct feature energy projection
            feature_energy = float(np.abs(features).sum()) + 1e-8
            for canvas_idx, weight in canvas_targets:
                coords[canvas_idx] += weight * (feature_energy / 16.0)

        # Normalize to [0, 1] range per dimension
        max_val = coords.max()
        if max_val > 1e-8:
            coords = coords / max_val

        # Determine dominant dimensions
        top_indices = np.argsort(coords)[-2:][::-1]
        dominant_dims = [CANVAS_DIMENSIONS[i]["key"]
                         for i in top_indices if coords[i] > 0.01]

        # Generate short algorithmic interpretation
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

        for i, dim in enumerate(CANVAS_DIMENSIONS):
            if coords[i] > 0.3:
                strength = "strong" if coords[i] > 0.7 else "moderate"
                parts.append(f"{strength} {dim['label']} signal ({coords[i]:.2f})")

        if sae_result:
            active = sae_result["active_count"]
            total = len(sae_result["active_mask"])
            parts.append(f"{active}/{total} concepts active")

        return "; ".join(parts)

    def get_accumulated_state(self) -> dict:
        """Return the full canvas state for narrative generation."""
        if not self.entries:
            return {"dimensions": CANVAS_DIMENSIONS, "coordinates": np.zeros(CANVAS_DIM),
                    "entries": [], "trajectory": []}

        # Normalize cumulative to [0, 1]
        total = self.cumulative.copy()
        max_val = total.max()
        if max_val > 1e-8:
            total = total / max_val

        trajectory = []
        running = np.zeros(CANVAS_DIM)
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

        # Identify the dominant narrative thread
        top_dims = np.argsort(total)[-3:][::-1]
        dominant_narrative = [
            {"dimension": CANVAS_DIMENSIONS[i]["key"],
             "label": CANVAS_DIMENSIONS[i]["label"],
             "value": float(total[i]),
             "desc": CANVAS_DIMENSIONS[i]["desc"]}
            for i in top_dims if total[i] > 0.05
        ]

        return {
            "dimensions": CANVAS_DIMENSIONS,
            "coordinates": total,
            "entries": self.entries,
            "trajectory": trajectory,
            "dominant_narrative": dominant_narrative,
        }

    def format_for_narrator(self) -> str:
        """Format the canvas state as structured text for the Tiny-LLM narrator.

        Returns a rich context string that the LLM can continue/complete.
        """
        state = self.get_accumulated_state()
        coords = state["coordinates"]
        entries = state["entries"]

        lines = [
            "=== Semantic Canvas: Cross-Domain Analysis Report ===",
            "",
            "The system analyzed data across financial markets, news information flows,",
            "geopolitical alliance structures, agent-based simulations, and physical-economic",
            "spatial data. The following semantic coordinates emerged from the analysis:",
            "",
        ]

        # Semantic coordinates
        for i, dim in enumerate(CANVAS_DIMENSIONS):
            val = coords[i]
            if val > 0.05:
                if val > 0.7:
                    strength = "STRONG"
                elif val > 0.3:
                    strength = "MODERATE"
                else:
                    strength = "WEAK"
                lines.append(f"  {dim['label']}: {strength} ({val:.2f}) — {dim['desc']}")

        lines.append("")
        lines.append("Layer-by-layer findings:")
        lines.append("")

        # Per-layer summaries
        for entry in entries:
            lines.append(f"  Step {entry.step} ({entry.block_name}): {entry.interpretation}")

        lines.append("")

        # Dominant narrative thread
        dominant = state.get("dominant_narrative", [])
        if dominant:
            dim_names = [d["label"] for d in dominant]
            lines.append(
                f"The dominant patterns are: {', '.join(dim_names)}."
            )
            lines.append("")

        lines.append(
            "Based on these cross-domain semantic coordinates, the key finding is that"
        )
        return "\n".join(lines)
