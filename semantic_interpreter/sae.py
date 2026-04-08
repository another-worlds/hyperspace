"""Sparse Autoencoders for concept discovery in neural network hidden layers.

SAEs decompose opaque hidden-layer activations into a small number of
interpretable "concepts." The sparsity constraint ensures that most concepts
are dormant for any given input — only the relevant ones activate. This is
what makes neural network internals interpretable.

Two variants:
  - StageSAE: lightweight, per-layer SAE (8 concepts per layer)
  - GlobalSAE: larger SAE trained on the full UKT matrix (32+ concepts)

Both are network-agnostic: they work on any feature vector regardless of
which architecture produced it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Stage SAE: per-layer concept discovery                                       #
# --------------------------------------------------------------------------- #

class StageSAE(nn.Module):
    """Lightweight SAE for a single network layer/source.

    Observes a layer's feature vector and learns sparse concepts that can
    be projected onto the semantic canvas.

    Args:
        input_dim: Dimension of the input feature vector.
        concept_dim: Number of sparse concepts to discover.
        sparsity_weight: L1 penalty weight on concept activations.
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


class GlobalSAE(nn.Module):
    """Larger SAE for concept discovery on the full feature matrix.

    Trained on the combined (n_blocks, feature_dim) UKT matrix to discover
    global cross-source concepts.

    Args:
        input_dim: Full feature dimension.
        hidden_dim: Number of latent concepts.
        sparsity_weight: L1 penalty weight.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32,
                 sparsity_weight: float = 0.01):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_weight = sparsity_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.encoder(x))
        x_hat = self.decoder(h)
        return x_hat, h

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x_hat, h = self.forward(x)
        recon_loss = nn.functional.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_weight * h.abs().mean()
        total = recon_loss + sparsity_loss
        return total, {
            "recon": recon_loss.item(),
            "sparsity": sparsity_loss.item(),
            "total": total.item(),
        }


# --------------------------------------------------------------------------- #
# Training functions                                                           #
# --------------------------------------------------------------------------- #

def train_stage_sae(
    features: np.ndarray,
    concept_dim: int = 8,
    epochs: int = 60,
    lr: float = 0.008,
) -> dict | None:
    """Train a per-stage SAE on a single layer/source's feature region.

    Args:
        features: (1, region_dim) or (n, region_dim) feature array.
        concept_dim: Number of sparse concepts to discover.
        epochs: Training epochs.
        lr: Learning rate.

    Returns:
        Dict with concept_vectors, activations, active flags, etc.
        None on failure.
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
            "concept_vectors": concept_vectors,
            "activations": act_np,
            "mean_activation": mean_act,
            "active_mask": active_mask,
            "active_count": int(active_mask.sum()),
            "model": model,
        }
    except Exception:
        return None


def train_global_sae(
    combined_features: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 0.005,
    registry: Any = None,
) -> dict | None:
    """Train a global SAE on the combined feature matrix from all sources.

    Args:
        combined_features: (n_samples, feature_dim) array from UKT.
        hidden_dim: Number of latent concepts.
        epochs: Training epochs.
        lr: Learning rate.
        registry: Optional FeatureRegionRegistry for concept labeling.

    Returns:
        Dict with concept vectors, activations, labels, loss history, etc.
        None on failure.
    """
    try:
        if combined_features.shape[0] < 2:
            return None

        input_dim = combined_features.shape[1]
        model = GlobalSAE(input_dim, hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X = torch.tensor(combined_features, dtype=torch.float32)
        noise = torch.randn_like(X) * 0.05
        X_aug = torch.cat([X, X + noise, X - noise], dim=0)

        loss_history = []
        for _ in range(epochs):
            loss, metrics = model.compute_loss(X_aug)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(metrics["total"])

        with torch.no_grad():
            _, activations = model.forward(X)
            concept_vectors = model.encoder.weight.detach().cpu().numpy()
            concept_activations = activations.cpu().numpy()

        mean_activation = concept_activations.mean(axis=0)
        active_mask = mean_activation >= mean_activation.mean()

        # Build concept labels
        concept_labels = _label_concepts(
            concept_vectors, mean_activation, active_mask,
            hidden_dim, registry,
        )

        return dict(
            concept_vectors=concept_vectors,
            concept_activations=concept_activations,
            mean_activation=mean_activation,
            active_concepts=int(active_mask.sum()),
            total_concepts=hidden_dim,
            final_loss=loss_history[-1] if loss_history else 0.0,
            loss_history=loss_history,
            concept_labels=concept_labels,
        )
    except Exception:
        return None


def _label_concepts(
    concept_vectors: np.ndarray,
    mean_activation: np.ndarray,
    active_mask: np.ndarray,
    hidden_dim: int,
    registry: Any = None,
) -> list[dict]:
    """Generate labels for discovered concepts.

    Uses the registry (if provided) to map concepts to named feature regions.
    Falls back to generic labels if no registry is available.
    """
    concept_labels = []
    for c in range(hidden_dim):
        # Preserve full cross-block signature (VISION invariant 10)
        cross_block_signature = {}
        dominant_region_hint = "features"  # Provenance hint only
        region_desc = ""
        top_features: list[dict] = []

        if registry is not None:
            # Compute signed contributions per region (preserve structure)
            for name, region in registry.regions.items():
                # Use signed sum to preserve direction, not just magnitude
                signed_contrib = float(concept_vectors[c, region.start:region.end].sum())
                cross_block_signature[name] = signed_contrib
            
            # Dominant region is just a provenance hint (never used as semantic label)
            if cross_block_signature:
                region_scores = {name: abs(contrib) for name, contrib in cross_block_signature.items()}
                dominant_region_hint = max(region_scores, key=region_scores.get)
                region_obj = registry.regions.get(dominant_region_hint)
                region_desc = region_obj.description if region_obj else ""

        # Top 3 feature loadings
        top_idx = np.argsort(np.abs(concept_vectors[c]))[-3:][::-1]
        for i in top_idx:
            name = registry.feature_name(int(i)) if registry else f"feature_{i}"
            top_features.append(dict(
                index=int(i),
                name=name,
                loading=float(concept_vectors[c, int(i)]),
            ))

        feat_strs = [f"'{f['name']}' ({f['loading']:+.3f})" for f in top_features]
        status = "ACTIVE" if active_mask[c] else "dormant"
        
        # Generate provenance-only narrative (no interpretive claims)
        narrative = (
            f"Concept C{c:02d} [{status}] — activation: {mean_activation[c]:.4f}. "
            f"Top feature loadings: {', '.join(feat_strs)}."
        )
        if region_desc:
            narrative += f" Provenance: {region_desc}"

        concept_labels.append(dict(
            concept_id=f"C{c:02d}",
            concept_idx=c,
            cross_block_signature=cross_block_signature,  # Full structured signature
            dominant_region_hint=dominant_region_hint,   # Provenance only
            mean_activation=float(mean_activation[c]),
            active=bool(active_mask[c]),
            top_features=top_features,
            label=f"C{c:02d} ({status})",  # No block name in label
            narrative=narrative,
        ))

    return concept_labels
