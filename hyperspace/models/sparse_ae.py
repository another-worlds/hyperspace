"""Sparse Autoencoder for unsupervised concept discovery on the UKT."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Minimal sparse autoencoder with L1 sparsity penalty."""

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


def train_sparse_ae(
    combined_features: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 0.005,
) -> dict | None:
    """Train sparse AE on combined block features.

    Args:
        combined_features: (n_samples, feature_dim) array from UKT.
        hidden_dim: Number of latent concepts to discover.
        epochs: Training epochs (kept small for CPU speed).
        lr: Learning rate.

    Returns:
        Dict with concept vectors, activations, active concept count, etc.
        None on failure.
    """
    try:
        if combined_features.shape[0] < 2:
            return None

        input_dim = combined_features.shape[1]
        model = SparseAutoencoder(input_dim, hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X = torch.tensor(combined_features, dtype=torch.float32)

        # Augment with noise copies for more training signal
        noise = torch.randn_like(X) * 0.05
        X_aug = torch.cat([X, X + noise, X - noise], dim=0)

        loss_history = []
        for _ in range(epochs):
            loss, metrics = model.compute_loss(X_aug)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(metrics["total"])

        # Extract learned concepts
        with torch.no_grad():
            _, activations = model.forward(X)
            concept_vectors = model.encoder.weight.detach().cpu().numpy()
            concept_activations = activations.cpu().numpy()

        # Identify active concepts (above mean activation)
        mean_activation = concept_activations.mean(axis=0)
        active_mask = mean_activation > mean_activation.mean()

        # Concept-kernel alignment: map each concept to feature regions
        from hyperspace.models.knowledge_matrix import (
            FEATURE_REGION_LABELS, _feature_name, _region_for_index,
        )
        from hyperspace.config import REGION_DESCRIPTIONS
        concept_labels = []
        for c in range(hidden_dim):
            region_scores = {}
            for (lo, hi), label in FEATURE_REGION_LABELS.items():
                region_scores[label] = float(np.abs(concept_vectors[c, lo:hi]).sum())
            dominant = max(region_scores, key=region_scores.get)

            # Find the top 3 features this concept loads on
            top_idx = np.argsort(np.abs(concept_vectors[c]))[-3:][::-1]
            top_feats = [
                dict(index=int(i), name=_feature_name(i),
                     loading=float(concept_vectors[c, i]))
                for i in top_idx
            ]
            feat_strs = [f"'{f['name']}' ({f['loading']:+.3f})" for f in top_feats]

            # Generate narrative
            status = "ACTIVE" if active_mask[c] else "dormant"
            narrative = (
                f"Concept C{c:02d} [{status}] — primarily encodes "
                f"{dominant.replace('-', ' ')} information "
                f"(activation: {mean_activation[c]:.4f}).\n"
                f"Top feature loadings: {', '.join(feat_strs)}.\n"
                f"Region: {REGION_DESCRIPTIONS.get(dominant, dominant)}"
            )

            concept_labels.append(dict(
                concept_id=f"C{c:02d}",
                dominant_region=dominant,
                mean_activation=float(mean_activation[c]),
                active=bool(active_mask[c]),
                top_features=top_feats,
                label=f"C{c:02d}: {dominant.replace('-', ' ')} ({status})",
                narrative=narrative,
            ))

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


def map_concepts_to_kernels(
    sae_result: dict,
    kernel_snapshot: dict,
) -> list[dict]:
    """Map SAE concepts to UKT kernels via projection.

    Returns list of dicts with concept-kernel correspondences.
    """
    try:
        concept_vectors = sae_result["concept_vectors"]   # (n_concepts, input_dim)
        Vt = kernel_snapshot["Vt"]                         # (n_kernels, feature_dim)
        importance = kernel_snapshot["importance"]

        # Project concept vectors into kernel space
        # Similarity = concepts @ kernels^T  -> (n_concepts, n_kernels)
        similarity = concept_vectors @ Vt.T

        rows = []
        for c in range(concept_vectors.shape[0]):
            best_kernel = int(np.argmax(np.abs(similarity[c])))
            coherence = float(np.abs(similarity[c, best_kernel]))
            mean_act = float(sae_result["mean_activation"][c])
            rows.append(dict(
                concept=f"C{c:02d}",
                best_kernel=f"K{best_kernel}",
                kernel_importance=float(importance[best_kernel]) if best_kernel < len(importance) else 0.0,
                coherence=round(coherence, 4),
                mean_activation=round(mean_act, 4),
                active=mean_act > sae_result["mean_activation"].mean(),
            ))
        return rows
    except Exception:
        return []
