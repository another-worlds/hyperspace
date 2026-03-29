"""Cross-modal contrastive alignment encoders (SPEC-4).

Per-block MLP encoders map block-specific features into a shared 32-dim latent
space.  An InfoNCE contrastive objective pulls semantically related block
representations together.  The contrastive kernels are blended with the SVD
kernels via ``CONTRASTIVE_WEIGHT``.

Disabled by default (``ENABLE_CONTRASTIVE_ALIGNMENT=False``).  When enabled,
runs as a parallel path alongside SVD — never replaces it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from hyperspace.config import (
    CONTRASTIVE_LATENT_DIM,
    CONTRASTIVE_TEMPERATURE,
)


# --------------------------------------------------------------------------- #
# Block Encoder (pure numpy MLP — no torch dependency at import time)          #
# --------------------------------------------------------------------------- #

class BlockEncoder:
    """Per-block MLP encoder: in_dim → 64 → ReLU → Dropout(0.1) → latent_dim.

    Uses numpy for CPU-only inference.  Weights initialised with Xavier uniform.
    """

    def __init__(self, in_dim: int, latent_dim: int = CONTRASTIVE_LATENT_DIM, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Xavier uniform initialisation
        limit1 = np.sqrt(6.0 / (in_dim + 64))
        self.W1 = rng.uniform(-limit1, limit1, (in_dim, 64)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)

        limit2 = np.sqrt(6.0 / (64 + latent_dim))
        self.W2 = rng.uniform(-limit2, limit2, (64, latent_dim)).astype(np.float32)
        self.b2 = np.zeros(latent_dim, dtype=np.float32)

        self._latent_dim = latent_dim
        self._dropout_rate = 0.1
        self._training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: (in_dim,) → (latent_dim,)."""
        h = x @ self.W1 + self.b1
        h = np.maximum(h, 0)  # ReLU
        if self._training:
            mask = (np.random.random(h.shape) > self._dropout_rate).astype(np.float32)
            h = h * mask / (1 - self._dropout_rate)
        z = h @ self.W2 + self.b2
        # L2 normalise
        norm = np.linalg.norm(z)
        if norm > 1e-8:
            z = z / norm
        return z

    def parameters(self) -> list[np.ndarray]:
        """Return all trainable parameters."""
        return [self.W1, self.b1, self.W2, self.b2]


# --------------------------------------------------------------------------- #
# Contrastive Encoder Bank                                                     #
# --------------------------------------------------------------------------- #

# Block name → (start_idx, end_idx) in the 80-dim UKT space
BLOCK_REGIONS: dict[str, tuple[int, int]] = {
    "Finance":  (0, 16),
    "Clusters": (16, 32),
    "Graph":    (32, 48),
    "Agents":   (48, 64),
    "Spatial":  (64, 80),
}


class ContrastiveEncoderBank:
    """Manages per-block encoders, InfoNCE training, and alignment scoring."""

    def __init__(self, latent_dim: int = CONTRASTIVE_LATENT_DIM, seed: int = 42) -> None:
        self._latent_dim = latent_dim
        self.encoders: dict[str, BlockEncoder] = {}
        for block_name, (start, end) in BLOCK_REGIONS.items():
            in_dim = end - start
            self.encoders[block_name] = BlockEncoder(in_dim, latent_dim, seed=seed)
        self._lr = 0.001
        self._step_count = 0

    def encode(self, block_name: str, features: np.ndarray) -> np.ndarray:
        """Encode block features → (latent_dim,) numpy array."""
        if block_name not in self.encoders:
            return np.zeros(self._latent_dim)
        start, end = BLOCK_REGIONS[block_name]
        block_features = features[start:end].astype(np.float32)
        return self.encoders[block_name].forward(block_features)

    def train_step(self, full_features: np.ndarray) -> float:
        """One training step using InfoNCE loss across all block pairs.

        Args:
            full_features: (80,) UKT feature vector.

        Returns:
            InfoNCE loss value.
        """
        # Encode all blocks
        embeddings: dict[str, np.ndarray] = {}
        for block_name in self.encoders:
            self.encoders[block_name]._training = True
            embeddings[block_name] = self.encode(block_name, full_features)
            self.encoders[block_name]._training = False

        block_names = list(embeddings.keys())
        n = len(block_names)
        if n < 2:
            return 0.0

        # InfoNCE: for each pair, compute similarity
        tau = CONTRASTIVE_TEMPERATURE
        total_loss = 0.0
        n_pairs = 0

        for i in range(n):
            z_i = embeddings[block_names[i]]
            # Positive: all other blocks (in contrastive learning, same-run
            # blocks are considered positive pairs)
            sims = []
            for j in range(n):
                if j == i:
                    continue
                z_j = embeddings[block_names[j]]
                sim = float(np.dot(z_i, z_j) / (tau + 1e-12))
                sims.append(sim)

            if sims:
                # Log-sum-exp for numerical stability
                max_sim = max(sims)
                log_sum_exp = max_sim + np.log(sum(np.exp(s - max_sim) for s in sims))
                # Loss: -log(exp(positive) / sum(exp(all)))
                # Since all are positive pairs, we use mean
                loss = -np.mean(sims) + log_sum_exp
                total_loss += loss
                n_pairs += 1

        avg_loss = total_loss / (n_pairs + 1e-12)

        # Simple SGD update (gradient approximation via finite differences)
        eps = 1e-4
        for block_name, encoder in self.encoders.items():
            for param in encoder.parameters():
                grad = np.zeros_like(param)
                # Stochastic gradient estimation: perturb a random subset
                n_perturb = min(10, param.size)
                indices = np.random.choice(param.size, n_perturb, replace=False)
                flat = param.flatten()
                for idx in indices:
                    old_val = flat[idx]
                    flat[idx] = old_val + eps
                    loss_plus = self._compute_loss(full_features)
                    flat[idx] = old_val - eps
                    loss_minus = self._compute_loss(full_features)
                    flat[idx] = old_val
                    grad.flat[idx] = (loss_plus - loss_minus) / (2 * eps)
                param -= self._lr * grad

        self._step_count += 1
        return avg_loss

    def _compute_loss(self, full_features: np.ndarray) -> float:
        """Compute InfoNCE loss for gradient estimation."""
        embeddings = {}
        for block_name in self.encoders:
            embeddings[block_name] = self.encode(block_name, full_features)

        block_names = list(embeddings.keys())
        n = len(block_names)
        tau = CONTRASTIVE_TEMPERATURE
        total_loss = 0.0
        n_pairs = 0

        for i in range(n):
            z_i = embeddings[block_names[i]]
            sims = []
            for j in range(n):
                if j == i:
                    continue
                z_j = embeddings[block_names[j]]
                sims.append(float(np.dot(z_i, z_j) / (tau + 1e-12)))
            if sims:
                max_sim = max(sims)
                log_sum_exp = max_sim + np.log(sum(np.exp(s - max_sim) for s in sims))
                total_loss += -np.mean(sims) + log_sum_exp
                n_pairs += 1

        return total_loss / (n_pairs + 1e-12)

    def alignment_score(self, full_features: np.ndarray) -> float:
        """Mean pairwise cosine similarity across all encoded block pairs (0-1)."""
        embeddings = {}
        for block_name in self.encoders:
            embeddings[block_name] = self.encode(block_name, full_features)

        block_names = list(embeddings.keys())
        n = len(block_names)
        if n < 2:
            return 0.0

        total_cos = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                cos = float(np.dot(embeddings[block_names[i]], embeddings[block_names[j]]))
                total_cos += cos
                n_pairs += 1

        return total_cos / (n_pairs + 1e-12)

    def get_contrastive_kernels(self, full_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode all blocks and run SVD on the encoded matrix.

        Returns:
            (importance, Vt) from SVD of the (n_blocks, latent_dim) encoded matrix.
        """
        rows = []
        for block_name in BLOCK_REGIONS:
            z = self.encode(block_name, full_features)
            rows.append(z)

        encoded_matrix = np.array(rows)  # (n_blocks, latent_dim)
        U, S, Vt = np.linalg.svd(encoded_matrix, full_matrices=False)
        importance = S / (S.sum() + 1e-12)
        return importance, Vt
