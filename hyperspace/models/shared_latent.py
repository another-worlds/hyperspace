"""Shared-latent shadow model for cross-modality alignment diagnostics.

This module projects modality-specific UKT features into a shared latent space
and reports alignment metrics. It is explicitly designed for shadow evaluation
and does not alter the legacy 80-d production path.

Architecture (v2 — nonlinear):
  Per-modality encoder: input -> Linear -> ReLU -> Linear -> L2-norm
  Contrastive training: InfoNCE-style loss over paired windows
  Metrics: retrieval@1, probe cosine, contrastive loss curve
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    """Element-wise ReLU activation."""
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 where x > 0, else 0."""
    return (x > 0).astype(x.dtype)


@dataclass
class SharedLatentHead:
    """Per-modality two-layer encoder with ReLU nonlinearity.

    Forward pass: x -> (x @ W1 + b1) -> ReLU -> (h @ W2 + b2) -> L2-norm
    """

    W1: np.ndarray      # (input_dim, hidden_dim)
    b1: np.ndarray      # (hidden_dim,)
    W2: np.ndarray      # (hidden_dim, latent_dim)
    b2: np.ndarray      # (latent_dim,)


@dataclass
class SharedLatentModel:
    """Collection of modality heads for shared-latent projection."""

    heads: dict[str, SharedLatentHead]
    latent_dim: int

    def encode(self, modality: str, x: np.ndarray) -> np.ndarray:
        """Project modality features into shared latent space.

        Returns L2-normalized embeddings of shape (n_samples, latent_dim).
        """
        head = self.heads[modality]
        h = _relu(x @ head.W1 + head.b1)
        z = h @ head.W2 + head.b2
        norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        return z / norm

    def _forward_with_cache(
        self, modality: str, x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning intermediate activations for backprop."""
        head = self.heads[modality]
        pre_relu = x @ head.W1 + head.b1
        h = _relu(pre_relu)
        z_raw = h @ head.W2 + head.b2
        norm = np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-8
        z = z_raw / norm
        return z, h, pre_relu


def _windowed_views(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    rows = []
    for start in range(0, max(1, x.shape[0] - window + 1), stride):
        rows.append(x[start:start + window])
    return np.asarray(rows)


def _prepare_aligned_windows(
    matrix: np.ndarray,
    block_names: list[str],
    window: int = 16,
    stride: int = 8,
) -> dict[str, np.ndarray]:
    by_modality: dict[str, np.ndarray] = {}
    for i, name in enumerate(block_names):
        wins = _windowed_views(matrix[i], window=window, stride=stride)
        by_modality[name] = wins
    return by_modality


def _build_model(
    modalities: list[str],
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    seed: int,
) -> SharedLatentModel:
    """Build a model with Xavier-initialized two-layer nonlinear encoders."""
    rng = np.random.default_rng(seed)
    heads: dict[str, SharedLatentHead] = {}

    # Xavier initialization scale
    scale_w1 = np.sqrt(2.0 / (input_dim + hidden_dim))
    scale_w2 = np.sqrt(2.0 / (hidden_dim + latent_dim))

    for m in modalities:
        heads[m] = SharedLatentHead(
            W1=rng.normal(0, scale_w1, size=(input_dim, hidden_dim)),
            b1=np.zeros(hidden_dim),
            W2=rng.normal(0, scale_w2, size=(hidden_dim, latent_dim)),
            b2=np.zeros(latent_dim),
        )
    return SharedLatentModel(heads=heads, latent_dim=latent_dim)


def _contrastive_epoch(
    model: SharedLatentModel,
    aligned_windows: dict[str, np.ndarray],
    learning_rate: float,
    temperature: float,
    weight_decay: float = 1e-4,
) -> float:
    """Run one contrastive training epoch with gradient-based updates.

    Uses InfoNCE loss with analytic gradients through the two-layer encoder.
    Includes L2 weight decay for regularization.
    """
    losses = []
    modalities = list(aligned_windows.keys())

    for ma, mb in combinations(modalities, 2):
        xa = aligned_windows[ma]
        xb = aligned_windows[mb]
        n = min(len(xa), len(xb))
        if n < 2:
            continue
        xa = xa[:n]
        xb = xb[:n]

        # Forward pass with cache for gradient computation
        za, ha, pre_relu_a = model._forward_with_cache(ma, xa)
        zb, hb, pre_relu_b = model._forward_with_cache(mb, xb)

        # InfoNCE loss
        sims = (za @ zb.T) / max(temperature, 1e-6)
        # Numerically stable softmax
        sims_max = sims.max(axis=1, keepdims=True)
        exp_sims = np.exp(sims - sims_max)
        softmax = exp_sims / (exp_sims.sum(axis=1, keepdims=True) + 1e-8)

        pos = np.diag(sims)
        denom = np.log(exp_sims.sum(axis=1) + 1e-8) + sims_max.squeeze()
        loss = float(np.mean(-(pos - denom)))
        losses.append(loss)

        # Gradient of loss w.r.t. za: d_loss/d_za_i = (softmax_i - e_i) @ zb / (n * tau)
        # where e_i is one-hot at position i
        targets = np.eye(n)
        grad_za = (softmax - targets) @ zb / (n * max(temperature, 1e-6))
        grad_zb = (softmax - targets).T @ za / (n * max(temperature, 1e-6))

        # Backprop through encoder a
        head_a = model.heads[ma]
        grad_W2_a = ha.T @ grad_za / n
        grad_b2_a = grad_za.mean(axis=0)
        grad_ha = grad_za @ head_a.W2.T
        grad_pre_a = grad_ha * _relu_grad(pre_relu_a)
        grad_W1_a = xa.T @ grad_pre_a / n
        grad_b1_a = grad_pre_a.mean(axis=0)

        # Backprop through encoder b
        head_b = model.heads[mb]
        grad_W2_b = hb.T @ grad_zb / n
        grad_b2_b = grad_zb.mean(axis=0)
        grad_hb = grad_zb @ head_b.W2.T
        grad_pre_b = grad_hb * _relu_grad(pre_relu_b)
        grad_W1_b = xb.T @ grad_pre_b / n
        grad_b1_b = grad_pre_b.mean(axis=0)

        # Update with weight decay
        head_a.W1 -= learning_rate * (grad_W1_a + weight_decay * head_a.W1)
        head_a.b1 -= learning_rate * grad_b1_a
        head_a.W2 -= learning_rate * (grad_W2_a + weight_decay * head_a.W2)
        head_a.b2 -= learning_rate * grad_b2_a

        head_b.W1 -= learning_rate * (grad_W1_b + weight_decay * head_b.W1)
        head_b.b1 -= learning_rate * grad_b1_b
        head_b.W2 -= learning_rate * (grad_W2_b + weight_decay * head_b.W2)
        head_b.b2 -= learning_rate * grad_b2_b

    return float(np.mean(losses)) if losses else 0.0


def _pairwise_retrieval_at_1(embeddings: dict[str, np.ndarray]) -> float:
    scores = []
    for ma, mb in combinations(embeddings.keys(), 2):
        ea, eb = embeddings[ma], embeddings[mb]
        n = min(len(ea), len(eb))
        if n == 0:
            continue
        sim = ea[:n] @ eb[:n].T
        pred = np.argmax(sim, axis=1)
        truth = np.arange(n)
        scores.append(float((pred == truth).mean()))
    return float(np.mean(scores)) if scores else 0.0


def _pairwise_probe_cosine(embeddings: dict[str, np.ndarray]) -> float:
    vals = []
    for ma, mb in combinations(embeddings.keys(), 2):
        ea, eb = embeddings[ma], embeddings[mb]
        n = min(len(ea), len(eb))
        if n == 0:
            continue
        cos = np.sum(ea[:n] * eb[:n], axis=1)
        vals.append(float(np.mean(cos)))
    return float(np.mean(vals)) if vals else 0.0


def compute_alignment_metrics(
    matrix: np.ndarray,
    block_names: list[str],
    *,
    latent_dim: int = 16,
    hidden_dim: int = 24,
    window: int = 16,
    stride: int = 8,
    epochs: int = 60,
    learning_rate: float = 0.005,
    temperature: float = 0.2,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict[str, object]:
    """Compute side-by-side legacy and shared-latent alignment metrics.

    The shared-latent path uses nonlinear two-layer encoders with ReLU
    activation, trained via InfoNCE contrastive loss with proper gradients
    and weight decay regularization.
    """
    aligned = _prepare_aligned_windows(matrix, block_names, window=window, stride=stride)

    legacy_embeddings = {
        m: w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
        for m, w in aligned.items()
    }
    legacy = {
        "retrieval_at_1": round(_pairwise_retrieval_at_1(legacy_embeddings), 4),
        "probe_cosine": round(_pairwise_probe_cosine(legacy_embeddings), 4),
    }

    model = _build_model(
        list(aligned.keys()),
        input_dim=window,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seed=seed,
    )
    loss_curve = []
    for _ in range(max(1, epochs)):
        loss_curve.append(
            _contrastive_epoch(model, aligned, learning_rate, temperature, weight_decay)
        )

    shared_embeddings = {m: model.encode(m, w) for m, w in aligned.items()}
    shared_latent = {
        "retrieval_at_1": round(_pairwise_retrieval_at_1(shared_embeddings), 4),
        "probe_cosine": round(_pairwise_probe_cosine(shared_embeddings), 4),
        "contrastive_loss_final": round(float(loss_curve[-1]) if loss_curve else 0.0, 4),
        "contrastive_loss_initial": round(float(loss_curve[0]) if loss_curve else 0.0, 4),
        "training_epochs": len(loss_curve),
        "training_windows": int(min((len(v) for v in aligned.values()), default=0)),
        "encoder_type": "nonlinear_2layer_relu",
        "shadow_only": True,
    }

    return {
        "legacy": legacy,
        "shared_latent": shared_latent,
        "parity_delta": {
            "retrieval_at_1": round(shared_latent["retrieval_at_1"] - legacy["retrieval_at_1"], 4),
            "probe_cosine": round(shared_latent["probe_cosine"] - legacy["probe_cosine"], 4),
        },
    }
