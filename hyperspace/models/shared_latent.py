"""Shared-latent shadow model for cross-modality alignment diagnostics.

This module adds a minimal prototype path that projects modality-specific UKT
features into a shared latent space and reports alignment metrics. It is
explicitly designed for shadow evaluation and does not alter the legacy 80-d
production path.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class SharedLatentHead:
    """Per-modality linear encoder + projector head."""

    encoder: np.ndarray
    projector: np.ndarray


@dataclass
class SharedLatentModel:
    """Collection of modality heads for shared-latent projection."""

    heads: dict[str, SharedLatentHead]
    latent_dim: int

    def encode(self, modality: str, x: np.ndarray) -> np.ndarray:
        head = self.heads[modality]
        h = x @ head.encoder
        z = h @ head.projector
        norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        return z / norm


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


def _build_model(modalities: list[str], input_dim: int, hidden_dim: int, latent_dim: int, seed: int) -> SharedLatentModel:
    rng = np.random.default_rng(seed)
    heads: dict[str, SharedLatentHead] = {}
    for m in modalities:
        encoder = rng.normal(0, 0.1, size=(input_dim, hidden_dim))
        projector = rng.normal(0, 0.1, size=(hidden_dim, latent_dim))
        heads[m] = SharedLatentHead(encoder=encoder, projector=projector)
    return SharedLatentModel(heads=heads, latent_dim=latent_dim)


def _contrastive_epoch(
    model: SharedLatentModel,
    aligned_windows: dict[str, np.ndarray],
    learning_rate: float,
    temperature: float,
) -> float:
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
        za = model.encode(ma, xa)
        zb = model.encode(mb, xb)

        sims = (za @ zb.T) / max(temperature, 1e-6)
        pos = np.diag(sims)
        denom = np.log(np.exp(sims).sum(axis=1) + 1e-8)
        losses.append(float(np.mean(-(pos - denom))))

        # Minimal, stable prototype update: encourage positive pairs by nudging
        # projector weights along correlated latent gradients.
        cross_cov = (xa.T @ xb) / n
        head_a = model.heads[ma]
        head_b = model.heads[mb]
        head_a.encoder += learning_rate * (cross_cov @ head_a.encoder)
        head_b.encoder += learning_rate * (cross_cov.T @ head_b.encoder)

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
    latent_dim: int = 12,
    hidden_dim: int = 16,
    window: int = 16,
    stride: int = 8,
    epochs: int = 40,
    learning_rate: float = 0.01,
    temperature: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    """Compute side-by-side legacy and shared-latent alignment metrics."""
    aligned = _prepare_aligned_windows(matrix, block_names, window=window, stride=stride)

    legacy_embeddings = {
        m: w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
        for m, w in aligned.items()
    }
    legacy = {
        "retrieval_at_1": round(_pairwise_retrieval_at_1(legacy_embeddings), 4),
        "probe_cosine": round(_pairwise_probe_cosine(legacy_embeddings), 4),
    }

    model = _build_model(list(aligned.keys()), input_dim=window, hidden_dim=hidden_dim,
                         latent_dim=latent_dim, seed=seed)
    loss_curve = []
    for _ in range(max(1, epochs)):
        loss_curve.append(_contrastive_epoch(model, aligned, learning_rate, temperature))

    shared_embeddings = {m: model.encode(m, w) for m, w in aligned.items()}
    shared_latent = {
        "retrieval_at_1": round(_pairwise_retrieval_at_1(shared_embeddings), 4),
        "probe_cosine": round(_pairwise_probe_cosine(shared_embeddings), 4),
        "contrastive_loss_final": round(float(loss_curve[-1]) if loss_curve else 0.0, 4),
        "training_windows": int(min((len(v) for v in aligned.values()), default=0)),
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
