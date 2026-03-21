"""Shared projection space for cross-block UKT feature mixing.

Instead of each block writing to non-overlapping slices of the feature vector,
the shared projection mixes features across blocks so that SVD discovers
genuine cross-block patterns rather than recovering the original block
structure.

The projection is ADAPTIVE — it learns from the data as blocks arrive:

- **Topology**: Which blocks CAN couple — structural prior that specifies
  which blocks are semantically related. This is the only non-data-driven
  component, a mild structural prior that says "these blocks are related"
  without specifying how much or how.

- **Strength**: How strongly two blocks couple — computed from the actual
  feature energy ratio: ``min(E_src, E_tgt) / max(E_src, E_tgt)``.
  If both blocks are strongly active, full coupling. If one block has
  near-zero features, coupling drops to zero. This is purely data-driven.

- **Direction**: Which specific features in the source map to which features
  in the target — rank-1 projection along each block's dominant feature
  direction: ``outer(v_tgt, v_src)`` where ``v = f / ||f||``.
  This operates on the full 80-dim vectors, connecting the strongest signal
  in each block across the monolithic feature space.
  If the data changes, the coupling direction changes accordingly.

The projection rebuilds every time a new block arrives, and all previously
stored blocks are re-projected through the updated matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Topology is OPEN by default — all block pairs can couple.
# ---------------------------------------------------------------------------
# The adaptive strength computation (energy ratio) already gates weak
# couplings to near-zero. If two blocks genuinely have nothing to do with
# each other, their features will have no structural similarity and coupling
# will be negligible. Let the data decide — don't gatekeep with topology.
#
# Pass a restricted topology set to __init__ to limit coupling if needed.

DEFAULT_TOPOLOGY: None = None  # Sentinel: means "all pairs"
FEATURE_DIM: int = 80  # Monolithic feature space dimension


def _random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix of size (n, n) via QR decomposition."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    return Q


@dataclass
class CouplingInfo:
    """Describes a single cross-block coupling in the projection."""
    source_block: str
    target_block: str
    weight: float
    data_driven: bool = True


class SharedProjection:
    """Adaptive projection that learns cross-block coupling from block features.

    The projection matrix ``P`` starts as identity and evolves as blocks arrive.
    Each new block updates the coupling strength and direction between itself
    and all other observed blocks, based on the actual feature values in the
    monolithic 80-dim feature space.

    The only structural prior is the coupling TOPOLOGY — which blocks can
    couple. Everything else (strength, direction) comes from data.

    Args:
        block_names: List of block names that may be observed (e.g.,
            ["finance", "clusters", "graph", "agents", "spatial"]).
        topology: Set of (source_block, target_block) pairs that can couple.
            If None, all block pairs can couple (data-driven strength gates weak).
        seed: Random seed for the orthogonal fallback basis.
    """

    def __init__(
        self,
        block_names: list[str] | None = None,
        topology: set[tuple[str, str]] | None = DEFAULT_TOPOLOGY,
        seed: int = 2025,
    ):
        self.block_names = block_names or []
        self.dim = FEATURE_DIM  # Monolithic 80-dim space
        self.seed = seed

        # Full topology: all directed block pairs. Data-driven strength
        # gates weak couplings to near-zero, so no need to restrict.
        if topology is None:
            self.topology = {
                (src, tgt)
                for src in self.block_names
                for tgt in self.block_names
                if src != tgt
            }
        else:
            self.topology = topology
        self._block_features: dict[str, np.ndarray] = {}
        self._P = np.eye(self.dim)
        self._P_inv: np.ndarray | None = None

        # Pre-compute fallback orthogonal bases for each block pair
        # Used when feature vectors are too weak for reliable direction estimation
        rng = np.random.default_rng(self.seed)
        self._fallback_bases: dict[tuple[str, str], np.ndarray] = {}
        for src in self.block_names:
            for tgt in self.block_names:
                if src != tgt:
                    self._fallback_bases[(src, tgt)] = _random_orthogonal(
                        self.dim, rng,
                    )

    def observe(self, block_name: str, features: np.ndarray) -> None:
        """Record a block's features and rebuild the projection.

        Args:
            block_name: The block name (e.g., "finance", "clusters", "graph", "agents", "spatial").
            features: The block's full 80-dim feature vector (sparse, with non-zeros in the block's feature range).
        """
        if len(features) != self.dim:
            raise ValueError(f"Expected feature vector of dimension {self.dim}, got {len(features)}")
        self._block_features[block_name] = features.copy()
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild the projection matrix from all observed block features."""
        P = np.eye(self.dim)

        for src_name, tgt_name in self.topology:
            if src_name not in self._block_features:
                continue
            if tgt_name not in self._block_features:
                continue

            f_src = self._block_features[src_name]
            f_tgt = self._block_features[tgt_name]

            e_src = float(np.linalg.norm(f_src))
            e_tgt = float(np.linalg.norm(f_tgt))

            # Strength from data: energy ratio (0 if one block is dead, 1 if equal)
            strength = min(e_src, e_tgt) / (max(e_src, e_tgt) + 1e-8)

            if strength < 1e-6:
                continue

            # Direction from data: rank-1 projection along dominant feature directions
            # across the full 80-dim monolithic space.
            # Blend ratio is data-driven: strong signals (strength≈1) get mostly
            # data-driven direction; weak signals get mostly orthogonal fallback.
            v_src = f_src / e_src
            v_tgt = f_tgt / e_tgt
            data_block = np.outer(v_tgt, v_src)
            fallback = self._fallback_bases[(src_name, tgt_name)]
            alpha = strength  # Data confidence tracks coupling strength
            block = alpha * data_block + (1.0 - alpha) * fallback

            # Add to full projection matrix (strength-scaled outer product)
            P += strength * block

        self._P = P
        self._P_inv = None  # Invalidate cache

    @property
    def inverse(self) -> np.ndarray:
        """Inverse of the projection matrix (cached)."""
        if self._P_inv is None:
            self._P_inv = np.linalg.inv(self._P)
        return self._P_inv

    def project(self, features: np.ndarray) -> np.ndarray:
        """Project a raw feature vector into the shared space.

        Args:
            features: ``(feature_dim,)`` raw block feature vector.

        Returns:
            ``(feature_dim,)`` projected vector with cross-region mixing.
        """
        return self._P @ features

    def back_project(self, projected: np.ndarray) -> np.ndarray:
        """Recover approximate original features from projected space."""
        return self.inverse @ projected

    @property
    def projection_matrix(self) -> np.ndarray:
        """The full projection matrix P (copy for safety)."""
        return self._P.copy()

    @property
    def active_couplings(self) -> list[CouplingInfo]:
        """List all active cross-block couplings with their data-derived weights."""
        couplings = []
        for src_name, tgt_name in sorted(self.topology):
            if src_name not in self._block_features or tgt_name not in self._block_features:
                continue
            f_src = self._block_features[src_name]
            f_tgt = self._block_features[tgt_name]
            e_src = float(np.linalg.norm(f_src))
            e_tgt = float(np.linalg.norm(f_tgt))
            strength = min(e_src, e_tgt) / (max(e_src, e_tgt) + 1e-8)
            if strength > 1e-6:
                couplings.append(CouplingInfo(
                    source_block=src_name,
                    target_block=tgt_name,
                    weight=round(strength, 4),
                ))
        return couplings

    def coupling_summary(self) -> dict[str, list[tuple[str, float]]]:
        """Summarize active couplings per source block."""
        summary: dict[str, list[tuple[str, float]]] = {}
        for c in self.active_couplings:
            summary.setdefault(c.source_block, []).append(
                (c.target_block, c.weight),
            )
        return summary
