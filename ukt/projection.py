"""Shared projection space for cross-domain UKT feature mixing.

Instead of each block writing to non-overlapping slices of the feature vector,
the shared projection mixes features across regions so that SVD discovers
genuine cross-domain patterns rather than recovering the original block
structure.

The mixing is governed by a cross-region coupling matrix derived from semantic
relationships between domains. Each block's features still land primarily in
their home region (preserving interpretability) but also partially project
into other regions proportional to their semantic coupling strength.

Math:
    Given a raw feature vector ``x`` with features in region ``[s, e)``,
    the projected vector is ``P @ x`` where ``P`` is a ``(d, d)`` matrix:

    - Diagonal blocks ``P[s:e, s:e] = I`` (identity — home region, full weight)
    - Off-diagonal blocks ``P[t_s:t_e, s:e] = w * Q`` where ``w`` is the
      coupling weight and ``Q`` is a seeded random orthogonal sub-matrix.

    After projection, every block's features span the full feature space,
    enabling SVD to discover genuine cross-domain correlations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ukt.registry import FeatureRegionRegistry


# ---------------------------------------------------------------------------
# Default cross-region coupling weights
# ---------------------------------------------------------------------------
# Derived from REGION_SEMANTIC_SPEC canvas coupling definitions.
# Key: (source_region, target_region) → weight.
# These represent how strongly a source block's features should project
# into a target region in the shared space.

DEFAULT_COUPLING: dict[tuple[str, str], float] = {
    # temporal-pattern → geospatial via systemic_stress
    ("temporal-pattern", "geospatial-kernel"): 0.30,
    # semantic-embedding → structural via power_concentration
    ("semantic-embedding", "structural-centrality"): 0.40,
    # structural-centrality → temporal via market_momentum
    ("structural-centrality", "temporal-pattern"): 0.30,
    # structural-centrality → geospatial via systemic_stress
    ("structural-centrality", "geospatial-kernel"): 0.40,
    # dynamic-agent → structural via power_concentration
    ("dynamic-agent", "structural-centrality"): 0.50,
    # dynamic-agent → semantic via narrative_diversity
    ("dynamic-agent", "semantic-embedding"): 0.30,
    # geospatial-kernel → structural via network_cohesion
    ("geospatial-kernel", "structural-centrality"): 0.30,
}


def _random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix of size (n, n) via QR decomposition."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # Ensure deterministic sign convention (positive diagonal of R)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    return Q


@dataclass
class CouplingInfo:
    """Describes a single cross-region coupling in the projection."""
    source_region: str
    target_region: str
    weight: float


class SharedProjection:
    """Projects block features into a shared embedding space.

    The projection matrix ``P ∈ R^{d × d}`` mixes features across regions:

    - Diagonal blocks: identity (home region, weight 1.0)
    - Off-diagonal blocks: ``coupling_weight * random_orthogonal_rotation``

    After projection, every block's features span the full feature space,
    so SVD discovers genuine cross-domain correlations instead of just
    recovering the original block structure.

    The projection is invertible (P is well-conditioned because it is
    diagonally dominant with unit diagonal blocks), allowing back-projection
    for interpretability.

    Args:
        registry: Feature region registry defining region boundaries.
        coupling: Cross-region coupling weights. If None, uses DEFAULT_COUPLING.
        seed: Random seed for reproducible orthogonal sub-matrices.
    """

    def __init__(
        self,
        registry: FeatureRegionRegistry,
        coupling: dict[tuple[str, str], float] | None = None,
        seed: int = 2025,
    ):
        self.registry = registry
        self.coupling = coupling if coupling is not None else DEFAULT_COUPLING
        self.dim = registry.total_dim
        self.seed = seed
        self._P = self._build_projection_matrix()
        self._P_inv: np.ndarray | None = None  # Lazy — computed on first use

    def _build_projection_matrix(self) -> np.ndarray:
        """Construct the shared projection matrix P."""
        rng = np.random.default_rng(self.seed)
        P = np.eye(self.dim)

        regions = self.registry.ordered_regions

        for source in regions:
            for target in regions:
                if source.name == target.name:
                    continue
                weight = self.coupling.get((source.name, target.name), 0.0)
                if weight < 1e-6:
                    continue

                sd, td = source.dim, target.dim
                min_dim = min(sd, td)

                # Random orthogonal sub-matrix for this coupling
                Q = _random_orthogonal(min_dim, rng)

                # Place: target rows, source columns
                block = np.zeros((td, sd))
                block[:min_dim, :min_dim] = Q[:min_dim, :min_dim]

                P[target.start:target.end, source.start:source.end] = weight * block

        return P

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
        """Recover approximate original features from projected space.

        Used for interpretability: trace a kernel's feature loadings back
        to the original block contributions.
        """
        return self.inverse @ projected

    @property
    def projection_matrix(self) -> np.ndarray:
        """The full projection matrix P (copy for safety)."""
        return self._P.copy()

    @property
    def active_couplings(self) -> list[CouplingInfo]:
        """List all active cross-region couplings."""
        return [
            CouplingInfo(source_region=src, target_region=tgt, weight=w)
            for (src, tgt), w in sorted(self.coupling.items())
            if w > 1e-6
        ]

    def coupling_summary(self) -> dict[str, list[tuple[str, float]]]:
        """Summarize active couplings per source region."""
        summary: dict[str, list[tuple[str, float]]] = {}
        for (src, tgt), weight in self.coupling.items():
            if weight > 1e-6:
                summary.setdefault(src, []).append((tgt, weight))
        return summary
