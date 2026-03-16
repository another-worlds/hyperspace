"""Shared projection space for cross-domain UKT feature mixing.

Instead of each block writing to non-overlapping slices of the feature vector,
the shared projection mixes features across regions so that SVD discovers
genuine cross-domain patterns rather than recovering the original block
structure.

The projection is ADAPTIVE — it learns from the data as blocks arrive:

- **Topology**: Which regions CAN couple — structural prior from the semantic
  spec (e.g., temporal-pattern and geospatial-kernel are semantically related).
  This is the only non-data-driven component, and it's a mild structural prior
  that says "these domains are related" without specifying how much or how.

- **Strength**: How strongly two regions couple — computed from the actual
  feature energy ratio: ``min(E_src, E_tgt) / max(E_src, E_tgt)``.
  If both blocks are strongly active, full coupling. If one block has
  near-zero features, coupling drops to zero. This is purely data-driven.

- **Direction**: Which specific features in the source map to which features
  in the target — rank-1 projection along each block's dominant feature
  direction: ``outer(v_tgt, v_src)`` where ``v = f / ||f||``.
  This means the coupling connects the strongest signal in each block.
  If the data changes (different attention patterns, different centrality
  structure), the coupling direction changes accordingly.

The projection rebuilds every time a new block arrives, and all previously
stored blocks are re-projected through the updated matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ukt.registry import FeatureRegionRegistry


# ---------------------------------------------------------------------------
# Default coupling topology
# ---------------------------------------------------------------------------
# Derived from REGION_SEMANTIC_SPEC canvas coupling definitions.
# This defines WHICH regions can couple (the graph edges), NOT how strongly
# (the edge weights — those come from data).

DEFAULT_TOPOLOGY: set[tuple[str, str]] = {
    # temporal-pattern ↔ geospatial via systemic_stress
    ("temporal-pattern", "geospatial-kernel"),
    # semantic-embedding → structural via power_concentration
    ("semantic-embedding", "structural-centrality"),
    # structural-centrality → temporal via market_momentum
    ("structural-centrality", "temporal-pattern"),
    # structural-centrality → geospatial via systemic_stress
    ("structural-centrality", "geospatial-kernel"),
    # dynamic-agent → structural via power_concentration
    ("dynamic-agent", "structural-centrality"),
    # dynamic-agent → semantic via narrative_diversity
    ("dynamic-agent", "semantic-embedding"),
    # geospatial-kernel → structural via network_cohesion
    ("geospatial-kernel", "structural-centrality"),
}


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
    """Describes a single cross-region coupling in the projection."""
    source_region: str
    target_region: str
    weight: float
    data_driven: bool = True


class SharedProjection:
    """Adaptive projection that learns cross-region coupling from block features.

    The projection matrix ``P`` starts as identity and evolves as blocks arrive.
    Each new block updates the coupling strength and direction between its
    region and all other observed regions, based on the actual feature values.

    The only structural prior is the coupling TOPOLOGY — which regions can
    couple. Everything else (strength, direction) comes from data.

    Args:
        registry: Feature region registry defining region boundaries.
        topology: Set of (source_region, target_region) pairs that can couple.
            If None, uses DEFAULT_TOPOLOGY derived from REGION_SEMANTIC_SPEC.
        seed: Random seed for the orthogonal fallback basis.
    """

    def __init__(
        self,
        registry: FeatureRegionRegistry,
        topology: set[tuple[str, str]] | None = None,
        seed: int = 2025,
    ):
        self.registry = registry
        self.topology = topology if topology is not None else DEFAULT_TOPOLOGY
        self.dim = registry.total_dim
        self.seed = seed
        self._block_features: dict[str, np.ndarray] = {}
        self._P = np.eye(self.dim)
        self._P_inv: np.ndarray | None = None

        # Pre-compute fallback orthogonal bases for each region pair
        # Used when feature vectors are too weak for reliable direction estimation
        rng = np.random.default_rng(self.seed)
        self._fallback_bases: dict[tuple[str, str], np.ndarray] = {}
        regions = self.registry.ordered_regions
        for src in regions:
            for tgt in regions:
                if src.name != tgt.name:
                    min_dim = min(src.dim, tgt.dim)
                    self._fallback_bases[(src.name, tgt.name)] = _random_orthogonal(
                        min_dim, rng,
                    )

    def observe(self, region_name: str, features: np.ndarray) -> None:
        """Record a block's native features and rebuild the projection.

        Args:
            region_name: The region this block writes to (e.g., "temporal-pattern").
            features: The block's native feature vector (extracted from its home region).
        """
        self._block_features[region_name] = features.copy()
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild the projection matrix from all observed block features."""
        P = np.eye(self.dim)

        for src_name, tgt_name in self.topology:
            if src_name not in self._block_features:
                continue
            if tgt_name not in self._block_features:
                continue

            src_region = self.registry.regions[src_name]
            tgt_region = self.registry.regions[tgt_name]
            f_src = self._block_features[src_name]
            f_tgt = self._block_features[tgt_name]

            e_src = float(np.linalg.norm(f_src))
            e_tgt = float(np.linalg.norm(f_tgt))

            # Strength from data: energy ratio (0 if one block is dead, 1 if equal)
            strength = min(e_src, e_tgt) / (max(e_src, e_tgt) + 1e-8)

            if strength < 1e-6:
                continue

            sd, td = src_region.dim, tgt_region.dim
            min_dim = min(sd, td)

            # Direction from data: rank-1 projection along dominant feature directions
            # Blended with orthogonal fallback for numerical stability
            if e_src > 0.1 and e_tgt > 0.1:
                v_src = f_src[:min_dim] / e_src
                v_tgt = f_tgt[:min_dim] / e_tgt
                # Data-driven rank-1 component
                data_block = np.outer(v_tgt, v_src)
                # Fallback orthogonal for the orthogonal complement
                fallback = self._fallback_bases[(src_name, tgt_name)][:min_dim, :min_dim]
                # Blend: 70% data-driven direction, 30% orthogonal spread
                # The blend ensures coupling isn't purely rank-1
                block = 0.7 * data_block + 0.3 * fallback
            else:
                block = self._fallback_bases[(src_name, tgt_name)][:min_dim, :min_dim]

            # Place in full projection matrix
            coupling_block = np.zeros((td, sd))
            coupling_block[:min_dim, :min_dim] = block
            P[tgt_region.start:tgt_region.end, src_region.start:src_region.end] = (
                strength * coupling_block
            )

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
        """List all active cross-region couplings with their data-derived weights."""
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
                    source_region=src_name,
                    target_region=tgt_name,
                    weight=round(strength, 4),
                ))
        return couplings

    def coupling_summary(self) -> dict[str, list[tuple[str, float]]]:
        """Summarize active couplings per source region."""
        summary: dict[str, list[tuple[str, float]]] = {}
        for c in self.active_couplings:
            summary.setdefault(c.source_region, []).append(
                (c.target_region, c.weight),
            )
        return summary
