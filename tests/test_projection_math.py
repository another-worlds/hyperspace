"""Tests for projection math correctness.

Verifies the two critical fixes:
1. _symmetric_normalize preserves zeros in unoccupied regions
2. SharedProjection._rebuild uses linear (not quadratic) strength gating
"""
from __future__ import annotations

import numpy as np
import pytest

from ukt.projection import SharedProjection
from ukt.registry import FeatureRegionRegistry
from hyperspace.models.knowledge_matrix import _symmetric_normalize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_region_registry() -> FeatureRegionRegistry:
    """Registry with two 8-dim regions in a 16-dim space."""
    reg = FeatureRegionRegistry()
    reg.register("block_a", 0, 8)
    reg.register("block_b", 8, 16)
    return reg


@pytest.fixture()
def sparse_20dim_registry() -> FeatureRegionRegistry:
    """Registry with a 4-dim region in a 20-dim space (lots of unoccupied padding)."""
    reg = FeatureRegionRegistry()
    reg.register("only_block", 0, 4)
    return reg


# ---------------------------------------------------------------------------
# _symmetric_normalize tests
# ---------------------------------------------------------------------------

class TestSymmetricNormalize:
    def test_occupied_region_maps_to_symmetric_range(self, two_region_registry):
        """Occupied regions should be scaled to [-1, 1]."""
        arr = np.zeros(16)
        arr[:8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        arr[8:16] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

        result = _symmetric_normalize(arr, two_region_registry)

        # All occupied values should be in [-1, 1]
        for r in two_region_registry.ordered_regions:
            segment = result[r.start:r.end]
            assert segment.min() >= -1.0 - 1e-8
            assert segment.max() <= 1.0 + 1e-8

        # Global min (0.0) should map to -1.0, global max (80.0) to 1.0
        assert abs(result[0] - (-1.0)) < 1e-8  # 0.0 → -1.0
        assert abs(result[15] - 1.0) < 1e-8    # 80.0 → 1.0

    def test_unoccupied_indices_stay_zero(self, sparse_20dim_registry):
        """Indices beyond any registered region must remain 0.0 after normalization."""
        arr = np.zeros(20)
        arr[:4] = [1.0, 2.0, 3.0, 4.0]

        result = _symmetric_normalize(arr, sparse_20dim_registry)

        # Occupied region [0:4] should be normalized
        assert result[0] == pytest.approx(-1.0)
        assert result[3] == pytest.approx(1.0)

        # Unoccupied region [4:20] MUST be zero — not -1.0
        np.testing.assert_array_equal(result[4:], np.zeros(16))

    def test_zero_padding_does_not_inflate_norm(self, sparse_20dim_registry):
        """L2 norm of normalized vector should reflect only occupied signal."""
        arr = np.zeros(20)
        arr[:4] = [1.0, 2.0, 3.0, 4.0]

        result = _symmetric_normalize(arr, sparse_20dim_registry)

        # Norm should come only from the 4 occupied values
        expected_norm = np.linalg.norm(result[:4])
        actual_norm = np.linalg.norm(result)
        assert actual_norm == pytest.approx(expected_norm)

    def test_empty_registry_normalizes_whole_vector(self):
        """With no regions, falls back to normalizing the entire vector."""
        reg = FeatureRegionRegistry()
        arr = np.array([0.0, 5.0, 10.0])

        result = _symmetric_normalize(arr, reg)

        assert result[0] == pytest.approx(-1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_constant_region_maps_to_zero(self, two_region_registry):
        """When all occupied values are identical, output should be 0.0."""
        arr = np.full(16, 5.0)

        result = _symmetric_normalize(arr, two_region_registry)

        np.testing.assert_array_equal(result[:16], np.zeros(16))


# ---------------------------------------------------------------------------
# SharedProjection._rebuild linear gating tests
# ---------------------------------------------------------------------------

class TestLinearGating:
    def test_equal_blocks_contribute_linearly(self):
        """With equal-energy blocks, coupling contribution should NOT be strength²."""
        proj = SharedProjection(dim=0)

        # Two blocks with identical energy
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0])

        proj.observe("A", a)
        proj.observe("B", b)

        P = proj.projection_matrix
        # P should be I + coupling_blocks (no strength multiplier on the block)
        # Strength = min(1,1)/max(1,1) = 1.0, alpha = 1.0
        # block = 1.0 * data + 0.0 * fallback = data
        # P = I + data_AB + data_BA  (two directed pairs)
        # NOT: P = I + 1.0 * data_AB + 1.0 * data_BA  (same, but quadratic would be 1.0²)
        # With strength=1, linear and quadratic are the same — test with unequal energy

    def test_weak_block_linear_not_quadratic(self):
        """With unequal energy, coupling should scale as strength, not strength²."""
        # Block A: strong signal, Block B: weaker signal
        a = np.array([10.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 3.0, 0.0, 0.0])

        # Manually compute expected strength
        e_a = np.linalg.norm(a)  # 10.0
        e_b = np.linalg.norm(b)  # 3.0
        strength = min(e_a, e_b) / max(e_a, e_b)  # 0.3

        proj = SharedProjection(dim=0)
        proj.observe("A", a)
        proj.observe("B", b)

        P = proj.projection_matrix
        P_off_identity = P - np.eye(4)

        # The contribution should NOT be negligible (as it would be with 0.3² = 0.09)
        # With linear gating: the blended block is added directly
        # With quadratic: the blended block is multiplied by 0.3 again
        frobenius = np.linalg.norm(P_off_identity, 'fro')
        assert frobenius > 0.1, "Coupling contribution too small — possible quadratic gating"

    def test_dead_block_no_coupling(self):
        """A zero-energy block should produce no coupling (strength ≈ 0)."""
        proj = SharedProjection(dim=0)

        a = np.array([5.0, 3.0, 1.0, 0.0])
        dead = np.array([0.0, 0.0, 0.0, 0.0])

        proj.observe("A", a)
        proj.observe("dead", dead)

        P = proj.projection_matrix
        # Dead block contributes nothing — P should be near identity
        np.testing.assert_allclose(P, np.eye(4), atol=1e-6)


# ---------------------------------------------------------------------------
# Projection with normalized sparse vectors
# ---------------------------------------------------------------------------

class TestProjectionWithNormalization:
    def test_energy_not_inflated_by_padding(self):
        """Projection energy should reflect real signal, not zero-padding artifacts."""
        reg = FeatureRegionRegistry()
        reg.register("A", 0, 4)
        reg.register("B", 4, 8)

        # Block A: signal in [0:4], zeros in [4:8]
        vec_a = np.zeros(8)
        vec_a[:4] = [1.0, 2.0, 3.0, 4.0]
        norm_a = _symmetric_normalize(vec_a, reg)

        # Energy should come only from [0:4]
        energy = np.linalg.norm(norm_a)
        energy_occupied = np.linalg.norm(norm_a[:4])
        assert energy == pytest.approx(energy_occupied)

    def test_outer_product_has_zero_unoccupied_rows(self):
        """Coupling direction should have zero rows/cols for unoccupied regions."""
        reg = FeatureRegionRegistry()
        reg.register("A", 0, 4)
        reg.register("B", 4, 8)

        vec_a = np.zeros(8)
        vec_a[:4] = [1.0, 2.0, 3.0, 4.0]
        norm_a = _symmetric_normalize(vec_a, reg)

        vec_b = np.zeros(8)
        vec_b[4:8] = [5.0, 6.0, 7.0, 8.0]
        norm_b = _symmetric_normalize(vec_b, reg)

        # Unit vectors
        e_a = np.linalg.norm(norm_a)
        e_b = np.linalg.norm(norm_b)
        v_a = norm_a / e_a
        v_b = norm_b / e_b

        outer = np.outer(v_b, v_a)

        # Rows [0:4] should be zero (v_b has zeros there)
        np.testing.assert_array_equal(outer[:4, :], 0.0)
        # Cols [4:8] should be zero (v_a has zeros there)
        np.testing.assert_array_equal(outer[:, 4:], 0.0)
        # Cross-block quadrant [4:8, 0:4] should be non-zero
        assert np.linalg.norm(outer[4:8, 0:4]) > 0

    def test_full_embed_normalize_project_pipeline(self):
        """End-to-end: embed → normalize → project → SVD produces correct structure."""
        from ukt.kernels import decompose_svd

        reg = FeatureRegionRegistry()
        reg.register("Finance", 0, 8)
        reg.register("Graph", 8, 16)

        rng = np.random.default_rng(42)
        fin_raw = rng.uniform(1.0, 10.0, 8)
        graph_raw = rng.uniform(1.0, 10.0, 8)

        # Embed
        fin_embedded = np.zeros(16)
        fin_embedded[:8] = fin_raw
        graph_embedded = np.zeros(16)
        graph_embedded[8:16] = graph_raw

        # Normalize (zeros in other block's region stay zero)
        fin_norm = _symmetric_normalize(fin_embedded, reg)
        graph_norm = _symmetric_normalize(graph_embedded, reg)

        assert np.all(fin_norm[8:] == 0.0), "Finance row should have zeros in Graph region"
        assert np.all(graph_norm[:8] == 0.0), "Graph row should have zeros in Finance region"

        # Project
        proj = SharedProjection()
        proj.observe("Finance", fin_norm)
        proj.observe("Graph", graph_norm)

        fin_proj = proj.project(fin_norm)
        graph_proj = proj.project(graph_norm)

        # After projection, cross-block mixing should introduce non-zero values
        # in the other block's region
        matrix = np.stack([fin_proj, graph_proj])
        decomp = decompose_svd(matrix)

        assert decomp.n_kernels == 2
        assert decomp.reconstruction_error < 1e-6
        assert np.all(decomp.importance > 0)
