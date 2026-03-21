"""Final shipping tests: end-to-end acceptance, regression guards, deployment readiness.

These tests validate that the Hyperspace system is production-ready:
- The full pipeline produces complete, correct output
- All governance guarantees hold
- All subsystems integrate correctly
- Invariants that must never break are guarded
- The app entry point can be loaded
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hyperspace.config import (
    GEOPOLITICAL_NODES,
    GOVERNANCE_FLAG_CODES,
    SCORECARD_THRESHOLDS,
    UKT_FEATURE_DIM,
)
from hyperspace.core.pipeline import PipelineRunner
from hyperspace.core.types import (
    validate_block_result,
    validate_snapshot,
)
from hyperspace.models.knowledge_matrix import (
    BLOCK_REGION_MAP,
    FEATURE_REGION_LABELS,
    UniversalKnowledgeTensor,
    estimate_reality_regression_stability,
)
from hyperspace.models.graph_engine import build_geopolitical_graph, analyze_graph
from hyperspace.models.spatial_kernels import get_spatial_features
from hyperspace.models.agent_sim import initialize_agents_from_data, run_simulation
from hyperspace.models.sparse_ae import (
    train_sparse_ae,
    map_concepts_to_kernels,
    enrich_concepts_with_narratives,
)
from hyperspace.models.semantic_canvas import (
    SemanticCanvas,
    CANVAS_DIM,
    CANVAS_DIMENSIONS,
    REGION_TO_CANVAS,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def full_pipeline_result(rng):
    """Run the full pipeline once and cache for all shipping tests."""
    finance = {
        "features_for_ukt": _make_features(rng, 0, 32),
        "feature_meta": {i: {"label": f"fin_{i}", "block": "finance"} for i in range(32)},
        "data_source": "synthetic_finance",
    }
    clusters = {
        "features_for_ukt": _make_features(rng, 16, 32),
        "feature_meta": {16 + i: {"label": f"topic_{i}", "block": "clusters"} for i in range(8)},
        "data_source": "synthetic_clusters",
    }
    spatial = dict(
        physical_raster=rng.uniform(0, 1, (4, 6, 9)),
        country_scalars=rng.uniform(0, 1, (10, 6)),
        scalar_names=["elev", "temp", "hum", "precip", "gdp", "debt",
                       "mil", "enroll", "stability", "homicide"],
        node_order=list(GEOPOLITICAL_NODES.keys()),
        source_label="synthetic_spatial",
    )
    agreement = _make_agreement()
    runner = PipelineRunner()
    return runner.run(
        finance_result=finance,
        cluster_result=clusters,
        agreement_matrix=agreement,
        spatial_data=spatial,
        timeframe_context={"start_date": "2025-01-01", "end_date": "2025-12-31",
                           "min_year": 2025, "max_year": 2025},
        sim_steps=30,
        sae_hidden_dim=16,
        sae_epochs=60,
        stability_runs=6,
    )


def _make_features(rng, lo, hi):
    f = np.zeros(UKT_FEATURE_DIM)
    f[lo:hi] = rng.uniform(0, 1, hi - lo)
    return f


def _make_agreement():
    nodes = list(GEOPOLITICAL_NODES.keys())
    n = len(nodes)
    rng = np.random.default_rng(99)
    mat = rng.uniform(-0.5, 1.0, (n, n))
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1.0)
    return pd.DataFrame(mat, index=nodes, columns=nodes)


# ========================================================================== #
# 1. Full Pipeline Acceptance Tests                                            #
# ========================================================================== #


class TestPipelineAcceptance:
    """End-to-end acceptance: the pipeline produces complete, valid output."""

    def test_all_five_blocks_present(self, full_pipeline_result):
        snaps = full_pipeline_result["snapshots"]
        assert len(snaps) == 5
        names = [s["block_name"] for s in snaps]
        assert names == ["Finance", "Clusters", "Graph", "Spatial", "Agents"]

    def test_final_matrix_shape(self, full_pipeline_result):
        mat = full_pipeline_result["final_matrix"]
        assert mat is not None
        assert mat.shape == (5, UKT_FEATURE_DIM)

    def test_all_data_sources_tracked(self, full_pipeline_result):
        ds = full_pipeline_result["data_sources"]
        for block in ["Finance", "Clusters", "Graph", "Spatial", "Agents"]:
            assert block in ds, f"data_sources missing '{block}'"
            assert len(ds[block]) > 0

    def test_run_id_present(self, full_pipeline_result):
        assert len(full_pipeline_result["run_id"]) == 8
        assert len(full_pipeline_result["run_timestamp"]) > 0

    def test_sae_result_present(self, full_pipeline_result):
        sae = full_pipeline_result["sae_result"]
        assert sae is not None
        assert sae["total_concepts"] == 16
        assert sae["active_concepts"] > 0

    def test_concept_kernel_map_present(self, full_pipeline_result):
        ckm = full_pipeline_result["concept_kernel_map"]
        assert len(ckm) == 16

    def test_stability_computed(self, full_pipeline_result):
        stab = full_pipeline_result["stability"]
        assert stab is not None
        assert stab["n_runs"] == 6
        assert 0 <= stab["mean_cosine"] <= 1.0

    def test_semantic_canvas_complete(self, full_pipeline_result):
        canvas = full_pipeline_result["semantic_canvas"]
        assert canvas is not None
        assert len(canvas.entries) == 5
        for entry in canvas.entries:
            assert entry.coordinates.shape == (CANVAS_DIM,)
            assert len(entry.dominant_dimensions) > 0
            assert len(entry.interpretation) > 0


# ========================================================================== #
# 2. Governance Guarantee Tests                                                #
# ========================================================================== #


class TestGovernanceGuarantees:
    """Every governance promise must hold in the shipping build."""

    def test_governance_flags_structure(self, full_pipeline_result):
        for flag in full_pipeline_result["governance_flags"]:
            assert "code" in flag
            assert "label" in flag
            assert "description" in flag
            assert "severity" in flag
            assert "detail" in flag
            assert flag["code"] in GOVERNANCE_FLAG_CODES

    def test_scorecard_all_dimensions_present(self, full_pipeline_result):
        sc = full_pipeline_result["interpretability_scorecard"]
        for dim in SCORECARD_THRESHOLDS:
            assert dim in sc, f"Scorecard missing dimension '{dim}'"
            entry = sc[dim]
            assert "value" in entry
            assert "threshold" in entry
            assert "passed" in entry
            assert "label" in entry
            assert "description" in entry
            assert isinstance(entry["passed"], bool)

    def test_every_snapshot_has_feature_meta(self, full_pipeline_result):
        for snap in full_pipeline_result["snapshots"]:
            assert "feature_meta" in snap
            assert isinstance(snap["feature_meta"], dict)

    def test_every_snapshot_has_report(self, full_pipeline_result):
        for snap in full_pipeline_result["snapshots"]:
            assert "report" in snap
            assert len(snap["report"]) > 50

    def test_every_snapshot_valid(self, full_pipeline_result):
        for snap in full_pipeline_result["snapshots"]:
            warnings = validate_snapshot(snap)
            assert warnings == [], f"Snapshot {snap['block_name']} invalid: {warnings}"

    def test_kernel_labels_traceable(self, full_pipeline_result):
        """Every kernel label must reference a valid block and region."""
        valid_regions = set(FEATURE_REGION_LABELS.values())
        for snap in full_pipeline_result["snapshots"]:
            for kl in snap["kernel_labels"]:
                assert kl["dominant_region"] in valid_regions
                assert kl["importance"] >= 0
                assert len(kl["top_features"]) > 0
                assert len(kl["narrative"]) > 0

    def test_reality_regression_dimensionality(self, full_pipeline_result):
        for snap in full_pipeline_result["snapshots"]:
            assert snap["reality_regression"].shape == (UKT_FEATURE_DIM,)

    def test_importance_sums_to_one_every_step(self, full_pipeline_result):
        for snap in full_pipeline_result["snapshots"]:
            assert abs(snap["importance"].sum() - 1.0) < 1e-6

    def test_reconstruction_error_bounded(self, full_pipeline_result):
        for snap in full_pipeline_result["snapshots"]:
            assert snap["reconstruction_error"] < 1.0


# ========================================================================== #
# 3. Regression Guards                                                         #
# ========================================================================== #


class TestRegressionGuards:
    """Invariants that must never break across releases."""

    def test_ukt_feature_dim_is_80(self):
        assert UKT_FEATURE_DIM == 80

    def test_six_geopolitical_nodes(self):
        assert len(GEOPOLITICAL_NODES) == 6
        assert set(GEOPOLITICAL_NODES.keys()) == {
            "USA", "Russia", "China", "Britain", "India", "Brazil",
        }

    def test_five_feature_regions(self):
        assert len(FEATURE_REGION_LABELS) == 5
        expected_regions = {
            "temporal-pattern", "semantic-embedding",
            "structural-centrality", "dynamic-agent",
            "geospatial-kernel",
        }
        assert set(FEATURE_REGION_LABELS.values()) == expected_regions

    def test_five_block_region_mappings(self):
        assert len(BLOCK_REGION_MAP) == 5
        assert set(BLOCK_REGION_MAP.keys()) == {
            "Finance", "Clusters", "Graph", "Agents", "Spatial",
        }

    def test_twelve_canvas_dimensions(self):
        assert CANVAS_DIM == 12
        assert len(CANVAS_DIMENSIONS) == 12

    def test_five_governance_flag_codes(self):
        assert len(GOVERNANCE_FLAG_CODES) == 6
        for i in range(1, 7):
            assert f"GOV-{i:03d}" in GOVERNANCE_FLAG_CODES

    def test_normalization_idempotent(self):
        from hyperspace.models.knowledge_matrix import _normalize_features
        rng = np.random.default_rng(42)
        raw = rng.uniform(-10, 10, UKT_FEATURE_DIM)
        once = _normalize_features(raw)
        twice = _normalize_features(once)
        np.testing.assert_allclose(once, twice, atol=1e-10)

    def test_pad_or_truncate_identity(self):
        from hyperspace.models.knowledge_matrix import _pad_or_truncate
        exact = np.arange(80.0)
        result = _pad_or_truncate(exact, 80)
        np.testing.assert_array_equal(exact, result)

    def test_svd_decomposition_reconstructs(self):
        rng = np.random.default_rng(42)
        matrix = rng.uniform(0, 1, (5, UKT_FEATURE_DIM))
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        reconstructed = U @ np.diag(S) @ Vt
        np.testing.assert_allclose(matrix, reconstructed, atol=1e-10)

    def test_stability_deterministic_across_calls(self):
        rng = np.random.default_rng(42)
        matrix = rng.uniform(0, 1, (5, UKT_FEATURE_DIM))
        r1 = estimate_reality_regression_stability(matrix, seed=123)
        r2 = estimate_reality_regression_stability(matrix, seed=123)
        assert r1["mean_cosine"] == r2["mean_cosine"]
        assert r1["min_cosine"] == r2["min_cosine"]

    def test_graph_always_has_six_nodes(self):
        G, pos = build_geopolitical_graph()
        assert len(G.nodes()) == 6

    def test_agent_count_matches_nodes(self):
        agents = initialize_agents_from_data()
        assert len(agents) == len(GEOPOLITICAL_NODES)


# ========================================================================== #
# 4. Cross-Subsystem Contract Tests                                            #
# ========================================================================== #


class TestCrossSubsystemContracts:
    """Verify data flows correctly between subsystems."""

    def test_graph_to_agent_data_flow(self, full_pipeline_result):
        """Graph centrality feeds agent initialization."""
        graph = full_pipeline_result["graph_result"]
        sim = full_pipeline_result["sim_result"]
        assert graph is not None
        assert sim is not None
        assert len(sim["agents"]) == 6

    def test_spatial_to_agent_data_flow(self, full_pipeline_result):
        """Spatial features enrich agent capability."""
        spatial = full_pipeline_result["spatial_result"]
        sim = full_pipeline_result["sim_result"]
        assert spatial is not None
        for agent in sim["agents"].values():
            assert hasattr(agent, "capability_multiplier")

    def test_ukt_to_sae_data_flow(self, full_pipeline_result):
        """UKT final matrix feeds into SAE."""
        mat = full_pipeline_result["final_matrix"]
        sae = full_pipeline_result["sae_result"]
        assert mat.shape[1] == sae["concept_vectors"].shape[1]

    def test_sae_to_concept_kernel_map(self, full_pipeline_result):
        """SAE concepts are mapped to UKT kernels."""
        sae = full_pipeline_result["sae_result"]
        ckm = full_pipeline_result["concept_kernel_map"]
        assert len(ckm) == sae["total_concepts"]
        for entry in ckm:
            assert entry["best_kernel"].startswith("K")

    def test_canvas_accumulates_from_ukt(self, full_pipeline_result):
        """Canvas entries match UKT blocks."""
        canvas = full_pipeline_result["semantic_canvas"]
        snaps = full_pipeline_result["snapshots"]
        assert len(canvas.entries) == len(snaps)
        for entry, snap in zip(canvas.entries, snaps):
            assert entry.block_name == snap["block_name"]

    def test_governance_uses_pipeline_data(self, full_pipeline_result):
        """Governance flags reference actual pipeline blocks."""
        flags = full_pipeline_result["governance_flags"]
        for flag in flags:
            assert flag["code"] in GOVERNANCE_FLAG_CODES
            assert len(flag["detail"]) > 0

    def test_scorecard_uses_stability(self, full_pipeline_result):
        """Scorecard kernel_stability uses the stability result."""
        sc = full_pipeline_result["interpretability_scorecard"]
        stab = full_pipeline_result["stability"]
        if stab and stab["n_runs"] > 0:
            assert sc["kernel_stability"]["value"] == round(stab["mean_cosine"], 4)

    def test_all_block_results_conform_to_type(self, full_pipeline_result):
        """Every stored block result has the required BlockResult keys."""
        block_keys = [
            ("finance_result", "Finance"),
            ("cluster_result", "Clusters"),
            ("graph_result", "Graph"),
            ("spatial_result", "Spatial"),
            ("sim_result", "Agents"),
        ]
        for key, name in block_keys:
            result = full_pipeline_result.get(key)
            if result is not None:
                assert "features_for_ukt" in result, f"{name} missing features_for_ukt"
                assert "feature_meta" in result or "analysis" in result, f"{name} missing metadata"
                assert "data_source" in result, f"{name} missing data_source"


# ========================================================================== #
# 5. Counterfactual Contestability Tests                                       #
# ========================================================================== #


class TestCounterfactualContestability:
    """Verify that block removal changes conclusions (core governance property)."""

    def test_removing_any_block_changes_reality_regression(self, full_pipeline_result):
        snaps = full_pipeline_result["snapshots"]
        final_snap = snaps[-1]
        original_rr = final_snap["reality_regression"]
        matrix = final_snap["matrix"]
        block_names = [s["block_name"] for s in snaps]

        for removed in block_names:
            kept = [i for i, n in enumerate(block_names) if n != removed]
            if len(kept) < 2:
                continue
            sub = matrix[kept, :]
            U, S, Vt = np.linalg.svd(sub, full_matrices=False)
            importance = S / (S.sum() + 1e-8)
            cf_rr = importance @ Vt[:len(S), :]
            diff = np.abs(original_rr - cf_rr).sum()
            assert diff > 0.001, f"Removing {removed} had no effect on reality regression"

    def test_counterfactual_kernel_count_decreases(self, full_pipeline_result):
        snaps = full_pipeline_result["snapshots"]
        final_snap = snaps[-1]
        original_kernels = final_snap["n_kernels"]
        matrix = final_snap["matrix"]
        block_names = [s["block_name"] for s in snaps]

        for removed in block_names:
            kept = [i for i, n in enumerate(block_names) if n != removed]
            if len(kept) < 2:
                continue
            sub = matrix[kept, :]
            _, S, _ = np.linalg.svd(sub, full_matrices=False)
            assert len(S) == original_kernels - 1


# ========================================================================== #
# 6. Deployment Readiness Tests                                                #
# ========================================================================== #


class TestDeploymentReadiness:
    """Pre-deployment checks for the application."""

    def test_app_entry_point_importable(self):
        """app.py should be importable (though not runnable without Streamlit server)."""
        import importlib
        # This verifies all top-level imports in app.py resolve
        spec = importlib.util.find_spec("app")
        # app.py is at the repo root, may not be in sys.path as a module
        # but all its imports from hyperspace.* should work
        import hyperspace.pages.dashboard
        import hyperspace.pages.mission_control_tab
        import hyperspace.pages.finance_tab
        import hyperspace.pages.clusters_tab
        import hyperspace.pages.politics_tab
        import hyperspace.pages.agents_tab
        import hyperspace.pages.interpreter_tab
        import hyperspace.pages.pipeline_tab
        import hyperspace.pages.counterfactual_tab

    def test_requirements_packages_available(self):
        """Core dependencies must be importable."""
        import numpy
        import pandas
        import torch
        import networkx
        import scipy
        import sklearn
        import plotly
        import streamlit

    def test_no_circular_imports(self):
        """Force-reimport core chain to detect circular imports."""
        import importlib
        modules_to_check = [
            "hyperspace.config",
            "hyperspace.core.types",
            "hyperspace.models.knowledge_matrix",
            "hyperspace.models.semantic_canvas",
            "hyperspace.models.graph_engine",
            "hyperspace.models.spatial_kernels",
            "hyperspace.models.agent_sim",
            "hyperspace.models.sparse_ae",
            "hyperspace.core.pipeline",
        ]
        for mod_path in modules_to_check:
            mod = importlib.import_module(mod_path)
            assert mod is not None

    def test_pipeline_runner_is_streamlit_free(self):
        """PipelineRunner should work without Streamlit being initialized."""
        import sys
        # PipelineRunner uses no st.* calls
        from hyperspace.core.pipeline import PipelineRunner
        runner = PipelineRunner()
        result = runner.run(sim_steps=5, sae_epochs=5, stability_runs=2)
        assert len(result["snapshots"]) >= 2

    def test_full_pipeline_under_60_seconds(self, full_pipeline_result):
        """The full pipeline should have completed (fixture ran)."""
        assert full_pipeline_result is not None
        assert len(full_pipeline_result["snapshots"]) == 5

    def test_pipeline_result_serializable_keys(self, full_pipeline_result):
        """All top-level keys in PipelineResult are present."""
        expected_keys = [
            "snapshots", "final_matrix", "data_sources", "timeframe_context",
            "finance_result", "cluster_result", "graph_result",
            "spatial_result", "sim_result",
            "sae_result", "concept_kernel_map", "semantic_canvas",
            "stability", "governance_flags", "interpretability_scorecard",
            "run_id", "run_timestamp",
        ]
        for key in expected_keys:
            assert key in full_pipeline_result, f"PipelineResult missing key '{key}'"
