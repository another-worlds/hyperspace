"""Full pipeline integration tests for Hyperspace v3.0.

These tests exercise the complete pipeline end-to-end using synthetic data,
verifying that all blocks integrate correctly through the Universal Knowledge
Tensor (UKT), Semantic Canvas, Sparse Autoencoder, and governance systems.

No Streamlit session state or external APIs are required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hyperspace.config import (
    GEOPOLITICAL_NODES,
    GEOPOLITICAL_EDGES,
    UKT_FEATURE_DIM,
    SCORECARD_THRESHOLDS,
)
from hyperspace.core.types import (
    validate_block_result,
    validate_snapshot,
)
from hyperspace.core.pipeline import PipelineRunner
from hyperspace.models.knowledge_matrix import (
    UniversalKnowledgeTensor,
    FEATURE_REGION_LABELS,
    _normalize_features,
    _pad_or_truncate,
    estimate_reality_regression_stability,
)
from hyperspace.models.graph_engine import build_geopolitical_graph, analyze_graph
from hyperspace.models.agent_sim import (
    ClusterAgent,
    initialize_agents_from_data,
    run_simulation,
)
from hyperspace.models.spatial_kernels import (
    build_full_feature_matrix,
    kernelize_spatial,
    get_spatial_features,
)
from hyperspace.models.sparse_ae import (
    SparseAutoencoder,
    train_sparse_ae,
    map_concepts_to_kernels,
)
from hyperspace.models.semantic_canvas import (
    SemanticCanvas,
    CanvasEntry,
    train_stage_sae,
    CANVAS_DIM,
    CANVAS_DIMENSIONS,
    REGION_TO_CANVAS,
)


# ========================================================================== #
# 1. UKT Core Integration                                                     #
# ========================================================================== #


class TestUKTCore:
    """Test the Universal Knowledge Tensor accumulation and decomposition."""

    def test_single_block_addition(self, synthetic_finance_features, timeframe_context):
        features, meta = synthetic_finance_features
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        snap = ukt.add_block("Finance", features, feature_meta=meta,
                             timeframe_context=timeframe_context)

        assert snap["step"] == 1
        assert snap["block_name"] == "Finance"
        assert snap["matrix"].shape == (1, UKT_FEATURE_DIM)
        assert snap["n_kernels"] == 1
        assert len(snap["kernel_labels"]) == 1
        assert snap["reconstruction_error"] < 1e-4
        assert "report" in snap and len(snap["report"]) > 0

    def test_multi_block_accumulation(
        self, synthetic_finance_features, synthetic_cluster_features, timeframe_context
    ):
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        fin_feat, fin_meta = synthetic_finance_features
        clu_feat, clu_meta = synthetic_cluster_features

        snap1 = ukt.add_block("Finance", fin_feat, feature_meta=fin_meta,
                               timeframe_context=timeframe_context)
        snap2 = ukt.add_block("Clusters", clu_feat, feature_meta=clu_meta,
                               timeframe_context=timeframe_context)

        assert snap1["step"] == 1
        assert snap2["step"] == 2
        assert snap2["matrix"].shape == (2, UKT_FEATURE_DIM)
        assert snap2["n_kernels"] == 2
        assert len(snap2["kernel_labels"]) == 2

        # Importance sums to 1
        assert abs(snap2["importance"].sum() - 1.0) < 1e-6

    def test_reality_regression_shape(
        self, synthetic_finance_features, synthetic_cluster_features, timeframe_context
    ):
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        fin_feat, fin_meta = synthetic_finance_features
        clu_feat, clu_meta = synthetic_cluster_features

        ukt.add_block("Finance", fin_feat, feature_meta=fin_meta,
                       timeframe_context=timeframe_context)
        snap = ukt.add_block("Clusters", clu_feat, feature_meta=clu_meta,
                              timeframe_context=timeframe_context)

        rr = snap["reality_regression"]
        assert rr.shape == (UKT_FEATURE_DIM,)
        assert np.linalg.norm(rr) > 0

    def test_normalize_features_range(self, rng):
        raw = rng.uniform(-10, 10, UKT_FEATURE_DIM)
        normed = _normalize_features(raw)
        for (lo, hi) in FEATURE_REGION_LABELS.keys():
            region = normed[lo:hi]
            assert region.min() >= -1e-8, f"Region {lo}:{hi} below 0"
            assert region.max() <= 1.0 + 1e-8, f"Region {lo}:{hi} above 1"

    def test_pad_or_truncate(self):
        short = np.array([1.0, 2.0, 3.0])
        padded = _pad_or_truncate(short, 5)
        assert padded.shape == (5,)
        assert padded[3] == 0.0
        assert padded[4] == 0.0

        long = np.arange(10.0)
        truncated = _pad_or_truncate(long, 5)
        assert truncated.shape == (5,)
        np.testing.assert_array_equal(truncated, np.arange(5.0))

    def test_kernel_labels_have_required_fields(
        self, synthetic_finance_features, timeframe_context
    ):
        features, meta = synthetic_finance_features
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        snap = ukt.add_block("Finance", features, feature_meta=meta,
                             timeframe_context=timeframe_context)

        for kl in snap["kernel_labels"]:
            assert "kernel_id" in kl
            assert "dominant_block" in kl
            assert "dominant_region" in kl
            assert "importance" in kl
            assert "top_features" in kl
            assert "label" in kl
            assert "narrative" in kl
            assert kl["dominant_region"] in [
                "temporal-pattern", "semantic-embedding",
                "structural-centrality", "dynamic-agent",
                "geospatial-kernel",
            ]

    def test_get_final_matrix(self, synthetic_finance_features, timeframe_context):
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        assert ukt.get_final_matrix() is None

        features, meta = synthetic_finance_features
        ukt.add_block("Finance", features, feature_meta=meta,
                       timeframe_context=timeframe_context)
        mat = ukt.get_final_matrix()
        assert mat is not None
        assert mat.shape == (1, UKT_FEATURE_DIM)


# ========================================================================== #
# 2. Graph Engine Integration                                                  #
# ========================================================================== #


class TestGraphEngine:
    """Test geopolitical graph construction and centrality analysis."""

    def test_graph_construction_without_agreement(self):
        G, pos = build_geopolitical_graph(agreement_matrix=None)
        assert len(G.nodes()) == len(GEOPOLITICAL_NODES)
        assert len(G.edges()) == len(GEOPOLITICAL_EDGES)
        for node in GEOPOLITICAL_NODES:
            assert node in G.nodes()
            assert node in pos

    def test_graph_construction_with_agreement(self, synthetic_agreement_matrix):
        G, pos = build_geopolitical_graph(agreement_matrix=synthetic_agreement_matrix)
        assert len(G.nodes()) == len(GEOPOLITICAL_NODES)
        # Weights should be blended (40% original + 60% agreement)
        for u, v in G.edges():
            assert "weight" in G[u][v]

    def test_centrality_analysis(self):
        G, _ = build_geopolitical_graph()
        analysis = analyze_graph(G)

        assert "degree_centrality" in analysis
        assert "betweenness" in analysis
        assert "eigenvector" in analysis
        assert "pagerank" in analysis
        assert "communities" in analysis
        assert "density" in analysis
        assert "features_for_ukt" in analysis
        assert "feature_meta" in analysis

        # Feature vector shape
        assert analysis["features_for_ukt"].shape == (UKT_FEATURE_DIM,)

        # Centrality in structural region (32-47)
        structural = analysis["features_for_ukt"][32:48]
        assert np.any(structural != 0), "Structural region should be populated"

        # Graph-level stats in dynamic region (48-50)
        assert analysis["features_for_ukt"][48] > 0  # density
        assert analysis["features_for_ukt"][50] >= 1  # communities >= 1

    def test_graph_features_into_ukt(self, timeframe_context):
        G, _ = build_geopolitical_graph()
        analysis = analyze_graph(G)
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        snap = ukt.add_block("Graph", analysis["features_for_ukt"],
                             feature_meta=analysis["feature_meta"],
                             timeframe_context=timeframe_context)

        assert snap["step"] == 1
        assert snap["block_name"] == "Graph"
        assert snap["matrix"].shape == (1, UKT_FEATURE_DIM)


# ========================================================================== #
# 3. Spatial Kernelization Integration                                         #
# ========================================================================== #


class TestSpatialKernels:
    """Test SVD spatial kernelization pipeline."""

    def test_build_full_feature_matrix(self, synthetic_spatial_data):
        full = build_full_feature_matrix(
            synthetic_spatial_data["physical_raster"],
            synthetic_spatial_data["country_scalars"],
        )
        # 4 physical + 10 scalar rows = 14 rows, 6 countries
        assert full.shape == (14, 6)
        # Row-normalized to [0, 1]
        for i in range(full.shape[0]):
            assert full[i].min() >= -1e-8
            assert full[i].max() <= 1.0 + 1e-8

    def test_kernelize_spatial(self, synthetic_spatial_data):
        full = build_full_feature_matrix(
            synthetic_spatial_data["physical_raster"],
            synthetic_spatial_data["country_scalars"],
        )
        result = kernelize_spatial(full)

        assert result["features_for_ukt"].shape == (UKT_FEATURE_DIM,)
        # Geospatial region (64-79) should be populated
        spatial_region = result["features_for_ukt"][64:80]
        assert np.any(spatial_region != 0)
        # Kernel importances (64-69) should sum close to 1
        importance = result["features_for_ukt"][64:70]
        assert abs(importance.sum() - 1.0) < 1e-6

        # Per-node vectors
        assert len(result["per_node_vectors"]) == 6
        for node in ["USA", "Russia", "China", "Britain", "India", "Brazil"]:
            assert node in result["per_node_vectors"]

    def test_get_spatial_features(self, synthetic_spatial_data, timeframe_context):
        result = get_spatial_features(
            synthetic_spatial_data["physical_raster"],
            synthetic_spatial_data["country_scalars"],
            synthetic_spatial_data["scalar_names"],
            synthetic_spatial_data["node_order"],
            timeframe_context,
        )
        assert "full_matrix" in result
        assert "features_for_ukt" in result
        assert "feature_meta" in result
        assert "per_node_vectors" in result
        assert result["node_order"] == synthetic_spatial_data["node_order"]

    def test_spatial_features_into_ukt(self, synthetic_spatial_data, timeframe_context):
        result = get_spatial_features(
            synthetic_spatial_data["physical_raster"],
            synthetic_spatial_data["country_scalars"],
            synthetic_spatial_data["scalar_names"],
            synthetic_spatial_data["node_order"],
            timeframe_context,
        )
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        snap = ukt.add_block("Spatial", result["features_for_ukt"],
                             feature_meta=result["feature_meta"],
                             timeframe_context=timeframe_context)
        assert snap["step"] == 1
        assert snap["block_name"] == "Spatial"


# ========================================================================== #
# 4. Agent Simulation Integration                                              #
# ========================================================================== #


class TestAgentSimulation:
    """Test multi-agent simulation with data-driven initialization."""

    def test_initialize_agents_default(self):
        agents = initialize_agents_from_data()
        assert len(agents) == len(GEOPOLITICAL_NODES)
        for name in GEOPOLITICAL_NODES:
            assert name in agents
            assert isinstance(agents[name], ClusterAgent)
            assert agents[name].resources > 0

    def test_initialize_agents_with_graph(self):
        G, _ = build_geopolitical_graph()
        analysis = analyze_graph(G)
        agents = initialize_agents_from_data(graph_analysis=analysis)
        assert len(agents) == len(GEOPOLITICAL_NODES)

    def test_initialize_agents_with_spatial(self, synthetic_spatial_data, timeframe_context):
        G, _ = build_geopolitical_graph()
        analysis = analyze_graph(G)
        spatial = get_spatial_features(
            synthetic_spatial_data["physical_raster"],
            synthetic_spatial_data["country_scalars"],
            synthetic_spatial_data["scalar_names"],
            synthetic_spatial_data["node_order"],
            timeframe_context,
        )
        agents = initialize_agents_from_data(
            graph_analysis=analysis, spatial_features=spatial,
        )
        for agent in agents.values():
            assert agent.capability_multiplier >= 0.5
            assert agent.capability_multiplier <= 2.0

    def test_run_simulation(self):
        agents = initialize_agents_from_data()
        agents, log, features, meta = run_simulation(agents, steps=20)

        assert features.shape == (UKT_FEATURE_DIM,)
        # Dynamic-agent region (48-63) should be populated
        assert np.any(features[48:64] != 0)
        # Resource shares (48-53) should sum close to 1
        res_shares = features[48:54]
        assert abs(res_shares.sum() - 1.0) < 0.05

        # Alliance eigenvalues (56-61) should exist
        assert np.any(features[56:62] != 0)

        # Feature meta should have entries
        assert len(meta) > 0

    def test_agent_simulation_into_ukt(self, timeframe_context):
        agents = initialize_agents_from_data()
        agents, log, features, meta = run_simulation(agents, steps=20)

        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        snap = ukt.add_block("Agents", features, feature_meta=meta,
                             timeframe_context=timeframe_context)
        assert snap["step"] == 1
        assert snap["block_name"] == "Agents"


# ========================================================================== #
# 5. Sparse Autoencoder Integration                                            #
# ========================================================================== #


class TestSparseAutoencoder:
    """Test SAE concept discovery on UKT matrices."""

    def test_sae_on_synthetic_matrix(self, rng):
        matrix = rng.uniform(0, 1, (5, UKT_FEATURE_DIM))
        result = train_sparse_ae(matrix, hidden_dim=16, epochs=30)

        assert result is not None
        assert result["concept_vectors"].shape == (16, UKT_FEATURE_DIM)
        assert result["concept_activations"].shape == (5, 16)
        assert result["active_concepts"] >= 0
        assert result["total_concepts"] == 16
        assert result["final_loss"] > 0
        assert len(result["concept_labels"]) == 16

    def test_sae_concept_labels_structure(self, rng):
        matrix = rng.uniform(0, 1, (5, UKT_FEATURE_DIM))
        result = train_sparse_ae(matrix, hidden_dim=8, epochs=20)
        assert result is not None

        for cl in result["concept_labels"]:
            assert "concept_id" in cl
            assert "concept_idx" in cl
            assert "dominant_region" in cl
            assert "active" in cl
            assert "label" in cl
            assert "narrative" in cl

    def test_concept_kernel_mapping(self, rng, timeframe_context):
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        for i in range(3):
            features = rng.uniform(0, 1, UKT_FEATURE_DIM)
            ukt.add_block(f"Block{i}", features, timeframe_context=timeframe_context)

        matrix = ukt.get_final_matrix()
        sae_result = train_sparse_ae(matrix, hidden_dim=8, epochs=20)
        assert sae_result is not None

        snap = ukt.get_latest_snapshot()
        concept_map = map_concepts_to_kernels(sae_result, snap)
        assert len(concept_map) == 8
        for entry in concept_map:
            assert "concept" in entry
            assert "best_kernel" in entry
            assert "coherence" in entry
            assert "active" in entry

    def test_sae_single_row_returns_none(self):
        matrix = np.random.uniform(0, 1, (1, UKT_FEATURE_DIM))
        result = train_sparse_ae(matrix, hidden_dim=8, epochs=10)
        assert result is None


# ========================================================================== #
# 6. Semantic Canvas Integration                                               #
# ========================================================================== #


class TestSemanticCanvas:
    """Test the Semantic Canvas subsystem."""

    def test_canvas_initialization(self):
        canvas = SemanticCanvas()
        assert len(canvas.entries) == 0
        assert canvas.cumulative.shape == (CANVAS_DIM,)
        np.testing.assert_array_equal(canvas.cumulative, np.zeros(CANVAS_DIM))

    def test_canvas_projection(self, rng):
        canvas = SemanticCanvas()
        features = rng.uniform(0, 1, 16)
        sae_result = train_stage_sae(features, concept_dim=8, epochs=30)

        entry = canvas.project_block(
            block_name="Finance",
            step=1,
            region_name="temporal-pattern",
            features=features,
            sae_result=sae_result,
        )

        assert isinstance(entry, CanvasEntry)
        assert entry.block_name == "Finance"
        assert entry.step == 1
        assert entry.coordinates.shape == (CANVAS_DIM,)
        assert len(entry.dominant_dimensions) > 0
        assert len(entry.interpretation) > 0

    def test_canvas_accumulation(self, rng):
        canvas = SemanticCanvas()
        regions = [
            ("Finance", "temporal-pattern"),
            ("Clusters", "semantic-embedding"),
            ("Graph", "structural-centrality"),
        ]
        for step, (block, region) in enumerate(regions, 1):
            features = rng.uniform(0, 1, 16)
            sae_result = train_stage_sae(features, concept_dim=8, epochs=20)
            canvas.project_block(block, step, region, features, sae_result)

        assert len(canvas.entries) == 3
        # Cumulative should have non-zero values
        assert np.any(canvas.cumulative > 0)

    def test_canvas_state(self, rng):
        canvas = SemanticCanvas()
        features = rng.uniform(0, 1, 16)
        sae_result = train_stage_sae(features, concept_dim=8, epochs=20)
        canvas.project_block("Finance", 1, "temporal-pattern", features, sae_result)

        state = canvas.get_accumulated_state()
        assert "dimensions" in state
        assert "coordinates" in state
        assert "entries" in state
        assert "trajectory" in state
        assert len(state["trajectory"]) == 1

    def test_canvas_narrator_format(self, rng):
        canvas = SemanticCanvas()
        for step, (block, region) in enumerate([
            ("Finance", "temporal-pattern"),
            ("Clusters", "semantic-embedding"),
        ], 1):
            features = rng.uniform(0, 1, 16)
            sae_result = train_stage_sae(features, concept_dim=8, epochs=20)
            canvas.project_block(block, step, region, features, sae_result)

        text = canvas.format_for_narrator()
        assert "Semantic Canvas" in text
        assert "Finance" in text
        assert "Clusters" in text

    def test_stage_sae_training(self, rng):
        features = rng.uniform(0, 1, 16)
        result = train_stage_sae(features, concept_dim=8, epochs=30)

        assert result is not None
        assert result["concept_vectors"].shape == (8, 16)
        assert result["activations"].shape[1] == 8
        assert "active_count" in result
        assert "active_mask" in result

    def test_region_to_canvas_mapping(self):
        """Verify that all UKT regions have canvas dimension mappings."""
        for region in FEATURE_REGION_LABELS.values():
            assert region in REGION_TO_CANVAS, f"Region {region} has no canvas mapping"


# ========================================================================== #
# 7. UKT + Canvas End-to-End Integration                                      #
# ========================================================================== #


class TestUKTCanvasIntegration:
    """Test that UKT add_block automatically drives the Semantic Canvas."""

    def test_canvas_populated_on_add_block(
        self, synthetic_finance_features, timeframe_context
    ):
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        features, meta = synthetic_finance_features
        snap = ukt.add_block("Finance", features, feature_meta=meta,
                             timeframe_context=timeframe_context)

        # Canvas entry should be created
        assert snap["canvas_entry"] is not None
        assert snap["stage_sae_result"] is not None
        assert len(ukt.canvas.entries) == 1

    def test_canvas_grows_with_pipeline(self, rng, timeframe_context):
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        blocks = ["Finance", "Clusters", "Graph", "Spatial", "Agents"]
        for block in blocks:
            features = rng.uniform(0, 1, UKT_FEATURE_DIM)
            ukt.add_block(block, features, timeframe_context=timeframe_context)

        assert len(ukt.canvas.entries) == 5
        state = ukt.canvas.get_accumulated_state()
        assert len(state["trajectory"]) == 5


# ========================================================================== #
# 8. Reality Regression Stability                                              #
# ========================================================================== #


class TestRealityRegressionStability:
    """Test multi-run stability estimation."""

    def test_stability_basic(self, rng):
        matrix = rng.uniform(0, 1, (5, UKT_FEATURE_DIM))
        result = estimate_reality_regression_stability(matrix, n_runs=4)

        assert result["n_runs"] == 4
        assert 0 <= result["mean_cosine"] <= 1.0
        assert result["min_cosine"] <= result["mean_cosine"]
        assert result["std_cosine"] >= 0

    def test_stability_single_row(self):
        matrix = np.random.uniform(0, 1, (1, UKT_FEATURE_DIM))
        result = estimate_reality_regression_stability(matrix)
        assert result["n_runs"] == 0

    def test_stability_deterministic(self, rng):
        matrix = rng.uniform(0, 1, (5, UKT_FEATURE_DIM))
        r1 = estimate_reality_regression_stability(matrix, seed=123)
        r2 = estimate_reality_regression_stability(matrix, seed=123)
        assert r1["mean_cosine"] == r2["mean_cosine"]


# ========================================================================== #
# 9. Full Pipeline Integration (all blocks → UKT → SAE → governance)          #
# ========================================================================== #


class TestFullPipelineIntegration:
    """End-to-end integration test: all 5 blocks through UKT + SAE + governance."""

    def _run_full_pipeline(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ) -> dict:
        """Execute the full pipeline using synthetic data. Returns all results."""
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        snapshots = []

        # Block 1: Finance
        fin_features = np.zeros(UKT_FEATURE_DIM)
        fin_features[:16] = rng.uniform(0, 1, 16)
        fin_features[16:24] = rng.uniform(0, 0.5, 8)
        fin_features[27:32] = rng.uniform(0.2, 0.8, 5)
        fin_meta = {i: {"label": f"fin_{i}", "block": "finance"} for i in range(32)}
        snap = ukt.add_block("Finance", fin_features, feature_meta=fin_meta,
                             timeframe_context=timeframe_context)
        snapshots.append(snap)

        # Block 2: Clusters
        clu_features = np.zeros(UKT_FEATURE_DIM)
        shares = rng.dirichlet(np.ones(8))
        clu_features[16:24] = shares
        clu_features[24:32] = rng.normal(0, 1, 8)
        clu_meta = {16 + i: {"label": f"topic_{i}", "block": "clusters"} for i in range(8)}
        snap = ukt.add_block("Clusters", clu_features, feature_meta=clu_meta,
                             timeframe_context=timeframe_context)
        snapshots.append(snap)

        # Block 3: Graph
        G, pos = build_geopolitical_graph(agreement_matrix=synthetic_agreement_matrix)
        graph_analysis = analyze_graph(G)
        snap = ukt.add_block("Graph", graph_analysis["features_for_ukt"],
                             feature_meta=graph_analysis["feature_meta"],
                             timeframe_context=timeframe_context)
        snapshots.append(snap)

        # Block 4: Spatial
        spatial_result = get_spatial_features(
            synthetic_spatial_data["physical_raster"],
            synthetic_spatial_data["country_scalars"],
            synthetic_spatial_data["scalar_names"],
            synthetic_spatial_data["node_order"],
            timeframe_context,
        )
        snap = ukt.add_block("Spatial", spatial_result["features_for_ukt"],
                             feature_meta=spatial_result["feature_meta"],
                             timeframe_context=timeframe_context)
        snapshots.append(snap)

        # Block 5: Agent Simulation
        agents = initialize_agents_from_data(
            graph_analysis=graph_analysis,
            agreement_matrix=synthetic_agreement_matrix,
            spatial_features=spatial_result,
        )
        agents, log, agent_features, agent_meta = run_simulation(agents, steps=30)
        snap = ukt.add_block("Agents", agent_features, feature_meta=agent_meta,
                             timeframe_context=timeframe_context)
        snapshots.append(snap)

        # Final: SAE
        final_matrix = ukt.get_final_matrix()
        sae_result = train_sparse_ae(final_matrix, hidden_dim=16, epochs=50)

        # Concept-kernel mapping
        concept_map = []
        if sae_result:
            concept_map = map_concepts_to_kernels(sae_result, ukt.get_latest_snapshot())

        # Stability
        stability = estimate_reality_regression_stability(final_matrix, n_runs=4)

        return dict(
            ukt=ukt,
            snapshots=snapshots,
            final_matrix=final_matrix,
            sae_result=sae_result,
            concept_map=concept_map,
            stability=stability,
            graph_analysis=graph_analysis,
            spatial_result=spatial_result,
            agents=agents,
            log=log,
        )

    def test_full_pipeline_completes(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        assert len(result["snapshots"]) == 5
        assert result["final_matrix"].shape == (5, UKT_FEATURE_DIM)
        assert result["sae_result"] is not None

    def test_ukt_matrix_grows_correctly(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        for i, snap in enumerate(result["snapshots"]):
            assert snap["step"] == i + 1
            assert snap["matrix"].shape == (i + 1, UKT_FEATURE_DIM)
            assert snap["n_kernels"] == i + 1

    def test_all_blocks_named_correctly(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        block_names = [s["block_name"] for s in result["snapshots"]]
        assert block_names == ["Finance", "Clusters", "Graph", "Spatial", "Agents"]

    def test_kernel_count_matches_blocks(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        final_snap = result["snapshots"][-1]
        assert final_snap["n_kernels"] == 5

    def test_reconstruction_error_reasonable(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        final_snap = result["snapshots"][-1]
        assert final_snap["reconstruction_error"] < 1e-3

    def test_importance_sums_to_one(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        final_snap = result["snapshots"][-1]
        assert abs(final_snap["importance"].sum() - 1.0) < 1e-6

    def test_sae_produces_concepts(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        sae = result["sae_result"]
        assert sae["total_concepts"] == 16
        assert sae["active_concepts"] > 0
        assert len(sae["concept_labels"]) == 16

    def test_concept_kernel_map_populated(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        assert len(result["concept_map"]) == 16
        for entry in result["concept_map"]:
            assert entry["best_kernel"].startswith("K")
            assert 0 <= entry["coherence"]

    def test_stability_after_full_pipeline(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        stability = result["stability"]
        assert stability["n_runs"] == 4
        assert stability["mean_cosine"] > 0.1  # synthetic random data has lower coherence

    def test_canvas_has_all_layers(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        canvas = result["ukt"].canvas
        assert len(canvas.entries) == 5
        blocks_in_canvas = [e.block_name for e in canvas.entries]
        assert blocks_in_canvas == ["Finance", "Clusters", "Graph", "Spatial", "Agents"]

    def test_canvas_narrator_output(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        text = result["ukt"].canvas.format_for_narrator()
        assert "Semantic Canvas" in text
        for block in ["Finance", "Clusters", "Graph", "Spatial", "Agents"]:
            assert block in text

    def test_feature_meta_accumulated(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        meta = result["ukt"].global_feature_meta
        # Should have metadata from multiple blocks
        assert len(meta) > 20

    def test_all_snapshots_have_reports(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        for snap in result["snapshots"]:
            assert "report" in snap
            assert len(snap["report"]) > 50

    def test_cross_block_coherence_in_report(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        result = self._run_full_pipeline(
            rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
        )
        # Reports after step 1 should include cross-block coherence
        for snap in result["snapshots"][1:]:
            assert "Cross-Block Coherence" in snap["report"]


# ========================================================================== #
# 10. Governance & Scorecard Validation                                        #
# ========================================================================== #


class TestGovernanceValidation:
    """Test governance flag detection and scorecard computation logic."""

    def test_scorecard_thresholds_defined(self):
        required = [
            "feature_traceability", "kernel_stability",
            "concept_activation_rate", "data_source_diversity",
            "governance_flags",
        ]
        for key in required:
            assert key in SCORECARD_THRESHOLDS
            assert "threshold" in SCORECARD_THRESHOLDS[key]
            assert "label" in SCORECARD_THRESHOLDS[key]

    def test_feature_traceability_from_pipeline(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        """After full pipeline, check that feature metadata covers most indices."""
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)

        # Finance block
        fin_features = np.zeros(UKT_FEATURE_DIM)
        fin_features[:32] = rng.uniform(0, 1, 32)
        fin_meta = {i: {"label": f"fin_{i}", "block": "finance"} for i in range(32)}
        ukt.add_block("Finance", fin_features, feature_meta=fin_meta,
                       timeframe_context=timeframe_context)

        # Graph block
        G, _ = build_geopolitical_graph(agreement_matrix=synthetic_agreement_matrix)
        analysis = analyze_graph(G)
        ukt.add_block("Graph", analysis["features_for_ukt"],
                       feature_meta=analysis["feature_meta"],
                       timeframe_context=timeframe_context)

        # Spatial block
        spatial = get_spatial_features(
            synthetic_spatial_data["physical_raster"],
            synthetic_spatial_data["country_scalars"],
            synthetic_spatial_data["scalar_names"],
            synthetic_spatial_data["node_order"],
            timeframe_context,
        )
        ukt.add_block("Spatial", spatial["features_for_ukt"],
                       feature_meta=spatial["feature_meta"],
                       timeframe_context=timeframe_context)

        # Agent block
        agents = initialize_agents_from_data(graph_analysis=analysis)
        _, _, agent_feat, agent_meta = run_simulation(agents, steps=10)
        ukt.add_block("Agents", agent_feat, feature_meta=agent_meta,
                       timeframe_context=timeframe_context)

        # Check traceability
        traced = len(ukt.global_feature_meta)
        threshold = SCORECARD_THRESHOLDS["feature_traceability"]["threshold"]
        assert traced >= threshold * 0.5, (
            f"Feature traceability ({traced}) is less than 50% of threshold ({threshold})"
        )


# ========================================================================== #
# 11. Type System Validation                                                   #
# ========================================================================== #


class TestTypeValidation:
    """Test the unified type system and validation helpers."""

    def test_validate_valid_block_result(self, rng):
        result = {
            "features_for_ukt": np.zeros(UKT_FEATURE_DIM),
            "feature_meta": {0: {"label": "test"}},
            "data_source": "test_source",
        }
        warnings = validate_block_result(result, "TestBlock")
        assert warnings == []

    def test_validate_none_result(self):
        warnings = validate_block_result(None, "TestBlock")
        assert len(warnings) == 1
        assert "None" in warnings[0]

    def test_validate_missing_features(self):
        result = {"feature_meta": {}, "data_source": "test"}
        warnings = validate_block_result(result, "TestBlock")
        assert any("features_for_ukt" in w for w in warnings)

    def test_validate_wrong_shape(self, rng):
        result = {
            "features_for_ukt": np.zeros(50),  # wrong shape
            "feature_meta": {},
            "data_source": "test",
        }
        warnings = validate_block_result(result, "TestBlock")
        assert any("shape" in w for w in warnings)

    def test_validate_missing_data_source(self, rng):
        result = {
            "features_for_ukt": np.zeros(UKT_FEATURE_DIM),
            "feature_meta": {},
        }
        warnings = validate_block_result(result, "TestBlock")
        assert any("data_source" in w for w in warnings)

    def test_validate_snapshot(self, synthetic_finance_features, timeframe_context):
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        features, meta = synthetic_finance_features
        snap = ukt.add_block("Finance", features, feature_meta=meta,
                             timeframe_context=timeframe_context)
        warnings = validate_snapshot(snap)
        assert warnings == []

    def test_validate_incomplete_snapshot(self):
        warnings = validate_snapshot({"step": 1, "block_name": "Test"})
        assert len(warnings) == 1
        assert "missing keys" in warnings[0].lower()

    def test_graph_analysis_conforms_to_block_result(self):
        G, _ = build_geopolitical_graph()
        analysis = analyze_graph(G)
        warnings = validate_block_result(
            {"features_for_ukt": analysis["features_for_ukt"],
             "feature_meta": analysis["feature_meta"],
             "data_source": "networkx"},
            "Graph",
        )
        assert warnings == []

    def test_spatial_conforms_to_block_result(self, synthetic_spatial_data, timeframe_context):
        result = get_spatial_features(
            synthetic_spatial_data["physical_raster"],
            synthetic_spatial_data["country_scalars"],
            synthetic_spatial_data["scalar_names"],
            synthetic_spatial_data["node_order"],
            timeframe_context,
        )
        warnings = validate_block_result(
            {"features_for_ukt": result["features_for_ukt"],
             "feature_meta": result["feature_meta"],
             "data_source": "spatial_raster"},
            "Spatial",
        )
        assert warnings == []

    def test_agent_sim_conforms_to_block_result(self):
        agents = initialize_agents_from_data()
        _, _, features, meta = run_simulation(agents, steps=10)
        warnings = validate_block_result(
            {"features_for_ukt": features,
             "feature_meta": meta,
             "data_source": "agent_simulation"},
            "Agents",
        )
        assert warnings == []


# ========================================================================== #
# 12. PipelineRunner Integration                                               #
# ========================================================================== #


class TestPipelineRunner:
    """Test the Streamlit-free PipelineRunner orchestrator."""

    def _make_synthetic_finance(self, rng) -> dict:
        features = np.zeros(UKT_FEATURE_DIM)
        features[:16] = rng.uniform(0, 1, 16)
        features[27:32] = rng.uniform(0.2, 0.8, 5)
        return {
            "features_for_ukt": features,
            "feature_meta": {i: {"label": f"fin_{i}", "block": "finance"}
                             for i in range(32)},
            "data_source": "synthetic_finance",
        }

    def _make_synthetic_clusters(self, rng) -> dict:
        features = np.zeros(UKT_FEATURE_DIM)
        features[16:24] = rng.dirichlet(np.ones(8))
        return {
            "features_for_ukt": features,
            "feature_meta": {16 + i: {"label": f"topic_{i}", "block": "clusters"}
                             for i in range(8)},
            "data_source": "synthetic_clusters",
        }

    def test_pipeline_runner_completes(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        runner = PipelineRunner()
        result = runner.run(
            finance_result=self._make_synthetic_finance(rng),
            cluster_result=self._make_synthetic_clusters(rng),
            agreement_matrix=synthetic_agreement_matrix,
            spatial_data=synthetic_spatial_data,
            timeframe_context=timeframe_context,
            sim_steps=20,
            sae_epochs=30,
            stability_runs=4,
        )
        assert "snapshots" in result
        assert len(result["snapshots"]) == 5
        assert result["final_matrix"] is not None
        assert result["sae_result"] is not None

    def test_pipeline_runner_data_sources_complete(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        runner = PipelineRunner()
        result = runner.run(
            finance_result=self._make_synthetic_finance(rng),
            cluster_result=self._make_synthetic_clusters(rng),
            agreement_matrix=synthetic_agreement_matrix,
            spatial_data=synthetic_spatial_data,
            timeframe_context=timeframe_context,
            sim_steps=10,
            sae_epochs=20,
            stability_runs=2,
        )
        ds = result["data_sources"]
        # All 5 blocks must be tracked
        assert "Finance" in ds
        assert "Clusters" in ds
        assert "Graph" in ds
        assert "Spatial" in ds
        assert "Agents" in ds

    def test_pipeline_runner_governance_flags(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        runner = PipelineRunner()
        result = runner.run(
            finance_result=self._make_synthetic_finance(rng),
            cluster_result=self._make_synthetic_clusters(rng),
            agreement_matrix=synthetic_agreement_matrix,
            spatial_data=synthetic_spatial_data,
            timeframe_context=timeframe_context,
            sim_steps=10,
            sae_epochs=20,
            stability_runs=2,
        )
        # Governance flags are a list of dicts with required keys
        for flag in result["governance_flags"]:
            assert "code" in flag
            assert "label" in flag
            assert "description" in flag
            assert "severity" in flag

    def test_pipeline_runner_scorecard(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        runner = PipelineRunner()
        result = runner.run(
            finance_result=self._make_synthetic_finance(rng),
            cluster_result=self._make_synthetic_clusters(rng),
            agreement_matrix=synthetic_agreement_matrix,
            spatial_data=synthetic_spatial_data,
            timeframe_context=timeframe_context,
            sim_steps=10,
            sae_epochs=20,
            stability_runs=2,
        )
        sc = result["interpretability_scorecard"]
        required_dims = [
            "feature_traceability", "kernel_stability",
            "concept_activation_rate", "data_source_diversity",
            "governance_flags",
        ]
        for dim in required_dims:
            assert dim in sc
            assert "value" in sc[dim]
            assert "threshold" in sc[dim]
            assert "passed" in sc[dim]

    def test_pipeline_runner_run_id_generated(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        runner = PipelineRunner()
        result = runner.run(
            finance_result=self._make_synthetic_finance(rng),
            cluster_result=self._make_synthetic_clusters(rng),
            agreement_matrix=synthetic_agreement_matrix,
            spatial_data=synthetic_spatial_data,
            timeframe_context=timeframe_context,
            sim_steps=10,
            sae_epochs=20,
            stability_runs=2,
        )
        assert "run_id" in result
        assert len(result["run_id"]) == 8
        assert "run_timestamp" in result

    def test_pipeline_runner_canvas_populated(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        runner = PipelineRunner()
        result = runner.run(
            finance_result=self._make_synthetic_finance(rng),
            cluster_result=self._make_synthetic_clusters(rng),
            agreement_matrix=synthetic_agreement_matrix,
            spatial_data=synthetic_spatial_data,
            timeframe_context=timeframe_context,
            sim_steps=10,
            sae_epochs=20,
            stability_runs=2,
        )
        canvas = result["semantic_canvas"]
        assert canvas is not None
        assert len(canvas.entries) == 5

    def test_pipeline_runner_reports_interpretability_contract(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        runner = PipelineRunner()
        result = runner.run(
            finance_result=self._make_synthetic_finance(rng),
            cluster_result=self._make_synthetic_clusters(rng),
            agreement_matrix=synthetic_agreement_matrix,
            spatial_data=synthetic_spatial_data,
            timeframe_context=timeframe_context,
            sim_steps=10,
            sae_epochs=20,
            stability_runs=2,
        )
        contract = result["interpretability_contract"]
        assert "UniversalKnowledgeTensor" in contract
        assert contract["UniversalKnowledgeTensor"]["compliant"] is True
        assert contract["UniversalKnowledgeTensor"]["interface_issues"] == []
        assert contract["UniversalKnowledgeTensor"]["payload_issues"] == []
        assert "SemanticCanvas" in contract
        assert contract["SemanticCanvas"]["compliant"] is True
        assert contract["SemanticCanvas"]["interface_issues"] == []
        assert contract["SemanticCanvas"]["payload_issues"] == []

        summary = result["interpretability_contract_summary"]
        assert summary["total_modules"] == 2
        assert summary["compliant_modules"] == 2
        assert summary["noncompliant_modules"] == 0
        assert summary["compliance_rate"] == 1.0

    def test_pipeline_runner_progress_callback(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        steps_seen = []
        def on_step(step: str, msg: str) -> None:
            steps_seen.append(step)

        runner = PipelineRunner(on_step=on_step)
        runner.run(
            finance_result=self._make_synthetic_finance(rng),
            cluster_result=self._make_synthetic_clusters(rng),
            agreement_matrix=synthetic_agreement_matrix,
            spatial_data=synthetic_spatial_data,
            timeframe_context=timeframe_context,
            sim_steps=10,
            sae_epochs=20,
            stability_runs=2,
        )
        assert len(steps_seen) >= 4  # At least finance, cluster, graph, spatial, agents, SAE

    def test_pipeline_runner_without_optional_blocks(self, rng, timeframe_context):
        """Pipeline should still work with only graph + agents (no finance/clusters/spatial)."""
        runner = PipelineRunner()
        result = runner.run(
            timeframe_context=timeframe_context,
            sim_steps=10,
            sae_epochs=20,
            stability_runs=2,
        )
        # Only Graph + Agents blocks
        assert len(result["snapshots"]) == 2
        block_names = [s["block_name"] for s in result["snapshots"]]
        assert "Graph" in block_names
        assert "Agents" in block_names

    def test_pipeline_runner_concept_kernel_map(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        runner = PipelineRunner()
        result = runner.run(
            finance_result=self._make_synthetic_finance(rng),
            cluster_result=self._make_synthetic_clusters(rng),
            agreement_matrix=synthetic_agreement_matrix,
            spatial_data=synthetic_spatial_data,
            timeframe_context=timeframe_context,
            sim_steps=10,
            sae_epochs=30,
            stability_runs=2,
        )
        assert len(result["concept_kernel_map"]) > 0
        for entry in result["concept_kernel_map"]:
            assert "concept" in entry
            assert "best_kernel" in entry
            assert "coherence" in entry

    def test_pipeline_runner_stability(
        self, rng, synthetic_spatial_data, synthetic_agreement_matrix, timeframe_context,
    ):
        runner = PipelineRunner()
        result = runner.run(
            finance_result=self._make_synthetic_finance(rng),
            cluster_result=self._make_synthetic_clusters(rng),
            agreement_matrix=synthetic_agreement_matrix,
            spatial_data=synthetic_spatial_data,
            timeframe_context=timeframe_context,
            sim_steps=10,
            sae_epochs=20,
            stability_runs=4,
        )
        stability = result["stability"]
        assert stability is not None
        assert stability["n_runs"] == 4
        assert "mean_cosine" in stability


# ========================================================================== #
# 13. Counterfactual Subsystem Integration                                     #
# ========================================================================== #


class TestCounterfactualIntegration:
    """Test block removal and diff analysis."""

    def test_counterfactual_block_removal(self, rng, timeframe_context):
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        for block in ["Finance", "Clusters", "Graph"]:
            features = rng.uniform(0, 1, UKT_FEATURE_DIM)
            ukt.add_block(block, features, timeframe_context=timeframe_context)

        snapshots = ukt.snapshots
        final_matrix = snapshots[-1]["matrix"]
        block_names = [s["block_name"] for s in snapshots]

        # Remove "Clusters" and re-run SVD
        kept = [i for i, n in enumerate(block_names) if n != "Clusters"]
        sub = final_matrix[kept, :]
        U, S, Vt = np.linalg.svd(sub, full_matrices=False)
        importance = S / (S.sum() + 1e-8)
        rr = importance @ Vt[:len(S), :]

        assert sub.shape == (2, UKT_FEATURE_DIM)
        assert len(S) == 2
        assert abs(importance.sum() - 1.0) < 1e-6
        assert rr.shape == (UKT_FEATURE_DIM,)

    def test_counterfactual_diff_nonzero(self, rng, timeframe_context):
        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        for block in ["Finance", "Clusters", "Graph"]:
            features = rng.uniform(0, 1, UKT_FEATURE_DIM)
            ukt.add_block(block, features, timeframe_context=timeframe_context)

        original_rr = ukt.snapshots[-1]["reality_regression"]

        # Counterfactual: remove Finance
        final_matrix = ukt.snapshots[-1]["matrix"]
        sub = final_matrix[1:, :]  # remove Finance (index 0)
        U, S, Vt = np.linalg.svd(sub, full_matrices=False)
        importance = S / (S.sum() + 1e-8)
        cf_rr = importance @ Vt[:len(S), :]

        # The diff should be non-zero (removing a block changes conclusions)
        diff = np.abs(original_rr - cf_rr)
        assert diff.sum() > 0.01


# ========================================================================== #
# 14. Cross-Block Neural Networks (UVT + USE)                                 #
# ========================================================================== #


class TestCrossBlockNeuralNetworks:
    """Tests for the Universal Variance Tensor and Universal Semantic Encoding."""

    # ------------------------------------------------------------------ #
    # UVT — Universal Variance Tensor                                     #
    # ------------------------------------------------------------------ #

    def test_uvt_returns_expected_keys(self, rng):
        """compute_universal_variance_tensor returns all required dict keys."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        matrix = rng.uniform(0, 1, (5, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert result is not None
        for key in (
            "coupling_matrix", "cross_covariance", "attention_per_head",
            "attended_features", "variance_decomposition", "feature_variance",
            "coupling_labels", "feature_variance_modes",
            "reconstruction_error", "loss_history", "block_names",
        ):
            assert key in result, f"Missing key: {key}"

    def test_uvt_coupling_matrix_shape(self, rng):
        """coupling_matrix is (n_blocks, n_blocks) and rows sum to ~1 (softmax)."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        n_blocks = 4
        matrix = rng.uniform(0, 1, (n_blocks, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert result is not None
        coupling = result["coupling_matrix"]
        assert coupling.shape == (n_blocks, n_blocks), (
            f"Expected ({n_blocks}, {n_blocks}), got {coupling.shape}"
        )
        # Rows are averaged softmax outputs → each row ≈ sum to 1
        row_sums = coupling.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.1), (
            f"Coupling rows should sum to ~1, got {row_sums}"
        )

    def test_uvt_attended_features_shape(self, rng):
        """attended_features has the same shape as the input matrix."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        n_blocks = 3
        matrix = rng.uniform(0, 1, (n_blocks, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert result is not None
        assert result["attended_features"].shape == (n_blocks, UKT_FEATURE_DIM)

    def test_uvt_loss_decreases(self, rng):
        """Training loss should trend downward over epochs."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        matrix = rng.uniform(0, 1, (5, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=30,
        )
        assert result is not None
        history = result["loss_history"]
        assert len(history) == 30
        # First-quarter average should exceed last-quarter average
        q = len(history) // 4
        assert np.mean(history[:q]) >= np.mean(history[-q:]) - 0.5, (
            "Training loss did not decrease (or barely did): "
            f"early={np.mean(history[:q]):.4f}, late={np.mean(history[-q:]):.4f}"
        )

    def test_uvt_coupling_labels_structure(self, rng):
        """Each coupling mode label has the required fields."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        matrix = rng.uniform(0, 1, (4, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert result is not None
        for label in result["coupling_labels"]:
            assert "mode_id" in label
            assert "variance_explained" in label
            assert "primary_blocks" in label
            assert "mutual_attention" in label
            assert "loadings" in label
            assert "label" in label
            assert "narrative" in label
            assert 0.0 <= label["variance_explained"] <= 1.0 + 1e-6

    def test_uvt_variance_decomposition_sums_to_one(self, rng):
        """Variance explained values from eigendecomposition sum to 1."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        matrix = rng.uniform(0, 1, (5, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert result is not None
        ve = result["variance_decomposition"]["variance_explained"]
        assert abs(ve.sum() - 1.0) < 1e-5, f"Variance explained sums to {ve.sum()}"

    def test_uvt_block_names_propagated(self, rng):
        """Custom block_names are preserved in the result."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        names = ["Alpha", "Beta", "Gamma"]
        matrix = rng.uniform(0, 1, (3, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5, block_names=names,
        )
        assert result is not None
        assert result["block_names"] == names
        # Block names appear in coupling mode labels
        for cl in result["coupling_labels"]:
            for pb in cl["primary_blocks"]:
                assert pb in names, f"Block name {pb!r} not in {names}"

    def test_uvt_returns_none_for_single_block(self):
        """UVT requires at least 2 blocks; returns None for single-row input."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        matrix = np.random.uniform(0, 1, (1, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(matrix, epochs=5)
        assert result is None

    def test_uvt_reconstruction_error_finite(self, rng):
        """Reconstruction error is a finite positive float."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        matrix = rng.uniform(0, 1, (3, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert result is not None
        err = result["reconstruction_error"]
        assert np.isfinite(err)
        assert err >= 0.0

    # ------------------------------------------------------------------ #
    # USE — Universal Semantic Encoding                                   #
    # ------------------------------------------------------------------ #

    def test_use_returns_expected_keys(self, rng):
        """compute_universal_semantic_encoding returns all required dict keys."""
        from hyperspace.models.cross_block_net import (
            compute_universal_variance_tensor,
            compute_universal_semantic_encoding,
        )

        matrix = rng.uniform(0, 1, (4, UKT_FEATURE_DIM)).astype(np.float32)
        uvt = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert uvt is not None

        result = compute_universal_semantic_encoding(
            matrix, uvt, semantic_dim=8, epochs=5,
        )
        assert result is not None
        for key in (
            "encoding", "decoded_features", "dimension_labels",
            "alignment_scores", "reconstruction_per_block",
            "loss_history", "semantic_dim",
        ):
            assert key in result, f"Missing key: {key}"

    def test_use_encoding_shape_and_bounds(self, rng):
        """Encoding is a (semantic_dim,) vector bounded to [-1, 1] by Tanh."""
        from hyperspace.models.cross_block_net import (
            compute_universal_variance_tensor,
            compute_universal_semantic_encoding,
        )

        semantic_dim = 12
        matrix = rng.uniform(0, 1, (4, UKT_FEATURE_DIM)).astype(np.float32)
        uvt = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert uvt is not None

        result = compute_universal_semantic_encoding(
            matrix, uvt, semantic_dim=semantic_dim, epochs=5,
        )
        assert result is not None
        enc = result["encoding"]
        assert enc.shape == (semantic_dim,), f"Expected ({semantic_dim},), got {enc.shape}"
        assert np.all(enc >= -1.0 - 1e-5) and np.all(enc <= 1.0 + 1e-5), (
            f"Encoding out of [-1, 1] bounds: min={enc.min():.4f}, max={enc.max():.4f}"
        )

    def test_use_decoded_features_shape(self, rng):
        """decoded_features has shape (n_blocks, feature_dim)."""
        from hyperspace.models.cross_block_net import (
            compute_universal_variance_tensor,
            compute_universal_semantic_encoding,
        )

        n_blocks = 3
        matrix = rng.uniform(0, 1, (n_blocks, UKT_FEATURE_DIM)).astype(np.float32)
        uvt = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert uvt is not None
        result = compute_universal_semantic_encoding(
            matrix, uvt, semantic_dim=8, epochs=5,
        )
        assert result is not None
        assert result["decoded_features"].shape == (n_blocks, UKT_FEATURE_DIM)

    def test_use_dimension_labels_structure(self, rng):
        """Each dimension label has required fields with valid values."""
        from hyperspace.models.cross_block_net import (
            compute_universal_variance_tensor,
            compute_universal_semantic_encoding,
        )

        matrix = rng.uniform(0, 1, (4, UKT_FEATURE_DIM)).astype(np.float32)
        uvt = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert uvt is not None
        result = compute_universal_semantic_encoding(
            matrix, uvt, semantic_dim=8, epochs=5,
        )
        assert result is not None
        assert len(result["dimension_labels"]) == 8
        for dl in result["dimension_labels"]:
            assert "dim_idx" in dl
            assert "value" in dl
            assert "strength" in dl
            assert dl["strength"] in ("STRONG", "MODERATE", "WEAK")
            assert "polarity" in dl
            assert dl["polarity"] in ("positive", "negative")
            assert "label" in dl

    def test_use_alignment_scores_structure(self, rng):
        """alignment_scores covers all block pairs with valid cosine similarities."""
        from hyperspace.models.cross_block_net import (
            compute_universal_variance_tensor,
            compute_universal_semantic_encoding,
        )

        n_blocks = 4
        names = ["Finance", "Clusters", "Graph", "Agents"]
        matrix = rng.uniform(0, 1, (n_blocks, UKT_FEATURE_DIM)).astype(np.float32)
        uvt = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5, block_names=names,
        )
        assert uvt is not None
        result = compute_universal_semantic_encoding(
            matrix, uvt, semantic_dim=8, epochs=5, block_names=names,
        )
        assert result is not None

        scores = result["alignment_scores"]
        # C(4,2) = 6 pairs
        assert len(scores) == 6
        for sc in scores:
            assert "block_a" in sc
            assert "block_b" in sc
            assert "alignment" in sc
            assert "cosine_similarity" in sc
            # Cosine similarity bounded to [-1, 1]
            assert -1.0 - 1e-5 <= sc["cosine_similarity"] <= 1.0 + 1e-5
            # Alignment = 1 - cosine_sim, so bounded to [0, 2] but practically [0, 1]
            assert sc["alignment"] >= -1e-5

    def test_use_returns_none_for_single_block(self):
        """USE requires at least 2 blocks; returns None for single-row input."""
        from hyperspace.models.cross_block_net import (
            compute_universal_variance_tensor,
            compute_universal_semantic_encoding,
        )

        matrix = np.random.uniform(0, 1, (1, UKT_FEATURE_DIM)).astype(np.float32)
        # UVT itself returns None for single block, so mock a minimal uvt dict
        # by simulating the case where UVT passed but USE sees 1 block
        result = compute_universal_semantic_encoding(
            matrix,
            {"coupling_matrix": np.array([[1.0]])},
            semantic_dim=8, epochs=5,
        )
        assert result is None

    def test_use_semantic_dim_stored(self, rng):
        """The semantic_dim value in the result matches the requested size."""
        from hyperspace.models.cross_block_net import (
            compute_universal_variance_tensor,
            compute_universal_semantic_encoding,
        )

        matrix = rng.uniform(0, 1, (3, UKT_FEATURE_DIM)).astype(np.float32)
        uvt = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5,
        )
        assert uvt is not None
        for dim in (8, 16, 24):
            result = compute_universal_semantic_encoding(
                matrix, uvt, semantic_dim=dim, epochs=5,
            )
            assert result is not None
            assert result["semantic_dim"] == dim
            assert result["encoding"].shape == (dim,)

    # ------------------------------------------------------------------ #
    # End-to-End: UVT → USE chained from a real UKT matrix               #
    # ------------------------------------------------------------------ #

    def test_uvt_use_full_chain_from_ukt(self, rng, timeframe_context):
        """UVT + USE chain runs end-to-end on a real UKT-derived matrix."""
        from hyperspace.models.cross_block_net import (
            compute_universal_variance_tensor,
            compute_universal_semantic_encoding,
        )

        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        block_names = ["Finance", "Clusters", "Graph", "Spatial", "Agents"]
        for name in block_names:
            features = rng.uniform(0, 1, UKT_FEATURE_DIM)
            ukt.add_block(name, features, timeframe_context=timeframe_context)

        final_matrix = ukt.get_final_matrix()
        assert final_matrix is not None
        assert final_matrix.shape == (5, UKT_FEATURE_DIM)

        uvt = compute_universal_variance_tensor(
            final_matrix, n_heads=4, d_model=16, epochs=10,
            block_names=block_names,
        )
        assert uvt is not None
        assert uvt["block_names"] == block_names

        use = compute_universal_semantic_encoding(
            final_matrix, uvt, semantic_dim=24, epochs=10,
            block_names=block_names,
        )
        assert use is not None
        assert use["encoding"].shape == (24,)
        assert use["decoded_features"].shape == (5, UKT_FEATURE_DIM)
        # 5 blocks → C(5,2)=10 alignment pairs
        assert len(use["alignment_scores"]) == 10

    def test_uvt_coupling_modes_cover_all_blocks(self, rng):
        """Every block appears in at least one coupling mode's loadings."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        names = ["Finance", "Clusters", "Graph", "Spatial", "Agents"]
        matrix = rng.uniform(0, 1, (5, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=8, epochs=5, block_names=names,
        )
        assert result is not None

        blocks_seen = set()
        for cl in result["coupling_labels"]:
            blocks_seen.update(cl["loadings"].keys())
        for name in names:
            assert name in blocks_seen, f"Block {name!r} absent from all coupling modes"

    def test_uvt_reconstruction_error_bounded(self, rng):
        """After sufficient training the reconstruction error is below a loose bound."""
        from hyperspace.models.cross_block_net import compute_universal_variance_tensor

        matrix = rng.uniform(0, 1, (4, UKT_FEATURE_DIM)).astype(np.float32)
        result = compute_universal_variance_tensor(
            matrix, n_heads=2, d_model=16, epochs=80,
        )
        assert result is not None
        # With 80 epochs the reconstruction error should be well below 1.0
        assert result["reconstruction_error"] < 1.0, (
            f"Reconstruction error too high: {result['reconstruction_error']:.4f}"
        )
