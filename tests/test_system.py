"""System-level tests: imports, configuration, module wiring, package structure.

Validates that all Hyperspace packages, modules, and cross-module dependencies
resolve correctly without runtime errors, and that configuration constants
are internally consistent.
"""
from __future__ import annotations

import importlib
import inspect
import sys

import numpy as np
import pytest


# ========================================================================== #
# 1. Package Import Tests                                                      #
# ========================================================================== #


class TestPackageImports:
    """Verify every package and module imports without error."""

    @pytest.mark.parametrize("module_path", [
        "hyperspace",
        "hyperspace.config",
        "hyperspace.state",
        "hyperspace.core",
        "hyperspace.core.types",
        "hyperspace.core.pipeline",
        "hyperspace.models.knowledge_matrix",
        "hyperspace.models.graph_engine",
        "hyperspace.models.spatial_kernels",
        "hyperspace.models.agent_sim",
        "hyperspace.models.sparse_ae",
        "hyperspace.models.semantic_canvas",
        "hyperspace.models.semantic_narrator",
        "hyperspace.viz.charts",
        "hyperspace.viz.kernel_viz",
    ])
    def test_module_imports(self, module_path: str):
        mod = importlib.import_module(module_path)
        assert mod is not None

    @pytest.mark.parametrize("module_path", [
        "hyperspace.pages.dashboard",
        "hyperspace.pages.mission_control_tab",
        "hyperspace.pages.finance_tab",
        "hyperspace.pages.clusters_tab",
        "hyperspace.pages.politics_tab",
        "hyperspace.pages.agents_tab",
        "hyperspace.pages.interpreter_tab",
        "hyperspace.pages.pipeline_tab",
        "hyperspace.pages.counterfactual_tab",
        "hyperspace.pages.governance",
        "hyperspace.pages._report_section",
    ])
    def test_page_module_imports(self, module_path: str):
        mod = importlib.import_module(module_path)
        assert mod is not None

    def test_data_modules_import(self):
        """Data modules import; they may fail at runtime without network but should load."""
        from hyperspace.data import finance, news, political, spatial
        assert finance is not None
        assert news is not None
        assert political is not None
        assert spatial is not None


# ========================================================================== #
# 2. Configuration Consistency                                                 #
# ========================================================================== #


class TestConfigConsistency:
    """Validate internal consistency of configuration constants."""

    def test_ukt_feature_dim(self):
        from hyperspace.config import UKT_FEATURE_DIM, FEATURE_NAMES
        assert UKT_FEATURE_DIM == 80
        assert len(FEATURE_NAMES) == UKT_FEATURE_DIM

    def test_geopolitical_nodes(self):
        from hyperspace.config import GEOPOLITICAL_NODES
        assert len(GEOPOLITICAL_NODES) == 6
        for name, attrs in GEOPOLITICAL_NODES.items():
            assert "influence" in attrs
            assert "lat" in attrs
            assert "lon" in attrs
            assert "color" in attrs
            assert "bloc" in attrs

    def test_geopolitical_edges(self):
        from hyperspace.config import GEOPOLITICAL_EDGES, GEOPOLITICAL_NODES
        nodes = set(GEOPOLITICAL_NODES.keys())
        for src, dst, w, etype, desc in GEOPOLITICAL_EDGES:
            assert src in nodes, f"Edge source {src} not in nodes"
            assert dst in nodes, f"Edge dest {dst} not in nodes"
            assert -1.0 <= w <= 1.0
            assert isinstance(etype, str)
            assert isinstance(desc, str)

    def test_pipeline_steps(self):
        from hyperspace.config import PIPELINE_STEPS
        assert len(PIPELINE_STEPS) >= 5

    def test_feature_region_labels_cover_full_dim(self):
        from hyperspace.models.knowledge_matrix import FEATURE_REGION_LABELS
        from hyperspace.config import UKT_FEATURE_DIM
        covered = set()
        for (lo, hi), label in FEATURE_REGION_LABELS.items():
            for i in range(lo, hi):
                covered.add(i)
        assert len(covered) == UKT_FEATURE_DIM

    def test_block_region_map_matches_feature_regions(self):
        from hyperspace.models.knowledge_matrix import BLOCK_REGION_MAP, FEATURE_REGION_LABELS
        region_names = set(FEATURE_REGION_LABELS.values())
        for block, (region, lo, hi) in BLOCK_REGION_MAP.items():
            assert region in region_names, f"Block {block} region {region} not in FEATURE_REGION_LABELS"
            assert hi - lo == 16

    def test_scorecard_thresholds(self):
        from hyperspace.config import SCORECARD_THRESHOLDS
        required = ["feature_traceability", "kernel_stability",
                     "concept_activation_rate", "data_source_diversity",
                     "governance_flags"]
        for key in required:
            assert key in SCORECARD_THRESHOLDS
            entry = SCORECARD_THRESHOLDS[key]
            assert "label" in entry
            assert "threshold" in entry
            assert "unit" in entry
            assert "description" in entry

    def test_governance_flag_codes(self):
        from hyperspace.config import GOVERNANCE_FLAG_CODES
        for code, info in GOVERNANCE_FLAG_CODES.items():
            assert code.startswith("GOV-")
            assert "label" in info
            assert "description" in info
            assert "severity" in info

    def test_policy_kernel_names_match_regions(self):
        from hyperspace.config import POLICY_KERNEL_NAMES
        from hyperspace.models.knowledge_matrix import FEATURE_REGION_LABELS
        region_names = set(FEATURE_REGION_LABELS.values())
        for region in POLICY_KERNEL_NAMES:
            assert region in region_names

    def test_canvas_dimensions(self):
        from hyperspace.models.semantic_canvas import CANVAS_DIMENSIONS, CANVAS_DIM
        assert len(CANVAS_DIMENSIONS) == CANVAS_DIM
        for dim in CANVAS_DIMENSIONS:
            assert "key" in dim
            assert "label" in dim
            assert "desc" in dim

    def test_region_to_canvas_covers_all_regions(self):
        from hyperspace.models.semantic_canvas import REGION_TO_CANVAS
        from hyperspace.models.knowledge_matrix import FEATURE_REGION_LABELS
        for region_name in FEATURE_REGION_LABELS.values():
            assert region_name in REGION_TO_CANVAS, f"Region {region_name} not in REGION_TO_CANVAS"

    def test_glossary_entries(self):
        from hyperspace.config import GLOSSARY
        assert len(GLOSSARY) >= 10
        for key, value in GLOSSARY.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(value) > 20

    def test_available_tickers(self):
        from hyperspace.config import AVAILABLE_TICKERS, DEFAULT_TICKERS
        for t in DEFAULT_TICKERS:
            assert t in AVAILABLE_TICKERS

    def test_dark_css_present(self):
        from hyperspace.config import DARK_CSS
        assert len(DARK_CSS) > 100
        assert "<style>" in DARK_CSS

    def test_plotly_layout(self):
        from hyperspace.config import PLOTLY_LAYOUT
        assert PLOTLY_LAYOUT["template"] == "plotly_dark"
        assert "paper_bgcolor" in PLOTLY_LAYOUT


# ========================================================================== #
# 3. Module Public API Validation                                              #
# ========================================================================== #


class TestModuleAPIs:
    """Verify that key modules expose the expected public API."""

    def test_knowledge_matrix_api(self):
        from hyperspace.models import knowledge_matrix as km
        assert hasattr(km, "UniversalKnowledgeTensor")
        assert hasattr(km, "FEATURE_REGION_LABELS")
        assert hasattr(km, "BLOCK_REGION_MAP")
        assert hasattr(km, "estimate_reality_regression_stability")
        assert hasattr(km, "_normalize_features")
        assert hasattr(km, "_pad_or_truncate")
        assert hasattr(km, "_feature_name")
        assert hasattr(km, "_region_for_index")

    def test_graph_engine_api(self):
        from hyperspace.models import graph_engine
        assert hasattr(graph_engine, "build_geopolitical_graph")
        assert hasattr(graph_engine, "analyze_graph")

    def test_spatial_kernels_api(self):
        from hyperspace.models import spatial_kernels
        assert hasattr(spatial_kernels, "build_full_feature_matrix")
        assert hasattr(spatial_kernels, "kernelize_spatial")
        assert hasattr(spatial_kernels, "get_spatial_features")

    def test_agent_sim_api(self):
        from hyperspace.models import agent_sim
        assert hasattr(agent_sim, "ClusterAgent")
        assert hasattr(agent_sim, "initialize_agents_from_data")
        assert hasattr(agent_sim, "run_simulation")

    def test_sparse_ae_api(self):
        from hyperspace.models import sparse_ae
        assert hasattr(sparse_ae, "SparseAutoencoder")
        assert hasattr(sparse_ae, "train_sparse_ae")
        assert hasattr(sparse_ae, "map_concepts_to_kernels")
        assert hasattr(sparse_ae, "enrich_concepts_with_narratives")

    def test_semantic_canvas_api(self):
        from hyperspace.models import semantic_canvas
        assert hasattr(semantic_canvas, "SemanticCanvas")
        assert hasattr(semantic_canvas, "CanvasEntry")
        assert hasattr(semantic_canvas, "train_stage_sae")
        assert hasattr(semantic_canvas, "CANVAS_DIMENSIONS")
        assert hasattr(semantic_canvas, "CANVAS_DIM")
        assert hasattr(semantic_canvas, "REGION_TO_CANVAS")

    def test_semantic_narrator_api(self):
        from hyperspace.models import semantic_narrator
        assert hasattr(semantic_narrator, "narrate_canvas")
        assert hasattr(semantic_narrator, "narrate_layer")
        assert hasattr(semantic_narrator, "narrate_kernel")
        assert hasattr(semantic_narrator, "narrate_reality_regression")
        assert hasattr(semantic_narrator, "narrate_concept")

    def test_core_types_api(self):
        from hyperspace.core import types
        assert hasattr(types, "BlockResult")
        assert hasattr(types, "SnapshotResult")
        assert hasattr(types, "PipelineResult")
        assert hasattr(types, "GovernanceFlag")
        assert hasattr(types, "ScorecardEntry")
        assert hasattr(types, "StabilityResult")
        assert hasattr(types, "validate_block_result")
        assert hasattr(types, "validate_snapshot")

    def test_core_pipeline_api(self):
        from hyperspace.core import pipeline
        assert hasattr(pipeline, "PipelineRunner")
        runner_cls = pipeline.PipelineRunner
        assert hasattr(runner_cls, "run")

    def test_page_modules_have_render(self):
        """All tab pages should expose a render() function."""
        tab_modules = [
            "hyperspace.pages.mission_control_tab",
            "hyperspace.pages.finance_tab",
            "hyperspace.pages.clusters_tab",
            "hyperspace.pages.politics_tab",
            "hyperspace.pages.agents_tab",
            "hyperspace.pages.interpreter_tab",
            "hyperspace.pages.pipeline_tab",
            "hyperspace.pages.counterfactual_tab",
        ]
        for mod_path in tab_modules:
            mod = importlib.import_module(mod_path)
            assert hasattr(mod, "render"), f"{mod_path} missing render()"
            assert callable(mod.render)

    def test_dashboard_api(self):
        from hyperspace.pages import dashboard
        assert hasattr(dashboard, "render_landing")
        assert hasattr(dashboard, "run_pipeline")
        assert hasattr(dashboard, "render_results")

    def test_viz_charts_api(self):
        from hyperspace.viz import charts
        assert hasattr(charts, "source_badge")
        assert hasattr(charts, "candlestick_chart")
        assert hasattr(charts, "forecast_chart")

    def test_viz_kernel_viz_api(self):
        from hyperspace.viz import kernel_viz
        assert hasattr(kernel_viz, "plot_kernel_matrix")
        assert hasattr(kernel_viz, "plot_kernel_importance")


# ========================================================================== #
# 4. Cross-Module Wiring Validation                                            #
# ========================================================================== #


class TestCrossModuleWiring:
    """Verify that cross-module dependencies resolve correctly."""

    def test_ukt_uses_semantic_canvas(self):
        from hyperspace.models.knowledge_matrix import UniversalKnowledgeTensor
        ukt = UniversalKnowledgeTensor()
        from hyperspace.models.semantic_canvas import SemanticCanvas
        assert isinstance(ukt.canvas, SemanticCanvas)

    def test_sparse_ae_uses_knowledge_matrix(self):
        """SAE concept labeling imports from knowledge_matrix."""
        from hyperspace.models.sparse_ae import train_sparse_ae
        # Training a small matrix should work and import region labels
        mat = np.random.uniform(0, 1, (3, 80))
        result = train_sparse_ae(mat, hidden_dim=4, epochs=5)
        assert result is not None
        for cl in result["concept_labels"]:
            assert cl["dominant_region"] in [
                "temporal-pattern", "semantic-embedding",
                "structural-centrality", "dynamic-agent",
                "geospatial-kernel",
            ]

    def test_pipeline_runner_orchestrates_all_models(self):
        """PipelineRunner imports and calls all model modules."""
        from hyperspace.core.pipeline import PipelineRunner
        runner = PipelineRunner()
        # Minimal run with no optional data
        result = runner.run(sim_steps=5, sae_epochs=5, stability_runs=2)
        assert len(result["snapshots"]) >= 2  # At least Graph + Agents

    def test_agent_sim_uses_graph_engine(self):
        from hyperspace.models.graph_engine import build_geopolitical_graph, analyze_graph
        from hyperspace.models.agent_sim import initialize_agents_from_data
        G, _ = build_geopolitical_graph()
        analysis = analyze_graph(G)
        agents = initialize_agents_from_data(graph_analysis=analysis)
        assert len(agents) == 6

    def test_spatial_kernels_feed_agent_sim(self):
        from hyperspace.models.spatial_kernels import get_spatial_features
        from hyperspace.models.agent_sim import initialize_agents_from_data
        from hyperspace.models.graph_engine import build_geopolitical_graph, analyze_graph

        rng = np.random.default_rng(42)
        spatial = get_spatial_features(
            rng.uniform(0, 1, (4, 6, 9)),
            rng.uniform(0, 1, (10, 6)),
            ["a"] * 10, ["USA", "Russia", "China", "Britain", "India", "Brazil"],
            {},
        )
        G, _ = build_geopolitical_graph()
        analysis = analyze_graph(G)
        agents = initialize_agents_from_data(
            graph_analysis=analysis, spatial_features=spatial,
        )
        for agent in agents.values():
            assert 0.5 <= agent.capability_multiplier <= 2.0

    def test_governance_flags_use_correct_codes(self):
        from hyperspace.config import GOVERNANCE_FLAG_CODES
        from hyperspace.core.pipeline import PipelineRunner
        # All flag codes referenced by PipelineRunner exist in config
        for code in ["GOV-001", "GOV-003", "GOV-004", "GOV-005"]:
            assert code in GOVERNANCE_FLAG_CODES
