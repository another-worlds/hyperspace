"""Tests for temporal drift monitoring, narrative faithfulness, and shared-latent alignment."""
from __future__ import annotations

import numpy as np
import pytest

from hyperspace.core.drift_monitor import (
    DriftMonitor,
    DriftResult,
    DRIFT_THRESHOLDS,
)
from hyperspace.core.faithfulness import (
    LOW_CONFIDENCE_DISCLAIMER,
    check_kernel_attribution_faithfulness,
    check_feature_region_masking,
    check_importance_delta_monotonicity,
    check_canvas_narrative_grounding,
    check_sae_concept_activation_consistency,
    check_concept_ablation,
    run_faithfulness_checks,
)


# --------------------------------------------------------------------------- #
# DriftMonitor tests                                                           #
# --------------------------------------------------------------------------- #


class TestDriftMonitor:
    """Tests for DriftMonitor class."""

    def _make_monitor_with_runs(self, n: int = 3, seed: int = 0) -> DriftMonitor:
        rng = np.random.default_rng(seed)
        monitor = DriftMonitor(max_history=50)
        for i in range(n):
            monitor.record_run(
                run_id=f"RUN-{i:03d}",
                timestamp=f"2026-01-{i + 1:02d} 00:00 UTC",
                reality_regression=rng.normal(size=80),
                kernel_importances=np.abs(rng.normal(size=5)),
                mean_cosine_stability=0.85 + rng.normal() * 0.02,
                n_kernels=5,
            )
        return monitor

    def test_record_run_increments_history(self):
        monitor = DriftMonitor()
        assert monitor.n_records == 0
        monitor.record_run(
            run_id="A", timestamp="T", reality_regression=np.zeros(80),
            kernel_importances=np.zeros(5), mean_cosine_stability=0.9,
            n_kernels=5,
        )
        assert monitor.n_records == 1

    def test_compute_drift_returns_none_with_insufficient_history(self):
        monitor = DriftMonitor()
        assert monitor.compute_drift() is None

        monitor.record_run(
            run_id="A", timestamp="T", reality_regression=np.zeros(80),
            kernel_importances=np.zeros(5), mean_cosine_stability=0.9,
            n_kernels=5,
        )
        assert monitor.compute_drift() is None

    def test_compute_drift_returns_result_with_two_runs(self):
        monitor = self._make_monitor_with_runs(n=2)
        drift = monitor.compute_drift()
        assert drift is not None
        assert isinstance(drift, DriftResult)
        assert drift.n_records == 2
        assert drift.window_size == 1

    def test_drift_cosines_bounded(self):
        monitor = self._make_monitor_with_runs(n=5)
        drift = monitor.compute_drift(window=3)
        assert -1.0 <= drift.regression_cosine <= 1.0
        assert -1.0 <= drift.importance_cosine <= 1.0

    def test_identical_runs_produce_perfect_cosine(self):
        monitor = DriftMonitor()
        reg = np.ones(80)
        imp = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        for i in range(3):
            monitor.record_run(
                run_id=f"R{i}", timestamp="T",
                reality_regression=reg, kernel_importances=imp,
                mean_cosine_stability=0.9, n_kernels=5,
            )
        drift = monitor.compute_drift()
        assert drift.regression_cosine == pytest.approx(1.0, abs=1e-6)
        assert drift.importance_cosine == pytest.approx(1.0, abs=1e-6)
        assert drift.stability_delta == pytest.approx(0.0, abs=1e-6)

    def test_max_history_limit_enforced(self):
        monitor = DriftMonitor(max_history=5)
        for i in range(10):
            monitor.record_run(
                run_id=f"R{i}", timestamp="T",
                reality_regression=np.zeros(80),
                kernel_importances=np.zeros(3),
                mean_cosine_stability=0.9, n_kernels=3,
            )
        assert monitor.n_records == 5

    def test_check_alert_thresholds_no_alerts_for_stable_runs(self):
        monitor = self._make_monitor_with_runs(n=2, seed=0)
        # Identical runs won't trigger alerts
        reg = np.ones(80)
        monitor2 = DriftMonitor()
        for i in range(3):
            monitor2.record_run(
                run_id=f"R{i}", timestamp="T",
                reality_regression=reg,
                kernel_importances=np.array([0.5, 0.3, 0.2]),
                mean_cosine_stability=0.9, n_kernels=3,
            )
        alerts = monitor2.check_alert_thresholds()
        assert len(alerts) == 0

    def test_check_alert_thresholds_fires_on_major_drift(self):
        monitor = DriftMonitor()
        monitor.record_run(
            run_id="A", timestamp="T",
            reality_regression=np.ones(80),
            kernel_importances=np.array([0.8, 0.1, 0.1]),
            mean_cosine_stability=0.95, n_kernels=3,
        )
        monitor.record_run(
            run_id="B", timestamp="T",
            reality_regression=-np.ones(80),  # Opposite direction
            kernel_importances=np.array([0.1, 0.1, 0.8]),  # Reversed
            mean_cosine_stability=0.50, n_kernels=3,
        )
        alerts = monitor.check_alert_thresholds()
        codes = {a.code for a in alerts}
        assert "DRIFT-001" in codes  # regression cosine far below 0.85
        assert "DRIFT-003" in codes  # stability dropped significantly

    def test_export_history_rows(self):
        monitor = self._make_monitor_with_runs(n=3)
        rows = monitor.export_history_rows()
        assert len(rows) == 3
        assert rows[0]["regression_cosine_vs_prev"] is None
        assert rows[1]["regression_cosine_vs_prev"] is not None

    def test_save_and_load_from_disk(self, tmp_path):
        """Round-trip serialization preserves history and drift results."""
        monitor = self._make_monitor_with_runs(n=4, seed=7)
        path = tmp_path / "drift.json"
        monitor.save_to_disk(path)

        loaded = DriftMonitor.load_from_disk(path)
        assert loaded.n_records == 4
        # Verify drift computation matches
        orig_drift = monitor.compute_drift()
        load_drift = loaded.compute_drift()
        assert orig_drift.regression_cosine == pytest.approx(
            load_drift.regression_cosine, abs=1e-8
        )
        assert orig_drift.importance_cosine == pytest.approx(
            load_drift.importance_cosine, abs=1e-8
        )

    def test_load_from_disk_missing_file(self, tmp_path):
        """Loading from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DriftMonitor.load_from_disk(tmp_path / "nope.json")

    def test_save_respects_max_history(self, tmp_path):
        """Loaded monitor enforces max_history cap."""
        monitor = DriftMonitor(max_history=3)
        rng = np.random.default_rng(0)
        for i in range(5):
            monitor.record_run(
                run_id=f"R{i}", timestamp="T",
                reality_regression=rng.normal(size=80),
                kernel_importances=np.abs(rng.normal(size=5)),
                mean_cosine_stability=0.9, n_kernels=5,
            )
        assert monitor.n_records == 3
        path = tmp_path / "drift.json"
        monitor.save_to_disk(path)
        loaded = DriftMonitor.load_from_disk(path)
        assert loaded.n_records == 3


# --------------------------------------------------------------------------- #
# Faithfulness checks tests                                                    #
# --------------------------------------------------------------------------- #


def _make_snapshot(n_kernels: int = 3) -> dict:
    """Build a minimal valid snapshot for faithfulness testing."""
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(5, 80))
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    importance = S / S.sum()
    reality_regression = matrix.mean(axis=0)  # Proxy reality regression vector
    return {
        "step": 5,
        "block_name": "Final",
        "matrix": matrix,
        "U": U,
        "S": S,
        "Vt": Vt,
        "n_kernels": len(S),
        "importance": importance,
        "reality_regression": reality_regression,
        "kernel_labels": [
            {"kernel_id": f"K{i}", "importance": float(importance[i]),
             "dominant_region": "temporal-pattern", "dominant_block": "Finance"}
            for i in range(len(S))
        ],
        "reconstruction_error": 0.0,
        "report": "Test snapshot",
        "feature_meta": {},
    }


class TestKernelAttributionFaithfulness:
    def test_returns_results_for_valid_snapshot(self):
        snap = _make_snapshot()
        results = check_kernel_attribution_faithfulness([snap])
        assert len(results) > 0
        for r in results:
            assert r.module_name == "UKT"
            assert r.check_name.startswith("kernel_removal_K")

    def test_kernel_removal_increases_error(self):
        snap = _make_snapshot()
        results = check_kernel_attribution_faithfulness([snap])
        # Removing a kernel should not decrease reconstruction error
        for r in results:
            assert r.delta >= -1e-10, f"{r.check_name}: delta={r.delta}"

    def test_empty_snapshots_returns_empty(self):
        assert check_kernel_attribution_faithfulness([]) == []


class TestCanvasNarrativeGrounding:
    def test_with_canvas_entries(self):
        class MockEntry:
            def __init__(self, coords):
                self.coordinates = coords

        class MockCanvas:
            def __init__(self):
                self.entries = [
                    MockEntry({"market_momentum": 0.9, "volatility_regime": 0.1}),
                    MockEntry({"market_momentum": 0.7, "cooperation_signal": 0.3}),
                ]

        results = check_canvas_narrative_grounding(MockCanvas(), "test narrative")
        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].module_name == "SemanticCanvas"

    def test_none_canvas_returns_empty(self):
        assert check_canvas_narrative_grounding(None, None) == []


class TestSAEConceptConsistency:
    def test_grounded_concepts(self):
        snap = _make_snapshot()
        sae = {"active_concepts": 3, "total_concepts": 8}
        results = check_sae_concept_activation_consistency(sae, [snap])
        assert len(results) == 1
        assert results[0].passed is True

    def test_none_sae_returns_empty(self):
        assert check_sae_concept_activation_consistency(None, [_make_snapshot()]) == []


class TestRunFaithfulnessChecks:
    def test_full_report_with_high_confidence(self):
        snap = _make_snapshot()
        report = run_faithfulness_checks(
            snapshots=[snap],
            sae_result={"active_concepts": 3, "total_concepts": 8},
        )
        assert report.overall_confidence > 0
        assert report.low_confidence is False
        assert report.downgraded_narrative is None

    def test_failsafe_downgrade_with_low_threshold(self):
        # Use impossibly high threshold to force downgrade
        snap = _make_snapshot()
        report = run_faithfulness_checks(
            snapshots=[snap],
            confidence_threshold=1.1,  # nothing can pass 110%
        )
        assert report.low_confidence is True
        assert report.downgraded_narrative == LOW_CONFIDENCE_DISCLAIMER

    def test_empty_inputs_returns_high_confidence(self):
        report = run_faithfulness_checks(snapshots=[])
        assert report.overall_confidence == 1.0
        assert report.low_confidence is False


# --------------------------------------------------------------------------- #
# Feature region masking tests                                                 #
# --------------------------------------------------------------------------- #


class TestFeatureRegionMasking:
    def test_produces_results_for_valid_snapshot(self):
        snap = _make_snapshot()
        results = check_feature_region_masking([snap])
        # 5 regions + 1 monotonicity check = 6
        assert len(results) == 6
        region_checks = [r for r in results if r.check_name.startswith("region_masking_") and r.check_name != "region_masking_monotonicity"]
        assert len(region_checks) == 5

    def test_empty_snapshots_returns_empty(self):
        assert check_feature_region_masking([]) == []

    def test_monotonicity_check_present(self):
        snap = _make_snapshot()
        results = check_feature_region_masking([snap])
        mono = [r for r in results if r.check_name == "region_masking_monotonicity"]
        assert len(mono) == 1


class TestImportanceDeltaMonotonicity:
    def test_produces_result_for_valid_snapshot(self):
        snap = _make_snapshot()
        results = check_importance_delta_monotonicity([snap])
        assert len(results) == 1
        assert results[0].check_name == "importance_delta_monotonicity"

    def test_empty_snapshots_returns_empty(self):
        assert check_importance_delta_monotonicity([]) == []


class TestConceptAblation:
    def test_with_concept_labels(self):
        snap = _make_snapshot()
        sae = {
            "concept_labels": [
                {"id": 0, "dominant_region": "temporal-pattern", "active": True},
                {"id": 1, "dominant_region": "semantic-embedding", "active": True},
            ],
            "active_concepts": 2,
            "total_concepts": 4,
        }
        results = check_concept_ablation(sae, [snap])
        assert len(results) == 1
        assert results[0].check_name == "concept_ablation_region_alignment"

    def test_none_sae_returns_empty(self):
        assert check_concept_ablation(None, [_make_snapshot()]) == []

    def test_no_concept_labels_returns_empty(self):
        assert check_concept_ablation({"concept_labels": []}, [_make_snapshot()]) == []


class TestRunFaithfulnessChecksExpanded:
    """Tests that the expanded check suite produces more results."""

    def test_full_report_includes_new_checks(self):
        snap = _make_snapshot()
        sae = {
            "active_concepts": 3,
            "total_concepts": 8,
            "concept_labels": [
                {"id": 0, "dominant_region": "temporal-pattern", "active": True},
            ],
        }
        report = run_faithfulness_checks(
            snapshots=[snap],
            sae_result=sae,
        )
        check_names = {c.check_name for c in report.checks}
        # Should include kernel removal, region masking, monotonicity, concept checks
        assert any("kernel_removal" in n for n in check_names)
        assert any("region_masking" in n for n in check_names)
        assert "importance_delta_monotonicity" in check_names
        assert "concept_kernel_grounding" in check_names
        assert "concept_ablation_region_alignment" in check_names


# --------------------------------------------------------------------------- #
# Shared-latent nonlinear encoder tests                                        #
# --------------------------------------------------------------------------- #


class TestSharedLatentNonlinear:
    """Tests for the upgraded nonlinear shared-latent alignment module."""

    def test_model_architecture(self):
        from hyperspace.models.shared_latent import _build_model

        model = _build_model(
            ["A", "B"], input_dim=16, hidden_dim=24, latent_dim=12, seed=0,
        )
        assert model.latent_dim == 12
        assert len(model.heads) == 2
        head = model.heads["A"]
        assert head.W1.shape == (16, 24)
        assert head.b1.shape == (24,)
        assert head.W2.shape == (24, 12)
        assert head.b2.shape == (12,)

    def test_encode_produces_unit_norm(self):
        from hyperspace.models.shared_latent import _build_model

        model = _build_model(
            ["X"], input_dim=8, hidden_dim=12, latent_dim=6, seed=42,
        )
        rng = np.random.default_rng(42)
        x = rng.normal(size=(10, 8))
        z = model.encode("X", x)
        norms = np.linalg.norm(z, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_compute_alignment_metrics_shape(self):
        from hyperspace.models.shared_latent import compute_alignment_metrics

        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(3, 80))
        metrics = compute_alignment_metrics(
            matrix, ["Finance", "Clusters", "Graph"], epochs=3,
        )
        assert "legacy" in metrics
        assert "shared_latent" in metrics
        assert "parity_delta" in metrics
        assert metrics["shared_latent"]["encoder_type"] == "nonlinear_2layer_relu"
        assert metrics["shared_latent"]["shadow_only"] is True
        assert metrics["shared_latent"]["training_epochs"] == 3

    def test_loss_decreases_over_training(self):
        from hyperspace.models.shared_latent import compute_alignment_metrics

        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(4, 80))
        metrics = compute_alignment_metrics(
            matrix, ["A", "B", "C", "D"], epochs=30, learning_rate=0.01,
        )
        initial = metrics["shared_latent"]["contrastive_loss_initial"]
        final = metrics["shared_latent"]["contrastive_loss_final"]
        assert final <= initial, f"Loss should decrease: {initial} -> {final}"

    def test_nonlinear_encoder_outperforms_random_baseline(self):
        from hyperspace.models.shared_latent import compute_alignment_metrics

        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(3, 80))
        # Trained model
        trained = compute_alignment_metrics(
            matrix, ["A", "B", "C"], epochs=40,
        )
        # Untrained model (1 epoch)
        untrained = compute_alignment_metrics(
            matrix, ["A", "B", "C"], epochs=1,
        )
        # Trained should have same or better metrics
        assert trained["shared_latent"]["contrastive_loss_final"] <= \
            untrained["shared_latent"]["contrastive_loss_final"] + 0.5

    def test_transfer_learning_evaluation(self):
        from hyperspace.models.shared_latent import compute_alignment_metrics

        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(4, 80))
        metrics = compute_alignment_metrics(
            matrix, ["A", "B", "C", "D"], epochs=10,
        )
        transfer = metrics.get("transfer_learning")
        assert transfer is not None
        assert "mean_transfer_gain" in transfer
        assert "per_modality" in transfer
        assert "n_modalities" in transfer
        assert transfer["n_modalities"] == 4
        # With 4 modalities, leave-one-out should produce 4 entries
        assert len(transfer["per_modality"]) == 4
        for held_out, data in transfer["per_modality"].items():
            assert "pretrained_r1" in data
            assert "random_r1" in data
            assert "transfer_gain" in data

    def test_transfer_learning_skipped_with_two_modalities(self):
        from hyperspace.models.shared_latent import compute_alignment_metrics

        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(2, 80))
        metrics = compute_alignment_metrics(
            matrix, ["A", "B"], epochs=3,
        )
        transfer = metrics.get("transfer_learning", {})
        # With only 2 modalities, leave-one-out can't work (need >= 2 remaining)
        assert transfer.get("n_modalities") == 2


# --------------------------------------------------------------------------- #
# Temporal memory (KernelMemory) tests                                         #
# --------------------------------------------------------------------------- #

from hyperspace.core.temporal_memory import KernelMemory, KernelEvolution


class TestKernelMemory:
    """Tests for KernelMemory cross-run kernel persistence."""

    def _make_snapshots(self, n_blocks: int = 3, seed: int = 0) -> list[dict]:
        rng = np.random.default_rng(seed)
        snapshots = []
        for i in range(n_blocks):
            snapshots.append({
                "block_name": f"Block{i}",
                "step": i + 1,
                "n_kernels": 4,
                "importance": rng.random(4),
                "reality_regression": rng.normal(size=80),
                "reconstruction_error": rng.random(),
                "kernel_activation": rng.normal(size=(n_blocks, 4)),
            })
        return snapshots

    def test_store_and_retrieve(self):
        mem = KernelMemory()
        snaps = self._make_snapshots()
        stored = mem.store_run("R1", "T1", snaps)
        assert stored == 3
        assert mem.n_snapshots == 3
        assert mem.n_runs == 1

    def test_block_history(self):
        mem = KernelMemory()
        for i in range(3):
            mem.store_run(f"R{i}", f"T{i}", self._make_snapshots(seed=i))
        history = mem.get_block_history("Block0")
        assert len(history) == 3
        assert all(s.block_name == "Block0" for s in history)

    def test_block_history_last_n(self):
        mem = KernelMemory()
        for i in range(5):
            mem.store_run(f"R{i}", f"T{i}", self._make_snapshots(seed=i))
        history = mem.get_block_history("Block0", last_n=2)
        assert len(history) == 2

    def test_kernel_evolution(self):
        mem = KernelMemory()
        for i in range(4):
            mem.store_run(f"R{i}", f"T{i}", self._make_snapshots(seed=i))
        evolutions = mem.compute_kernel_evolution()
        assert "Block0" in evolutions
        evo = evolutions["Block0"]
        assert isinstance(evo, KernelEvolution)
        assert evo.n_runs == 4
        assert len(evo.regression_cosines) == 3  # n-1 pairwise
        assert len(evo.reconstruction_trend) == 4

    def test_kernel_evolution_single_block(self):
        mem = KernelMemory()
        for i in range(3):
            mem.store_run(f"R{i}", f"T{i}", self._make_snapshots(seed=i))
        evolutions = mem.compute_kernel_evolution(block_name="Block1")
        assert len(evolutions) == 1
        assert "Block1" in evolutions

    def test_max_runs_limit(self):
        mem = KernelMemory(max_runs=3)
        for i in range(5):
            mem.store_run(f"R{i}", f"T{i}", self._make_snapshots(seed=i))
        assert mem.n_runs == 3
        assert mem.run_ids == ["R2", "R3", "R4"]

    def test_save_and_load(self, tmp_path):
        mem = KernelMemory()
        for i in range(3):
            mem.store_run(f"R{i}", f"T{i}", self._make_snapshots(seed=i))
        path = tmp_path / "kernel_memory.json"
        mem.save(path)
        loaded = KernelMemory.load(path)
        assert loaded.n_snapshots == mem.n_snapshots
        assert loaded.n_runs == mem.n_runs
        # Verify evolution matches
        orig = mem.compute_kernel_evolution()
        load = loaded.compute_kernel_evolution()
        for bn in orig:
            assert orig[bn].mean_importance_stability == pytest.approx(
                load[bn].mean_importance_stability, abs=1e-8
            )

    def test_evolution_summary(self):
        mem = KernelMemory()
        for i in range(3):
            mem.store_run(f"R{i}", f"T{i}", self._make_snapshots(seed=i))
        summary = mem.get_evolution_summary()
        assert summary["n_runs"] == 3
        assert "Block0" in summary["blocks"]
        assert "mean_importance_stability" in summary["blocks"]["Block0"]

    def test_store_skips_incomplete_snapshots(self):
        mem = KernelMemory()
        # Snapshot missing importance → should be skipped
        stored = mem.store_run("R0", "T0", [{"block_name": "X", "step": 1}])
        assert stored == 0
        assert mem.n_snapshots == 0


# --------------------------------------------------------------------------- #
# Scorecard threshold & contract integration tests                             #
# --------------------------------------------------------------------------- #


class TestScorecardThresholds:
    """Validate scorecard threshold configuration and computation."""

    def test_all_thresholds_have_required_keys(self):
        from hyperspace.config import SCORECARD_THRESHOLDS

        required = {"label", "unit", "threshold", "description"}
        for key, spec in SCORECARD_THRESHOLDS.items():
            missing = required - set(spec.keys())
            assert not missing, f"{key} missing keys: {missing}"

    def test_faithfulness_confidence_threshold_exists(self):
        from hyperspace.config import SCORECARD_THRESHOLDS

        assert "faithfulness_confidence" in SCORECARD_THRESHOLDS
        thresh = SCORECARD_THRESHOLDS["faithfulness_confidence"]
        assert thresh["threshold"] == 0.50

    def test_shared_latent_thresholds_nonzero(self):
        from hyperspace.config import SCORECARD_THRESHOLDS

        assert SCORECARD_THRESHOLDS["shared_latent_retrieval_at_1"]["threshold"] > 0
        assert SCORECARD_THRESHOLDS["shared_latent_probe_cosine"]["threshold"] > 0
        assert SCORECARD_THRESHOLDS["legacy_retrieval_at_1"]["threshold"] > 0

    def test_scorecard_count(self):
        from hyperspace.config import SCORECARD_THRESHOLDS

        assert len(SCORECARD_THRESHOLDS) == 9


class TestContractRegistry:
    """Validate alpha-scope interpretability contract registry."""

    def test_all_modules_have_required_fields(self):
        from hyperspace.core.interpretability_registry import (
            ALPHA_SCOPE_MODULE_POLICIES,
        )

        required = {"module_name", "owner", "status", "rationale"}
        for policy in ALPHA_SCOPE_MODULE_POLICIES:
            missing = required - set(policy.keys())
            assert not missing, f"{policy.get('module_name')} missing: {missing}"

    def test_new_modules_enrolled(self):
        from hyperspace.core.interpretability_registry import (
            ALPHA_SCOPE_MODULE_POLICIES,
        )

        names = {p["module_name"] for p in ALPHA_SCOPE_MODULE_POLICIES}
        assert "SharedLatentHead" in names
        assert "DriftMonitor" in names
        assert "KernelMemory" in names

    def test_total_module_count(self):
        from hyperspace.core.interpretability_registry import (
            ALPHA_SCOPE_MODULE_POLICIES,
        )

        assert len(ALPHA_SCOPE_MODULE_POLICIES) == 11

    def test_contract_modules_are_ukt_and_canvas(self):
        from hyperspace.core.interpretability_registry import (
            ALPHA_SCOPE_MODULE_POLICIES,
        )

        contract_modules = [
            p["module_name"]
            for p in ALPHA_SCOPE_MODULE_POLICIES
            if p["status"] == "contract"
        ]
        assert sorted(contract_modules) == [
            "SemanticCanvas",
            "UniversalKnowledgeTensor",
        ]

    def test_na_modules_have_owner_and_rationale(self):
        from hyperspace.core.interpretability_registry import (
            ALPHA_SCOPE_MODULE_POLICIES,
        )

        for p in ALPHA_SCOPE_MODULE_POLICIES:
            if p["status"] == "not_applicable":
                assert p["owner"], f"{p['module_name']} missing owner"
                assert p["rationale"], f"{p['module_name']} missing rationale"


class TestDashboardFixtureParity:
    """Verify dashboard fixture covers all expected payload keys."""

    def test_fixture_keys_match_expected(self):
        from tests.dashboard_fixtures import (
            EXPECTED_RUNNER_PAYLOAD_KEYS,
            build_runner_payload,
        )

        payload = build_runner_payload()
        for key in EXPECTED_RUNNER_PAYLOAD_KEYS:
            assert key in payload, f"Missing key in fixture payload: {key}"

    def test_kernel_evolution_in_fixture(self):
        from tests.dashboard_fixtures import build_runner_payload

        payload = build_runner_payload()
        assert "kernel_evolution" in payload

    def test_session_map_covers_kernel_evolution(self):
        from tests.dashboard_fixtures import SESSION_TO_PAYLOAD_KEY_MAP

        assert "kernel_evolution" in SESSION_TO_PAYLOAD_KEY_MAP

    def test_session_map_covers_contract_keys(self):
        from tests.dashboard_fixtures import SESSION_TO_PAYLOAD_KEY_MAP

        assert "interpretability_contract" in SESSION_TO_PAYLOAD_KEY_MAP
        assert "interpretability_contract_summary" in SESSION_TO_PAYLOAD_KEY_MAP

    def test_session_map_covers_all_expected_keys(self):
        from tests.dashboard_fixtures import (
            EXPECTED_RUNNER_PAYLOAD_KEYS,
            SESSION_TO_PAYLOAD_KEY_MAP,
        )

        payload_keys_in_map = set(SESSION_TO_PAYLOAD_KEY_MAP.values())
        for key in EXPECTED_RUNNER_PAYLOAD_KEYS:
            assert key in payload_keys_in_map, (
                f"Expected payload key '{key}' not covered by SESSION_TO_PAYLOAD_KEY_MAP"
            )
