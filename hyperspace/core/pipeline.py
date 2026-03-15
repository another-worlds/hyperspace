"""Streamlit-free pipeline runner for the Hyperspace system.

Orchestrates all 5 pipeline blocks through the UKT, Semantic Canvas,
Sparse Autoencoder, and governance systems without any dependency on
Streamlit session state or UI components.

Usage:
    runner = PipelineRunner()
    result = runner.run(finance_data=..., docs=..., agreement=..., spatial=...)

This module is the canonical integration point — dashboard.py delegates
to this runner and maps the result into session state.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from hyperspace.config import (
    GEOPOLITICAL_NODES,
    GOVERNANCE_FLAG_CODES,
    SCORECARD_THRESHOLDS,
    UKT_FEATURE_DIM,
    FEATURE_FLAGS,
)
from hyperspace.core.types import (
    GovernanceFlag,
    PipelineResult,
    ScorecardEntry,
    validate_block_result,
    summarize_interpretable_reports,
)
from hyperspace.core.faithfulness import run_faithfulness_checks
from hyperspace.core.drift_monitor import DriftMonitor
from hyperspace.core.temporal_memory import KernelMemory
from hyperspace.core.interpretability_registry import (
    build_alpha_scope_contract_reports,
    enforce_alpha_scope_contract_coverage,
)
from hyperspace.core.latent_versioning import (
    LatentVersionTrail,
    compute_latent_version,
    build_concept_audit_record,
)
from hyperspace.models.knowledge_matrix import (
    UniversalKnowledgeTensor,
    estimate_reality_regression_stability,
    FEATURE_REGION_LABELS,
    HYPERSPACE_REGISTRY,
)


class PipelineRunner:
    """Orchestrate the full Hyperspace pipeline.

    Accepts pre-fetched data and runs all blocks in sequence through the UKT.
    Returns a PipelineResult dict with everything needed for rendering.
    """

    def __init__(
        self,
        on_step: Callable[[str, str], None] | None = None,
        drift_monitor: DriftMonitor | None = None,
        kernel_memory: KernelMemory | None = None,
        version_trail: LatentVersionTrail | None = None,
    ):
        """
        Args:
            on_step: Optional callback(step_name, message) for progress reporting.
            drift_monitor: Optional DriftMonitor for temporal drift tracking.
                When provided, each run is recorded and drift is computed.
            kernel_memory: Optional KernelMemory for cross-run kernel persistence.
                When provided, kernel snapshots are stored and evolution is tracked.
            version_trail: Optional LatentVersionTrail for latent space versioning.
                When provided, captures latent space fingerprints and concept
                vocabulary snapshots per run for governance auditing.
        """
        self._on_step = on_step or (lambda s, m: None)
        self._warnings: list[str] = []
        self._drift_monitor = drift_monitor
        self._kernel_memory = kernel_memory
        self._version_trail = version_trail

    def _report(self, step: str, msg: str) -> None:
        self._on_step(step, msg)

    def run(
        self,
        *,
        finance_result: dict | None = None,
        cluster_result: dict | None = None,
        agreement_matrix: pd.DataFrame | None = None,
        spatial_data: dict | None = None,
        timeframe_context: dict | None = None,
        sim_steps: int = 50,
        sae_hidden_dim: int = 16,
        sae_epochs: int = 80,
        stability_runs: int = 8,
        compute_cross_block: bool = False,
        cross_block_epochs_uvt: int = 120,
        cross_block_epochs_use: int = 150,
        enable_shared_latent_shadow: bool | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline.

        All data inputs are pre-fetched — this runner does not call external APIs.

        Args:
            finance_result: Output from fit_tft() (must contain features_for_ukt).
            cluster_result: Output from fit_topic_model() (must contain features_for_ukt).
            agreement_matrix: Pairwise country agreement DataFrame.
            spatial_data: Output from fetch_all_spatial_data().
            timeframe_context: Date range metadata.
            sim_steps: Number of agent simulation steps.
            sae_hidden_dim: Hidden dim for final SAE.
            sae_epochs: Training epochs for final SAE.
            stability_runs: Number of noisy runs for stability estimation.
            compute_cross_block: Whether to run the UVT + USE neural networks.
                Disabled by default because each adds ~120–150 training epochs;
                enable explicitly in UI contexts or when cross-block analysis is needed.
            cross_block_epochs_uvt: Training epochs for the UVT network.
            cross_block_epochs_use: Training epochs for the USE network.
            enable_shared_latent_shadow: Optional override for shared-latent
                prototype feature flag. When enabled, computes shadow-only
                paired-window contrastive alignment metrics.

        Returns:
            PipelineResult with all outputs.
        """
        run_id = str(uuid.uuid4())[:8].upper()
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        timeframe_context = timeframe_context or {}

        ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
        snapshots: list[dict] = []
        data_sources: dict[str, str] = {}

        # ---- Block 1: Finance ----
        if finance_result is not None:
            self._report("finance_block", "Adding finance block to UKT...")
            warnings = validate_block_result(finance_result, "Finance")
            self._warnings.extend(warnings)

            data_sources["Finance"] = finance_result.get("data_source", "unknown")
            snap = ukt.add_block(
                "Finance",
                finance_result["features_for_ukt"],
                feature_meta=finance_result.get("feature_meta", {}),
                timeframe_context=timeframe_context,
            )
            snapshots.append(snap)

        # ---- Block 2: Clusters ----
        if cluster_result is not None:
            self._report("cluster_block", "Adding cluster block to UKT...")
            warnings = validate_block_result(cluster_result, "Clusters")
            self._warnings.extend(warnings)

            data_sources["Clusters"] = cluster_result.get("data_source", "unknown")
            snap = ukt.add_block(
                "Clusters",
                cluster_result["features_for_ukt"],
                feature_meta=cluster_result.get("feature_meta", {}),
                timeframe_context=timeframe_context,
            )
            snapshots.append(snap)

        # ---- Block 3: Graph ----
        from hyperspace.models.graph_engine import (
            build_geopolitical_graph,
            analyze_graph,
        )

        self._report("graph_block", "Building geopolitical graph...")
        G, pos = build_geopolitical_graph(agreement_matrix=agreement_matrix)
        graph_analysis = analyze_graph(G)

        data_sources["Graph"] = "networkx_centrality"
        if agreement_matrix is not None:
            data_sources["Graph"] = "UN_voting + networkx_centrality"

        snap = ukt.add_block(
            "Graph",
            graph_analysis["features_for_ukt"],
            feature_meta=graph_analysis.get("feature_meta", {}),
            timeframe_context=timeframe_context,
        )
        snapshots.append(snap)

        graph_result_full = dict(
            G=G, pos=pos, analysis=graph_analysis,
            features_for_ukt=graph_analysis["features_for_ukt"],
            feature_meta=graph_analysis.get("feature_meta", {}),
            data_source=data_sources["Graph"],
        )

        # ---- Block 4: Spatial ----
        spatial_result = None
        if spatial_data is not None:
            self._report("spatial_block", "Computing spatial kernels...")
            from hyperspace.models.spatial_kernels import get_spatial_features

            spatial_result = get_spatial_features(
                spatial_data["physical_raster"],
                spatial_data["country_scalars"],
                spatial_data.get("scalar_names", []),
                spatial_data.get("node_order", list(GEOPOLITICAL_NODES.keys())),
                timeframe_context,
            )
            data_sources["Spatial"] = spatial_data.get("source_label", "spatial_raster")
            spatial_result["data_source"] = data_sources["Spatial"]

            snap = ukt.add_block(
                "Spatial",
                spatial_result["features_for_ukt"],
                feature_meta=spatial_result.get("feature_meta", {}),
                timeframe_context=timeframe_context,
            )
            snapshots.append(snap)

        # ---- Block 5: Agent Simulation ----
        self._report("agent_sim", f"Running agent simulation ({sim_steps} steps)...")
        from hyperspace.models.agent_sim import (
            initialize_agents_from_data,
            run_simulation,
        )

        agents = initialize_agents_from_data(
            graph_analysis=graph_analysis,
            agreement_matrix=agreement_matrix,
            spatial_features=spatial_result,
        )
        agents, log_entries, agent_features, agent_meta = run_simulation(
            agents, steps=sim_steps,
        )
        data_sources["Agents"] = "agent_simulation (derived from Graph + Spatial)"

        snap = ukt.add_block(
            "Agents",
            agent_features,
            feature_meta=agent_meta,
            timeframe_context=timeframe_context,
        )
        snapshots.append(snap)

        sim_result = dict(
            agents=agents,
            log=log_entries,
            features_for_ukt=agent_features,
            feature_meta=agent_meta,
            data_source=data_sources["Agents"],
        )

        # ---- Final: SAE Concept Discovery ----
        self._report("final_interpretation", "Running sparse autoencoder...")
        from hyperspace.models.sparse_ae import (
            train_sparse_ae,
            map_concepts_to_kernels,
            enrich_concepts_with_narratives,
        )

        final_matrix = ukt.get_final_matrix()
        sae_result = None
        concept_kernel_map: list[dict] = []

        alignment_metrics: dict[str, Any] = {
            "legacy": {},
            "shared_latent": {
                "enabled": False,
                "shadow_only": True,
                "reason": "feature_flag_disabled",
            },
            "parity_delta": {},
        }
        shared_latent_enabled = (
            FEATURE_FLAGS.get("shared_latent_shadow", False)
            if enable_shared_latent_shadow is None
            else enable_shared_latent_shadow
        )

        if final_matrix is not None:
            sae_result = train_sparse_ae(
                final_matrix, hidden_dim=sae_hidden_dim, epochs=sae_epochs,
            )
            if sae_result is not None:
                final_snap = ukt.get_latest_snapshot()
                if final_snap:
                    concept_kernel_map = map_concepts_to_kernels(
                        sae_result, final_snap,
                    )

        # ---- Cross-Block Interconnection: UVT + USE ----
        uvt_result = None
        use_result = None
        if compute_cross_block and final_matrix is not None and final_matrix.shape[0] >= 2:
            self._report("cross_block", "Computing Universal Variance Tensor...")
            from hyperspace.models.cross_block_net import (
                compute_universal_variance_tensor,
                compute_universal_semantic_encoding,
            )
            from hyperspace.models.knowledge_matrix import HYPERSPACE_REGISTRY

            active_block_names = [s["block_name"] for s in snapshots]
            uvt_result = compute_universal_variance_tensor(
                final_matrix, n_heads=4, d_model=32, epochs=cross_block_epochs_uvt,
                registry=HYPERSPACE_REGISTRY,
                block_names=active_block_names,
            )

            if uvt_result is not None:
                self._report("cross_block", "Computing Universal Semantic Encoding...")
                use_result = compute_universal_semantic_encoding(
                    final_matrix, uvt_result,
                    canvas=ukt.canvas, semantic_dim=24, epochs=cross_block_epochs_use,
                    registry=HYPERSPACE_REGISTRY,
                    block_names=active_block_names,
                )

        # ---- Semantic Narratives ----
        canvas_narrative = None
        reality_narrative = None
        try:
            from hyperspace.models.semantic_narrator import (
                narrate_canvas,
                narrate_reality_regression,
            )

            canvas_narrative = narrate_canvas(ukt.canvas)
            if snapshots:
                reality_narrative = narrate_reality_regression(
                    snapshots[-1], ukt.canvas,
                )
            if sae_result is not None:
                enrich_concepts_with_narratives(sae_result, ukt.canvas)
        except Exception as exc:
            self._warnings.append(f"Narrator unavailable: {exc}")

        # ---- Stability ----
        stability = None
        if final_matrix is not None:
            stability = estimate_reality_regression_stability(
                final_matrix, n_runs=stability_runs, noise_std=0.01, seed=42,
            )


        # ---- Shared-latent shadow alignment MVP ----
        if shared_latent_enabled and final_matrix is not None and final_matrix.shape[0] >= 2:
            self._report("shared_latent", "Computing shared-latent shadow alignment metrics...")
            from hyperspace.models.shared_latent import compute_alignment_metrics

            alignment_metrics = compute_alignment_metrics(
                final_matrix, [s["block_name"] for s in snapshots],
            )
        elif final_matrix is not None and final_matrix.shape[0] >= 2:
            # Always emit legacy metrics section for side-by-side governance reporting.
            from hyperspace.models.shared_latent import compute_alignment_metrics

            alignment_metrics = compute_alignment_metrics(
                final_matrix, [s["block_name"] for s in snapshots], epochs=1,
            )
            alignment_metrics["shared_latent"] = {
                "enabled": False,
                "shadow_only": True,
                "reason": "feature_flag_disabled",
                "retrieval_at_1": alignment_metrics.get("shared_latent", {}).get("retrieval_at_1", 0.0),
                "probe_cosine": alignment_metrics.get("shared_latent", {}).get("probe_cosine", 0.0),
            }

        # ---- Faithfulness checks (H-003) ----
        self._report("faithfulness", "Running narrative faithfulness checks...")
        faithfulness = run_faithfulness_checks(
            snapshots=snapshots,
            canvas=ukt.canvas,
            canvas_narrative=canvas_narrative,
            sae_result=sae_result,
        )
        faithfulness_report = {
            "checks": [
                {
                    "module_name": c.module_name,
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "original_value": c.original_value,
                    "intervened_value": c.intervened_value,
                    "delta": c.delta,
                    "detail": c.detail,
                }
                for c in faithfulness.checks
            ],
            "overall_confidence": faithfulness.overall_confidence,
            "low_confidence": faithfulness.low_confidence,
            "downgraded_narrative": faithfulness.downgraded_narrative,
        }

        # Fail-safe downgrade: replace narratives if confidence is below threshold
        if faithfulness.low_confidence:
            canvas_narrative = faithfulness.downgraded_narrative
            reality_narrative = faithfulness.downgraded_narrative

        # ---- Governance ----
        governance_flags = self._compute_governance_flags(
            data_sources, snapshots, sae_result, graph_result_full,
            timeframe_context,
        )
        scorecard = self._compute_scorecard(
            snapshots, sae_result, data_sources, stability, governance_flags,
            alignment_metrics, faithfulness_report,
        )

        interpretability_contract = build_alpha_scope_contract_reports({
            "UniversalKnowledgeTensor": ukt,
            "SemanticCanvas": ukt.canvas,
        })
        enforce_alpha_scope_contract_coverage(interpretability_contract)
        interpretability_contract_summary = summarize_interpretable_reports(
            interpretability_contract,
        )

        # ---- Kernel memory persistence (Milestone C) ----
        kernel_evolution = None
        if self._kernel_memory is not None and snapshots:
            self._kernel_memory.store_run(run_id, run_timestamp, snapshots)
            kernel_evolution = self._kernel_memory.get_evolution_summary()

        # ---- Latent space versioning (Phase 3 governance) ----
        latent_version_summary = None
        if self._version_trail is not None:
            n_kernels = 0
            if snapshots:
                n_kernels = snapshots[-1].get("n_kernels", 0)

            version = compute_latent_version(
                run_id=run_id,
                timestamp=run_timestamp,
                feature_dim=HYPERSPACE_REGISTRY.total_dim,
                region_labels=FEATURE_REGION_LABELS,
                n_kernels=n_kernels,
                shared_latent_active=use_shared_latent,
                sae_result=sae_result,
            )
            concept_record = build_concept_audit_record(
                run_id=run_id,
                timestamp=run_timestamp,
                sae_result=sae_result,
            )
            self._version_trail.record(version, concept_record)
            latent_version_summary = self._version_trail.get_summary()

        # ---- Temporal drift (H-002) ----
        drift_result = None
        if self._drift_monitor is not None and snapshots:
            final_snap = snapshots[-1]
            mean_cos = 0.0
            if stability and stability.get("n_runs", 0) > 0:
                mean_cos = stability["mean_cosine"]

            self._drift_monitor.record_run(
                run_id=run_id,
                timestamp=run_timestamp,
                reality_regression=final_snap.get(
                    "reality_regression", np.zeros(UKT_FEATURE_DIM)
                ),
                kernel_importances=final_snap.get("importance", np.array([])),
                mean_cosine_stability=mean_cos,
                n_kernels=final_snap.get("n_kernels", 0),
            )
            drift = self._drift_monitor.compute_drift()
            if drift is not None:
                alerts = self._drift_monitor.check_alert_thresholds(drift)
                drift_result = {
                    "regression_cosine": drift.regression_cosine,
                    "regression_l2": drift.regression_l2,
                    "importance_cosine": drift.importance_cosine,
                    "importance_l2": drift.importance_l2,
                    "stability_delta": drift.stability_delta,
                    "n_kernel_delta": drift.n_kernel_delta,
                    "window_size": drift.window_size,
                    "n_records": drift.n_records,
                    "alerts": [
                        {
                            "code": a.code,
                            "label": a.label,
                            "severity": a.severity,
                            "detail": a.detail,
                            "measured": a.measured,
                            "threshold": a.threshold,
                        }
                        for a in alerts
                    ],
                }

        return PipelineResult(
            snapshots=snapshots,
            final_matrix=final_matrix,
            data_sources=data_sources,
            timeframe_context=timeframe_context,
            finance_result=finance_result,
            cluster_result=cluster_result,
            graph_result=graph_result_full,
            spatial_result=spatial_result,
            sim_result=sim_result,
            sae_result=sae_result,
            concept_kernel_map=concept_kernel_map,
            semantic_canvas=ukt.canvas,
            canvas_narrative=canvas_narrative,
            reality_narrative=reality_narrative,
            uvt_result=uvt_result,
            use_result=use_result,
            stability=stability,
            governance_flags=governance_flags,
            interpretability_scorecard=scorecard,
            alignment_metrics=alignment_metrics,
            interpretability_contract=interpretability_contract,
            interpretability_contract_summary=interpretability_contract_summary,
            faithfulness_report=faithfulness_report,
            drift_result=drift_result,
            kernel_evolution=kernel_evolution,
            latent_version=latent_version_summary,
            run_id=run_id,
            run_timestamp=run_timestamp,
        )

    # ------------------------------------------------------------------ #
    # Governance flags                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_governance_flags(
        data_sources: dict[str, str],
        snapshots: list[dict],
        sae_result: dict | None,
        graph_result: dict | None,
        timeframe_context: dict,
    ) -> list[GovernanceFlag]:
        """Auto-detect governance issues from pipeline outputs."""
        flags: list[GovernanceFlag] = []

        # GOV-001: Modality imbalance — one block dominates >50% of active features
        if snapshots:
            final_snap = snapshots[-1]
            importance = final_snap.get("importance", np.array([]))
            if len(importance) > 0 and importance.max() > 0.50:
                dominant_idx = int(np.argmax(importance))
                labels = final_snap.get("kernel_labels", [])
                dominant_block = (
                    labels[dominant_idx]["dominant_block"]
                    if dominant_idx < len(labels) else "unknown"
                )
                code_info = GOVERNANCE_FLAG_CODES["GOV-001"]
                flags.append(GovernanceFlag(
                    code="GOV-001",
                    label=code_info["label"],
                    description=code_info["description"],
                    severity=code_info["severity"],
                    detail=(
                        f"Kernel K{dominant_idx} ({dominant_block}) explains "
                        f"{importance.max():.1%} of total variance."
                    ),
                ))

        # GOV-002: Temporal coverage gap
        finance_src = data_sources.get("Finance", "")
        cluster_src = data_sources.get("Clusters", "")
        if "Live" in finance_src and "Fallback" in cluster_src:
            code_info = GOVERNANCE_FLAG_CODES["GOV-002"]
            flags.append(GovernanceFlag(
                code="GOV-002",
                label=code_info["label"],
                description=code_info["description"],
                severity=code_info["severity"],
                detail=(
                    "Finance data is live but news/cluster data is from static "
                    "fallback snippets. Cross-modal conclusions span different "
                    "observation windows."
                ),
            ))

        # GOV-003: Geopolitical centrality skew
        if graph_result and "analysis" in graph_result:
            eigenvector = graph_result["analysis"].get("eigenvector", {})
            if eigenvector:
                vals = list(eigenvector.values())
                mean_c = np.mean(vals) if vals else 0
                for node, val in eigenvector.items():
                    if mean_c > 0 and val > 2 * mean_c:
                        code_info = GOVERNANCE_FLAG_CODES["GOV-003"]
                        flags.append(GovernanceFlag(
                            code="GOV-003",
                            label=code_info["label"],
                            description=code_info["description"],
                            severity=code_info["severity"],
                            detail=(
                                f"{node} eigenvector centrality ({val:.3f}) "
                                f"exceeds 2× network average ({mean_c:.3f})."
                            ),
                        ))
                        break

        # GOV-004: Low concept coverage
        if sae_result is not None:
            active = sae_result.get("active_concepts", 0)
            total = sae_result.get("total_concepts", 1)
            if total > 0 and (active / total) < 0.40:
                code_info = GOVERNANCE_FLAG_CODES["GOV-004"]
                flags.append(GovernanceFlag(
                    code="GOV-004",
                    label=code_info["label"],
                    description=code_info["description"],
                    severity=code_info["severity"],
                    detail=(
                        f"{active}/{total} concepts active "
                        f"({active / total:.0%})."
                    ),
                ))

        # GOV-005: Synthetic data active
        synthetic_blocks = [
            k for k, v in data_sources.items()
            if any(x in v.lower() for x in ["synthetic", "fallback", "mock"])
        ]
        if synthetic_blocks:
            code_info = GOVERNANCE_FLAG_CODES["GOV-005"]
            flags.append(GovernanceFlag(
                code="GOV-005",
                label=code_info["label"],
                description=code_info["description"],
                severity=code_info["severity"],
                detail=f"Blocks using synthetic data: {', '.join(synthetic_blocks)}.",
            ))

        return flags

    # ------------------------------------------------------------------ #
    # Interpretability scorecard                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_scorecard(
        snapshots: list[dict],
        sae_result: dict | None,
        data_sources: dict[str, str],
        stability: dict | None,
        governance_flags: list[dict],
        alignment_metrics: dict[str, Any] | None = None,
        faithfulness_report: dict | None = None,
    ) -> dict[str, ScorecardEntry]:
        """Compute the interpretability scorecard."""
        scorecard: dict[str, ScorecardEntry] = {}

        # Feature traceability
        traced = 0
        if snapshots:
            meta = snapshots[-1].get("feature_meta", {})
            traced = len(meta)
        thresh = SCORECARD_THRESHOLDS["feature_traceability"]
        scorecard["feature_traceability"] = ScorecardEntry(
            label=thresh["label"],
            value=float(traced),
            threshold=float(thresh["threshold"]),
            unit=thresh["unit"],
            passed=traced >= thresh["threshold"],
            description=thresh["description"],
        )

        # Kernel stability
        mean_cos = 0.0
        if stability and stability.get("n_runs", 0) > 0:
            mean_cos = stability["mean_cosine"]
        thresh = SCORECARD_THRESHOLDS["kernel_stability"]
        scorecard["kernel_stability"] = ScorecardEntry(
            label=thresh["label"],
            value=round(mean_cos, 4),
            threshold=float(thresh["threshold"]),
            unit=thresh["unit"],
            passed=mean_cos >= thresh["threshold"],
            description=thresh["description"],
        )

        # Concept activation rate
        activation_rate = 0.0
        if sae_result:
            active = sae_result.get("active_concepts", 0)
            total = sae_result.get("total_concepts", 1)
            activation_rate = (active / total) * 100 if total > 0 else 0.0
        thresh = SCORECARD_THRESHOLDS["concept_activation_rate"]
        scorecard["concept_activation_rate"] = ScorecardEntry(
            label=thresh["label"],
            value=round(activation_rate, 1),
            threshold=float(thresh["threshold"]),
            unit=thresh["unit"],
            passed=activation_rate >= thresh["threshold"],
            description=thresh["description"],
        )

        # Data source diversity
        synthetic_keys = {"synthetic", "fallback", "mock"}
        live_count = sum(
            1 for v in data_sources.values()
            if not any(x in v.lower() for x in synthetic_keys)
        )
        thresh = SCORECARD_THRESHOLDS["data_source_diversity"]
        scorecard["data_source_diversity"] = ScorecardEntry(
            label=thresh["label"],
            value=float(live_count),
            threshold=float(thresh["threshold"]),
            unit=thresh["unit"],
            passed=live_count >= thresh["threshold"],
            description=thresh["description"],
        )


        # Alignment comparison (legacy vs shared latent shadow path)
        alignment_metrics = alignment_metrics or {}
        legacy_metrics = alignment_metrics.get("legacy", {})
        shared_metrics = alignment_metrics.get("shared_latent", {})

        legacy_r1 = float(legacy_metrics.get("retrieval_at_1", 0.0))
        thresh = SCORECARD_THRESHOLDS["legacy_retrieval_at_1"]
        scorecard["legacy_retrieval_at_1"] = ScorecardEntry(
            label=thresh["label"],
            value=round(legacy_r1, 4),
            threshold=float(thresh["threshold"]),
            unit=thresh["unit"],
            passed=legacy_r1 >= thresh["threshold"],
            description=thresh["description"],
        )

        shared_r1 = float(shared_metrics.get("retrieval_at_1", 0.0))
        thresh = SCORECARD_THRESHOLDS["shared_latent_retrieval_at_1"]
        scorecard["shared_latent_retrieval_at_1"] = ScorecardEntry(
            label=thresh["label"],
            value=round(shared_r1, 4),
            threshold=float(thresh["threshold"]),
            unit=thresh["unit"],
            passed=shared_r1 >= thresh["threshold"],
            description=thresh["description"],
        )

        shared_probe = float(shared_metrics.get("probe_cosine", 0.0))
        thresh = SCORECARD_THRESHOLDS["shared_latent_probe_cosine"]
        scorecard["shared_latent_probe_cosine"] = ScorecardEntry(
            label=thresh["label"],
            value=round(shared_probe, 4),
            threshold=float(thresh["threshold"]),
            unit=thresh["unit"],
            passed=shared_probe >= thresh["threshold"],
            description=thresh["description"],
        )

        # Faithfulness confidence
        faith_confidence = 0.0
        if faithfulness_report:
            faith_confidence = float(
                faithfulness_report.get("overall_confidence", 0.0)
            )
        thresh = SCORECARD_THRESHOLDS["faithfulness_confidence"]
        scorecard["faithfulness_confidence"] = ScorecardEntry(
            label=thresh["label"],
            value=round(faith_confidence, 4),
            threshold=float(thresh["threshold"]),
            unit=thresh["unit"],
            passed=faith_confidence >= thresh["threshold"],
            description=thresh["description"],
        )

        # Governance flags count
        n_flags = len(governance_flags)
        thresh = SCORECARD_THRESHOLDS["governance_flags"]
        scorecard["governance_flags"] = ScorecardEntry(
            label=thresh["label"],
            value=float(n_flags),
            threshold=float(thresh["threshold"]),
            unit=thresh["unit"],
            passed=n_flags <= thresh["threshold"],
            description=thresh["description"],
        )

        return scorecard
