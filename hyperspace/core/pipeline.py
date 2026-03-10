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
)
from hyperspace.core.types import (
    GovernanceFlag,
    PipelineResult,
    ScorecardEntry,
    validate_block_result,
)
from hyperspace.models.knowledge_matrix import (
    UniversalKnowledgeTensor,
    estimate_reality_regression_stability,
)


class PipelineRunner:
    """Orchestrate the full Hyperspace pipeline.

    Accepts pre-fetched data and runs all blocks in sequence through the UKT.
    Returns a PipelineResult dict with everything needed for rendering.
    """

    def __init__(
        self,
        on_step: Callable[[str, str], None] | None = None,
    ):
        """
        Args:
            on_step: Optional callback(step_name, message) for progress reporting.
        """
        self._on_step = on_step or (lambda s, m: None)
        self._warnings: list[str] = []

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
        except Exception:
            pass  # Narrator unavailable — graceful degradation

        # ---- Stability ----
        stability = None
        if final_matrix is not None:
            stability = estimate_reality_regression_stability(
                final_matrix, n_runs=stability_runs, noise_std=0.01, seed=42,
            )

        # ---- Governance ----
        governance_flags = self._compute_governance_flags(
            data_sources, snapshots, sae_result, graph_result_full,
            timeframe_context,
        )
        scorecard = self._compute_scorecard(
            snapshots, sae_result, data_sources, stability, governance_flags,
        )

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
            stability=stability,
            governance_flags=governance_flags,
            interpretability_scorecard=scorecard,
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
