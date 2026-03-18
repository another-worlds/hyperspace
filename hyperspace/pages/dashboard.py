"""Dashboard: landing page, pipeline orchestration, progress tracking, results display."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hyperspace.config import (
    DEFAULT_TICKERS, GEOPOLITICAL_NODES, PIPELINE_STEPS, PLOTLY_LAYOUT,
    UKT_FEATURE_DIM,
)
from hyperspace.models.knowledge_matrix import estimate_reality_regression_stability
from hyperspace.viz.charts import source_badge
from hyperspace.core.pipeline import PipelineRunner
from hyperspace.core.drift_monitor import DriftMonitor
from hyperspace.core.temporal_memory import KernelMemory
from hyperspace.core.latent_versioning import LatentVersionTrail
from hyperspace.core.parallel_fetch import fetch_all_data_parallel, fetch_models_parallel
from hyperspace.core.logging import StructuredLogger, log_pipeline_step


_DRIFT_PERSIST_PATH = ".hyperspace/drift_history.json"


def _get_drift_monitor() -> DriftMonitor:
    """Get or create a session-scoped DriftMonitor instance.

    On first access, attempts to load persisted drift history from disk
    (``_DRIFT_PERSIST_PATH``).  This allows drift tracking to survive
    across separate Streamlit sessions, not just re-runs within a session.
    """
    if "drift_monitor" not in st.session_state:
        try:
            monitor = DriftMonitor.load_from_disk(_DRIFT_PERSIST_PATH)
        except (FileNotFoundError, Exception):
            monitor = DriftMonitor(max_history=50)
        st.session_state.drift_monitor = monitor
    return st.session_state.drift_monitor


def _save_drift_monitor() -> None:
    """Persist the current DriftMonitor to disk after a pipeline run."""
    monitor = st.session_state.get("drift_monitor")
    if monitor is not None:
        try:
            monitor.save_to_disk(_DRIFT_PERSIST_PATH)
        except Exception:
            pass  # Non-critical — session state still has the data


_KERNEL_MEMORY_PATH = ".hyperspace/kernel_memory.json"


def _get_kernel_memory() -> KernelMemory:
    """Get or create a session-scoped KernelMemory instance.

    Loads persisted kernel history from disk on first access.
    """
    if "kernel_memory" not in st.session_state:
        try:
            memory = KernelMemory.load(_KERNEL_MEMORY_PATH)
        except (FileNotFoundError, Exception):
            memory = KernelMemory(max_runs=100)
        st.session_state.kernel_memory = memory
    return st.session_state.kernel_memory


def _save_kernel_memory() -> None:
    """Persist kernel memory to disk after a pipeline run."""
    memory = st.session_state.get("kernel_memory")
    if memory is not None:
        try:
            memory.save(_KERNEL_MEMORY_PATH)
        except Exception:
            pass


_VERSION_TRAIL_PATH = ".hyperspace/latent_versions.json"


def _get_version_trail() -> LatentVersionTrail:
    """Get or create a session-scoped LatentVersionTrail instance."""
    if "version_trail" not in st.session_state:
        try:
            trail = LatentVersionTrail.load(_VERSION_TRAIL_PATH)
        except (FileNotFoundError, Exception):
            trail = LatentVersionTrail(max_entries=100)
        st.session_state.version_trail = trail
    return st.session_state.version_trail


def _save_version_trail() -> None:
    """Persist latent version trail to disk after a pipeline run."""
    trail = st.session_state.get("version_trail")
    if trail is not None:
        try:
            trail.save(_VERSION_TRAIL_PATH)
        except Exception:
            pass  # Non-critical — session state still has the data


# --------------------------------------------------------------------------- #
# Landing page                                                                  #
# --------------------------------------------------------------------------- #

def render_landing() -> None:
    """Render a compact governance-framed dashboard landing page."""
    st.markdown(
        """
        <div class="governance-header">
            <h2 style="margin:0;color:#64ffda;">Hyperspace</h2>
            <p style="margin:8px 0 0;color:#4a6880;">
                Accountability Infrastructure for AI Governance.
                Launch the full cycle to run all modality blocks and generate
                traceable UKT + interpretability outputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Feature Space", f"{UKT_FEATURE_DIM} dims")
    c2.metric("Pipeline Steps", str(len(PIPELINE_STEPS)))
    c3.metric("Geopolitical Nodes", str(len(GEOPOLITICAL_NODES)))



# --------------------------------------------------------------------------- #
# Pipeline execution                                                            #
# --------------------------------------------------------------------------- #

def run_pipeline() -> None:
    """Execute dashboard pipeline via canonical PipelineRunner orchestration."""
    with st.status("Running Hyperspace Pipeline...", expanded=True) as status:
        # ---- Step 1: Fetch Data in Parallel ----
        st.write("📥 Fetching real data sources (finance, news, political, spatial) in parallel...")
        tickers = st.session_state.get("tickers", DEFAULT_TICKERS) or DEFAULT_TICKERS
        import time
        fetch_start = time.time()

        from hyperspace.data.finance import get_ohlcv
        def fetch_ohlcv_wrapper():
            return get_ohlcv(tickers)

        # Parallel fetch all data sources
        fetch_result = fetch_all_data_parallel(
            tickers=tuple(tickers),
            fetch_ohlcv_fn=fetch_ohlcv_wrapper,
            max_workers=4,
        )

        # Check for errors and report
        if fetch_result.errors:
            for source, error in fetch_result.errors.items():
                log_pipeline_step(f"data_fetch_{source}", "failed", error_msg=str(error))
                st.warning(f"⚠️ {source.upper()}: {error}")

        if not fetch_result.is_complete():
            missing = fetch_result.get_missing()
            status.update(
                label=f"Pipeline blocked: missing {', '.join(missing)}",
                state="error"
            )
            st.error(f"Cannot proceed without: {', '.join(missing)}")
            return

        fetch_duration = time.time() - fetch_start
        st.write(f"✓ Data fetching complete ({fetch_duration:.1f}s): {fetch_result.ohlcv_source} | {fetch_result.docs_source} | {fetch_result.agreement_source}")
        log_pipeline_step("data_fetch", "completed", duration_sec=fetch_duration)

        # Extract fetched data
        ohlcv_df = fetch_result.ohlcv_df
        docs = fetch_result.docs
        docs_src = fetch_result.docs_source
        agreement = fetch_result.agreement
        pol_src = fetch_result.agreement_source
        fin_src = fetch_result.ohlcv_source
        raw_spatial = fetch_result.spatial_data

        # Compute timeframe from finance data
        finance_start = None
        finance_end = None
        if "Date" in ohlcv_df.columns and len(ohlcv_df) > 0:
            finance_start = pd.to_datetime(ohlcv_df["Date"]).min().to_pydatetime()
            finance_end = pd.to_datetime(ohlcv_df["Date"]).max().to_pydatetime()

        min_year = finance_start.year if finance_start else 2000
        max_year = finance_end.year if finance_end else None

        timeframe_context = {
            "start_date": finance_start.date().isoformat() if finance_start else None,
            "end_date": finance_end.date().isoformat() if finance_end else None,
            "min_year": min_year,
            "max_year": max_year,
        }
        st.session_state.raw_ohlcv = ohlcv_df
        st.session_state.raw_docs = docs
        st.session_state.timeframe_context = timeframe_context

        # ---- Step 2: Train Models in Parallel ----
        st.write("⚙️ Training models (TFT, BERTopic) in parallel...")
        model_start = time.time()

        model_results = fetch_models_parallel(
            tickers=tuple(tickers),
            docs=docs,
            docs_source=docs_src,
            max_workers=2,
        )

        # Check model training results
        tft_result = model_results.get("tft_result")
        cluster_result = model_results.get("bertopic_result")

        if "tft_error" in model_results:
            status.update(label="Pipeline blocked: TFT fitting failed", state="error")
            st.error(f"TFT fitting failed: {model_results['tft_error']}")
            log_pipeline_step("tft_training", "failed", error_msg=str(model_results.get("tft_error")))
            return

        if "bertopic_error" in model_results:
            status.update(label="Pipeline blocked: BERTopic unavailable", state="error")
            st.error(f"BERTopic fitting failed: {model_results['bertopic_error']}")
            log_pipeline_step("bertopic_fitting", "failed", error_msg=str(model_results.get("bertopic_error")))
            return

        model_duration = time.time() - model_start
        st.write(f"✓ Model training complete ({model_duration:.1f}s)")
        log_pipeline_step("model_training", "completed", duration_sec=model_duration)

        # ---- Step 3: Canonical orchestration (single path) ----
        st.write("Running canonical pipeline runner for graph/spatial/agents/interpreter...")
        runner = PipelineRunner(
            drift_monitor=_get_drift_monitor(),
            kernel_memory=_get_kernel_memory(),
            version_trail=_get_version_trail(),
        )
        result = runner.run(
            finance_result=tft_result,
            cluster_result=cluster_result,
            agreement_matrix=agreement,
            spatial_data=raw_spatial,
            timeframe_context=timeframe_context,
            compute_cross_block=True,
        )

        st.session_state.run_id = result.get("run_id")
        st.session_state.run_timestamp = result.get("run_timestamp")
        st.session_state.finance_result = result.get("finance_result")
        st.session_state.cluster_result = result.get("cluster_result")
        st.session_state.graph_result = result.get("graph_result")
        st.session_state.spatial_result = result.get("spatial_result")
        st.session_state.sim_result = result.get("sim_result")
        st.session_state.sae_result = result.get("sae_result")
        st.session_state.concept_kernel_map = result.get("concept_kernel_map", [])
        st.session_state.ukt_snapshots = result.get("snapshots", [])
        st.session_state.data_sources = result.get("data_sources", {})
        st.session_state.semantic_canvas = result.get("semantic_canvas")
        st.session_state.canvas_narrative = result.get("canvas_narrative")
        st.session_state.reality_narrative = result.get("reality_narrative")
        st.session_state.uvt_result = result.get("uvt_result")
        st.session_state.use_result = result.get("use_result")
        st.session_state.ukt_multirun_stability = result.get("stability")
        st.session_state.governance_flags = result.get("governance_flags", [])
        st.session_state.interpretability_scorecard = result.get("interpretability_scorecard", {})
        st.session_state.alignment_metrics = result.get("alignment_metrics", {})
        st.session_state.interpretability_contract = result.get("interpretability_contract", {})
        st.session_state.interpretability_contract_summary = result.get(
            "interpretability_contract_summary", {},
        )
        st.session_state.faithfulness_report = result.get("faithfulness_report")
        st.session_state.drift_result = result.get("drift_result")
        st.session_state.kernel_evolution = result.get("kernel_evolution")
        st.session_state.latent_version = result.get("latent_version")

        # Persist drift history, kernel memory, and version trail to disk
        _save_drift_monitor()
        _save_kernel_memory()
        _save_version_trail()

        status.update(label="Pipeline complete!", state="complete")
        st.session_state.pipeline_complete = True



def _build_governance_report_markdown(
    *,
    run_id: str,
    run_ts: str,
    scorecard: dict,
    governance_flags: list[dict],
    snapshots: list[dict],
    canvas_narrative: str | None,
    reality_narrative: str | None,
    interpretability_contract: dict,
    interpretability_contract_summary: dict,
    alignment_metrics: dict | None = None,
    faithfulness_report: dict | None = None,
    drift_result: dict | None = None,
) -> str:
    """Build exportable markdown governance report with compliance artifacts."""
    report_lines = [
        "# Hyperspace Governance Report",
        f"## Run ID: {run_id} | {run_ts}",
        "",
        "## Interpretability Score Card",
    ]

    if scorecard:
        for item in scorecard.values():
            status_str = "PASS" if item.get("passed") else "WARN"
            report_lines.append(
                f"- {item.get('label')}: {item.get('value')}{item.get('unit','')} "
                f"(threshold ≥{item.get('threshold')}{item.get('unit','')}) — {status_str}"
            )

    report_lines.append("")
    report_lines.append("## Interpretability Contract Compliance")
    if interpretability_contract_summary:
        report_lines.append(
            "- Summary: "
            f"{interpretability_contract_summary.get('compliant_modules', 0)} compliant, "
            f"{interpretability_contract_summary.get('na_modules', 0)} N/A, "
            f"{interpretability_contract_summary.get('noncompliant_modules', 0)} non-compliant "
            f"out of {interpretability_contract_summary.get('total_modules', 0)} "
            f"(rate={interpretability_contract_summary.get('compliance_rate', 0.0):.2f})"
        )

    if interpretability_contract:
        for module_name, module_report in interpretability_contract.items():
            report_lines.append(
                f"- {module_name}: status={module_report.get('status', 'unknown')}; "
                f"compliant={module_report.get('compliant', False)}; "
                f"interface_issues={len(module_report.get('interface_issues', []))}; "
                f"payload_issues={len(module_report.get('payload_issues', []))}; "
                f"na_owner={module_report.get('na_owner')}; "
                f"na_reason={module_report.get('na_reason')}"
            )


    alignment_metrics = alignment_metrics or {}
    if alignment_metrics:
        legacy = alignment_metrics.get("legacy", {})
        shared = alignment_metrics.get("shared_latent", {})
        report_lines.append("")
        report_lines.append("## Alignment Metrics (Legacy vs Shared-Latent Shadow)")
        report_lines.append(
            f"- Legacy retrieval@1: {legacy.get('retrieval_at_1', 0.0)} | "
            f"probe cosine: {legacy.get('probe_cosine', 0.0)}"
        )
        report_lines.append(
            f"- Shared-latent retrieval@1: {shared.get('retrieval_at_1', 0.0)} | "
            f"probe cosine: {shared.get('probe_cosine', 0.0)} | "
            f"shadow_only: {shared.get('shadow_only', True)}"
        )

    if governance_flags:
        report_lines.append("")
        report_lines.append("## Governance Flags")
        for flag in governance_flags:
            report_lines.append(
                f"- [{flag.get('code')}] {flag.get('label')}: {flag.get('detail', '')}"
            )

    if canvas_narrative or reality_narrative:
        report_lines.append("")
        report_lines.append("## Semantic Narratives (Tiny-LLM)")
        if canvas_narrative:
            report_lines.append(f"**Cross-Domain Narrative:** {canvas_narrative}")
        if reality_narrative:
            report_lines.append(f"**Reality Assessment:** {reality_narrative}")
        report_lines.append("")
        for snap in snapshots:
            if snap.get("layer_narrative"):
                report_lines.append(
                    f"**Step {snap['step']} ({snap['block_name']}):** "
                    f"{snap['layer_narrative']}"
                )

    # Faithfulness checks (H-003)
    faithfulness_report = faithfulness_report or {}
    if faithfulness_report:
        report_lines.append("")
        report_lines.append("## Narrative Faithfulness Assessment")
        report_lines.append(
            f"- Overall confidence: {faithfulness_report.get('overall_confidence', 0.0):.2%}"
        )
        report_lines.append(
            f"- Low confidence flag: {faithfulness_report.get('low_confidence', False)}"
        )
        checks = faithfulness_report.get("checks", [])
        passed = sum(1 for c in checks if c.get("passed"))
        report_lines.append(f"- Checks: {passed}/{len(checks)} passed")
        for c in checks:
            status_str = "PASS" if c.get("passed") else "FAIL"
            report_lines.append(
                f"  - [{status_str}] {c.get('module_name')}/{c.get('check_name')}: {c.get('detail', '')}"
            )

    # Temporal drift (H-002)
    drift_result = drift_result or {}
    if drift_result:
        report_lines.append("")
        report_lines.append("## Temporal Drift Analysis")
        report_lines.append(
            f"- Regression cosine: {drift_result.get('regression_cosine', 0.0):.4f}"
        )
        report_lines.append(
            f"- Importance cosine: {drift_result.get('importance_cosine', 0.0):.4f}"
        )
        report_lines.append(
            f"- Stability delta: {drift_result.get('stability_delta', 0.0):+.4f}"
        )
        alerts = drift_result.get("alerts", [])
        if alerts:
            for a in alerts:
                report_lines.append(
                    f"  - [{a.get('code')}] {a.get('label')}: {a.get('detail', '')}"
                )

    # Kernel evolution (temporal memory)
    kernel_evolution = st.session_state.get("kernel_evolution")
    if kernel_evolution and kernel_evolution.get("n_runs", 0) >= 2:
        report_lines.append("")
        report_lines.append("## Kernel Evolution (Temporal Memory)")
        report_lines.append(
            f"- Total runs tracked: {kernel_evolution['n_runs']}"
        )
        report_lines.append(
            f"- Total snapshots: {kernel_evolution.get('n_snapshots', 0)}"
        )
        for bn, bd in kernel_evolution.get("blocks", {}).items():
            report_lines.append(
                f"- {bn}: stability={bd.get('mean_importance_stability', 0.0):.4f}, "
                f"runs={bd.get('n_runs', 0)}"
            )

    report_lines.append("")
    report_lines.append("## Pipeline Reports")
    for snap in snapshots:
        report_lines.append(snap["report"])

    return "\n".join(report_lines)


def _build_diagnostics_export_rows(
    *,
    run_id: str,
    run_ts: str,
    alignment_metrics: dict | None,
    faithfulness_report: dict | None,
    drift_result: dict | None,
    kernel_evolution: dict | None = None,
) -> list[dict[str, object]]:
    """Flatten alignment, faithfulness, and drift data for CSV export."""
    rows: list[dict[str, object]] = []
    alignment_metrics = alignment_metrics or {}
    legacy = alignment_metrics.get("legacy", {})
    shared = alignment_metrics.get("shared_latent", {})
    parity = alignment_metrics.get("parity_delta", {})

    rows.append({
        "RunID": run_id,
        "Timestamp": run_ts,
        "Category": "alignment",
        "Metric": "legacy_retrieval_at_1",
        "Value": legacy.get("retrieval_at_1", 0.0),
    })
    rows.append({
        "RunID": run_id,
        "Timestamp": run_ts,
        "Category": "alignment",
        "Metric": "legacy_probe_cosine",
        "Value": legacy.get("probe_cosine", 0.0),
    })
    rows.append({
        "RunID": run_id,
        "Timestamp": run_ts,
        "Category": "alignment",
        "Metric": "shared_retrieval_at_1",
        "Value": shared.get("retrieval_at_1", 0.0),
    })
    rows.append({
        "RunID": run_id,
        "Timestamp": run_ts,
        "Category": "alignment",
        "Metric": "shared_probe_cosine",
        "Value": shared.get("probe_cosine", 0.0),
    })
    rows.append({
        "RunID": run_id,
        "Timestamp": run_ts,
        "Category": "alignment",
        "Metric": "parity_delta_retrieval",
        "Value": parity.get("retrieval_at_1", 0.0),
    })

    faithfulness_report = faithfulness_report or {}
    if faithfulness_report:
        rows.append({
            "RunID": run_id,
            "Timestamp": run_ts,
            "Category": "faithfulness",
            "Metric": "overall_confidence",
            "Value": faithfulness_report.get("overall_confidence", 0.0),
        })
        rows.append({
            "RunID": run_id,
            "Timestamp": run_ts,
            "Category": "faithfulness",
            "Metric": "low_confidence",
            "Value": 1.0 if faithfulness_report.get("low_confidence") else 0.0,
        })
        for c in faithfulness_report.get("checks", []):
            rows.append({
                "RunID": run_id,
                "Timestamp": run_ts,
                "Category": "faithfulness_check",
                "Metric": f"{c.get('module_name')}/{c.get('check_name')}",
                "Value": 1.0 if c.get("passed") else 0.0,
            })

    drift_result = drift_result or {}
    if drift_result:
        for key in ("regression_cosine", "importance_cosine", "stability_delta"):
            rows.append({
                "RunID": run_id,
                "Timestamp": run_ts,
                "Category": "drift",
                "Metric": key,
                "Value": drift_result.get(key, 0.0),
            })

    kernel_evolution = kernel_evolution or {}
    if kernel_evolution.get("n_runs", 0) >= 1:
        rows.append({
            "RunID": run_id,
            "Timestamp": run_ts,
            "Category": "kernel_evolution",
            "Metric": "total_runs",
            "Value": kernel_evolution["n_runs"],
        })
        for bn, bd in kernel_evolution.get("blocks", {}).items():
            rows.append({
                "RunID": run_id,
                "Timestamp": run_ts,
                "Category": "kernel_evolution",
                "Metric": f"{bn}/importance_stability",
                "Value": bd.get("mean_importance_stability", 0.0),
            })

    return rows


def _build_contract_export_rows(
    *,
    run_id: str,
    run_ts: str,
    interpretability_contract: dict,
    interpretability_contract_summary: dict,
) -> list[dict[str, object]]:
    """Flatten interpretability contract reports for CSV export."""
    rows: list[dict[str, object]] = []
    for module_name, module_report in interpretability_contract.items():
        rows.append({
            "RunID": run_id,
            "Timestamp": run_ts,
            "Module": module_name,
            "Status": module_report.get("status", "unknown"),
            "Compliant": module_report.get("compliant", False),
            "InterfaceIssueCount": len(module_report.get("interface_issues", [])),
            "PayloadIssueCount": len(module_report.get("payload_issues", [])),
            "InterfaceIssues": " | ".join(module_report.get("interface_issues", [])),
            "PayloadIssues": " | ".join(module_report.get("payload_issues", [])),
            "NAOwner": module_report.get("na_owner"),
            "NAReason": module_report.get("na_reason"),
            "TotalModules": interpretability_contract_summary.get("total_modules", 0),
            "CompliantModules": interpretability_contract_summary.get("compliant_modules", 0),
            "NAModules": interpretability_contract_summary.get("na_modules", 0),
            "NoncompliantModules": interpretability_contract_summary.get("noncompliant_modules", 0),
            "ComplianceRate": interpretability_contract_summary.get("compliance_rate", 0.0),
        })
    return rows



# --------------------------------------------------------------------------- #
# Results rendering                                                             #
# --------------------------------------------------------------------------- #

def render_results() -> None:
    """Render the full results dashboard after pipeline completes."""
    snapshots = st.session_state.get("ukt_snapshots", [])
    data_sources = st.session_state.get("data_sources", {})

    if not snapshots:
        st.info("No pipeline results yet. Click 'Launch' to run.")
        return

    # D1: Run ID watermark
    run_id = st.session_state.get("run_id")
    run_ts = st.session_state.get("run_timestamp")
    if run_id:
        st.markdown(
            f'<span class="run-id-watermark">Run ID: {run_id} · {run_ts}</span>',
            unsafe_allow_html=True,
        )

    # D3: Synthetic data banner
    synthetic_blocks = [
        k for k, v in data_sources.items()
        if any(x in v for x in ["Synthetic", "synthetic", "Fallback", "fallback", "Mock", "mock"])
    ]
    if synthetic_blocks:
        st.markdown(
            '<div class="synthetic-banner">'
            '⚠️ <strong>SYNTHETIC DATA ACTIVE</strong> — '
            f'Blocks using simulated data: {", ".join(synthetic_blocks)}. '
            'Conclusions from these blocks are <strong>illustrative only</strong> '
            'and do not represent real-world observations.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Data source badges
    src_html = " ".join(source_badge(v) for v in data_sources.values())
    st.markdown(f"**Data Sources**: {src_html}", unsafe_allow_html=True)

    # Gather all session state data for use across sections
    final_snap = snapshots[-1]
    gov_flags = st.session_state.get("governance_flags", [])
    scorecard = st.session_state.get("interpretability_scorecard", {})
    canvas_narrative = st.session_state.get("canvas_narrative")
    reality_narrative = st.session_state.get("reality_narrative")
    alignment_metrics = st.session_state.get("alignment_metrics", {})
    faithfulness_report = st.session_state.get("faithfulness_report")
    drift_result = st.session_state.get("drift_result")
    kernel_evolution = st.session_state.get("kernel_evolution")
    interpretability_contract = st.session_state.get("interpretability_contract", {})
    interpretability_contract_summary = st.session_state.get(
        "interpretability_contract_summary", {},
    )

    # ================================================================== #
    #  EXECUTIVE SUMMARY — prominent, non-collapsible                     #
    # ================================================================== #
    st.markdown("### Executive Summary")

    # Key metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Active Kernels", str(final_snap["n_kernels"]))

    finance_result = st.session_state.get("finance_result", {})
    if isinstance(finance_result, dict) and "model_params" in finance_result:
        m2.metric("TFT Params", f"{finance_result['model_params']:,}")
    else:
        m2.metric("TFT Params", "N/A (mock)")

    graph_result = st.session_state.get("graph_result", {})
    if isinstance(graph_result, dict) and "analysis" in graph_result:
        density = graph_result["analysis"].get("density", 0.0)
        m3.metric("Graph Density", f"{density:.3f}")
    else:
        m3.metric("Graph Density", "N/A")

    sae_result = st.session_state.get("sae_result")
    if sae_result:
        m4.metric("Active Concepts", f"{sae_result['active_concepts']}/{sae_result['total_concepts']}")
    else:
        m4.metric("Active Concepts", "N/A")

    m5.metric("Recon Error", f"{final_snap['reconstruction_error']:.6f}")

    # Scorecard pass/fail headline
    if scorecard:
        pass_count = sum(1 for item in scorecard.values() if item.get("passed", False))
        total_count = len(scorecard)
        all_pass = pass_count == total_count
        if all_pass:
            st.success(f"Scorecard: **{pass_count}/{total_count} PASS** — System meets minimum interpretability standards.")
        else:
            st.warning(f"Scorecard: **{pass_count}/{total_count} PASS** — Review flagged items before citing conclusions.")

    # Governance flags headline
    if gov_flags:
        n_warn = sum(1 for f in gov_flags if f.get("severity") == "warning")
        n_info = len(gov_flags) - n_warn
        st.warning(f"Governance: **{len(gov_flags)} flag(s)** detected ({n_warn} warning, {n_info} info).")
    else:
        st.markdown(
            '<span class="gov-pass">✓ No governance flags detected</span>',
            unsafe_allow_html=True,
        )

    # Brief narrative summary
    if canvas_narrative:
        st.info(f"**Narrative summary:** {canvas_narrative}")

    stability = st.session_state.get("ukt_multirun_stability")
    if isinstance(stability, dict) and stability.get("n_runs", 0) > 0:
        st.caption(
            "UKT multi-run reality-regression stability "
            f"(n={stability['n_runs']}): mean cosine={stability['mean_cosine']:.3f}, "
            f"min cosine={stability['min_cosine']:.3f}, std={stability['std_cosine']:.3f}"
        )

    st.markdown("---")

    # ================================================================== #
    #  SECTION 1: Governance & Accountability (expanded by default)       #
    # ================================================================== #
    with st.expander("Governance & Accountability", expanded=True):
        # Governance Flags
        if gov_flags:
            st.markdown("#### Governance Flags")
            st.caption(
                f"{len(gov_flags)} issue(s) detected. These are auto-generated alerts "
                "indicating potential data quality, bias, or coverage concerns."
            )
            for flag in gov_flags:
                severity = flag.get("severity", "warning")
                code = flag.get("code", "GOV-???")
                label = flag.get("label", "Unknown")
                description = flag.get("description", "")
                detail = flag.get("detail", "")
                icon = "⚠️" if severity == "warning" else "ℹ️"
                with st.expander(f'{icon} [{code}] {label}', expanded=True):
                    st.markdown(f"**Definition:** {description}")
                    if detail:
                        st.markdown(
                            f'<div class="contest-note"><strong>Detected:</strong> {detail}</div>',
                            unsafe_allow_html=True,
                        )

        # Interpretability Score Card
        if scorecard:
            st.markdown("#### Interpretability Accountability Score Card")
            st.caption(
                "Formal pass/fail audit against minimum governance thresholds. "
                "This score card is included in all exported reports."
            )
            sc_rows = []
            for key, item in scorecard.items():
                passed = item.get("passed", False)
                status_icon = "✅ PASS" if passed else "⚠️ WARN"
                sc_rows.append({
                    "Dimension": item.get("label", key),
                    "Value": f"{item.get('value', 'N/A')}{item.get('unit', '')}",
                    "Threshold": f"≥{item.get('threshold', 'N/A')}{item.get('unit', '')}",
                    "Status": status_icon,
                    "Description": item.get("description", ""),
                })
            sc_df = pd.DataFrame(sc_rows)

            def _style_status(val: str) -> str:
                if "PASS" in val:
                    return "color: #64ffda; font-weight: bold"
                return "color: #ffaa00; font-weight: bold"

            try:
                styled = sc_df.style.map(_style_status, subset=["Status"])
            except AttributeError:
                styled = sc_df.style.applymap(_style_status, subset=["Status"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # Faithfulness Report
        if faithfulness_report:
            st.markdown("#### Narrative Faithfulness Checks")
            confidence = faithfulness_report.get("overall_confidence", 1.0)
            low_conf = faithfulness_report.get("low_confidence", False)
            checks = faithfulness_report.get("checks", [])

            if low_conf:
                st.warning(
                    f"Explanation confidence is **{confidence:.0%}** — below minimum threshold. "
                    "Narratives have been downgraded to avoid unsupported claims."
                )
            else:
                st.success(
                    f"Explanation confidence: **{confidence:.0%}** — "
                    f"{sum(1 for c in checks if c.get('passed'))}/{len(checks)} checks passed."
                )

            if checks:
                with st.expander("Intervention check details", expanded=False):
                    fc_rows = []
                    for c in checks:
                        fc_rows.append({
                            "Module": c.get("module_name", ""),
                            "Check": c.get("check_name", ""),
                            "Passed": "PASS" if c.get("passed") else "FAIL",
                            "Delta": f"{c.get('delta', 0.0):+.6f}",
                            "Detail": c.get("detail", ""),
                        })
                    st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)

    # ================================================================== #
    #  SECTION 2: Semantic Narratives                                     #
    # ================================================================== #
    with st.expander("Semantic Narratives", expanded=False):
        if canvas_narrative or reality_narrative:
            if canvas_narrative:
                st.success(f"**Cross-Domain Narrative:** {canvas_narrative}")
            if reality_narrative:
                st.info(f"**Reality Assessment:** {reality_narrative}")
        else:
            st.caption("No narratives generated. Run the Semantic Interpreter tab for detailed analysis.")

        st.caption(
            "For detailed kernel analysis, concept discovery, and per-block interpretability "
            "reports, see the **Semantic Interpreter** tab."
        )

    # ================================================================== #
    #  SECTION 3: Temporal Stability                                      #
    # ================================================================== #
    with st.expander("Temporal Stability", expanded=False):
        # Temporal Drift Panel
        if drift_result:
            st.markdown("#### Temporal Drift Monitor")
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric(
                "Regression Cosine",
                f"{drift_result.get('regression_cosine', 0.0):.4f}",
            )
            dc2.metric(
                "Importance Cosine",
                f"{drift_result.get('importance_cosine', 0.0):.4f}",
            )
            dc3.metric(
                "Stability Delta",
                f"{drift_result.get('stability_delta', 0.0):+.4f}",
            )
            st.caption(
                f"Window size: {drift_result.get('window_size', 0)} | "
                f"Total records: {drift_result.get('n_records', 0)}"
            )

            drift_alerts = drift_result.get("alerts", [])
            if drift_alerts:
                st.markdown("##### Drift Alerts")
                for alert in drift_alerts:
                    st.warning(
                        f"[{alert.get('code')}] {alert.get('label')}: "
                        f"{alert.get('detail', '')}"
                    )

        # Kernel Evolution Panel
        if kernel_evolution and kernel_evolution.get("n_runs", 0) >= 2:
            st.markdown("#### Kernel Evolution (Cross-Run Memory)")
            st.caption(
                f"Tracking kernel structure across **{kernel_evolution['n_runs']}** "
                f"pipeline runs ({kernel_evolution.get('n_snapshots', 0)} total snapshots)."
            )
            blocks_data = kernel_evolution.get("blocks", {})
            if blocks_data:
                evo_cols = st.columns(min(len(blocks_data), 4))
                for i, (block_name, block_evo) in enumerate(blocks_data.items()):
                    with evo_cols[i % len(evo_cols)]:
                        stab = block_evo.get("mean_importance_stability", 0.0)
                        n_runs = block_evo.get("n_runs", 0)
                        st.metric(
                            label=f"{block_name}",
                            value=f"{stab:.3f}",
                            delta=f"{n_runs} runs",
                            help="Mean cosine similarity of importance vectors across consecutive runs.",
                        )
                # Reconstruction trend chart
                _recon_traces = []
                for block_name, block_evo in blocks_data.items():
                    recon = block_evo.get("reconstruction_trend", [])
                    if len(recon) >= 2:
                        _recon_traces.append(go.Scatter(
                            x=list(range(1, len(recon) + 1)),
                            y=recon,
                            mode="lines+markers",
                            name=block_name,
                            line=dict(width=2),
                            marker=dict(size=6),
                        ))
                if _recon_traces:
                    _recon_fig = go.Figure(data=_recon_traces)
                    _recon_fig.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Reconstruction Error Across Runs",
                        xaxis_title="Run",
                        yaxis_title="Reconstruction Error",
                        height=320,
                        margin=dict(l=48, r=24, t=44, b=40),
                        legend=dict(
                            orientation="h", y=-0.2, x=0.5, xanchor="center",
                            font=dict(size=11),
                        ),
                    )
                    st.plotly_chart(_recon_fig, use_container_width=True)

                # Importance stability chart
                _imp_traces = []
                for block_name, block_evo in blocks_data.items():
                    imp_trend = block_evo.get("importance_trend", [])
                    if len(imp_trend) >= 2:
                        cosines = []
                        for idx in range(1, len(imp_trend)):
                            a = np.asarray(imp_trend[idx - 1])
                            b = np.asarray(imp_trend[idx])
                            na, nb = np.linalg.norm(a), np.linalg.norm(b)
                            cos = float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0
                            cosines.append(cos)
                        _imp_traces.append(go.Scatter(
                            x=list(range(2, len(imp_trend) + 1)),
                            y=cosines,
                            mode="lines+markers",
                            name=block_name,
                            line=dict(width=2),
                            marker=dict(size=6),
                        ))
                if _imp_traces:
                    _imp_fig = go.Figure(data=_imp_traces)
                    _imp_fig.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Importance Vector Stability (Consecutive Cosine)",
                        xaxis_title="Run",
                        yaxis_title="Cosine Similarity",
                        height=320,
                        yaxis_range=[0, 1.05],
                        margin=dict(l=48, r=24, t=44, b=40),
                        legend=dict(
                            orientation="h", y=-0.2, x=0.5, xanchor="center",
                            font=dict(size=11),
                        ),
                    )
                    _imp_fig.add_hline(
                        y=0.80, line_dash="dash", line_color="rgba(248,113,113,0.5)",
                        annotation_text="Stability threshold",
                        annotation_position="top right",
                        annotation_font_color="rgba(248,113,113,0.8)",
                    )
                    st.plotly_chart(_imp_fig, use_container_width=True)

                with st.expander("Reconstruction error trends"):
                    trend_rows = []
                    for block_name, block_evo in blocks_data.items():
                        recon = block_evo.get("reconstruction_trend", [])
                        for j, val in enumerate(recon):
                            trend_rows.append({
                                "Block": block_name,
                                "Run": j + 1,
                                "Reconstruction Error": round(val, 6),
                            })
                    if trend_rows:
                        import pandas as _pd
                        st.dataframe(_pd.DataFrame(trend_rows), use_container_width=True)

        if not drift_result and not (kernel_evolution and kernel_evolution.get("n_runs", 0) >= 2):
            st.caption("No temporal stability data available yet. Run the pipeline multiple times to populate.")

    # ================================================================== #
    #  SECTION 4: Technical Details & Exports (collapsed by default)      #
    # ================================================================== #
    with st.expander("Technical Details & Exports", expanded=False):
        # Alignment Comparison
        if alignment_metrics:
            st.markdown("#### Alignment Comparison (Legacy vs Shared-Latent)")
            legacy = alignment_metrics.get("legacy", {})
            shared = alignment_metrics.get("shared_latent", {})
            parity = alignment_metrics.get("parity_delta", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Legacy Retrieval@1", f"{legacy.get('retrieval_at_1', 0.0):.3f}")
            c2.metric("Shared Retrieval@1 (shadow)", f"{shared.get('retrieval_at_1', 0.0):.3f}")
            c3.metric("Delta (shared-legacy)", f"{parity.get('retrieval_at_1', 0.0):+.3f}")
            st.caption("Shared-latent path is feature-flagged and shadow-only. Default decisions remain on legacy UKT outputs.")

        # Contract Compliance
        if interpretability_contract:
            st.markdown("#### Interpretability Contract Governance View")
            st.caption(
                "Alpha-scope modules must be either contract-compliant or explicitly "
                "marked N/A with owner + rationale."
            )
            st.write(
                "Summary:",
                {
                    "total": interpretability_contract_summary.get("total_modules", 0),
                    "compliant": interpretability_contract_summary.get("compliant_modules", 0),
                    "na": interpretability_contract_summary.get("na_modules", 0),
                    "noncompliant": interpretability_contract_summary.get("noncompliant_modules", 0),
                    "rate": interpretability_contract_summary.get("compliance_rate", 0.0),
                },
            )

            rows = []
            for module_name, module_report in interpretability_contract.items():
                rows.append({
                    "Module": module_name,
                    "Status": module_report.get("status", "unknown"),
                    "Compliant": module_report.get("compliant", False),
                    "NAOwner": module_report.get("na_owner"),
                    "NAReason": module_report.get("na_reason"),
                    "InterfaceIssues": len(module_report.get("interface_issues", [])),
                    "PayloadIssues": len(module_report.get("payload_issues", [])),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Export buttons
        st.markdown("#### Export")
        run_id = st.session_state.get("run_id", "UNKNOWN")
        run_ts = st.session_state.get("run_timestamp", "")
        exp1, exp2, exp3, exp4, exp5 = st.columns(5)

        report_md = _build_governance_report_markdown(
            run_id=run_id,
            run_ts=run_ts,
            scorecard=scorecard,
            governance_flags=gov_flags,
            snapshots=snapshots,
            canvas_narrative=canvas_narrative,
            reality_narrative=reality_narrative,
            interpretability_contract=interpretability_contract,
            interpretability_contract_summary=interpretability_contract_summary,
            alignment_metrics=alignment_metrics,
            faithfulness_report=faithfulness_report,
            drift_result=drift_result,
        )

        exp1.download_button(
            "Download Report (Markdown)", report_md,
            f"hyperspace_report_{run_id}.md", "text/markdown",
            key="dashboard_download_report",
        )

        metrics_rows = []
        for snap in snapshots:
            for kl in snap["kernel_labels"]:
                metrics_rows.append({
                    "RunID": run_id, "Timestamp": run_ts,
                    "Step": snap["step"], "Block": snap["block_name"],
                    "Kernel": kl["kernel_id"], "Importance": kl["importance"],
                    "Region": kl["dominant_region"],
                })
        if metrics_rows:
            exp2.download_button(
                "Download Metrics (CSV)",
                pd.DataFrame(metrics_rows).to_csv(index=False),
                f"hyperspace_metrics_{run_id}.csv", "text/csv",
                key="dashboard_download_metrics",
            )

        if scorecard:
            sc_export_rows = [
                {
                    "RunID": run_id, "Timestamp": run_ts,
                    "Dimension": item.get("label"), "Value": item.get("value"),
                    "Threshold": item.get("threshold"), "Unit": item.get("unit", ""),
                    "Pass": item.get("passed"), "Description": item.get("description"),
                }
                for item in scorecard.values()
            ]
            exp3.download_button(
                "Download Score Card (CSV)",
                pd.DataFrame(sc_export_rows).to_csv(index=False),
                f"hyperspace_scorecard_{run_id}.csv", "text/csv",
                key="dashboard_download_scorecard",
            )

        contract_export_rows = _build_contract_export_rows(
            run_id=run_id,
            run_ts=run_ts,
            interpretability_contract=interpretability_contract,
            interpretability_contract_summary=interpretability_contract_summary,
        )
        if contract_export_rows:
            exp4.download_button(
                "Download Contract Compliance (CSV)",
                pd.DataFrame(contract_export_rows).to_csv(index=False),
                f"hyperspace_contract_{run_id}.csv", "text/csv",
                key="dashboard_download_interpretability_contract",
            )

        diag_rows = _build_diagnostics_export_rows(
            run_id=run_id,
            run_ts=run_ts,
            alignment_metrics=alignment_metrics,
            faithfulness_report=faithfulness_report,
            drift_result=drift_result,
            kernel_evolution=kernel_evolution,
        )
        if diag_rows:
            exp5.download_button(
                "Download Diagnostics (CSV)",
                pd.DataFrame(diag_rows).to_csv(index=False),
                f"hyperspace_diagnostics_{run_id}.csv", "text/csv",
                key="dashboard_download_diagnostics",
            )
