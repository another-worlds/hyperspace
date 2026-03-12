"""Dashboard: landing page, pipeline orchestration, progress tracking, results display."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.config import (
    DEFAULT_TICKERS, GEOPOLITICAL_NODES, PIPELINE_STEPS, UKT_FEATURE_DIM,
)
from hyperspace.models.knowledge_matrix import estimate_reality_regression_stability
from hyperspace.viz.charts import source_badge
from hyperspace.core.pipeline import PipelineRunner


# --------------------------------------------------------------------------- #
# Internal helpers: governance flag computation                                #
# --------------------------------------------------------------------------- #

def _compute_governance_flags(
    data_sources: dict,
    ukt_snapshots: list,
    sae_result: dict | None,
    graph_result: dict | None,
    timeframe_context: dict,
) -> list[dict]:
    """Auto-detect governance flags from pipeline results.

    Delegates to PipelineRunner._compute_governance_flags to ensure
    consistent flag detection logic across headless and UI pipelines.
    """
    return PipelineRunner._compute_governance_flags(
        data_sources=data_sources,
        snapshots=ukt_snapshots,
        sae_result=sae_result,
        graph_result=graph_result,
        timeframe_context=timeframe_context,
    )


def _compute_scorecard(
    ukt_snapshots: list,
    sae_result: dict | None,
    data_sources: dict,
    stability: dict | None,
    governance_flags: list,
    alignment_metrics: dict | None = None,
) -> dict:
    """Compute interpretability scorecard via canonical PipelineRunner logic.

    Delegates to PipelineRunner._compute_scorecard to guarantee parity between
    headless pipeline runs and Streamlit dashboard execution.
    """
    return PipelineRunner._compute_scorecard(
        snapshots=ukt_snapshots,
        sae_result=sae_result,
        data_sources=data_sources,
        stability=stability,
        governance_flags=governance_flags,
        alignment_metrics=alignment_metrics,
    )



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
        # ---- Step 1: Fetch Data ----
        st.write("Fetching real data sources...")
        from hyperspace.data.finance import get_ohlcv
        from hyperspace.data.news import get_text_data
        from hyperspace.data.political import get_political_data

        tickers = st.session_state.get("tickers", DEFAULT_TICKERS) or DEFAULT_TICKERS
        try:
            ohlcv_df, fin_src = get_ohlcv(tickers)
        except RuntimeError as exc:
            status.update(label="Pipeline blocked: finance data unavailable", state="error")
            st.error(str(exc))
            return

        finance_start = None
        finance_end = None
        if "Date" in ohlcv_df.columns and len(ohlcv_df) > 0:
            finance_start = pd.to_datetime(ohlcv_df["Date"]).min().to_pydatetime()
            finance_end = pd.to_datetime(ohlcv_df["Date"]).max().to_pydatetime()

        try:
            docs, docs_src = get_text_data(start_date=finance_start, end_date=finance_end)
        except RuntimeError as exc:
            status.update(label="Pipeline blocked: no live news source", state="error")
            st.error(str(exc))
            return

        min_year = finance_start.year if finance_start else 2000
        max_year = finance_end.year if finance_end else None
        try:
            _, agreement, pol_src = get_political_data(min_year=min_year, max_year=max_year)
        except RuntimeError as exc:
            status.update(label="Pipeline blocked: political data unavailable", state="error")
            st.error(str(exc))
            return

        timeframe_context = {
            "start_date": finance_start.date().isoformat() if finance_start else None,
            "end_date": finance_end.date().isoformat() if finance_end else None,
            "min_year": min_year,
            "max_year": max_year,
        }
        st.session_state.raw_ohlcv = ohlcv_df
        st.session_state.raw_docs = docs
        st.session_state.timeframe_context = timeframe_context
        st.write(f"Data fetched: {fin_src} | {docs_src} | {pol_src}")

        # ---- Step 2: Build precomputed block inputs ----
        st.write("Training Temporal Fusion Transformer (3 epochs, CPU)...")
        from hyperspace.models.tft_forecast import fit_tft

        tft_result = fit_tft(
            tickers=tuple(tickers),
            hidden=32, encoder_len=48, prediction_len=12,
        )
        if tft_result is None:
            status.update(label="Pipeline blocked: TFT fitting failed", state="error")
            st.error(
                "TFT fitting failed. Check the error above — common causes: "
                "missing pytorch-forecasting/lightning packages, or unreachable market data."
            )
            return

        st.write("Fitting BERTopic on real documents...")
        from hyperspace.models.topic_model import fit_topic_model
        import hashlib

        docs_hash = hashlib.md5("".join(docs[:5]).encode()).hexdigest()[:8]
        cluster_result = fit_topic_model(docs_hash, docs=docs, data_source=docs_src)
        if cluster_result is None:
            status.update(label="Pipeline blocked: BERTopic unavailable", state="error")
            st.error("BERTopic is unavailable; cluster fallback was intentionally removed.")
            return

        st.write("Fetching multimodal spatial rasters (elevation, climate, economics, conflict)...")
        from hyperspace.data.spatial import fetch_all_spatial_data

        try:
            raw_spatial = fetch_all_spatial_data()
        except RuntimeError as exc:
            status.update(label="Pipeline blocked: spatial data unavailable", state="error")
            st.error(str(exc))
            return

        # ---- Step 3: Canonical orchestration (single path) ----
        st.write("Running canonical pipeline runner for graph/spatial/agents/interpreter...")
        runner = PipelineRunner()
        result = runner.run(
            finance_result=tft_result,
            cluster_result=cluster_result,
            agreement_matrix=agreement,
            spatial_data=raw_spatial,
            timeframe_context=timeframe_context,
            compute_cross_block=True,
        )

        st.session_state.run_id = result["run_id"]
        st.session_state.run_timestamp = result["run_timestamp"]
        st.session_state.finance_result = result["finance_result"]
        st.session_state.cluster_result = result["cluster_result"]
        st.session_state.graph_result = result["graph_result"]
        st.session_state.spatial_result = result["spatial_result"]
        st.session_state.sim_result = result["sim_result"]
        st.session_state.sae_result = result["sae_result"]
        st.session_state.concept_kernel_map = result["concept_kernel_map"]
        st.session_state.ukt_snapshots = result["snapshots"]
        st.session_state.data_sources = result["data_sources"]
        st.session_state.semantic_canvas = result["semantic_canvas"]
        st.session_state.canvas_narrative = result["canvas_narrative"]
        st.session_state.reality_narrative = result["reality_narrative"]
        st.session_state.uvt_result = result["uvt_result"]
        st.session_state.use_result = result["use_result"]
        st.session_state.ukt_multirun_stability = result["stability"]
        st.session_state.governance_flags = result["governance_flags"]
        st.session_state.interpretability_scorecard = result["interpretability_scorecard"]
        st.session_state.alignment_metrics = result.get("alignment_metrics", {})
        st.session_state.interpretability_contract = result["interpretability_contract"]
        st.session_state.interpretability_contract_summary = result[
            "interpretability_contract_summary"
        ]

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

    report_lines.append("")
    report_lines.append("## Pipeline Reports")
    for snap in snapshots:
        report_lines.append(snap["report"])

    return "\n".join(report_lines)


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

    # Summary metrics from real computations
    final_snap = snapshots[-1]
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

    stability = st.session_state.get("ukt_multirun_stability")
    if isinstance(stability, dict) and stability.get("n_runs", 0) > 0:
        st.caption(
            "UKT multi-run reality-regression stability "
            f"(n={stability['n_runs']}): mean cosine={stability['mean_cosine']:.3f}, "
            f"min cosine={stability['min_cosine']:.3f}, std={stability['std_cosine']:.3f}"
        )

    st.markdown("---")

    # ---- A3: Governance Flags Panel ----
    gov_flags = st.session_state.get("governance_flags", [])
    if gov_flags:
        st.markdown("### ⚠️ Governance Flags")
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
    else:
        st.markdown(
            '<span class="gov-pass">✓ No governance flags detected</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---- B3: Interpretability Score Card ----
    alignment_metrics = st.session_state.get("alignment_metrics", {})
    if alignment_metrics:
        st.markdown("### Alignment Comparison (Legacy vs Shared-Latent)")
        legacy = alignment_metrics.get("legacy", {})
        shared = alignment_metrics.get("shared_latent", {})
        parity = alignment_metrics.get("parity_delta", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Legacy Retrieval@1", f"{legacy.get('retrieval_at_1', 0.0):.3f}")
        c2.metric("Shared Retrieval@1 (shadow)", f"{shared.get('retrieval_at_1', 0.0):.3f}")
        c3.metric("Delta (shared-legacy)", f"{parity.get('retrieval_at_1', 0.0):+.3f}")
        st.caption("Shared-latent path is feature-flagged and shadow-only. Default decisions remain on legacy UKT outputs.")

    scorecard = st.session_state.get("interpretability_scorecard", {})
    if scorecard:
        st.markdown("### 📊 Interpretability Accountability Score Card")
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

        # Style: color Status column
        def _style_status(val: str) -> str:
            if "PASS" in val:
                return "color: #64ffda; font-weight: bold"
            return "color: #ffaa00; font-weight: bold"

        try:
            styled = sc_df.style.map(_style_status, subset=["Status"])
        except AttributeError:
            styled = sc_df.style.applymap(_style_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Overall pass/fail summary
        all_pass = all(item.get("passed", False) for item in scorecard.values())
        pass_count = sum(1 for item in scorecard.values() if item.get("passed", False))
        total_count = len(scorecard)
        if all_pass:
            st.success(f"Overall: {pass_count}/{total_count} criteria PASS — System meets minimum interpretability standards.")
        else:
            st.warning(f"Overall: {pass_count}/{total_count} criteria PASS — Review flagged items before citing conclusions.")

    st.markdown("---")

    # Semantic Canvas summary
    canvas_narrative = st.session_state.get("canvas_narrative")
    reality_narrative = st.session_state.get("reality_narrative")
    if canvas_narrative or reality_narrative:
        st.markdown("### Semantic Interpretability — LLM Narratives")
        if canvas_narrative:
            st.success(f"**Cross-Domain Narrative:** {canvas_narrative}")
        if reality_narrative:
            st.info(f"**Reality Assessment:** {reality_narrative}")

    st.caption(
        "For detailed kernel analysis, concept discovery, and per-block interpretability "
        "reports, see the **Semantic Interpreter** tab."
    )

    interpretability_contract = st.session_state.get("interpretability_contract", {})
    interpretability_contract_summary = st.session_state.get(
        "interpretability_contract_summary", {},
    )

    if interpretability_contract:
        st.markdown("### Interpretability Contract Governance View")
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

    # Export — D1: stamped with run ID
    st.markdown("### Export")
    run_id = st.session_state.get("run_id", "UNKNOWN")
    run_ts = st.session_state.get("run_timestamp", "")
    exp1, exp2, exp3, exp4 = st.columns(4)

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

    # Score card CSV export
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
        alignment_metrics=alignment_metrics,
    )
    if contract_export_rows:
        exp4.download_button(
            "Download Contract Compliance (CSV)",
            pd.DataFrame(contract_export_rows).to_csv(index=False),
            f"hyperspace_contract_{run_id}.csv", "text/csv",
            key="dashboard_download_interpretability_contract",
        )
