"""Full Pipeline tab: end-to-end orchestration with UKT evolution display."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from hyperspace.config import GEOPOLITICAL_NODES
from hyperspace.models.graph_engine import build_geopolitical_graph, plot_geopolitical_graph
from hyperspace.viz import kernel_viz
from hyperspace.viz.charts import forecast_chart, source_badge


def render() -> None:
    """Render the Full Pipeline tab."""
    st.markdown("## Hyperspace Pipeline — Governance Accountability View")
    st.markdown(
        "End-to-end cycle: Finance → Clustering → Graph → **Spatial Raster Kernelization** → "
        "Agentic Sim → Semantic Interpretation. Each step updates the 80-dim Universal "
        "Knowledge Tensor. All conclusions are traceable, stability-tested, and contestable."
    )

    snapshots = st.session_state.get("ukt_snapshots", [])
    data_sources = st.session_state.get("data_sources", {})
    run_id = st.session_state.get("run_id", "UNKNOWN")
    run_ts = st.session_state.get("run_timestamp", "")
    policy_mode = st.session_state.get("policy_language_mode", False)

    if not snapshots:
        st.info(
            "No pipeline results available. Launch the full pipeline from the "
            "dashboard to see results here."
        )
        return

    # D1: Run watermark
    if run_id and run_id != "UNKNOWN":
        st.markdown(
            f'<span class="run-id-watermark">Run ID: {run_id} · {run_ts}</span>',
            unsafe_allow_html=True,
        )

    # B1: Policy mode banner
    if policy_mode:
        st.info(
            "🗂️ **Governance Language Mode is active.** "
            "Visit the Semantic Interpreter tab to see policy-language kernel briefings."
        )

    # Data source summary
    src_html = " ".join(source_badge(v) for v in data_sources.values())
    st.markdown(f"**Data Sources**: {src_html}", unsafe_allow_html=True)
    st.success("Hyperspace cycle complete. All blocks synchronized.")

    st.markdown("---")

    # B3: Compact scorecard summary
    scorecard = st.session_state.get("interpretability_scorecard", {})
    if scorecard:
        st.markdown("### Interpretability Score Card Summary")
        sc_cols = st.columns(len(scorecard))
        for col, (key, item) in zip(sc_cols, scorecard.items()):
            passed = item.get("passed", False)
            status_icon = "✅" if passed else "⚠️"
            col.metric(
                item.get("label", key),
                f"{item.get('value', 'N/A')}{item.get('unit', '')}",
                f"{status_icon} {'PASS' if passed else 'WARN'}",
            )
        st.markdown("---")

    st.markdown("### Unified Dashboard Summary")

    # Metrics row
    final_snap = snapshots[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Active Kernels", str(final_snap["n_kernels"]))
    m2.metric("Recon Error", f"{final_snap['reconstruction_error']:.6f}")

    finance_result = st.session_state.get("finance_result", {})
    if isinstance(finance_result, dict) and "model_params" in finance_result:
        m3.metric("TFT Params", f"{finance_result['model_params']:,}")
    else:
        m3.metric("TFT Params", "N/A")

    sae_result = st.session_state.get("sae_result")
    if sae_result:
        m4.metric("SAE Concepts", f"{sae_result['active_concepts']}/{sae_result['total_concepts']}")
    else:
        m4.metric("SAE Concepts", "N/A")

    graph_result = st.session_state.get("graph_result", {})
    if isinstance(graph_result, dict) and "analysis" in graph_result:
        m5.metric("Communities", str(len(graph_result["analysis"]["communities"])))
    else:
        m5.metric("Communities", "N/A")

    # Kernel evolution chart
    st.markdown("### Kernel Evolution Across Pipeline Steps")
    fig_evo = kernel_viz.plot_kernel_evolution(snapshots)
    st.plotly_chart(fig_evo, use_container_width=True)

    # Key outputs
    st.markdown("### Key Outputs Across All Blocks")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        # Finance forecast
        if isinstance(finance_result, dict) and "q50" in finance_result:
            x_ax = list(range(len(finance_result["q50"])))
            fig = forecast_chart(
                x_ax, finance_result["q10"], finance_result["q50"],
                finance_result["q90"], title="Finance: Multi-Horizon Forecast",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif isinstance(finance_result, dict) and "quantiles" in finance_result:
            q = finance_result["quantiles"]
            if len(q.shape) == 3:
                q_mean = q.mean(axis=0)
                if q_mean.shape[0] == 0 or q_mean.shape[1] < 2:
                    st.warning("TFT quantile output has insufficient shape for forecast chart.")
                else:
                    x_ax = list(range(q_mean.shape[0]))
                    fig = forecast_chart(
                        x_ax, q_mean[:, 0], q_mean[:, q_mean.shape[1] // 2],
                        q_mean[:, -1], title="Finance: TFT Forecast",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Finance forecast unavailable — run the full pipeline first.")

    with r1c2:
        # Graph
        if isinstance(graph_result, dict) and "G" in graph_result:
            fig_g = plot_geopolitical_graph(graph_result["G"], graph_result["pos"])
            fig_g.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        # Agent resources
        sim_result = st.session_state.get("sim_result")
        if sim_result and "agents" in sim_result:
            import plotly.graph_objects as go
            agents = sim_result["agents"]
            names = list(agents.keys())
            resources = [agents[n].resources for n in names]
            colors = [GEOPOLITICAL_NODES.get(n, {}).get("color", "#888") for n in names]
            fig = go.Figure(go.Bar(x=names, y=resources, marker_color=colors))
            fig.update_layout(
                template="plotly_dark", title="Agents: Final Resources",
                height=300, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        # Final kernel matrix (compact)
        fig_km = kernel_viz.plot_kernel_matrix(
            final_snap, [s["block_name"] for s in snapshots],
        )
        fig_km.update_layout(height=300)
        st.plotly_chart(fig_km, use_container_width=True)

    # C1: Annotation summary
    annotations = st.session_state.get("stakeholder_annotations", [])
    if annotations:
        st.markdown("---")
        st.markdown(f"### Stakeholder Annotations ({len(annotations)})")
        from hyperspace.pages.governance import render_annotations_summary
        render_annotations_summary()

    # Export — D1: stamped with run ID
    st.markdown("---")
    st.markdown("### Export")
    exp1, exp2, exp3 = st.columns(3)

    # Build report
    report_lines = [
        f"# Hyperspace Governance Report",
        f"## Run ID: {run_id} | {run_ts}",
        "",
    ]

    # Include scorecard in report
    if scorecard:
        report_lines.append("## Interpretability Score Card")
        for key, item in scorecard.items():
            status_str = "PASS" if item.get("passed") else "WARN"
            report_lines.append(
                f"- {item.get('label')}: {item.get('value')}{item.get('unit','')} "
                f"(≥{item.get('threshold')}{item.get('unit','')}) — {status_str}"
            )
        report_lines.append("")

    # Include governance flags
    gov_flags = st.session_state.get("governance_flags", [])
    if gov_flags:
        report_lines.append("## Governance Flags")
        for flag in gov_flags:
            report_lines.append(
                f"- [{flag.get('code')}] {flag.get('label')}: {flag.get('detail', '')}"
            )
        report_lines.append("")

    report_lines.append("## Pipeline Reports")
    for snap in snapshots:
        report_lines.append(snap["report"])

    # Include annotations
    if annotations:
        report_lines.append("")
        report_lines.append("## Stakeholder Annotations")
        for ann in annotations:
            report_lines.append(
                f"- [{ann.get('role')}] {ann.get('timestamp', '')}: {ann.get('text', '')}"
            )

    report_md = "\n".join(report_lines)
    exp1.download_button(
        "Download Report (Markdown)", report_md,
        f"hyperspace_report_{run_id}.md", "text/markdown",
        key="pipeline_download_report",
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
            key="pipeline_download_metrics",
        )

    # Scorecard CSV
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
            key="pipeline_download_scorecard",
        )
