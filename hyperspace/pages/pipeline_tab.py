"""Full Pipeline tab: end-to-end orchestration with UKT evolution display."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from hyperspace.config import GEOPOLITICAL_NODES
from hyperspace.models.graph_engine import build_geopolitical_graph, plot_geopolitical_graph
from hyperspace.models.tft_forecast import mock_forecast
from hyperspace.viz import kernel_viz
from hyperspace.viz.charts import forecast_chart, source_badge


def render() -> None:
    """Render the Full Pipeline tab."""
    st.markdown("## Hyperspace Pipeline -- End-to-End Cycle")
    st.markdown(
        "Full orchestration: Finance -> Clustering -> Graph -> Agentic Sim -> "
        "Semantic Interpretation. Each step updates the Universal Knowledge Tensor."
    )

    snapshots = st.session_state.get("ukt_snapshots", [])
    data_sources = st.session_state.get("data_sources", {})

    if not snapshots:
        st.info(
            "No pipeline results available. Launch the full pipeline from the "
            "dashboard to see results here."
        )
        return

    # Data source summary
    src_html = " ".join(source_badge(v) for v in data_sources.values())
    st.markdown(f"**Data Sources**: {src_html}", unsafe_allow_html=True)
    st.success("Hyperspace cycle complete. All blocks synchronized.")

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
                x_ax = list(range(q_mean.shape[0]))
                fig = forecast_chart(
                    x_ax, q_mean[:, 0], q_mean[:, q_mean.shape[1] // 2],
                    q_mean[:, -1], title="Finance: TFT Forecast",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            mock = mock_forecast(20)
            x_ax = list(range(20))
            fig = forecast_chart(
                x_ax, mock["q10"], mock["q50"], mock["q90"],
                title="Finance: Mock Forecast",
            )
            st.plotly_chart(fig, use_container_width=True)

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

    # Export
    st.markdown("### Export")
    exp1, exp2 = st.columns(2)
    report_lines = []
    for snap in snapshots:
        report_lines.append(snap["report"])
    report_md = f"# Hyperspace Pipeline Report\n## Date: {datetime.now().date()}\n\n" + "\n\n".join(report_lines)
    exp1.download_button("Download Report (Markdown)", report_md,
                         "hyperspace_report.md", "text/markdown")

    metrics_rows = []
    for snap in snapshots:
        for kl in snap["kernel_labels"]:
            metrics_rows.append({
                "Step": snap["step"], "Block": snap["block_name"],
                "Kernel": kl["kernel_id"], "Importance": kl["importance"],
                "Region": kl["dominant_region"],
            })
    if metrics_rows:
        exp2.download_button("Download Metrics (CSV)",
                             pd.DataFrame(metrics_rows).to_csv(index=False),
                             "hyperspace_metrics.csv", "text/csv")
