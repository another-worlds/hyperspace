"""Dashboard: landing page, pipeline orchestration, progress tracking, results display."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.config import (
    DEFAULT_TICKERS, GEOPOLITICAL_NODES, PIPELINE_STEPS, UKT_FEATURE_DIM,
)
from hyperspace.models.knowledge_matrix import UniversalKnowledgeTensor
from hyperspace.viz.charts import source_badge
from hyperspace.viz import kernel_viz


def render_landing() -> None:
    """Render the dashboard landing page (shown before pipeline launch)."""
    st.markdown("## Hyperspace -- Predictive Polymath System v3.0")
    st.markdown("""
> **Mission**: A modular, continuously learning hybrid AI system integrating
> semantic interpretability, persistent knowledge reuse, financial forecasting,
> informational clustering, geopolitical graph simulation, and agentic
> macro-modeling into a single coherent pipeline.

**Core architecture**: Each pipeline block feeds features into the **Universal
Knowledge Tensor (UKT)**, which is decomposed at every step via SVD to produce
a **universal reality regression** -- a combined basis of reality dimensions --
with full **semantic interpretability** at each stage.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipeline Steps", "6", "")
    c2.metric("Feature Dimensions", str(UKT_FEATURE_DIM), "")
    c3.metric("Blocks", "4", "Finance + Clusters + Graph + Agents")
    c4.metric("Kernels (max)", "4", "grows with blocks")

    st.markdown("---")

    # Cold-start kernel matrix placeholder
    st.markdown("### Universal Kernel Matrix")
    st.caption("The kernel matrix will populate as each pipeline block completes.")
    placeholder_matrix = np.zeros((1, UKT_FEATURE_DIM))
    import plotly.express as px
    fig = px.imshow(
        placeholder_matrix,
        color_continuous_scale="RdBu_r",
        template="plotly_dark",
        title="UKT: Awaiting pipeline launch...",
    )
    fig.update_layout(height=150, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Architecture diagram
    with st.expander("v3.0 Pipeline Architecture"):
        st.code("""
Data Fetch (yfinance + RSS + UN Votes + 20newsgroups)
          |
    [Finance Block] --> TFT train --> UKT row 1 --> SVD --> Interpret
          |
    [Cluster Block] --> BERTopic  --> UKT row 2 --> SVD --> Interpret
          |
    [Graph Block]   --> Centrality --> UKT row 3 --> SVD --> Interpret
          |
    [Agent Sim]     --> Simulate  --> UKT row 4 --> SVD --> Interpret
          |
    [Final SAE]     --> Concept discovery on full UKT
          |
    Universal Reality Regression + Semantic Interpretability Report
        """, language="text")


def run_pipeline() -> None:
    """Execute the full pipeline with step-by-step UKT updates."""
    ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
    snapshots: list[dict] = []
    data_sources: dict[str, str] = {}

    with st.status("Running Hyperspace Pipeline...", expanded=True) as status:
        # ---- Step 1: Fetch Data ----
        st.write("Fetching real data sources...")
        from hyperspace.data.finance import get_ohlcv
        from hyperspace.data.news import get_text_data
        from hyperspace.data.political import get_political_data

        tickers = st.session_state.get("tickers", DEFAULT_TICKERS)
        ohlcv_df, fin_src = get_ohlcv(tickers)
        docs, docs_src = get_text_data()
        un_df, agreement, pol_src = get_political_data()

        data_sources["Finance"] = fin_src
        data_sources["Clusters"] = docs_src
        data_sources["Graph"] = pol_src
        st.session_state.data_sources = data_sources
        st.session_state.raw_ohlcv = ohlcv_df
        st.session_state.raw_docs = docs
        st.write(f"Data fetched: {fin_src} | {docs_src} | {pol_src}")

        # ---- Step 2: Finance Block ----
        st.write("Training Temporal Fusion Transformer (3 epochs, CPU)...")
        from hyperspace.models.tft_forecast import fit_tft, mock_forecast

        tft_result = fit_tft(
            tickers=tuple(tickers),
            hidden=32, encoder_len=48, prediction_len=12,
        )
        if tft_result is not None:
            finance_features = tft_result["features_for_ukt"]
            data_sources["Finance"] = tft_result["data_source"]
        else:
            mock = mock_forecast(12)
            finance_features = mock["features_for_ukt"]
            tft_result = mock
            data_sources["Finance"] = mock["data_source"]

        snap = ukt.add_block("Finance", finance_features)
        snapshots.append(snap)
        st.session_state.finance_result = tft_result
        st.write(f"Finance: {snap['report'].split(chr(10))[0]}")

        # ---- Step 3: Cluster Block ----
        st.write("Fitting BERTopic on real documents...")
        from hyperspace.models.topic_model import fit_topic_model, mock_clusters

        import hashlib
        docs_hash = hashlib.md5("".join(docs[:5]).encode()).hexdigest()[:8]
        cluster_result = fit_topic_model(docs_hash)
        if cluster_result is not None:
            cluster_features = cluster_result["features_for_ukt"]
            data_sources["Clusters"] = cluster_result["data_source"]
        else:
            cluster_df, cluster_features = mock_clusters(docs)
            cluster_result = dict(
                mock_df=cluster_df, docs=docs,
                features_for_ukt=cluster_features,
                data_source="Fallback: keyword clusters",
            )
            data_sources["Clusters"] = cluster_result["data_source"]

        snap = ukt.add_block("Clusters", cluster_features)
        snapshots.append(snap)
        st.session_state.cluster_result = cluster_result
        st.write(f"Clusters: {snap['report'].split(chr(10))[0]}")

        # ---- Step 4: Graph Block ----
        st.write("Analyzing geopolitical graph + centrality...")
        from hyperspace.models.graph_engine import build_geopolitical_graph, analyze_graph

        G, pos = build_geopolitical_graph(agreement_matrix=agreement)
        graph_analysis = analyze_graph(G)
        graph_features = graph_analysis["features_for_ukt"]

        snap = ukt.add_block("Graph", graph_features)
        snapshots.append(snap)
        st.session_state.graph_result = dict(
            G=G, pos=pos, analysis=graph_analysis,
            features_for_ukt=graph_features,
            data_source=pol_src,
        )
        st.write(f"Graph: {snap['report'].split(chr(10))[0]}")

        # ---- Step 5: Agent Simulation ----
        st.write("Running agent simulation (50 steps)...")
        from hyperspace.models.agent_sim import (
            initialize_agents_from_data, run_simulation,
        )

        agents = initialize_agents_from_data(graph_analysis, agreement)
        agents, log_entries, agent_features = run_simulation(agents, steps=50)

        snap = ukt.add_block("Agents", agent_features)
        snapshots.append(snap)
        st.session_state.sim_result = dict(
            agents=agents, log=log_entries,
            features_for_ukt=agent_features,
        )
        st.write(f"Agents: {snap['report'].split(chr(10))[0]}")

        # ---- Step 6: Final Interpretation ----
        st.write("Running sparse autoencoder for concept discovery...")
        from hyperspace.models.sparse_ae import train_sparse_ae, map_concepts_to_kernels

        final_matrix = ukt.get_final_matrix()
        sae_result = None
        concept_kernel_map = []
        if final_matrix is not None:
            sae_result = train_sparse_ae(final_matrix, hidden_dim=16, epochs=80)
            if sae_result is not None:
                final_snap = ukt.get_latest_snapshot()
                if final_snap:
                    concept_kernel_map = map_concepts_to_kernels(sae_result, final_snap)

        st.session_state.sae_result = sae_result
        st.session_state.concept_kernel_map = concept_kernel_map
        st.session_state.ukt_snapshots = snapshots
        st.session_state.data_sources = data_sources

        status.update(label="Pipeline complete!", state="complete")

    st.session_state.pipeline_complete = True


def render_results() -> None:
    """Render the full results dashboard after pipeline completes."""
    snapshots = st.session_state.get("ukt_snapshots", [])
    data_sources = st.session_state.get("data_sources", {})

    if not snapshots:
        st.info("No pipeline results yet. Click 'Launch' to run.")
        return

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
        density = graph_result["analysis"]["density"]
        m3.metric("Graph Density", f"{density:.3f}")
    else:
        m3.metric("Graph Density", "N/A")

    sae_result = st.session_state.get("sae_result")
    if sae_result:
        m4.metric("Active Concepts", f"{sae_result['active_concepts']}/{sae_result['total_concepts']}")
    else:
        m4.metric("Active Concepts", "N/A")

    m5.metric("Recon Error", f"{final_snap['reconstruction_error']:.6f}")

    st.markdown("---")

    # Kernel matrix visualization
    st.markdown("### Universal Kernel Matrix")
    block_names = [s["block_name"] for s in snapshots]
    fig_km = kernel_viz.plot_kernel_matrix(final_snap, block_names)
    st.plotly_chart(fig_km, use_container_width=True)

    # Kernel importance + reality regression
    col1, col2 = st.columns(2)
    with col1:
        fig_imp = kernel_viz.plot_kernel_importance(final_snap)
        st.plotly_chart(fig_imp, use_container_width=True)
    with col2:
        fig_rr = kernel_viz.plot_reality_regression(final_snap)
        st.plotly_chart(fig_rr, use_container_width=True)

    # Kernel evolution
    st.markdown("### Kernel Evolution Across Pipeline Steps")
    fig_evo = kernel_viz.plot_kernel_evolution(snapshots)
    st.plotly_chart(fig_evo, use_container_width=True)

    # Semantic interpretation log
    st.markdown("### Semantic Interpretability Report")
    for snap in snapshots:
        with st.expander(f"Step {snap['step']}: {snap['block_name']}", expanded=False):
            st.markdown(snap["report"])
            for kl in snap["kernel_labels"]:
                st.markdown(
                    f'<span class="concept-badge">{kl["label"]}</span>',
                    unsafe_allow_html=True,
                )
                if kl.get("narrative"):
                    st.caption(kl["narrative"])

    # Concept-kernel map (from SAE)
    concept_kernel_map = st.session_state.get("concept_kernel_map", [])
    if concept_kernel_map:
        st.markdown("### Concept-Kernel Correspondence (Sparse Autoencoder)")
        fig_ck = kernel_viz.plot_concept_kernel_map(concept_kernel_map)
        st.plotly_chart(fig_ck, use_container_width=True)
        st.dataframe(pd.DataFrame(concept_kernel_map), use_container_width=True)

    # Export
    st.markdown("### Export")
    exp1, exp2 = st.columns(2)
    report_lines = []
    for snap in snapshots:
        report_lines.append(snap["report"])
    report_md = "# Hyperspace Pipeline Report\n\n" + "\n\n".join(report_lines)
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
