"""Politics-Military tab: graph engine with real centrality analysis."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from hyperspace.config import PLOTLY_LAYOUT
from hyperspace.data.political import get_political_data
from hyperspace.models.graph_engine import (
    analyze_graph, build_geopolitical_graph, plot_geopolitical_graph,
)
from hyperspace.viz.charts import source_badge


def render() -> None:
    """Render the Politics-Military Graph tab."""
    st.markdown("## Politics-Military Graph Engine")
    st.markdown(
        "Multi-relational geopolitical graph with real centrality analysis, "
        "community detection, and optional UN voting data integration."
    )

    graph_btn = st.button("Build & Analyze Graph", type="primary", key="graph_build")

    graph_result = st.session_state.get("graph_result")

    if graph_btn or graph_result:
        if graph_btn:
            with st.spinner("Fetching political data and analyzing graph..."):
                un_df, agreement, pol_src = get_political_data()
                G, pos = build_geopolitical_graph(agreement_matrix=agreement)
                analysis = analyze_graph(G)
                graph_result = dict(
                    G=G, pos=pos, analysis=analysis,
                    features_for_ukt=analysis["features_for_ukt"],
                    data_source=pol_src,
                    un_df=un_df, agreement=agreement,
                )
                st.session_state.graph_result = graph_result

        if graph_result:
            src = graph_result.get("data_source", "unknown")
            st.markdown(f"**Data source**: {source_badge(src)}", unsafe_allow_html=True)

            G = graph_result["G"]
            pos = graph_result["pos"]
            analysis = graph_result["analysis"]

            # Graph visualization
            st.markdown("### Geopolitical Graph")
            fig = plot_geopolitical_graph(G, pos)
            st.plotly_chart(fig, use_container_width=True)

            # Centrality metrics
            st.markdown("### Centrality Analysis")
            cent_df = pd.DataFrame({
                "Node": analysis["node_names"],
                "Degree": [analysis["degree_centrality"][n] for n in analysis["node_names"]],
                "Betweenness": [analysis["betweenness"][n] for n in analysis["node_names"]],
                "Eigenvector": [analysis["eigenvector"][n] for n in analysis["node_names"]],
                "PageRank": [analysis["pagerank"][n] for n in analysis["node_names"]],
            })
            st.dataframe(cent_df.style.format({
                "Degree": "{:.3f}", "Betweenness": "{:.3f}",
                "Eigenvector": "{:.3f}", "PageRank": "{:.3f}",
            }), use_container_width=True)

            # Centrality bar chart
            fig = px.bar(
                cent_df.melt(id_vars="Node", var_name="Metric", value_name="Score"),
                x="Node", y="Score", color="Metric", barmode="group",
                title="Centrality Measures by Node",
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Graph metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Density", f"{analysis['density']:.3f}")
            c2.metric("Avg Clustering", f"{analysis['avg_clustering']:.3f}")
            c3.metric("Communities", str(len(analysis["communities"])))

            # Community membership
            with st.expander("Community Detection Results"):
                for i, comm in enumerate(analysis["communities"]):
                    st.markdown(f"**Community {i + 1}**: {', '.join(comm)}")

            # Voting agreement heatmap (if available)
            agreement = graph_result.get("agreement")
            if agreement is not None and isinstance(agreement, pd.DataFrame):
                st.markdown("### Voting Agreement Matrix")
                fig = px.imshow(
                    agreement,
                    color_continuous_scale="RdBu_r", text_auto=".2f",
                    title="Pairwise Agreement (Geopolitical Nodes)",
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=400)
                st.plotly_chart(fig, use_container_width=True)

            # UKT contribution
            if "features_for_ukt" in graph_result:
                with st.expander("UKT Contribution (Graph Feature Vector)"):
                    fv = graph_result["features_for_ukt"]
                    st.bar_chart(pd.DataFrame(fv, columns=["Value"]))
