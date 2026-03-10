"""Politics-Military tab: graph engine with real centrality analysis."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from hyperspace.config import PLOTLY_LAYOUT
from hyperspace.data.political import get_political_data
from hyperspace.data.map import get_country_stats
from hyperspace.models.graph_engine import (
    analyze_graph, build_geopolitical_graph,
    plot_geopolitical_graph, plot_geopolitical_map,
)
from hyperspace.pages._report_section import render_interpretability_report
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
                try:
                    un_df, agreement, pol_src = get_political_data()
                except RuntimeError as exc:
                    st.error(str(exc))
                    return
                G, pos = build_geopolitical_graph(agreement_matrix=agreement)
                analysis = analyze_graph(G)
                try:
                    country_stats, map_src = get_country_stats()
                except RuntimeError:
                    country_stats, map_src = None, "unavailable"
                graph_result = dict(
                    G=G, pos=pos, analysis=analysis,
                    features_for_ukt=analysis["features_for_ukt"],
                    data_source=pol_src,
                    un_df=un_df, agreement=agreement,
                    country_stats=country_stats, map_src=map_src,
                )
                st.session_state.graph_result = graph_result

        if graph_result:
            src = graph_result.get("data_source", "unknown")
            st.markdown(f"**Data source**: {source_badge(src)}", unsafe_allow_html=True)

            G = graph_result.get("G")
            pos = graph_result.get("pos")
            analysis = graph_result.get("analysis", {})
            if G is None or not analysis:
                st.warning("Graph data incomplete. Try rebuilding the graph.")
                return

            # Geographic map
            country_stats = graph_result.get("country_stats")
            map_src = graph_result.get("map_src", "")
            st.markdown("### Geographic View")
            if map_src and map_src != "unavailable":
                st.markdown(
                    f"**Map data**: {source_badge(map_src)}", unsafe_allow_html=True
                )
            fig_map = plot_geopolitical_map(G, country_stats=country_stats)
            st.plotly_chart(fig_map, use_container_width=True, key="politics_geo_map")
            st.caption(
                "v3.0 — Geographic projection of the geopolitical graph. Node "
                "positions and edge weights feed into UKT structural-centrality "
                "features (indices 32–47), enabling spatial provenance tracing."
            )

            # Abstract graph
            st.markdown("### Relation Graph")
            fig = plot_geopolitical_graph(G, pos)
            st.plotly_chart(fig, use_container_width=True, key="politics_relation_graph")
            st.caption(
                "v3.0 — Abstract relation graph with edge weights derived from "
                "voting agreement data. Centrality measures from this graph "
                "populate the structural-centrality UKT region."
            )

            # Centrality analysis
            st.markdown("### Influence Distribution")
            node_names = analysis.get("node_names", [])
            degree_cent = analysis.get("degree_centrality", {})
            betweenness = analysis.get("betweenness", {})
            eigenvector = analysis.get("eigenvector", {})
            pagerank = analysis.get("pagerank", {})
            cent_df = pd.DataFrame({
                "Node": node_names,
                "Degree": [degree_cent.get(n, 0.0) for n in node_names],
                "Betweenness": [betweenness.get(n, 0.0) for n in node_names],
                "Eigenvector": [eigenvector.get(n, 0.0) for n in node_names],
                "PageRank": [pagerank.get(n, 0.0) for n in node_names],
            })
            fig = px.bar(
                cent_df.melt(id_vars="Node", var_name="Metric", value_name="Score"),
                x="Node", y="Score", color="Metric", barmode="group",
                title="Influence Measures by Actor",
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True, key="politics_centrality_bar")
            st.caption(
                "**Degree** — how many direct relationships an actor has. "
                "**Betweenness** — how often an actor sits on the shortest path between others "
                "(high betweenness = gatekeeper / broker role). "
                "**PageRank** — recursive influence; an actor connected to influential actors "
                "scores higher regardless of raw degree. "
                "Concentrated scores on one actor trigger Governance Flag GOV-003 (centrality skew)."
            )

            # Community structure summary
            communities = analysis.get("communities", [])
            if communities:
                st.markdown("---")
                if len(communities) == 1:
                    st.warning(
                        f"**Alliance Structure: UNIPOLAR** — all actors form a single community. "
                        "No distinct opposing blocs detected in this analysis period."
                    )
                elif len(communities) == 2:
                    blocs = [", ".join(c) for c in communities]
                    st.info(
                        f"**Alliance Structure: BIPOLAR** — two distinct blocs detected: "
                        f"[{blocs[0]}] vs [{blocs[1]}]."
                    )
                else:
                    blocs = [", ".join(c) for c in communities]
                    st.success(
                        f"**Alliance Structure: MULTIPOLAR** — {len(communities)} distinct blocs: "
                        + " | ".join(f"[{b}]" for b in blocs) + "."
                    )

            # ----------------------------------------------------------- #
            # Interpretability Report                                      #
            # ----------------------------------------------------------- #
            st.markdown("---")
            render_interpretability_report("Graph")
