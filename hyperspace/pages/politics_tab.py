"""Politics-Military tab: graph engine with real centrality analysis."""
from __future__ import annotations

import numpy as np
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

            # Country stats table
            if country_stats:
                rows = []
                for node, cs in country_stats.items():
                    rows.append({
                        "Node": node,
                        "Flag": cs["flag_emoji"],
                        "Capital": cs["capital"],
                        "Population (M)": round(cs["population"] / 1_000_000, 1),
                        "Area (k km²)": round(cs["area_km2"] / 1_000, 0),
                        "Region": cs["subregion"] or cs["region"],
                        "UN Member": "✓" if cs["un_member"] else "✗",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Abstract graph
            st.markdown("### Relation Graph")
            fig = plot_geopolitical_graph(G, pos)
            st.plotly_chart(fig, use_container_width=True, key="politics_relation_graph")
            st.caption(
                "v3.0 — Abstract relation graph with edge weights derived from "
                "voting agreement data. Centrality measures from this graph "
                "populate the structural-centrality UKT region."
            )

            # Centrality metrics
            st.markdown("### Centrality Analysis")
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
            st.plotly_chart(fig, use_container_width=True, key="politics_centrality_bar")
            st.caption(
                "v3.0 — Centrality measures (degree, betweenness, eigenvector, "
                "PageRank) are flattened into UKT indices 32–47. High betweenness "
                "flags bridge nodes; skew triggers governance flag GOV-003."
            )

            # Graph metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Density", f"{analysis.get('density', 0.0):.3f}")
            c2.metric("Avg Clustering", f"{analysis.get('avg_clustering', 0.0):.3f}")
            communities = analysis.get("communities", [])
            c3.metric("Communities", str(len(communities)))

            # Community membership
            with st.expander("Community Detection Results"):
                for i, comm in enumerate(communities):
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
                st.plotly_chart(fig, use_container_width=True, key="politics_voting_agreement")
                st.caption(
                    "v3.0 — Pairwise voting agreement informs edge weights in the "
                    "geopolitical graph. This matrix is fully traceable to the "
                    "Harvard Dataverse UN votes source (or config-edge fallback)."
                )

            # UKT contribution
            if "features_for_ukt" in graph_result:
                with st.expander("UKT Contribution (Graph Feature Vector)"):
                    fv = graph_result["features_for_ukt"]
                    st.bar_chart(pd.DataFrame(fv, columns=["Value"]))

            # ----------------------------------------------------------- #
            # Fitting Metrics                                              #
            # ----------------------------------------------------------- #
            st.markdown("---")
            st.markdown("### Fitting Metrics")
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            max_edges = n_nodes * (n_nodes - 1) // 2

            fm1, fm2, fm3, fm4 = st.columns(4)
            fm1.metric("Nodes", str(n_nodes))
            fm2.metric("Edges", str(n_edges))
            fm3.metric("Edge Coverage", f"{n_edges / max(max_edges, 1):.1%}")
            fm4.metric("Graph Density", f"{analysis.get('density', 0.0):.4f}")

            # Modularity (quality of community structure)
            communities = analysis.get("communities", [])
            if communities:
                try:
                    import networkx.algorithms.community as nx_comm
                    modularity = nx_comm.modularity(G, [set(c) for c in communities],
                                                    weight="weight")
                    st.metric("Modularity", f"{modularity:.4f}",
                              help="Quality of community partition; "
                                   "higher = stronger community structure")
                except Exception:
                    pass

            # Edge weight statistics
            weights = [d["weight"] for _, _, d in G.edges(data=True)]
            if weights:
                w_arr = np.array(weights)
                wm1, wm2, wm3 = st.columns(3)
                wm1.metric("Mean Edge Weight", f"{np.mean(w_arr):+.3f}")
                wm2.metric("Edge Weight Std", f"{np.std(w_arr):.3f}")
                wm3.metric("Negative Edges", str(int(np.sum(w_arr < 0))))

            # ----------------------------------------------------------- #
            # Test Metrics                                                 #
            # ----------------------------------------------------------- #
            st.markdown("### Test Metrics")
            degree_cent = analysis.get("degree_centrality", {})
            betweenness = analysis.get("betweenness", {})
            eigenvector = analysis.get("eigenvector", {})
            pagerank = analysis.get("pagerank", {})

            if degree_cent:
                # Centrality Gini — measures how evenly distributed influence is
                for metric_name, metric_vals in [
                    ("Degree", degree_cent),
                    ("Betweenness", betweenness),
                    ("Eigenvector", eigenvector),
                    ("PageRank", pagerank),
                ]:
                    vals = np.array(list(metric_vals.values()))
                    if len(vals) > 1:
                        gini = float(
                            np.sum(np.abs(np.subtract.outer(vals, vals)))
                            / (2 * len(vals) * (np.sum(vals) + 1e-8))
                        )
                    else:
                        gini = 0.0
                    st.metric(
                        f"{metric_name} Centrality Gini", f"{gini:.3f}",
                        help="0 = uniform influence, 1 = single node dominates",
                    )

            # Community balance
            if communities and len(communities) > 1:
                comm_sizes = np.array([len(c) for c in communities])
                balance = float(comm_sizes.min() / (comm_sizes.max() + 1e-8))
                st.metric("Community Balance", f"{balance:.2f}",
                          help="1.0 = perfectly balanced, near 0 = one "
                               "community dominates")

            # ----------------------------------------------------------- #
            # Interpretability Report                                      #
            # ----------------------------------------------------------- #
            st.markdown("---")
            render_interpretability_report("Graph")
