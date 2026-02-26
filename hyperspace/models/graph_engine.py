"""Geopolitical graph: build, analyze, and extract centrality features for UKT."""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from hyperspace.config import (
    GEOPOLITICAL_EDGES, GEOPOLITICAL_NODES, PLOTLY_LAYOUT, UKT_FEATURE_DIM,
)


def build_geopolitical_graph(
    agreement_matrix: pd.DataFrame | None = None,
) -> tuple[nx.Graph, dict]:
    """Build the geopolitical graph, optionally updating weights from UN voting data.

    Args:
        agreement_matrix: If provided, blends real voting agreement into edge weights.

    Returns:
        (graph, pos_dict)
    """
    G = nx.Graph()
    for name, attrs in GEOPOLITICAL_NODES.items():
        G.add_node(name, **attrs)
    for src, dst, w, etype, desc in GEOPOLITICAL_EDGES:
        G.add_edge(src, dst, weight=w, edge_type=etype, description=desc)

    # Blend in real voting agreement if available
    if agreement_matrix is not None:
        for u, v in G.edges():
            # Map node names to country names in UN data
            name_map = {
                "USA": "United States", "Russia": "Russia",
                "China": "China", "Britain": "United Kingdom",
                "India": "India", "Brazil": "Brazil",
            }
            un_u = name_map.get(u)
            un_v = name_map.get(v)
            if (un_u and un_v and un_u in agreement_matrix.columns
                    and un_v in agreement_matrix.columns):
                real_w = agreement_matrix.loc[un_u, un_v]
                if not np.isnan(real_w):
                    old_w = G[u][v]["weight"]
                    G[u][v]["weight"] = 0.4 * old_w + 0.6 * real_w

    pos = nx.spring_layout(G, seed=42, k=2.5)
    return G, pos


def analyze_graph(G: nx.Graph) -> dict:
    """Run centrality and community analysis.

    Returns dict with centrality measures, communities, and UKT feature vector.
    """
    # Centrality measures
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight")
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigenvector = degree_cent
    try:
        # PageRank requires non-negative weights; use absolute values
        G_abs = G.copy()
        for u, v in G_abs.edges():
            G_abs[u][v]["weight"] = abs(G_abs[u][v]["weight"])
        pagerank = nx.pagerank(G_abs, weight="weight")
    except Exception:
        pagerank = degree_cent

    # Community detection
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = [list(c) for c in greedy_modularity_communities(G, weight="weight")]
    except Exception:
        communities = [list(G.nodes())]

    # Graph-level metrics
    density = nx.density(G)
    try:
        avg_clustering = nx.average_clustering(G, weight="weight")
    except Exception:
        avg_clustering = 0.0

    # Build node feature matrix: nodes x 4 centrality measures
    node_names = list(G.nodes())
    feature_matrix = np.array([
        [degree_cent[n], betweenness[n], eigenvector[n], pagerank[n]]
        for n in node_names
    ])

    # Build UKT feature vector (structural-centrality region: indices 32-47)
    features_for_ukt = np.zeros(UKT_FEATURE_DIM)
    feature_meta: dict[int, dict] = {}
    metric_names = ["degree_centrality", "betweenness", "eigenvector", "pagerank"]
    fm_flat = feature_matrix.flatten()
    n_fill = min(16, len(fm_flat))
    features_for_ukt[32:32 + n_fill] = fm_flat[:n_fill]

    for local_idx in range(n_fill):
        node_idx = local_idx // 4
        metric_idx = local_idx % 4
        if node_idx < len(node_names):
            node = node_names[node_idx]
            metric = metric_names[metric_idx]
            feature_meta[32 + local_idx] = {
                "label": f"{node}_{metric}",
                "entity": node,
                "metric": metric,
                "block": "graph",
                "source": "networkx_centrality",
            }

    # Add graph-level stats
    features_for_ukt[48] = density
    features_for_ukt[49] = avg_clustering
    features_for_ukt[50] = len(communities)
    feature_meta[48] = {"label": "graph_density", "block": "graph", "metric": "density"}
    feature_meta[49] = {"label": "graph_avg_clustering", "block": "graph", "metric": "avg_clustering"}
    feature_meta[50] = {"label": "graph_n_communities", "block": "graph", "metric": "n_communities"}

    return dict(
        degree_centrality=degree_cent,
        betweenness=betweenness,
        eigenvector=eigenvector,
        pagerank=pagerank,
        communities=communities,
        density=density,
        avg_clustering=avg_clustering,
        feature_matrix=feature_matrix,
        node_names=node_names,
        features_for_ukt=features_for_ukt,
        feature_meta=feature_meta,
    )


def plot_geopolitical_graph(G: nx.Graph, pos: dict) -> go.Figure:
    """Create an interactive Plotly figure from the geopolitical graph."""
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data["weight"]
        color = "#64ffda" if w > 0 else "#ff6b6b"
        width = abs(w) * 4
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines", line=dict(width=width, color=color),
            hoverinfo="text",
            hovertext=(f"{u} <-> {v}<br>Weight: {w:+.2f}<br>"
                       f"Type: {data['edge_type']}<br>{data['description']}"),
            showlegend=False,
        ))

    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for node, attrs in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"{node}<br>Influence: {attrs['influence']:.2f}<br>Bloc: {attrs['bloc']}")
        node_size.append(attrs["influence"] * 50 + 10)
        node_color.append(attrs["color"])

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=2, color="#ffffff")),
        text=[n for n in G.nodes()], textposition="top center",
        textfont=dict(size=11, color="#e0e0ff"),
        hovertext=node_text, hoverinfo="text", showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Multi-Relational Geopolitical Graph (v3.0 Politics-Military Engine)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=520, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
