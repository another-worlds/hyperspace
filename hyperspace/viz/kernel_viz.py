"""Kernel matrix visualization: heatmaps, importance bars, evolution plots."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hyperspace.config import PLOTLY_LAYOUT


def plot_kernel_matrix(snapshot: dict, block_names: list[str] | None = None) -> go.Figure:
    """Plot the cross-block kernel activation heatmap."""
    ka = snapshot["kernel_activation"]
    n_blocks, n_kernels = ka.shape
    if block_names is None:
        block_names = [f"Block {i}" for i in range(n_blocks)]

    fig = px.imshow(
        ka,
        x=[f"K{i}" for i in range(n_kernels)],
        y=block_names[:n_blocks],
        color_continuous_scale="RdBu_r",
        text_auto=".2f",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Universal Kernel Matrix (Cross-Block Feature Decomposition)",
        height=max(250, 80 * n_blocks + 100),
    )
    return fig


def plot_kernel_importance(snapshot: dict) -> go.Figure:
    """Bar chart of kernel importance (normalized singular values)."""
    importance = snapshot["importance"]
    n = len(importance)
    fig = go.Figure(go.Bar(
        x=[f"K{i}" for i in range(n)],
        y=importance,
        marker_color="#64ffda",
        text=[f"{v:.1%}" for v in importance],
        textposition="auto",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Kernel Importance (Normalized Singular Values)",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Explained Variance",
    )
    return fig


def plot_reality_regression(snapshot: dict) -> go.Figure:
    """Visualize the universal reality regression vector."""
    rr = snapshot["reality_regression"]
    n = len(rr)
    # Color by feature region
    colors = []
    for i in range(n):
        if i < 16:
            colors.append("#3498db")  # temporal
        elif i < 32:
            colors.append("#e67e22")  # semantic
        elif i < 48:
            colors.append("#2ecc71")  # structural
        elif i < 64:
            colors.append("#e74c3c")  # dynamic
        else:
            colors.append("#9b59b6")  # geospatial

    fig = go.Figure(go.Bar(
        x=list(range(n)), y=rr,
        marker_color=colors,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Universal Reality Regression (Feature-Space Basis)",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Feature Index",
        yaxis_title="Weight",
    )
    # Add region annotations (use np.max + np.abs to avoid ambiguous array truth value)
    annotation_y = float(np.max(np.abs(rr))) * 1.1
    fig.add_annotation(x=8,  y=annotation_y, text="Temporal",
                       showarrow=False, font=dict(color="#3498db", size=10))
    fig.add_annotation(x=24, y=annotation_y, text="Semantic",
                       showarrow=False, font=dict(color="#e67e22", size=10))
    fig.add_annotation(x=40, y=annotation_y, text="Structural",
                       showarrow=False, font=dict(color="#2ecc71", size=10))
    fig.add_annotation(x=56, y=annotation_y, text="Dynamic",
                       showarrow=False, font=dict(color="#e74c3c", size=10))
    fig.add_annotation(x=72, y=annotation_y, text="Geospatial",
                       showarrow=False, font=dict(color="#9b59b6", size=10))
    return fig


def plot_kernel_evolution(snapshots: list[dict]) -> go.Figure:
    """Show how kernel importance evolves step-by-step."""
    if not snapshots:
        return go.Figure()

    max_kernels = max(s["n_kernels"] for s in snapshots)
    fig = go.Figure()

    for k in range(max_kernels):
        x_vals, y_vals = [], []
        for s in snapshots:
            if k < s["n_kernels"]:
                x_vals.append(f"Step {s['step']}: {s['block_name']}")
                y_vals.append(s["importance"][k])
        if x_vals:
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="lines+markers",
                name=f"K{k}", line=dict(width=2),
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Kernel Importance Evolution Across Pipeline Steps",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Importance",
        xaxis_title="Pipeline Step",
    )
    return fig


def plot_concept_kernel_map(concept_kernel_rows: list[dict]) -> go.Figure:
    """Scatter plot of concepts mapped to kernels."""
    if not concept_kernel_rows:
        return go.Figure()

    active = [r for r in concept_kernel_rows if r.get("active")]
    inactive = [r for r in concept_kernel_rows if not r.get("active")]

    fig = go.Figure()
    if active:
        fig.add_trace(go.Scatter(
            x=[r["coherence"] for r in active],
            y=[r["mean_activation"] for r in active],
            mode="markers+text",
            marker=dict(size=12, color="#64ffda", symbol="diamond"),
            text=[r["concept"] for r in active],
            textposition="top center",
            name="Active Concepts",
        ))
    if inactive:
        fig.add_trace(go.Scatter(
            x=[r["coherence"] for r in inactive],
            y=[r["mean_activation"] for r in inactive],
            mode="markers",
            marker=dict(size=8, color="#555555", symbol="circle"),
            name="Inactive Concepts",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Concept-Kernel Coherence Map",
        height=350,
        xaxis_title="Coherence (concept-kernel alignment)",
        yaxis_title="Mean Activation",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
