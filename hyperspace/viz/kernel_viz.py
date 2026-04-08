"""Kernel matrix visualization: heatmaps, importance bars, evolution plots."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hyperspace.config import FEATURE_NAMES, PLOTLY_LAYOUT


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
    fig.update_traces(
        hovertemplate="Block: %{y}<br>Kernel: %{x}<br>Activation: %{z:.3f}<extra></extra>",
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
        hovertemplate="<b>%{x}</b><br>Variance Explained: %{y:.3f} (%{text})<extra></extra>",
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

    # Build block coloring from registry in snapshot or use defaults
    default_regions = {
        "temporal-pattern": (0, 16, "#3498db"),
        "semantic-embedding": (16, 24, "#e67e22"), 
        "tft-extended": (24, 27, "#2ecc71"),
        "macro": (27, 32, "#e74c3c"),
        "structural-centrality": (32, 48, "#9b59b6"),
        "dynamic-agent": (48, 64, "#1abc9c"),
        "geospatial-kernel": (64, 80, "#f39c12")
    }
    
    _BLOCK_PALETTE = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6",
                      "#1abc9c", "#f39c12", "#8e44ad", "#d35400", "#27ae60"]
    
    if 'registry' in snapshot and snapshot['registry'] is not None:
        registry = snapshot['registry']
        # Build mapping from actual registry
        block_map = []
        for i, (name, region) in enumerate(registry.regions.items()):
            color = _BLOCK_PALETTE[i % len(_BLOCK_PALETTE)]
            block_map.append((region.end, color, name))
        block_map.sort(key=lambda x: x[0])  # Sort by end index
    else:
        # Use default regions
        block_map = [
            (end, color, name) 
            for name, (start, end, color) in default_regions.items()
        ]
        block_map.sort(key=lambda x: x[0])

    colors = []
    blocks = []
    for i in range(n):
        for bound, color, label in block_map:
            if i < bound:
                colors.append(color)
                blocks.append(label)
                break
        else:
            # If no region found, use default
            colors.append("#95a5a6")
            blocks.append("unknown")

    feature_labels = FEATURE_NAMES[:n] if n <= len(FEATURE_NAMES) else [
        FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
        for i in range(n)
    ]

    fig = go.Figure(go.Bar(
        x=feature_labels, y=rr,
        marker_color=colors,
        customdata=list(zip(range(n), blocks)),
        hovertemplate=(
            "<b>%{x}</b> (idx %{customdata[0]})<br>"
            "Source Block: %{customdata[1]}<br>"
            "Weight: %{y:.4f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Universal Reality Regression (Feature-Space Basis)",
        height=280,
        margin=dict(l=20, r=20, t=40, b=60),
        xaxis_title="Feature",
        yaxis_title="Weight",
        xaxis_tickangle=-45,
    )
    # Add block annotations dynamically
    annotation_y = float(np.max(np.abs(rr))) * 1.1
    for i, (bname, (start, end)) in enumerate(_sorted_blocks):
        mid = (start + end) / 2
        color = _BLOCK_PALETTE[i % len(_BLOCK_PALETTE)]
        fig.add_annotation(x=mid, y=annotation_y, text=bname,
                           showarrow=False, font=dict(color=color, size=10))
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
                hovertemplate="<b>K%{k}</b><br>%{x}<br>Importance: %{y:.3f}<extra></extra>"
                .replace("%{k}", str(k)),
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
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Coherence: %{x:.3f}<br>"
                "Mean Activation: %{y:.3f}<extra>Active</extra>"
            ),
        ))
    if inactive:
        fig.add_trace(go.Scatter(
            x=[r["coherence"] for r in inactive],
            y=[r["mean_activation"] for r in inactive],
            mode="markers",
            marker=dict(size=8, color="#555555", symbol="circle"),
            name="Inactive Concepts",
            customdata=[r["concept"] for r in inactive],
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Coherence: %{x:.3f}<br>"
                "Mean Activation: %{y:.3f}<extra>Inactive</extra>"
            ),
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
