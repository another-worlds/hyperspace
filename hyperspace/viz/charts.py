"""Reusable Plotly chart builders with dark theme defaults."""
from __future__ import annotations

import plotly.graph_objects as go

from hyperspace.config import PLOTLY_LAYOUT


def dark_layout(**kwargs) -> dict:
    """Merge PLOTLY_LAYOUT defaults with overrides."""
    merged = {**PLOTLY_LAYOUT}
    merged.update(kwargs)
    return merged


def candlestick_chart(df, ticker: str) -> go.Figure:
    """Create a candlestick chart for a single ticker."""
    tdf = df[df.Ticker == ticker] if "Ticker" in df.columns else df
    fig = go.Figure(data=[go.Candlestick(
        x=tdf.Date, open=tdf.Open, high=tdf.High, low=tdf.Low, close=tdf.Close,
        increasing_line_color="#64ffda", decreasing_line_color="#f87171",
        increasing_fillcolor="rgba(100,255,218,0.25)",
        decreasing_fillcolor="rgba(248,113,113,0.25)",
        hoverinfo="x+y",
    )])
    fig.update_layout(**dark_layout(
        title=f"{ticker} — Price History",
        height=360,
        xaxis_rangeslider_visible=False,
        xaxis_title=None,
        yaxis_title="Price (USD)",
    ))
    return fig


def forecast_chart(x_axis, q10, q50, q90, title: str = "Forecast") -> go.Figure:
    """Create a multi-quantile forecast chart with confidence interval."""
    fig = go.Figure()
    x_list = list(x_axis)
    q10_list = list(q10)
    q50_list = list(q50)
    q90_list = list(q90)
    # Confidence band first (drawn behind)
    fig.add_trace(go.Scatter(
        x=x_list + x_list[::-1],
        y=q90_list + q10_list[::-1],
        fill="toself", fillcolor="rgba(100,255,218,0.10)",
        line=dict(width=0), name="80% CI", showlegend=True,
        hovertemplate="Step: %{x}<br>Value: %{y:.2f}<extra>80% CI</extra>",
    ))
    # Median forecast line
    fig.add_trace(go.Scatter(
        x=x_list, y=q50_list, mode="lines",
        name="Median forecast",
        line=dict(color="#64ffda", width=2.5),
        hovertemplate="Step: %{x}<br>Forecast: %{y:.2f}<extra>Median</extra>",
    ))
    fig.update_layout(**dark_layout(
        title=title, height=320,
        margin=dict(l=24, r=24, t=44, b=24),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                    font=dict(size=11)),
    ))
    return fig


def source_badge(label: str) -> str:
    """Return HTML for a data source badge."""
    css_class = "source-badge" if "Live" in label or "Offline" in label else "source-badge fallback"
    return f'<span class="{css_class}">{label}</span>'
