"""
Hyperspace -- Predictive Polymath System v3.0
=============================================
Entry point for the Streamlit dashboard.

Architecture:
  - Dashboard landing page with launch button and UKT visualization
  - 6 tabbed blocks: Finance, Clusters, Politics, Agents, Interpreter, Pipeline
  - Universal Knowledge Tensor (UKT) updated at every step with SVD decomposition
  - Semantic interpretability at each pipeline stage

Run:  pip install -r requirements.txt && streamlit run app.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from hyperspace.config import DARK_CSS, DEFAULT_TICKERS
from hyperspace.state import init_session_state

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hyperspace -- PPS v3.0",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────
init_session_state()

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# Hyperspace")
    st.markdown("### Predictive Polymath System v3.0")
    st.markdown("---")
    st.markdown(
        '<span class="concept-badge">UKT: Online</span> '
        '<span class="concept-badge">Semantic Interpreter: Ready</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Ticker selection (shared across tabs)
    tickers = st.multiselect(
        "Finance Tickers",
        ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN"],
        default=DEFAULT_TICKERS,
        key="tickers",
    )

    st.markdown("---")
    st.markdown("**v3.0 Architecture**")
    st.markdown(
        "Hierarchical JEPA semantics -- Realist geopolitical constraints -- "
        "Lifelong kernel reuse -- Multi-horizon calibration"
    )
    st.caption("Hyperspace Prototype -- February 2026")

    # Show data source status if pipeline has run
    data_sources = st.session_state.get("data_sources", {})
    if data_sources:
        st.markdown("---")
        st.markdown("**Data Sources**")
        for block, src in data_sources.items():
            icon = "+" if "Live" in src or "Offline" in src else "!"
            st.caption(f"[{icon}] {block}: {src}")

# ── Main area: Dashboard or Tabs ─────────────────────────────────────
if not st.session_state.pipeline_launched:
    # Landing page
    from hyperspace.pages import dashboard
    dashboard.render_landing()

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        launch = st.button(
            "Launch Full Hyperspace Cycle",
            type="primary",
            use_container_width=True,
            key="launch_btn",
        )
    if launch:
        st.session_state.pipeline_launched = True
        st.rerun()

else:
    # Pipeline has been launched
    from hyperspace.pages import (
        dashboard,
        finance_tab,
        clusters_tab,
        politics_tab,
        agents_tab,
        interpreter_tab,
        pipeline_tab,
    )

    # Run pipeline if not yet complete
    if not st.session_state.pipeline_complete:
        dashboard.run_pipeline()

    # Show results header
    dashboard.render_results()

    st.markdown("---")

    # Tabs for individual block exploration
    tabs = st.tabs([
        "Finance-Neural",
        "Info Clusters",
        "Politics-Military",
        "Agentic Sim",
        "Semantic Interpreter",
        "Full Pipeline",
    ])

    with tabs[0]:
        finance_tab.render()
    with tabs[1]:
        clusters_tab.render()
    with tabs[2]:
        politics_tab.render()
    with tabs[3]:
        agents_tab.render()
    with tabs[4]:
        interpreter_tab.render()
    with tabs[5]:
        pipeline_tab.render()

    # Reset button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Reset Pipeline", use_container_width=True, key="reset_btn"):
            from hyperspace.state import reset_pipeline
            reset_pipeline()
            st.rerun()
