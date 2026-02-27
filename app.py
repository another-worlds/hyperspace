"""
Hyperspace -- Predictive Polymath System v3.0
=============================================
Entry point for the Streamlit dashboard.

Architecture:
  - Governance-framed landing page with launch button and UKT visualization
  - 7 tabbed blocks: Finance, Clusters, Politics, Agents, Interpreter, Pipeline, Counterfactual
  - Universal Knowledge Tensor (UKT) updated at every step with SVD decomposition
  - Semantic interpretability at each pipeline stage
  - Governance features: flags, score card, provenance tracing, annotations, jurisdiction labels

Run:  pip install -r requirements.txt && streamlit run app.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from hyperspace.config import DARK_CSS, DEFAULT_TICKERS, GLOSSARY
from hyperspace.state import init_session_state
from hyperspace.pages import governance as governance_module

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hyperspace -- AI Governance Demo",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────
init_session_state()

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# Hyperspace")
    st.markdown("### AI Accountability Infrastructure")
    st.markdown(
        '<span class="concept-badge">UKT: Online</span> '
        '<span class="concept-badge">Governance: Active</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # B1: Policy Language Mode toggle
    policy_mode = st.toggle(
        "🗂️ Governance Language Mode",
        value=st.session_state.get("policy_language_mode", False),
        key="policy_language_mode",
        help=(
            "Switch kernel narratives from technical notation (loading values, "
            "feature indices) to plain-English policy briefing language. "
            "Designed for legal, diplomatic, and civil society delegates."
        ),
    )

    st.markdown("---")

    # Ticker selection (shared across tabs)
    tickers = st.multiselect(
        "Finance Tickers (Country ETFs)",
        ["SPY", "EWZ", "INDA", "FXI", "EWU", "ERUS", "RSX"],
        default=DEFAULT_TICKERS,
        key="tickers",
    )

    st.markdown("---")

    # D1: Run ID display (post-pipeline)
    run_id = st.session_state.get("run_id")
    run_ts = st.session_state.get("run_timestamp")
    if run_id:
        st.markdown("**Run Identifier**")
        st.markdown(
            f'<span class="run-id-watermark">ID: {run_id}</span>  \n'
            f'<span class="run-id-watermark">{run_ts}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

    # Data source status + C2 jurisdiction badges (post-pipeline)
    data_sources = st.session_state.get("data_sources", {})
    if data_sources:
        st.markdown("**Data Sources**")
        for block, src in data_sources.items():
            icon = "+" if "Live" in src or "Offline" in src else "!"
            st.caption(f"[{icon}] {block}: {src}")

        # C2: Jurisdiction badges
        governance_module.render_jurisdiction_badges(data_sources)

        # Governance flags summary in sidebar
        gov_flags = st.session_state.get("governance_flags", [])
        if gov_flags:
            st.markdown(
                f'<span class="gov-flag">⚠ {len(gov_flags)} governance flag(s)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="gov-pass">✓ No flags</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

    # A2: Provenance Trace Panel (post-pipeline)
    snapshots = st.session_state.get("ukt_snapshots", [])
    if snapshots:
        st.markdown("**🔍 Feature Provenance Trace**")
        governance_module.render_provenance_panel(snapshots)
        st.markdown("---")

    # D2: Glossary
    with st.expander("📖 Glossary", expanded=False):
        st.markdown(
            "Plain-English definitions for all technical terms used in this system. "
            "Intended for legal, policy, and diplomatic delegates."
        )
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}**")
            st.caption(definition)
            st.markdown("")

    st.markdown("---")
    st.markdown("**v3.0 Architecture**")
    st.markdown(
        "Interpretability · Traceability · Contestability as first-class requirements"
    )
    st.caption("UN Global Dialogue on AI Governance · February 2026")

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
        counterfactual_tab,
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
        "⚖ Counterfactual",
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
    with tabs[6]:
        counterfactual_tab.render()

    # Reset button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Reset Pipeline", use_container_width=True, key="reset_btn"):
            from hyperspace.state import reset_pipeline
            reset_pipeline()
            st.rerun()
