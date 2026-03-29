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

from hyperspace.config import (
    AVAILABLE_COUNTRIES,
    COUNTRY_DISPLAY,
    COUNTRY_TICKER_MAP,
    DARK_CSS,
    DEFAULT_COUNTRIES,
    GLOSSARY,
    countries_to_tickers,
)
from hyperspace.state import init_session_state
from hyperspace.pages import governance as governance_module
from hyperspace.viz.sidebar_kanban import render_sidebar_kanban

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
    st.markdown(
        '<div style="padding: 4px 0 14px;">'
        '<div style="font-family:Inter,sans-serif; font-size:1.25em; font-weight:700; '
        'color:#dce8f0; letter-spacing:-0.01em; line-height:1.2;">Hyperspace</div>'
        '<div style="font-family:Inter,sans-serif; font-size:0.68em; font-weight:600; '
        'color:#8ab4cc; letter-spacing:0.12em; text-transform:uppercase; margin-top:4px;">'
        'AI Accountability Infrastructure</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="concept-badge">UKT: Online</span> '
        '<span class="concept-badge">Governance: Active</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Analysis Scope ──────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section-header">Analysis Scope</div>',
        unsafe_allow_html=True,
    )

    # Country selection (replaces ticker selection)
    selected_countries = st.multiselect(
        "Countries",
        AVAILABLE_COUNTRIES,
        default=st.session_state.get("selected_countries", DEFAULT_COUNTRIES),
        key="selected_countries",
        label_visibility="collapsed",
    )

    # Render country chips with metadata
    if selected_countries:
        chips_html = []
        for country in selected_countries:
            info = COUNTRY_DISPLAY.get(country, {})
            flag = info.get("flag", "")
            etf = info.get("etf", "")
            bloc = info.get("bloc", "")
            chips_html.append(
                f'<div class="country-chip selected">'
                f'<span class="chip-flag">{flag}</span>'
                f'<span class="chip-name">{country}</span>'
                f'<span class="chip-bloc">{bloc}</span>'
                f'<span class="chip-etf">{etf}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div class="country-grid">{"".join(chips_html)}</div>',
            unsafe_allow_html=True,
        )

    # Sync tickers from country selection for downstream consumers
    st.session_state["tickers"] = countries_to_tickers(selected_countries)

    # B1: Policy Language Mode toggle
    st.toggle(
        "Governance Language Mode",
        value=st.session_state.get("policy_language_mode", False),
        key="policy_language_mode",
        help=(
            "Switch kernel narratives from technical notation (loading values, "
            "feature indices) to plain-English policy briefing language. "
            "Designed for legal, diplomatic, and civil society delegates."
        ),
    )

    st.markdown("---")

    # ── Pipeline Status ──────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section-header">Pipeline Status</div>',
        unsafe_allow_html=True,
    )
    render_sidebar_kanban()

    st.markdown("---")

    # ── Governance & Provenance ──────────────────────────────────────
    data_sources = st.session_state.get("data_sources", {})
    snapshots = st.session_state.get("ukt_snapshots", [])

    if data_sources or snapshots:
        st.markdown(
            '<div class="sidebar-section-header">Governance & Provenance</div>',
            unsafe_allow_html=True,
        )

        # C2: Jurisdiction badges (post-pipeline)
        if data_sources:
            governance_module.render_jurisdiction_badges(data_sources)

        # A2: Provenance Trace Panel (post-pipeline)
        if snapshots:
            st.markdown("**Feature Provenance Trace**")
            governance_module.render_provenance_panel(snapshots)

        st.markdown("---")

    # D2: Glossary
    with st.expander("Glossary", expanded=False):
        st.markdown(
            "Plain-English definitions for all technical terms used in this system. "
            "Intended for legal, policy, and diplomatic delegates."
        )
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}**")
            st.caption(definition)
            st.markdown("")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; padding: 4px 0;">'
        '<div style="font-size:0.72rem; font-weight:600; color:#8ab4cc; '
        'letter-spacing:0.05em;">v3.0 Architecture</div>'
        '<div style="font-size:0.65rem; color:#64748b; margin-top:2px;">'
        'Interpretability · Traceability · Contestability</div>'
        '<div style="font-size:0.60rem; color:#4a5568; margin-top:4px;">'
        'UN Global Dialogue on AI Governance · Feb 2026</div>'
        '</div>',
        unsafe_allow_html=True,
    )

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
        mission_control_tab,
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

    # Tabs — spec-aligned: 0-6 per CLAUDE.md + Counterfactual (contestability)
    tabs = st.tabs([
        "Mission Control",
        "Finance-Neural Block",
        "Informational Cluster Mapping",
        "Politics-Military Block",
        "Agentic Simulation",
        "Semantic Interpreter",
        "Hyperspace Pipeline",
        "⚖ Counterfactual",
    ])

    tab_modules = [
        (tabs[0], "Mission Control", mission_control_tab),
        (tabs[1], "Finance-Neural Block", finance_tab),
        (tabs[2], "Informational Cluster Mapping", clusters_tab),
        (tabs[3], "Politics-Military Block", politics_tab),
        (tabs[4], "Agentic Simulation", agents_tab),
        (tabs[5], "Semantic Interpreter", interpreter_tab),
        (tabs[6], "Hyperspace Pipeline", pipeline_tab),
        (tabs[7], "Counterfactual", counterfactual_tab),
    ]
    for tab, name, module in tab_modules:
        with tab:
            try:
                module.render()
            except Exception as exc:
                st.error(f"{name} tab encountered an error: {exc}")

    # Reset button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Reset Pipeline", use_container_width=True, key="reset_btn"):
            from hyperspace.state import reset_pipeline
            reset_pipeline()
            st.rerun()
