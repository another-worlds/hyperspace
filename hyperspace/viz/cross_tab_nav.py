"""Cross-tab navigation hints for contextual linking between tabs.

Renders styled pill badges at the top of each tab, pointing users
to related tabs with short context descriptions. Since Streamlit
does not support programmatic tab switching, these are informational
cues — users still click the main tab bar to navigate.
"""
from __future__ import annotations

import streamlit as st

# Mapping: tab name → list of (related_tab_name, short_context)
TAB_RELATIONSHIPS: dict[str, list[tuple[str, str]]] = {
    "Mission Control": [
        ("Finance-Neural Block", "Market forecasts"),
        ("Informational Cluster Mapping", "Topic clusters"),
        ("Politics-Military Block", "Graph centrality"),
        ("Agentic Simulation", "Resource dynamics"),
        ("Semantic Interpreter", "Kernel narratives"),
    ],
    "Finance-Neural Block": [
        ("Semantic Interpreter", "Kernel interpretation"),
        ("\u2696 Counterfactual", "What-if: remove finance block"),
    ],
    "Informational Cluster Mapping": [
        ("Semantic Interpreter", "Topic kernel interpretation"),
        ("\u2696 Counterfactual", "What-if: remove info block"),
    ],
    "Politics-Military Block": [
        ("Semantic Interpreter", "Centrality kernels"),
        ("\u2696 Counterfactual", "What-if: remove politics block"),
    ],
    "Agentic Simulation": [
        ("Semantic Interpreter", "Resource kernels"),
        ("\u2696 Counterfactual", "What-if: remove agents block"),
    ],
    "Semantic Interpreter": [
        ("Mission Control", "Governance overview"),
        ("Finance-Neural Block", "Finance provenance"),
        ("Politics-Military Block", "Graph provenance"),
        ("Agentic Simulation", "Agent provenance"),
    ],
    "Hyperspace Pipeline": [
        ("Mission Control", "Governance scorecard"),
        ("\u2696 Counterfactual", "Contestability analysis"),
    ],
    "\u2696 Counterfactual": [
        ("Mission Control", "Compare with baseline"),
        ("Semantic Interpreter", "Kernel diff interpretation"),
    ],
}


def render_related_tabs(current_tab: str) -> None:
    """Render a row of navigation pill badges for the current tab.

    Args:
        current_tab: The name of the currently active tab (must match
            a key in TAB_RELATIONSHIPS).
    """
    related = TAB_RELATIONSHIPS.get(current_tab, [])
    if not related:
        return

    pills = []
    for tab_name, context in related:
        pills.append(
            f'<span class="nav-pill">'
            f'<span class="nav-pill-tab">{tab_name}</span>'
            f' &mdash; {context}'
            f'</span>'
        )

    html = f'<div class="nav-row">{"".join(pills)}</div>'
    st.markdown(html, unsafe_allow_html=True)
