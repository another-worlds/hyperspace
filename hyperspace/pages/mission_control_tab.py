"""Mission Control tab (Tab 0): system overview, live metrics, pipeline status.

Provides a single-page summary of the Hyperspace system state: data source
health, UKT kernel status, governance scorecard, and semantic canvas snapshot.
Designed as the first tab a delegate or analyst sees after the pipeline runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.config import (
    PLOTLY_LAYOUT, UKT_FEATURE_DIM, GEOPOLITICAL_NODES,
)
from hyperspace.viz import kernel_viz
from hyperspace.viz.charts import source_badge


def render() -> None:
    """Render the Mission Control tab: system overview, metrics, status."""
    st.markdown("## Mission Control")
    st.markdown(
        "System overview and governance health dashboard. All metrics update "
        "after each pipeline run."
    )

    snapshots = st.session_state.get("ukt_snapshots", [])
    data_sources = st.session_state.get("data_sources", {})
    sae_result = st.session_state.get("sae_result")
    stability = st.session_state.get("ukt_multirun_stability")
    gov_flags = st.session_state.get("governance_flags", [])
    scorecard = st.session_state.get("interpretability_scorecard", {})

    if not snapshots:
        st.info(
            "No pipeline results yet. Launch the pipeline from the dashboard "
            "to populate Mission Control."
        )
        return

    final_snap = snapshots[-1]

    # ── Run identity ──────────────────────────────────────────────────
    run_id = st.session_state.get("run_id", "—")
    run_ts = st.session_state.get("run_timestamp", "")
    st.caption(f"Run ID: {run_id}  ·  {run_ts}")

    # ── Key metrics row ───────────────────────────────────────────────
    st.markdown("### System Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Active Kernels", str(final_snap["n_kernels"]))

    finance_result = st.session_state.get("finance_result", {})
    tft_params = (
        f"{finance_result['model_params']:,}"
        if isinstance(finance_result, dict) and "model_params" in finance_result
        else "—"
    )
    m2.metric("TFT Parameters", tft_params)

    graph_result = st.session_state.get("graph_result", {})
    if isinstance(graph_result, dict) and "analysis" in graph_result:
        density = graph_result["analysis"].get("density", 0.0)
        m3.metric("Graph Density", f"{density:.3f}")
    else:
        m3.metric("Graph Density", "—")

    if sae_result:
        m4.metric(
            "Active Concepts",
            f"{sae_result['active_concepts']}/{sae_result['total_concepts']}",
        )
    else:
        m4.metric("Active Concepts", "—")

    m5.metric("Recon Error", f"{final_snap['reconstruction_error']:.6f}")

    st.caption(
        "v3.0 — These metrics summarise the Universal Knowledge Tensor state. "
        "Each kernel is an SVD-derived latent direction that spans multiple data "
        "domains; reconstruction error measures how much information the current "
        "kernel set preserves."
    )

    # ── Stability summary ─────────────────────────────────────────────
    if isinstance(stability, dict) and stability.get("n_runs", 0) > 0:
        st.markdown("### Regression Stability")
        s1, s2, s3 = st.columns(3)
        s1.metric("Mean Cosine", f"{stability['mean_cosine']:.3f}")
        s2.metric("Min Cosine", f"{stability['min_cosine']:.3f}")
        s3.metric("Std", f"{stability['std_cosine']:.3f}")
        st.caption(
            "v3.0 — Reality regression stability is tested by adding small "
            "Gaussian noise to the UKT matrix and re-computing the regression "
            f"{stability['n_runs']} times. High cosine similarity (>0.9) means "
            "conclusions are robust to small data perturbations."
        )

    st.markdown("---")

    # ── Data source health ────────────────────────────────────────────
    st.markdown("### Data Source Status")
    src_cols = st.columns(len(data_sources) or 1)
    for i, (block, src) in enumerate(data_sources.items()):
        with src_cols[i % len(src_cols)]:
            is_live = "Live" in src or "Offline" in src
            status_icon = "+" if is_live else "!"
            st.markdown(f"**{block}**")
            st.markdown(source_badge(src), unsafe_allow_html=True)

    st.caption(
        "v3.0 — Full feature provenance requires tracing every UKT dimension "
        "back to its raw data source. This panel shows whether each pipeline "
        "block drew from live or fallback data, which is recorded in the "
        "governance flags below."
    )

    st.markdown("---")

    # ── Governance flags ──────────────────────────────────────────────
    st.markdown("### Governance Flags")
    if gov_flags:
        for flag in gov_flags:
            severity = flag.get("severity", "warning")
            icon = "!" if severity == "warning" else "i"
            st.warning(
                f"**[{flag.get('code', '?')}] {flag.get('label', '?')}** — "
                f"{flag.get('detail', flag.get('description', ''))}"
            )
    else:
        st.success("No governance flags detected.")

    st.caption(
        "v3.0 — Governance flags are auto-generated alerts for modality "
        "imbalance, temporal coverage gaps, centrality skew, low concept "
        "coverage, and synthetic data usage. They support the contestability "
        "guarantee by surfacing potential issues before conclusions are cited."
    )

    st.markdown("---")

    # ── Scorecard summary ─────────────────────────────────────────────
    if scorecard:
        st.markdown("### Interpretability Score Card")
        sc_rows = []
        for key, item in scorecard.items():
            passed = item.get("passed", False)
            sc_rows.append({
                "Dimension": item.get("label", key),
                "Value": f"{item.get('value', '—')}{item.get('unit', '')}",
                "Threshold": f"\u2265{item.get('threshold', '—')}{item.get('unit', '')}",
                "Status": "PASS" if passed else "WARN",
            })
        sc_df = pd.DataFrame(sc_rows)

        def _style_status(val: str) -> str:
            if "PASS" in val:
                return "color: #64ffda; font-weight: bold"
            return "color: #ffaa00; font-weight: bold"

        try:
            styled = sc_df.style.map(_style_status, subset=["Status"])
        except AttributeError:
            styled = sc_df.style.applymap(_style_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        pass_count = sum(1 for v in scorecard.values() if v.get("passed"))
        total = len(scorecard)
        if pass_count == total:
            st.success(f"All {total} criteria pass.")
        else:
            st.warning(f"{pass_count}/{total} criteria pass — review flagged items.")

        st.caption(
            "v3.0 — The score card is a formal pass/fail audit against minimum "
            "interpretability thresholds. It operationalises the traceability "
            "guarantee by quantifying feature coverage, kernel stability, concept "
            "activation, and data diversity."
        )

    st.markdown("---")

    # ── Kernel matrix snapshot ────────────────────────────────────────
    st.markdown("### Universal Kernel Matrix")
    block_names = [s["block_name"] for s in snapshots]
    fig_km = kernel_viz.plot_kernel_matrix(final_snap, block_names)
    st.plotly_chart(fig_km, use_container_width=True, key="mc_kernel_matrix")
    st.caption(
        "v3.0 — The kernel matrix shows how each pipeline block activates each "
        "SVD-derived kernel. Strong cross-block activation indicates patterns "
        "that span multiple data modalities — the core of the UKT's cross-domain "
        "synthesis capability."
    )

    # ── Semantic canvas snapshot ──────────────────────────────────────
    canvas = st.session_state.get("semantic_canvas")
    canvas_narrative = st.session_state.get("canvas_narrative")
    reality_narrative = st.session_state.get("reality_narrative")

    if canvas is not None:
        st.markdown("### Semantic Canvas Summary")
        canvas_state = canvas.get_accumulated_state()
        dominant = canvas_state.get("dominant_narrative", [])
        if dominant:
            for d in dominant:
                st.markdown(f"- **{d['label']}** ({d['value']:.2f}): {d['desc']}")

        if canvas_narrative:
            st.success(f"**Cross-Domain Narrative:** {canvas_narrative}")
        if reality_narrative:
            st.info(f"**Reality Assessment:** {reality_narrative}")

        st.caption(
            "v3.0 — The Semantic Canvas projects each block's SAE-discovered "
            "concepts onto 12 named interpretive dimensions. The dominant themes "
            "above are the system's best summary of what matters most across all "
            "data domains."
        )
