"""Mission Control tab (Tab 0): system overview, live metrics, pipeline status.

Provides a single-page summary of the Hyperspace system state: data source
health, UKT kernel status, governance scorecard, and semantic canvas snapshot.
Designed as the first tab a delegate or analyst sees after the pipeline runs.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

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
    st.markdown("### System Status")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pipeline Blocks", str(len(snapshots)),
              help="Number of data domains successfully processed")

    data_live = sum(
        1 for v in data_sources.values()
        if "Live" in v or ("Offline" in v and "Synthetic" not in v)
    )
    m2.metric("Live Data Sources", f"{data_live}/{len(data_sources)}",
              help="Data blocks drawing from real sources vs. synthetic fallbacks")

    if sae_result:
        m3.metric(
            "Interpretable Concepts",
            f"{sae_result['active_concepts']}/{sae_result['total_concepts']}",
            help="Named concepts discovered by the Sparse Autoencoder — higher = more interpretable",
        )
    else:
        m3.metric("Interpretable Concepts", "—")

    pass_count = sum(1 for v in scorecard.values() if v.get("passed"))
    total_sc = len(scorecard)
    m4.metric(
        "Accountability Score",
        f"{pass_count}/{total_sc} PASS" if total_sc else "—",
        help="Number of interpretability criteria passing the minimum governance threshold",
    )

    st.caption(
        "v3.0 — These vitals reflect the current governance health of the system. "
        "All four dimensions are traced to specific data sources and are included "
        "in the exported governance report."
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


