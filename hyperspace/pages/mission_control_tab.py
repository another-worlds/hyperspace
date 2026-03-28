"""Mission Control tab (Tab 0): system overview with governance-first organization.

Restructured for policy stakeholders: governance outputs first (flags, scorecard, compliance),
then data provenance, then technical diagnostics (collapsed by default).

This redesign implements the vision hierarchy:
1. Executive Summary (top)
2. Governance Status (prominent, governance-focused)
3. Data Provenance (for auditing)
4. Technical Diagnostics (collapsed, for experts)
5. Export options (sticky/prominent)
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hyperspace.config import PLOTLY_LAYOUT
from hyperspace.core.caching import get_or_compute_figure, hash_list
from hyperspace.viz.charts import source_badge
from hyperspace.viz.cross_tab_nav import render_related_tabs


def render() -> None:
    """Render Mission Control with governance-first hierarchy."""
    st.markdown("## Mission Control")
    render_related_tabs("Mission Control")
    st.markdown(
        "**System overview and accountability dashboard.** All metrics update after each pipeline run. "
        "This page is designed for policy officers, auditors, and governance stakeholders. "
        "[View technical diagnostics below](#technical-diagnostics)"
    )

    snapshots = st.session_state.get("ukt_snapshots", [])
    data_sources = st.session_state.get("data_sources", {})
    sae_result = st.session_state.get("sae_result")
    gov_flags = st.session_state.get("governance_flags", [])
    scorecard = st.session_state.get("interpretability_scorecard", {})
    contract_summary = st.session_state.get("interpretability_contract_summary", {})
    alignment_metrics = st.session_state.get("alignment_metrics", {})

    if not snapshots:
        st.info(
            "No pipeline results yet. Launch the pipeline from the dashboard "
            "to populate Mission Control."
        )
        return

    final_snap = snapshots[-1]
    run_id = st.session_state.get("run_id", "—")
    run_ts = st.session_state.get("run_timestamp", "")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1: EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    with st.container(border=True):
        st.markdown("### 🎯 Executive Summary")
        summary_cols = st.columns([2, 3, 1])

        with summary_cols[0]:
            st.metric("Run ID", run_id, label_visibility="collapsed")
            st.caption(f"Timestamp: {run_ts}")

        with summary_cols[1]:
            pass_count = sum(1 for v in scorecard.values() if v.get("passed"))
            total_sc = len(scorecard)
            status_text = f"{pass_count}/{total_sc} Governance Criteria PASS"
            if pass_count == total_sc and total_sc > 0:
                st.success(f"✓ {status_text}")
            elif pass_count >= total_sc * 0.75:
                st.warning(f"⚠ {status_text}")
            else:
                st.error(f"✗ {status_text}")

            flag_status = "No governance flags detected" if not gov_flags else f"{len(gov_flags)} flag(s) raised"
            st.caption(f"Governance Flags: {flag_status}")

        with summary_cols[2]:
            data_live = sum(
                1 for v in data_sources.values()
                if "Live" in v or ("Offline" in v and "Synthetic" not in v)
            )
            st.metric("Data Sources", f"{data_live}/{len(data_sources)}", label_visibility="collapsed")
            st.caption(f"Live: {data_live}; Synthetic: {len(data_sources) - data_live}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2: GOVERNANCE STATUS (Prominent, Expanded by Default)
    # ═══════════════════════════════════════════════════════════════════════

    with st.expander("🔒 **Governance Status** (Accountability Scorecard)", expanded=True):
        gov_cols = st.columns([1, 3])

        with gov_cols[0]:
            st.markdown("**Governance Flags:**")
            if gov_flags:
                for flag in gov_flags:
                    severity = flag.get("severity", "warning")
                    code = flag.get("code", "?")
                    label = flag.get("label", "?")
                    detail = flag.get("detail", flag.get("description", ""))

                    if severity == "warning":
                        st.warning(f"**{code}** — {label}\n\n{detail}")
                    else:
                        st.info(f"**{code}** — {label}\n\n{detail}")
            else:
                st.success("✓ No governance flags detected.")

        with gov_cols[1]:
            st.markdown("**Accountability Score Card:**")
            if scorecard:
                sc_rows = []
                for key, item in scorecard.items():
                    passed = item.get("passed", False)
                    sc_rows.append({
                        "Criterion": item.get("label", key),
                        "Current": f"{item.get('value', '—')}{item.get('unit', '')}",
                        "Threshold": f"\u2265{item.get('threshold', '—')}{item.get('unit', '')}",
                        "Status": "✓ PASS" if passed else "✗ WARN",
                    })
                sc_df = pd.DataFrame(sc_rows)

                def _style_status(val: str) -> str:
                    if "PASS" in val:
                        return "color: #34d399; font-weight: bold"
                    return "color: #ff6b6b; font-weight: bold"

                styled_df = sc_df.style.applymap(_style_status, subset=["Status"])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # Contract compliance
                st.markdown("**Interpretability Contract:**")
                contract_rate = float(contract_summary.get("compliance_rate", 0.0))
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.metric("Compliance Rate", f"{contract_rate:.0%}", label_visibility="collapsed")
                with col2:
                    st.progress(contract_rate, text=f"{contract_rate:.0%} modules pass interface checks")
                with col3:
                    if contract_rate >= 0.9:
                        st.success("Strong")
                    elif contract_rate >= 0.7:
                        st.warning("Moderate")
                    else:
                        st.error("Weak")

            else:
                st.info("Run the pipeline to generate scorecard.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3: DATA PROVENANCE (For Auditing)
    # ═══════════════════════════════════════════════════════════════════════

    with st.expander("📊 **Data Provenance** (Source Integrity & Coverage)", expanded=True):
        st.markdown("**Pipeline Block Status:**")
        st.metric(
            "Pipeline Blocks Processed",
            str(len(snapshots)),
            help="Number of data domains successfully integrated"
        )

        st.markdown("**Data Source Status:**")
        src_cols = st.columns(min(4, len(data_sources) or 1))
        for i, (block, src) in enumerate(data_sources.items()):
            with src_cols[i % len(src_cols)]:
                st.markdown(f"**{block}**")
                st.markdown(source_badge(src), unsafe_allow_html=True)

        st.markdown("**Key Metrics:**")
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric(
                "Live Data Sources",
                f"{sum(1 for v in data_sources.values() if 'Live' in v or ('Offline' in v and 'Synthetic' not in v))}/{len(data_sources)}",
                help="Blocks drawing from real sources vs. synthetic fallbacks"
            )
        with metric_cols[1]:
            if sae_result:
                st.metric(
                    "Interpretable Concepts",
                    f"{sae_result['active_concepts']}/{sae_result['total_concepts']}",
                    help="Named concepts discovered by Sparse Autoencoder"
                )
            else:
                st.metric("Interpretable Concepts", "—")
        with metric_cols[2]:
            st.metric(
                "UKT Dimensions",
                "80",
                help="Features in the shared Universal Knowledge Tensor"
            )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4: TECHNICAL DIAGNOSTICS (Collapsed, For Experts)
    # ═══════════════════════════════════════════════════════════════════════

    with st.expander("🔬 **Technical Diagnostics** (Expert View)", expanded=False):
        diag_tabs = st.tabs([
            "Kernel Evolution",
            "Reconstruction",
            "Stability",
            "Drift",
            "Alignment",
        ])

        with diag_tabs[0]:
            kernel_evolution = st.session_state.get("kernel_evolution", {})
            if kernel_evolution:
                evo_cols = st.columns(2)
                with evo_cols[0]:
                    st.markdown("**Kernel Stability Over Blocks:**")
                    stability_info = kernel_evolution.get("stability_trend", [])
                    if stability_info:
                        steps = list(range(1, len(stability_info) + 1))
                        mean_vals = [s.get("mean", 0) for s in stability_info]
                        min_vals = [s.get("min", 0) for s in stability_info]
                        _stab_hash = hash_list(mean_vals + min_vals, "mc_stab")

                        def _make_stab_fig():
                            fig_stab = go.Figure()
                            fig_stab.add_trace(go.Scatter(
                                x=steps, y=mean_vals, mode="lines+markers",
                                name="Mean Cosine", line=dict(color="#64ffda", width=2),
                                hovertemplate="Step %{x}<br>Mean Cosine: %{y:.3f}<extra></extra>",
                            ))
                            fig_stab.add_trace(go.Scatter(
                                x=steps, y=min_vals, mode="lines+markers",
                                name="Min Cosine", line=dict(color="#ff6b6b", width=2),
                                hovertemplate="Step %{x}<br>Min Cosine: %{y:.3f}<extra></extra>",
                            ))
                            fig_stab.update_layout(
                                **PLOTLY_LAYOUT, height=280,
                                title="Kernel Stability Over Blocks",
                                xaxis_title="Block Step", yaxis_title="Cosine Similarity",
                                margin=dict(l=20, r=20, t=40, b=20),
                            )
                            return fig_stab

                        fig_stab = get_or_compute_figure(
                            f"mc_stab_{_stab_hash}", _make_stab_fig,
                        )
                        st.plotly_chart(fig_stab, use_container_width=True,
                                        key="mc_kernel_stability")
                with evo_cols[1]:
                    st.markdown("**Kernel Count and Importance:**")
                    kernel_counts = kernel_evolution.get("kernel_counts", [])
                    if kernel_counts:
                        steps = list(range(1, len(kernel_counts) + 1))
                        _kc_hash = hash_list(kernel_counts, "mc_kc")

                        def _make_kc_fig():
                            fig_kc = go.Figure(go.Bar(
                                x=steps, y=kernel_counts,
                                marker_color="#64ffda",
                                text=[str(k) for k in kernel_counts],
                                textposition="auto",
                                hovertemplate="Step %{x}<br>Kernels: %{y}<extra></extra>",
                            ))
                            fig_kc.update_layout(
                                **PLOTLY_LAYOUT, height=280,
                                title="Kernel Count per Block",
                                xaxis_title="Block Step", yaxis_title="Kernel Count",
                                margin=dict(l=20, r=20, t=40, b=20),
                            )
                            return fig_kc

                        fig_kc = get_or_compute_figure(
                            f"mc_kc_{_kc_hash}", _make_kc_fig,
                        )
                        st.plotly_chart(fig_kc, use_container_width=True,
                                        key="mc_kernel_counts")
            else:
                st.info("Kernel evolution data not available.")

        with diag_tabs[1]:
            if final_snap:
                recon_error = final_snap.get("reconstruction_error", 0)
                st.metric("SVD Reconstruction Error", f"{recon_error:.6f}", help="Lower = better kernel fit")
            else:
                st.info("No reconstruction data available yet.")

        with diag_tabs[2]:
            stability = st.session_state.get("ukt_multirun_stability", {})
            if stability:
                stab_cols = st.columns(3)
                with stab_cols[0]:
                    st.metric(
                        "Mean Cosine Similarity",
                        f"{stability.get('mean_cosine', 0):.3f}",
                        help="Reality regression stability across 8 noise runs"
                    )
                with stab_cols[1]:
                    st.metric(
                        "Min Cosine Similarity",
                        f"{stability.get('min_cosine', 0):.3f}",
                    )
                with stab_cols[2]:
                    st.metric(
                        "Std Dev",
                        f"{stability.get('std_cosine', 0):.3f}",
                    )
            else:
                st.info("No stability data available yet.")

        with diag_tabs[3]:
            drift_result = st.session_state.get("drift_result", {})
            if drift_result:
                drift_cols = st.columns(2)
                with drift_cols[0]:
                    st.markdown("**Drift Summary:**")
                    st.write(f"Detection: {drift_result.get('alert_type', 'None')}")
                    st.write(f"Severity: {drift_result.get('severity', 'N/A')}")
                with drift_cols[1]:
                    st.markdown("**Metrics:**")
                    metrics = drift_result.get("drift_metrics", {})
                    for k, v in metrics.items():
                        st.metric(k, f"{v:.3f}", label_visibility="collapsed")
            else:
                st.info("No drift data available yet.")

        with diag_tabs[4]:
            if alignment_metrics:
                st.json(alignment_metrics)
            else:
                st.info("No alignment data available yet.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5: EXPORT & ACTIONS (Sticky/Prominent)
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("### 💾 Export & Actions")
    export_cols = st.columns([1, 1, 1])
    with export_cols[0]:
        if st.button("📄 Export Governance Report", use_container_width=True):
            st.info("Governance report export would be generated here (full implementation pending).")

    with export_cols[1]:
        if st.button("📋 Download Raw Data", use_container_width=True):
            st.info("Raw snapshot data download pending implementation.")

    with export_cols[2]:
        if st.button("🔄 Refresh Metrics", use_container_width=True):
            st.rerun()

    st.caption(
        "**v3.0 Governance Promise:** Every metric on this page traces back to specific "
        "data sources and is auditable. The scorecard, flags, and provenance chain "
        "enable contestability: any governance decision can be challenged by examining "
        "the evidence, and the system provides mechanistic explanations."
    )
