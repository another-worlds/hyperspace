"""Full Pipeline tab: end-to-end orchestration with UKT evolution display."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from hyperspace.config import KERNEL_EXPANDER_THRESHOLD
from hyperspace.core.caching import get_or_compute_figure, hash_list
from hyperspace.viz import kernel_viz
from hyperspace.viz.charts import source_badge


def render() -> None:
    """Render the Full Pipeline tab."""
    st.markdown("## Hyperspace Pipeline — Governance Accountability View")
    st.markdown(
        "End-to-end cycle: Finance → Clustering → Graph → **Spatial Raster Kernelization** → "
        "Agentic Sim → Semantic Interpretation. Each step updates the 80-dim Universal "
        "Knowledge Tensor. All conclusions are traceable, stability-tested, and contestable."
    )

    snapshots = st.session_state.get("ukt_snapshots", [])
    data_sources = st.session_state.get("data_sources", {})
    run_id = st.session_state.get("run_id", "UNKNOWN")
    run_ts = st.session_state.get("run_timestamp", "")
    policy_mode = st.session_state.get("policy_language_mode", False)
    scorecard = st.session_state.get("interpretability_scorecard", {})

    if not snapshots:
        st.info(
            "No pipeline results available. Launch the full pipeline from the "
            "dashboard to see results here."
        )
        return

    # D1: Run watermark
    if run_id and run_id != "UNKNOWN":
        st.markdown(
            f'<span class="run-id-watermark">Run ID: {run_id} · {run_ts}</span>',
            unsafe_allow_html=True,
        )

    # B1: Policy mode banner
    if policy_mode:
        st.info(
            "🗂️ **Governance Language Mode is active.** "
            "Kernel narratives below use policy-friendly language."
        )

    # Data source summary
    src_html = " ".join(source_badge(v) for v in data_sources.values())
    st.markdown(f"**Data Sources**: {src_html}", unsafe_allow_html=True)
    st.success("Hyperspace cycle complete. All blocks synchronized.")

    st.markdown("---")

    # ── Pipeline Step Status Table ────────────────────────────────────
    st.markdown("### Pipeline Step Status")
    st.caption(
        "Overview of each processing block: data source, governance health, "
        "and key analytical finding. For detailed visualizations, visit each block's tab."
    )

    gov_flags = st.session_state.get("governance_flags", [])
    finance_result = st.session_state.get("finance_result", {})
    cluster_result = st.session_state.get("cluster_result", {})
    graph_result = st.session_state.get("graph_result", {})
    sim_result = st.session_state.get("sim_result", {})

    # Build status rows
    step_rows = []
    for snap in snapshots:
        block = snap["block_name"]
        flags_for_block = [
            f.get("code", "?") for f in gov_flags
            if block.lower() in f.get("label", "").lower()
            or block.lower() in f.get("detail", "").lower()
        ]
        flag_str = ", ".join(flags_for_block) if flags_for_block else "None"
        status = "PASS" if not flags_for_block else "FLAG"

        # Key finding per block
        if block == "Finance" and isinstance(finance_result, dict):
            finding = f"Forecast generated · data: {finance_result.get('data_source', '?')}"
        elif block == "Clusters" and isinstance(cluster_result, dict):
            topics = cluster_result.get("topics", [])
            n_topics = len(set(t for t in topics if t != -1))
            finding = f"{n_topics} topics discovered"
        elif block == "Graph" and isinstance(graph_result, dict):
            comms = graph_result.get("analysis", {}).get("communities", [])
            finding = f"{len(comms)} alliance bloc(s) detected"
        elif block == "Agents" and isinstance(sim_result, dict):
            agents = sim_result.get("agents", {})
            if agents:
                max_a = max(agents.values(), key=lambda a: a.resources)
                finding = f"Dominant actor: {max_a.name}"
            else:
                finding = "Simulation complete"
        elif block == "Spatial":
            finding = "Spatial raster kernels integrated"
        else:
            finding = snap.get("report", "")[:80].split("\n")[0]

        step_rows.append({
            "Block": block,
            "Data Source": data_sources.get(block, "—"),
            "Governance": status,
            "Flags": flag_str,
            "Key Finding": finding,
        })

    if step_rows:
        step_df = pd.DataFrame(step_rows)

        def _style_gov(val: str) -> str:
            if val == "PASS":
                return "color: #64ffda; font-weight: bold"
            return "color: #ffaa00; font-weight: bold"

        try:
            styled_steps = step_df.style.map(_style_gov, subset=["Governance"])
        except AttributeError:
            styled_steps = step_df.style.applymap(_style_gov, subset=["Governance"])
        st.dataframe(styled_steps, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Kernel Evolution (unique to this tab) ────────────────────────
    st.markdown("### Kernel Evolution Across Pipeline Steps")
    st.caption(
        "This chart is unique to the Pipeline tab. It shows how the system's "
        "cross-domain patterns (kernels) evolve as each data block is added. "
        "A kernel that spikes when a new block is added indicates that block "
        "introduced a dominant new pattern into the analysis."
    )
    _snap_hash = hash_list([s["block_name"] for s in snapshots], "pipe_evo")
    fig_evo = get_or_compute_figure(
        f"pipe_evo_{_snap_hash}",
        lambda: kernel_viz.plot_kernel_evolution(snapshots),
    )
    st.plotly_chart(fig_evo, use_container_width=True, key="pipeline_kernel_evolution")

    # ── Semantic Kernel Narratives ────────────────────────────────────
    # Each UKT kernel gets a short (data-grounded) and extended (LLM-
    # generated) narrative explanation, surfaced with full governance
    # widgets for contestability and stakeholder annotation.
    final_snap = snapshots[-1] if snapshots else None
    if final_snap and final_snap.get("kernel_labels"):
        st.markdown("---")
        st.markdown("### Semantic Kernel Narratives")
        st.caption(
            "Each kernel is a cross-domain covariance pattern discovered by SVD "
            "decomposition of the Universal Knowledge Tensor. The short narrative "
            "is data-grounded — every claim backed by a measured quantity. The "
            "extended interpretation provides richer semantic context generated "
            "by the Tiny-LLM narrator when available."
        )

        if policy_mode:
            from hyperspace.pages.governance import render_kernel_policy_mode
            render_kernel_policy_mode(final_snap["kernel_labels"])
        else:
            stability = st.session_state.get("ukt_multirun_stability")
            from hyperspace.pages.governance import (
                render_contest_popover,
                render_annotation_widget,
            )
            for kl in final_snap["kernel_labels"]:
                with st.expander(
                    kl["label"],
                    expanded=kl["importance"] > KERNEL_EXPANDER_THRESHOLD,
                ):
                    # Short narrative (data-grounded, always present)
                    st.markdown(kl["narrative"])

                    # Extended narrative (semantic, LLM-generated, optional)
                    if kl.get("semantic_narrative"):
                        with st.expander(
                            "Extended Semantic Interpretation", expanded=False
                        ):
                            st.info(kl["semantic_narrative"])

                    # Governance widgets: contest + annotate
                    st.markdown("---")
                    col_contest, col_annotate = st.columns([1, 2])
                    with col_contest:
                        render_contest_popover(
                            kl, stability=stability,
                            key_suffix=f"pipeline_{kl['kernel_id']}",
                        )
                    with col_annotate:
                        render_annotation_widget(
                            kernel_id=kl["kernel_id"],
                            label=kl.get("label", kl["kernel_id"]),
                            key_suffix=f"pipeline_{kl['kernel_id']}",
                        )

    # ── Governance Narrative ─────────────────────────────────────────
    canvas_narrative = st.session_state.get("canvas_narrative")
    reality_narrative = st.session_state.get("reality_narrative")
    if canvas_narrative or reality_narrative:
        st.markdown("---")
        st.markdown("### System-Level Governance Narrative")
        if canvas_narrative:
            st.success(f"**Cross-Domain Summary:** {canvas_narrative}")
        if reality_narrative:
            st.info(f"**Reality Assessment:** {reality_narrative}")

    # C1: Annotation summary
    annotations = st.session_state.get("stakeholder_annotations", [])
    if annotations:
        st.markdown("---")
        st.markdown(f"### Stakeholder Annotations ({len(annotations)})")
        from hyperspace.pages.governance import render_annotations_summary
        render_annotations_summary()

    # Export — D1: stamped with run ID
    st.markdown("---")
    st.markdown("### Export")
    exp1, exp2, exp3 = st.columns(3)

    # Build report
    report_lines = [
        f"# Hyperspace Governance Report",
        f"## Run ID: {run_id} | {run_ts}",
        "",
    ]

    # Include scorecard in report
    if scorecard:
        report_lines.append("## Interpretability Score Card")
        for key, item in scorecard.items():
            status_str = "PASS" if item.get("passed") else "WARN"
            report_lines.append(
                f"- {item.get('label')}: {item.get('value')}{item.get('unit','')} "
                f"(≥{item.get('threshold')}{item.get('unit','')}) — {status_str}"
            )
        report_lines.append("")

    # Include governance flags
    gov_flags = st.session_state.get("governance_flags", [])
    if gov_flags:
        report_lines.append("## Governance Flags")
        for flag in gov_flags:
            report_lines.append(
                f"- [{flag.get('code')}] {flag.get('label')}: {flag.get('detail', '')}"
            )
        report_lines.append("")

    report_lines.append("## Pipeline Reports")
    for snap in snapshots:
        report_lines.append(snap["report"])

    # Include kernel narratives (short + extended)
    if final_snap and final_snap.get("kernel_labels"):
        report_lines.append("")
        report_lines.append("## Kernel Narratives")
        for kl in final_snap["kernel_labels"]:
            report_lines.append(f"### {kl.get('label', kl['kernel_id'])}")
            report_lines.append(kl.get("narrative", ""))
            if kl.get("semantic_narrative"):
                report_lines.append(
                    f"\n**Semantic interpretation:** {kl['semantic_narrative']}"
                )
            report_lines.append("")

    # Include annotations
    if annotations:
        report_lines.append("")
        report_lines.append("## Stakeholder Annotations")
        for ann in annotations:
            report_lines.append(
                f"- [{ann.get('role')}] {ann.get('timestamp', '')}: {ann.get('text', '')}"
            )

    report_md = "\n".join(report_lines)
    exp1.download_button(
        "Download Report (Markdown)", report_md,
        f"hyperspace_report_{run_id}.md", "text/markdown",
        key="pipeline_download_report",
    )

    metrics_rows = []
    for snap in snapshots:
        for kl in snap["kernel_labels"]:
            metrics_rows.append({
                "RunID": run_id, "Timestamp": run_ts,
                "Step": snap["step"], "Block": snap["block_name"],
                "Kernel": kl["kernel_id"], "Importance": kl["importance"],
                "Region": kl["dominant_region"],
            })
    if metrics_rows:
        exp2.download_button(
            "Download Metrics (CSV)",
            pd.DataFrame(metrics_rows).to_csv(index=False),
            f"hyperspace_metrics_{run_id}.csv", "text/csv",
            key="pipeline_download_metrics",
        )

    # Scorecard CSV
    if scorecard:
        sc_export_rows = [
            {
                "RunID": run_id, "Timestamp": run_ts,
                "Dimension": item.get("label"), "Value": item.get("value"),
                "Threshold": item.get("threshold"), "Unit": item.get("unit", ""),
                "Pass": item.get("passed"), "Description": item.get("description"),
            }
            for item in scorecard.values()
        ]
        exp3.download_button(
            "Download Score Card (CSV)",
            pd.DataFrame(sc_export_rows).to_csv(index=False),
            f"hyperspace_scorecard_{run_id}.csv", "text/csv",
            key="pipeline_download_scorecard",
        )
