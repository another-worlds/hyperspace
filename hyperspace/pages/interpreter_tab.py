"""Semantic Interpreter tab: concept discovery, semantic canvas, and LLM narratives.

Audience: legal, diplomatic, and civil society delegates.
Top 3 visualizations (audience-aligned):
  1. Semantic Canvas Radar Chart — cross-domain theme fingerprint
  2. Concept Activation Heatmap — which concepts active per pipeline block
  3. Reality Regression Bar Chart — feature importance by domain region

Everything else moved to an Advanced Diagnostics expander.
"""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hyperspace.config import KERNEL_EXPANDER_THRESHOLD, PLOTLY_LAYOUT
from hyperspace.core.caching import (
    get_or_compute_sae,
    get_cache_stats,
    make_sae_cache_key,
)
from hyperspace.core.logging import StructuredLogger, log_sae_training
from hyperspace.models.sparse_ae import train_sparse_ae, map_concepts_to_kernels
from hyperspace.models.semantic_canvas import CANVAS_DIMENSIONS
from hyperspace.viz import kernel_viz


def render() -> None:
    """Render the Semantic Interpreter & Persistent Knowledge Matrix tab."""
    st.markdown("## Semantic Interpreter")
    st.markdown(
        "What has the system learned, and can those conclusions be trusted? "
        "This tab answers three questions every governance review requires: "
        "**What themes dominate?** (Radar) — **Which concepts are active?** (Heatmap) — "
        "**What drives the conclusions?** (Feature Importance)"
    )

    # SAE caching control
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        interpret_btn = st.button(
            "Run Interpretability Scan", type="primary", key="interpret_run"
        )
    with col2:
        force_retrain = st.button("🔄 Retrain", key="sae_retrain_btn",
                                 help="Ignore cache and retrain SAE (stochastic mode)")
    with col3:
        cache_stats = get_cache_stats()
        st.metric("Cache Items", cache_stats["total"])

    snapshots = st.session_state.get("ukt_snapshots", [])
    sae_result = st.session_state.get("sae_result")
    policy_mode = st.session_state.get("policy_language_mode", False)

    if interpret_btn or sae_result or force_retrain:
        if interpret_btn or force_retrain:
            with st.spinner("Running concept extraction pipeline..."):
                if snapshots:
                    final_snap = snapshots[-1]
                    matrix = final_snap["matrix"]

                    # Use improved caching system with hyperparameter hashing
                    import time
                    start_time = time.time()
                    cache_key = make_sae_cache_key(matrix, 16, 100)
                    was_cached = f"cache_{cache_key}" in st.session_state

                    def _train_sae(m, hidden_dim, epochs):
                        return train_sparse_ae(m, hidden_dim=hidden_dim, epochs=epochs)

                    sae_result = get_or_compute_sae(
                        matrix,
                        hidden_dim=16,
                        epochs=100,
                        compute_fn=_train_sae,
                        force_retrain=force_retrain,
                    )

                    duration = time.time() - start_time
                    log_sae_training(cached=was_cached, hidden_dim=16, epochs=100, duration_sec=duration)

                    if sae_result and final_snap:
                        concept_kernel_map = map_concepts_to_kernels(
                            sae_result, final_snap,
                        )
                        st.session_state.sae_result = sae_result
                        st.session_state.concept_kernel_map = concept_kernel_map
                else:
                    st.warning("Run the full pipeline first to populate the UKT.")
                    return

        if not snapshots:
            st.info(
                "No UKT data available. Run the full pipeline from the dashboard first."
            )
            return

        final_snap = snapshots[-1]
        block_names = [s["block_name"] for s in snapshots]

        # ================================================================ #
        # 1. SEMANTIC CANVAS RADAR CHART                                   #
        # The system's interpretive fingerprint across 12 named dimensions  #
        # ================================================================ #
        canvas = st.session_state.get("semantic_canvas")
        if canvas is not None:
            st.markdown("### Cross-Domain Theme Fingerprint")
            st.caption(
                "Each axis is a named interpretive dimension. The solid teal shape "
                "is the system's accumulated reading across all data domains. "
                "Dashed traces show each pipeline block's individual contribution. "
                "Large spikes indicate dimensions where multiple data sources agree."
            )

            canvas_state = canvas.get_accumulated_state()
            coords = canvas_state["coordinates"]
            dim_labels = [d["label"] for d in CANVAS_DIMENSIONS]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=coords.tolist() + [coords[0]],
                theta=dim_labels + [dim_labels[0]],
                fill="toself",
                name="Accumulated (all domains)",
                line=dict(color="#64ffda", width=3),
                fillcolor="rgba(100, 255, 218, 0.15)",
            ))
            colors = ["#3498db", "#e67e22", "#e74c3c", "#2ecc71", "#9b59b6"]
            for i, entry in enumerate(canvas_state["entries"]):
                c = entry.coordinates
                fig_radar.add_trace(go.Scatterpolar(
                    r=c.tolist() + [c[0]],
                    theta=dim_labels + [dim_labels[0]],
                    name=entry.block_name,
                    line=dict(color=colors[i % len(colors)], dash="dot", width=1.5),
                    opacity=0.6,
                ))
            fig_radar.update_layout(
                **PLOTLY_LAYOUT,
                polar=dict(
                    bgcolor="#0d1117",
                    radialaxis=dict(range=[0, 1], showticklabels=True,
                                    gridcolor="#1a2332"),
                    angularaxis=dict(gridcolor="#1a2332"),
                ),
                title="Semantic Canvas — Cross-Domain Theme Fingerprint",
                height=480,
                showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True,
                            key="interp_semantic_radar")

            # Dominant narrative (plain-language summary)
            dominant = canvas_state.get("dominant_narrative", [])
            if dominant:
                st.markdown("**Dominant themes detected across all data domains:**")
                for d in dominant[:3]:
                    st.markdown(
                        f"- **{d['label']}** — {d['desc']}"
                    )

            # Cross-domain narrative from Tiny-LLM
            canvas_narrative = st.session_state.get("canvas_narrative")
            reality_narrative = st.session_state.get("reality_narrative")
            if canvas_narrative:
                st.success(f"**System Summary:** {canvas_narrative}")
                # Provenance: surface the concrete features behind the narrative
                all_evidence = []
                for entry in canvas_state.get("entries", []):
                    for ev in getattr(entry, "feature_evidence", []):
                        all_evidence.append(dict(ev, block=entry.block_name))
                if all_evidence:
                    with st.expander(
                        "Evidence: features driving this summary", expanded=False
                    ):
                        st.caption(
                            "Top features (by absolute loading) from each pipeline block, "
                            "linked to the canvas dimensions they influence. "
                            "Use feature indices to trace back to kernel labels below."
                        )
                        for ev in all_evidence:
                            dim_list = ", ".join(f"`{d}`" for d in ev.get("canvas_dims", []))
                            st.markdown(
                                f"- **{ev['name']}** (idx {ev['index']}, "
                                f"`{ev['region']}`, source: `{ev['source']}`) — "
                                f"loading {ev['loading']:+.4f} · block: {ev['block']} "
                                f"· dims: {dim_list}"
                            )
            if reality_narrative:
                st.info(f"**Reality Assessment:** {reality_narrative}")
                # Reuse same evidence block for reality narrative
                all_evidence_r = []
                for entry in canvas_state.get("entries", []):
                    for ev in getattr(entry, "feature_evidence", []):
                        all_evidence_r.append(dict(ev, block=entry.block_name))
                if all_evidence_r:
                    with st.expander(
                        "Evidence: features driving the reality assessment",
                        expanded=False,
                    ):
                        st.caption(
                            "Same feature provenance as system summary. "
                            "Cross-reference with 'Feature Importance by Domain Region' "
                            "chart below for full 80-feature detail."
                        )
                        for ev in all_evidence_r:
                            dim_list = ", ".join(f"`{d}`" for d in ev.get("canvas_dims", []))
                            st.markdown(
                                f"- **{ev['name']}** (idx {ev['index']}, "
                                f"`{ev['region']}`, source: `{ev['source']}`) — "
                                f"loading {ev['loading']:+.4f} · block: {ev['block']} "
                                f"· dims: {dim_list}"
                            )

        st.markdown("---")

        # ================================================================ #
        # 2. CONCEPT ACTIVATION HEATMAP                                    #
        # Which named concepts are active in which pipeline blocks          #
        # ================================================================ #
        if sae_result:
            act = sae_result.get("concept_activations")
            if act is not None and act.shape[0] > 1:
                st.markdown("### Structural Feature Patterns")
                st.caption(
                    "Each row is a pipeline block (Finance, Clusters, Graph, etc.). "
                    "Each column is a structural pattern identified by the Sparse Autoencoder "
                    "across the full 80-dimensional feature space. "
                    "Bright cells = that pattern is strongly active in that block. "
                    "Patterns active across multiple blocks indicate cross-domain structure. "
                    f"**Note:** Computed from N={act.shape[0]} blocks (single-run, augmented training). "
                    "Treat as a structural decomposition of the current run, "
                    "not a statistically robust concept discovery."
                )

                # Use concept labels where available
                concept_labels = sae_result.get("concept_labels", [])
                col_labels = []
                for i in range(act.shape[1]):
                    if i < len(concept_labels):
                        col_labels.append(
                            concept_labels[i].get("label", f"C{i:02d}")[:18]
                        )
                    else:
                        col_labels.append(f"C{i:02d}")

                fig_act = px.imshow(
                    act,
                    x=col_labels,
                    y=block_names[:act.shape[0]],
                    color_continuous_scale="Viridis",
                    title="Concept Activations — Data Domain × Named Concept",
                )
                fig_act.update_layout(
                    **PLOTLY_LAYOUT,
                    height=max(280, 60 * act.shape[0]),
                    xaxis_title="Named Concept",
                    yaxis_title="Pipeline Block",
                )
                st.plotly_chart(fig_act, use_container_width=True,
                                key="interp_concept_activations")

            # Active pattern summaries (plain-English)
            active_concepts = [
                cl for cl in sae_result.get("concept_labels", []) if cl["active"]
            ]
            if active_concepts:
                st.markdown("**Active structural patterns — plain-language summaries:**")
                from hyperspace.pages.governance import render_annotation_widget
                for cl in active_concepts:
                    with st.expander(cl.get("label", cl["concept_id"])):
                        st.markdown(cl.get("narrative", ""))
                        if cl.get("semantic_narrative"):
                            st.info(f"**Semantic:** {cl['semantic_narrative']}")
                        render_annotation_widget(
                            kernel_id=f"concept_{cl['concept_id']}",
                            label=cl.get("label", cl["concept_id"]),
                            key_suffix=f"concept_{cl['concept_id']}",
                        )

        st.markdown("---")

        # ================================================================ #
        # 3. REALITY REGRESSION BAR CHART                                  #
        # Feature-level importance, color-coded by domain region            #
        # ================================================================ #
        st.markdown("### Feature Importance by Domain Region")
        st.caption(
            "Each bar represents one of the system's 80 input features. "
            "Height = how much that feature drives the system's cross-domain conclusions. "
            "Color indicates which data domain the feature comes from: "
            "blue=market/temporal, orange=news/semantic, green=geopolitical/structural, "
            "red=agent/dynamic, purple=spatial/geospatial."
        )
        fig_rr = kernel_viz.plot_reality_regression(final_snap)
        st.plotly_chart(fig_rr, use_container_width=True, key="interp_reality_regression")

        # Policy language mode: named kernel briefings
        if policy_mode:
            from hyperspace.pages.governance import render_kernel_policy_mode
            st.markdown("---")
            render_kernel_policy_mode(final_snap["kernel_labels"])
        else:
            # Standard kernel view with contest/annotate widgets
            st.markdown("---")
            st.markdown("### SVD Kernels — Cross-Domain Covariance Patterns")
            stability = st.session_state.get("ukt_multirun_stability")
            from hyperspace.pages.governance import (
                render_contest_popover,
                render_annotation_widget,
            )
            for kl in final_snap["kernel_labels"]:
                with st.expander(
                    f"{kl['label']}",
                    expanded=kl["importance"] > KERNEL_EXPANDER_THRESHOLD,
                ):
                    st.markdown(kl["narrative"])
                    if kl.get("semantic_narrative"):
                        st.info(f"**Semantic interpretation:** {kl['semantic_narrative']}")
                    st.markdown("---")
                    col_contest, col_annotate = st.columns([1, 2])
                    with col_contest:
                        render_contest_popover(
                            kl, stability=stability,
                            key_suffix=f"interpreter_{kl['kernel_id']}",
                        )
                    with col_annotate:
                        render_annotation_widget(
                            kernel_id=kl["kernel_id"],
                            label=kl.get("label", kl["kernel_id"]),
                            key_suffix=f"interpreter_{kl['kernel_id']}",
                        )

        # ================================================================ #
        # Stakeholder Annotations                                           #
        # ================================================================ #
        st.markdown("---")
        st.markdown("### Stakeholder Annotation Record")
        from hyperspace.pages.governance import render_annotations_summary
        render_annotations_summary()

        # ================================================================ #
        # ADVANCED DIAGNOSTICS (collapsed by default)                      #
        # ML-engineering detail for expert review only                      #
        # ================================================================ #
        st.markdown("---")
        with st.expander("Advanced Diagnostics (ML Engineering Detail)", expanded=False):
            st.caption(
                "The following diagnostics are intended for ML engineers reviewing "
                "the system's internal state. They are not required for governance "
                "or policy analysis."
            )

            # Kernel matrix
            st.markdown("#### Universal Kernel Matrix")
            fig_km = kernel_viz.plot_kernel_matrix(final_snap, block_names)
            st.plotly_chart(fig_km, use_container_width=True,
                            key="interp_adv_kernel_matrix")
            st.caption(
                "Each cell: activation strength of a pipeline block on a UKT kernel. "
                "Cross-block activation indicates shared latent structure."
            )

            # Kernel importance
            col1, col2 = st.columns(2)
            with col1:
                fig_imp = kernel_viz.plot_kernel_importance(final_snap)
                st.plotly_chart(fig_imp, use_container_width=True,
                                key="interp_adv_kernel_importance")
                st.caption("Fraction of total variance explained per SVD kernel.")
            with col2:
                # Canvas trajectory heatmap
                _canvas_adv = st.session_state.get("semantic_canvas")
                if _canvas_adv is not None:
                    _canvas_state_adv = _canvas_adv.get_accumulated_state()
                    _dim_labels_adv = [d["label"] for d in CANVAS_DIMENSIONS]
                    trajectory = _canvas_state_adv.get("trajectory", [])
                    if len(trajectory) > 1:
                        traj_matrix = np.array([t["state"] for t in trajectory])
                        traj_block_labels = [t["block"] for t in trajectory]
                        fig_traj = px.imshow(
                            traj_matrix,
                            x=_dim_labels_adv,
                            y=traj_block_labels,
                            color_continuous_scale="Viridis",
                            title="Canvas Evolution per Step",
                        )
                        fig_traj.update_layout(**PLOTLY_LAYOUT, height=280)
                        st.plotly_chart(fig_traj, use_container_width=True,
                                        key="interp_adv_canvas_trajectory")
                        st.caption(
                            "Cumulative semantic canvas state after each pipeline step."
                        )

            # SAE metrics + loss curve
            if sae_result:
                st.markdown("#### Sparse Autoencoder Diagnostics")
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Active Concepts",
                    f"{sae_result['active_concepts']}/{sae_result['total_concepts']}",
                )
                c2.metric("Final Loss", f"{sae_result['final_loss']:.4f}")
                c3.metric("Total Concepts", str(sae_result["total_concepts"]))

                loss_hist = sae_result.get("loss_history", [])
                if loss_hist:
                    fig_loss = px.line(
                        x=list(range(len(loss_hist))), y=loss_hist,
                        title="SAE Training Loss Curve",
                    )
                    fig_loss.update_layout(
                        **PLOTLY_LAYOUT, height=220,
                        xaxis_title="Epoch", yaxis_title="Loss",
                    )
                    st.plotly_chart(fig_loss, use_container_width=True,
                                    key="interp_adv_sae_loss")

                # Dormant concepts
                dormant_concepts = [
                    cl for cl in sae_result.get("concept_labels", [])
                    if not cl["active"]
                ]
                if dormant_concepts:
                    st.caption(
                        f"{len(dormant_concepts)} dormant concepts (below-mean activation): "
                        + ", ".join(
                            cl.get("label", cl["concept_id"])
                            for cl in dormant_concepts
                        )
                    )

            # Concept-kernel map
            concept_kernel_map = st.session_state.get("concept_kernel_map", [])
            if concept_kernel_map:
                st.markdown("#### Concept-Kernel Correspondence")
                fig_ck = kernel_viz.plot_concept_kernel_map(concept_kernel_map)
                st.plotly_chart(fig_ck, use_container_width=True,
                                key="interp_adv_concept_kernel_map")

            # UVT coupling matrix + per-head attention
            uvt_result = st.session_state.get("uvt_result")
            if uvt_result is not None:
                st.markdown("#### Universal Variance Tensor — Cross-Block Coupling")
                coupling = uvt_result["coupling_matrix"]
                uvt_block_names = uvt_result.get("block_names") or block_names
                coupling_block_names = uvt_block_names[:coupling.shape[0]]
                if len(coupling_block_names) != coupling.shape[0]:
                    coupling_block_names = [
                        f"Block_{i}" for i in range(coupling.shape[0])
                    ]
                fig_coup = px.imshow(
                    coupling,
                    x=coupling_block_names,
                    y=coupling_block_names,
                    color_continuous_scale="Viridis",
                    title="Cross-Block Coupling Matrix (Attention Weights)",
                    text_auto=".3f",
                )
                fig_coup.update_layout(**PLOTLY_LAYOUT, height=350)
                st.plotly_chart(fig_coup, use_container_width=True,
                                key="interp_adv_uvt_coupling")

                attn_per_head = uvt_result.get("attention_per_head")
                if attn_per_head is not None and attn_per_head.shape[0] > 1:
                    with st.expander("Per-Head Attention Detail", expanded=False):
                        head_cols = st.columns(min(4, attn_per_head.shape[0]))
                        for h in range(attn_per_head.shape[0]):
                            with head_cols[h % len(head_cols)]:
                                fig_h = px.imshow(
                                    attn_per_head[h],
                                    x=coupling_block_names,
                                    y=coupling_block_names,
                                    color_continuous_scale="Viridis",
                                    title=f"Head {h}",
                                    text_auto=".2f",
                                )
                                fig_h.update_layout(
                                    **PLOTLY_LAYOUT, height=250,
                                    coloraxis_showscale=False,
                                )
                                st.plotly_chart(fig_h, use_container_width=True,
                                                key=f"interp_adv_uvt_head_{h}")

                coupling_labels = uvt_result.get("coupling_labels", [])
                if coupling_labels:
                    fig_var = go.Figure(go.Bar(
                        x=[cl["mode_id"] for cl in coupling_labels],
                        y=[cl["variance_explained"] for cl in coupling_labels],
                        marker_color=["#64ffda" if cl["variance_explained"] > KERNEL_EXPANDER_THRESHOLD
                                      else "#3498db" for cl in coupling_labels],
                    ))
                    fig_var.update_layout(
                        **PLOTLY_LAYOUT, height=220,
                        title="Variance Explained per Coupling Mode",
                    )
                    st.plotly_chart(fig_var, use_container_width=True,
                                    key="interp_adv_uvt_variance")

            # USE encoding
            use_result = st.session_state.get("use_result")
            if use_result is not None:
                st.markdown("#### Universal Semantic Encoding Vector")
                encoding = use_result["encoding"]
                dim_labels_use = [dl["label"] for dl in use_result["dimension_labels"]]
                colors_use = [
                    "#64ffda" if dl["strength"] == "STRONG"
                    else "#3498db" if dl["strength"] == "MODERATE"
                    else "#555"
                    for dl in use_result["dimension_labels"]
                ]
                fig_enc = go.Figure(go.Bar(
                    x=list(range(len(encoding))),
                    y=encoding,
                    marker_color=colors_use,
                ))
                fig_enc.update_layout(
                    **PLOTLY_LAYOUT, height=260,
                    title="USE: Unified Multi-Modal State Vector",
                    yaxis=dict(range=[-1.1, 1.1]),
                )
                st.plotly_chart(fig_enc, use_container_width=True,
                                key="interp_adv_use_encoding")

                use_loss = use_result.get("loss_history", [])
                if use_loss:
                    fig_use_loss = px.line(
                        x=list(range(len(use_loss))), y=use_loss,
                        title="USE Training Loss",
                    )
                    fig_use_loss.update_layout(**PLOTLY_LAYOUT, height=200)
                    st.plotly_chart(fig_use_loss, use_container_width=True,
                                    key="interp_adv_use_loss")

            # Step-by-step interpretability logs
            st.markdown("#### Step-by-Step Interpretability Reports")
            for snap in snapshots:
                with st.expander(f"Step {snap['step']}: {snap['block_name']}"):
                    st.markdown(snap["report"])
