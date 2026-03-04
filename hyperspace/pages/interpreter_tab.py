"""Semantic Interpreter tab: real concept discovery via sparse AE + UKT analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from hyperspace.config import PLOTLY_LAYOUT, UKT_FEATURE_DIM
from hyperspace.models.knowledge_matrix import UniversalKnowledgeTensor
from hyperspace.models.sparse_ae import train_sparse_ae, map_concepts_to_kernels
from hyperspace.viz import kernel_viz
from hyperspace.viz.charts import source_badge


def render() -> None:
    """Render the Semantic Interpreter & Knowledge Matrix tab."""
    st.markdown("## Semantic Interpreter & Persistent Knowledge Matrix")
    st.markdown(
        "Concept discovery via Sparse Autoencoder on the Universal Knowledge Tensor. "
        "Every pipeline block's features are decomposed into interpretable concepts "
        "and mapped to SVD-derived kernels."
    )

    interpret_btn = st.button("Run Interpretability Scan", type="primary",
                              key="interpret_run")

    snapshots = st.session_state.get("ukt_snapshots", [])
    sae_result = st.session_state.get("sae_result")
    policy_mode = st.session_state.get("policy_language_mode", False)

    if interpret_btn or sae_result:
        if interpret_btn:
            with st.spinner("Running concept extraction pipeline..."):
                # Re-run SAE if we have snapshots
                if snapshots:
                    # Reconstruct UKT from snapshots
                    final_snap = snapshots[-1]
                    matrix = final_snap["matrix"]
                    sae_result = train_sparse_ae(
                        matrix, hidden_dim=16, epochs=100,
                    )
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
            st.info("No UKT data available. Run the full pipeline from the dashboard first.")
            return

        final_snap = snapshots[-1]
        stability = st.session_state.get("ukt_multirun_stability")

        # Kernel matrix from UKT
        st.markdown("### Universal Kernel Matrix")
        block_names = [s["block_name"] for s in snapshots]
        fig = kernel_viz.plot_kernel_matrix(final_snap, block_names)
        st.plotly_chart(fig, use_container_width=True, key="interp_kernel_matrix")

        # Kernel importance
        col1, col2 = st.columns(2)
        with col1:
            fig_imp = kernel_viz.plot_kernel_importance(final_snap)
            st.plotly_chart(fig_imp, use_container_width=True, key="interp_kernel_importance")
        with col2:
            fig_rr = kernel_viz.plot_reality_regression(final_snap)
            st.plotly_chart(fig_rr, use_container_width=True, key="interp_reality_regression")

        # B1: Policy language mode branch
        if policy_mode:
            from hyperspace.pages.governance import render_kernel_policy_mode
            st.markdown("---")
            render_kernel_policy_mode(final_snap["kernel_labels"])
        else:
            # Standard technical view with A1 + C1 enhancements
            st.markdown("### Discovered Kernels — Semantic Interpretation")
            from hyperspace.pages.governance import (
                render_contest_popover,
                render_annotation_widget,
            )
            for kl in final_snap["kernel_labels"]:
                with st.expander(
                    f"{kl['label']}",
                    expanded=kl["importance"] > 0.2,
                ):
                    st.markdown(kl["narrative"])

                    # Show top features as a mini table
                    if kl.get("top_features"):
                        feat_df = pd.DataFrame(kl["top_features"])
                        st.dataframe(
                            feat_df[["name", "region", "loading"]].rename(
                                columns={"name": "Feature", "region": "Region",
                                         "loading": "Loading"}
                            ),
                            use_container_width=True, hide_index=True,
                        )

                    # Region breakdown
                    if kl.get("region_scores"):
                        scores = kl["region_scores"]
                        fig_rs = px.bar(
                            x=list(scores.keys()), y=list(scores.values()),
                            color=list(scores.values()),
                            color_continuous_scale="Viridis",
                            title="Region Energy Distribution",
                        )
                        fig_rs.update_layout(
                            **PLOTLY_LAYOUT, height=200, showlegend=False,
                            xaxis_title="Region", yaxis_title="Abs. Loading Sum",
                        )
                        st.plotly_chart(fig_rs, use_container_width=True,
                                        key=f"interp_region_energy_{kl['kernel_id']}")

                    # A1: Contest This button
                    st.markdown("---")
                    col_contest, col_annotate = st.columns([1, 2])
                    with col_contest:
                        render_contest_popover(
                            kl, stability=stability,
                            key_suffix=f"interpreter_{kl['kernel_id']}",
                        )

                    # C1: Multi-stakeholder annotation widget
                    with col_annotate:
                        render_annotation_widget(
                            kernel_id=kl["kernel_id"],
                            label=kl.get("label", kl["kernel_id"]),
                            key_suffix=f"interpreter_{kl['kernel_id']}",
                        )

        # SAE concept discovery
        if sae_result:
            st.markdown("---")
            st.markdown("### Sparse Autoencoder Concept Discovery")

            c1, c2, c3 = st.columns(3)
            c1.metric("Active Concepts",
                       f"{sae_result['active_concepts']}/{sae_result['total_concepts']}")
            c2.metric("Final Loss", f"{sae_result['final_loss']:.4f}")
            c3.metric("Total Concepts", str(sae_result["total_concepts"]))

            # Concept labels — full semantic output
            st.markdown("#### Discovered Concepts")
            active_concepts = [cl for cl in sae_result.get("concept_labels", [])
                               if cl["active"]]
            dormant_concepts = [cl for cl in sae_result.get("concept_labels", [])
                                if not cl["active"]]

            if active_concepts:
                st.markdown("**Active concepts** (above-mean activation):")
                from hyperspace.pages.governance import render_annotation_widget
                for cl in active_concepts:
                    with st.expander(cl.get("label", cl["concept_id"])):
                        st.markdown(cl.get("narrative", ""))
                        if cl.get("top_features"):
                            feat_df = pd.DataFrame(cl["top_features"])
                            st.dataframe(
                                feat_df[["name", "loading"]].rename(
                                    columns={"name": "Feature",
                                             "loading": "Loading"}
                                ),
                                use_container_width=True, hide_index=True,
                            )
                        # C1: Allow annotations on concepts too
                        render_annotation_widget(
                            kernel_id=f"concept_{cl['concept_id']}",
                            label=cl.get("label", cl["concept_id"]),
                            key_suffix=f"concept_{cl['concept_id']}",
                        )

            if dormant_concepts:
                with st.expander(
                    f"Dormant concepts ({len(dormant_concepts)})",
                    expanded=False,
                ):
                    for cl in dormant_concepts:
                        st.caption(cl.get("label", cl["concept_id"]))

            # Loss curve
            loss_hist = sae_result.get("loss_history", [])
            if loss_hist:
                st.markdown("#### SAE Training Loss")
                fig = px.line(
                    x=list(range(len(loss_hist))), y=loss_hist,
                    title="Sparse Autoencoder Loss Curve",
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT, height=250,
                    xaxis_title="Epoch", yaxis_title="Loss",
                )
                st.plotly_chart(fig, use_container_width=True, key="interp_sae_loss_curve")

            # Concept activation heatmap
            act = sae_result.get("concept_activations")
            if act is not None and act.shape[0] > 1:
                st.markdown("#### Concept Activation per Block")
                fig = px.imshow(
                    act,
                    x=[f"C{i:02d}" for i in range(act.shape[1])],
                    y=block_names[:act.shape[0]],
                    color_continuous_scale="Viridis",
                    title="Concept Activations (Blocks x Concepts)",
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=300)
                st.plotly_chart(fig, use_container_width=True, key="interp_concept_activations")

        # Concept-kernel mapping
        concept_kernel_map = st.session_state.get("concept_kernel_map", [])
        if concept_kernel_map:
            st.markdown("### Concept-Kernel Correspondence")
            fig_ck = kernel_viz.plot_concept_kernel_map(concept_kernel_map)
            st.plotly_chart(fig_ck, use_container_width=True, key="interp_concept_kernel_map")
            st.dataframe(pd.DataFrame(concept_kernel_map), use_container_width=True)

        # Stakeholder annotation summary
        st.markdown("---")
        st.markdown("### Stakeholder Annotation Record")
        from hyperspace.pages.governance import render_annotations_summary
        render_annotations_summary()

        # Step-by-step interpretation logs (now with rich text)
        st.markdown("### Step-by-Step Interpretability Reports")
        for snap in snapshots:
            with st.expander(f"Step {snap['step']}: {snap['block_name']}"):
                st.markdown(snap["report"])
