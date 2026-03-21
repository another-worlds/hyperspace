"""Shared interpretability report section for domain tabs.

Each domain tab (Finance, Clusters, Politics, Agents) renders three
standardized sections at the bottom:
  1. Fitting Metrics — how well the block's model/analysis fit
  2. Test Metrics — out-of-sample / validation quality indicators
  3. Interpretability Report — UKT kernel contribution + semantic narrative
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hyperspace.config import KERNEL_NARRATOR_IMPORTANCE_MIN, PLOTLY_LAYOUT

# UKT block-name → (region label, feature start, feature end, color)
_BLOCK_REGIONS: dict[str, tuple[str, int, int, str]] = {
    "Finance":  ("temporal-pattern",      0,  16, "#3498db"),
    "Clusters": ("semantic-embedding",    16, 32, "#e67e22"),
    "Graph":    ("structural-centrality", 32, 48, "#2ecc71"),
    "Agents":   ("dynamic-agent",         48, 64, "#e74c3c"),
    "Spatial":  ("geospatial-kernel",     64, 80, "#9b59b6"),
}


def render_interpretability_report(block_name: str) -> None:
    """Render the UKT interpretability report for a given block.

    Reads UKT snapshots from session state and renders:
      - Kernel contribution of this block
      - Region energy breakdown
      - Semantic narrative (if available)
      - Stage SAE concept summary (if available)
    """
    snapshots = st.session_state.get("ukt_snapshots", [])
    if not snapshots:
        st.info("Run the full pipeline to generate the UKT interpretability report.")
        return

    # Find snapshot for this block
    snap = None
    for s in snapshots:
        if s["block_name"] == block_name:
            snap = s
            break
    if snap is None:
        st.info(f"No UKT snapshot found for block '{block_name}'.")
        return

    region_info = _BLOCK_REGIONS.get(block_name)
    if region_info is None:
        return
    region_label, feat_lo, feat_hi, color = region_info

    st.markdown("### Interpretability Report")
    st.caption(
        f"UKT analysis for the **{block_name}** block — feature region "
        f"**{region_label}** (indices {feat_lo}–{feat_hi - 1})."
    )

    # --- Kernel contribution of this block ---
    kernel_labels = snap.get("kernel_labels", [])
    importance = snap.get("importance", np.array([]))
    ka = snap.get("kernel_activation")

    if ka is not None and len(kernel_labels) > 0:
        # Find which row in the kernel activation matrix corresponds to this block
        block_idx = snap["step"] - 1  # 0-indexed row
        if block_idx < ka.shape[0]:
            activations = ka[block_idx]
            n_k = len(activations)

            col_k1, col_k2 = st.columns(2)
            with col_k1:
                # Block's kernel activation profile
                fig = go.Figure(go.Bar(
                    x=[f"K{i}" for i in range(n_k)],
                    y=activations,
                    marker_color=color,
                    text=[f"{v:+.3f}" for v in activations],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Activation: %{y:+.4f}<extra>" + block_name + "</extra>",
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT, height=260,
                    title=f"{block_name} — Kernel Activation Profile",
                    yaxis_title="Activation",
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True,
                                key=f"report_{block_name}_kernel_act")

            with col_k2:
                # Kernel importance at this step
                fig_imp = go.Figure(go.Bar(
                    x=[f"K{i}" for i in range(len(importance))],
                    y=importance,
                    marker_color="#64ffda",
                    text=[f"{v:.1%}" for v in importance],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Variance: %{y:.3f} (%{text})<extra></extra>",
                ))
                fig_imp.update_layout(
                    **PLOTLY_LAYOUT, height=260,
                    title=f"Kernel Importance (after {block_name})",
                    yaxis_title="Explained Variance",
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_imp, use_container_width=True,
                                key=f"report_{block_name}_kernel_imp")

    # --- Region energy in reality regression ---
    rr = snap.get("reality_regression")
    if rr is not None:
        region_slice = rr[feat_lo:feat_hi]
        energy = float(np.sum(np.abs(region_slice)))
        total_energy = float(np.sum(np.abs(rr))) + 1e-8
        pct = energy / total_energy

        # Show all region energies for context
        region_defs = [
            ("temporal-pattern", 0, 16),
            ("semantic-embedding", 16, 32),
            ("structural-centrality", 32, 48),
            ("dynamic-agent", 48, 64),
            ("geospatial-kernel", 64, 80),
        ]
        region_names = [r[0] for r in region_defs]
        region_energies = [
            float(np.sum(np.abs(rr[lo:hi]))) / total_energy
            for _, lo, hi in region_defs
        ]
        region_colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]

        fig_re = go.Figure(go.Bar(
            x=region_names,
            y=region_energies,
            marker_color=region_colors,
            text=[f"{v:.1%}" for v in region_energies],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Energy Share: %{y:.3f} (%{text})<extra></extra>",
        ))
        fig_re.update_layout(
            **PLOTLY_LAYOUT, height=260,
            title="Reality Regression — Region Energy Share",
            yaxis_title="Share of Total |RR| Energy",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_re, use_container_width=True,
                        key=f"report_{block_name}_region_energy")

        st.markdown(
            f"**{block_name}** contributes **{pct:.1%}** of total reality "
            f"regression energy ({region_label} region)."
        )

    # --- Reconstruction error ---
    recon_err = snap.get("reconstruction_error")
    if recon_err is not None:
        st.metric("Reconstruction Error (after this block)", f"{recon_err:.6f}")

    # --- Stage SAE summary ---
    sae = snap.get("stage_sae_result")
    if sae is not None:
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Stage SAE Active Concepts", str(sae.get("active_concepts", "—")))
        sc2.metric("Stage SAE Loss", f"{sae.get('final_loss', 0):.4f}")
        sc3.metric("Stage SAE Total Concepts", str(sae.get("total_concepts", "—")))

    # --- Semantic narrative ---
    layer_narr = snap.get("layer_narrative")
    if layer_narr:
        st.info(f"**Semantic narrative:** {layer_narr}")

    # --- Kernel narratives relevant to this block ---
    for kl in kernel_labels:
        if kl.get("semantic_narrative") and kl["importance"] > KERNEL_NARRATOR_IMPORTANCE_MIN:
            with st.expander(f"{kl['label']} — narrative"):
                st.markdown(kl["narrative"])
                if kl.get("semantic_narrative"):
                    st.info(kl["semantic_narrative"])

    st.caption(
        "v3.0 — This report traces the block's contribution through the UKT "
        "decomposition: kernel activations show cross-block latent structure, "
        "region energy reveals the block's weight in reality regression, "
        "and stage-SAE concepts provide interpretable decomposition."
    )
