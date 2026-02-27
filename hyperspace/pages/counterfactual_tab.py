"""Counterfactual Scenario Engine (B2): block removal + diff analysis.

Allows users to remove one pipeline block from the UKT and see how
conclusions change — operationalising contestability for governance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from hyperspace.config import PLOTLY_LAYOUT, POLICY_KERNEL_NAMES
from hyperspace.models.knowledge_matrix import estimate_reality_regression_stability
from hyperspace.models.sparse_ae import train_sparse_ae, map_concepts_to_kernels
from hyperspace.viz import kernel_viz


def _run_counterfactual_ukt(
    snapshots: list[dict],
    removed_block: str,
) -> dict | None:
    """Re-run UKT SVD with one block removed.

    Returns a minimal result dict compatible with the main snapshot format,
    or None if not enough blocks remain.
    """
    # Filter snapshots to exclude the removed block
    remaining = [s for s in snapshots if s["block_name"] != removed_block]
    if len(remaining) < 1:
        return None

    # Reconstruct the sub-matrix
    # More reliably: reconstruct from scratch using the rows stored in the final matrix
    final_snap = snapshots[-1]
    block_names = [s["block_name"] for s in snapshots]
    full_matrix = final_snap["matrix"]  # shape (n_blocks, 64)

    kept_indices = [i for i, name in enumerate(block_names) if name != removed_block]
    if len(kept_indices) < 2:
        return None

    sub_matrix = full_matrix[kept_indices, :]  # (n_remaining, 64)
    kept_names = [block_names[i] for i in kept_indices]

    # SVD
    U, S, Vt = np.linalg.svd(sub_matrix, full_matrices=False)
    n_kernels = len(S)
    total = S.sum() + 1e-8
    importance = S / total
    kernel_activation = U * S[np.newaxis, :]
    reality_regression = importance @ Vt[:n_kernels, :]

    # Reconstruction error
    recon = U @ np.diag(S) @ Vt[:n_kernels, :]
    recon_error = float(np.linalg.norm(sub_matrix - recon))

    # SAE on reduced matrix
    sae_result = None
    concept_kernel_map = []
    try:
        sae_result = train_sparse_ae(sub_matrix, hidden_dim=16, epochs=60)
    except Exception:
        pass

    # Stability
    stability = estimate_reality_regression_stability(sub_matrix, n_runs=6, noise_std=0.01, seed=99)

    return dict(
        removed_block=removed_block,
        kept_blocks=kept_names,
        sub_matrix=sub_matrix,
        U=U,
        S=S,
        Vt=Vt,
        n_kernels=n_kernels,
        importance=importance,
        kernel_activation=kernel_activation,
        reality_regression=reality_regression,
        reconstruction_error=recon_error,
        stability=stability,
        sae_result=sae_result,
    )


def _plot_rr_diff(
    rr_original: np.ndarray,
    rr_counterfactual: np.ndarray,
    title: str = "Reality Regression: Original vs. Counterfactual",
) -> go.Figure:
    """Plot side-by-side reality regression comparison."""
    n = len(rr_original)
    n_cf = len(rr_counterfactual)
    max_n = max(n, n_cf)
    rr_orig_padded = np.pad(rr_original, (0, max_n - n))
    rr_cf_padded = np.pad(rr_counterfactual, (0, max_n - n_cf))

    # Color by region
    colors_orig = []
    colors_cf = []
    region_colors = {
        "temporal": "#3498db",
        "semantic": "#e67e22",
        "structural": "#2ecc71",
        "dynamic": "#e74c3c",
    }
    for i in range(max_n):
        c = (region_colors["temporal"] if i < 16
             else region_colors["semantic"] if i < 32
             else region_colors["structural"] if i < 48
             else region_colors["dynamic"])
        colors_orig.append(c)
        colors_cf.append(c)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Original", "Counterfactual", "Difference (CF − Original)"],
        vertical_spacing=0.10,
    )

    fig.add_trace(go.Bar(x=list(range(max_n)), y=rr_orig_padded,
                          marker_color=colors_orig, name="Original",
                          showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(max_n)), y=rr_cf_padded,
                          marker_color=colors_cf, name="Counterfactual",
                          showlegend=False), row=2, col=1)

    diff = rr_cf_padded - rr_orig_padded
    diff_colors = ["#64ffda" if d >= 0 else "#ff6b6b" for d in diff]
    fig.add_trace(go.Bar(x=list(range(max_n)), y=diff,
                          marker_color=diff_colors, name="Difference",
                          showlegend=False), row=3, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _plot_kernel_importance_diff(
    importance_orig: np.ndarray,
    importance_cf: np.ndarray,
    n_kernels_orig: int,
    n_kernels_cf: int,
) -> go.Figure:
    """Compare kernel importance distributions."""
    max_k = max(n_kernels_orig, n_kernels_cf)
    orig_padded = np.pad(importance_orig, (0, max_k - len(importance_orig)))
    cf_padded = np.pad(importance_cf, (0, max_k - len(importance_cf)))

    fig = go.Figure()
    x = [f"K{i}" for i in range(max_k)]
    fig.add_trace(go.Bar(
        x=x, y=orig_padded, name="Original",
        marker_color="#64ffda", opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=x, y=cf_padded, name="Counterfactual",
        marker_color="#ff6b6b", opacity=0.8,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title="Kernel Importance: Original vs. Counterfactual",
        height=280,
        yaxis_title="Explained Variance",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def render() -> None:
    """Render the Counterfactual Scenario Engine tab."""
    st.markdown("## Counterfactual Scenario Engine")
    st.markdown(
        "Test the robustness of AI conclusions by removing one data block from the analysis. "
        "This operationalises **contestability** — asking: *What would the system have concluded "
        "if it hadn't relied on this data source?*"
    )

    snapshots = st.session_state.get("ukt_snapshots", [])
    if not snapshots:
        st.info(
            "No pipeline results available. Launch the full pipeline first, "
            "then return here to run counterfactual scenarios."
        )
        return

    block_names = [s["block_name"] for s in snapshots]

    if len(block_names) < 3:
        st.warning(
            "At least 3 pipeline blocks are required to run a meaningful counterfactual "
            "(removing one block while keeping ≥2 for SVD)."
        )
        return

    st.markdown("---")

    # Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        removed_block = st.selectbox(
            "Remove this block from the analysis:",
            block_names,
            help="The selected block's feature vector will be excluded from the UKT before re-computing SVD kernels.",
            key="cf_removed_block",
        )
    with col2:
        st.markdown("")
        st.markdown("")
        run_cf = st.button(
            "Run Counterfactual",
            type="primary",
            use_container_width=True,
            key="cf_run_btn",
        )

    # Optional: shock injection
    with st.expander("⚡ Optional: Inject a structural shock", expanded=False):
        st.caption(
            "In addition to removing a block, you can flip the sign of a specific geopolitical "
            "relationship in the graph features (indices 32–47). This simulates a sudden alliance "
            "reversal or adversarial shift."
        )
        inject_shock = st.checkbox("Inject sign flip on structural features", key="cf_inject_shock")
        if inject_shock:
            shock_feature = st.slider(
                "Feature index to flip (structural region: 32–47):",
                min_value=32, max_value=47, value=35,
                key="cf_shock_feature",
            )
            st.caption(
                f"Feature {shock_feature} will have its sign flipped in the counterfactual matrix."
            )

    if run_cf:
        with st.spinner(f"Computing counterfactual (removing '{removed_block}')..."):
            cf_result = _run_counterfactual_ukt(snapshots, removed_block)
            if cf_result is None:
                st.error(
                    "Cannot compute counterfactual: not enough blocks remain after removal."
                )
                st.stop()

            # Apply shock if requested
            if st.session_state.get("cf_inject_shock"):
                shock_f = st.session_state.get("cf_shock_feature", 35)
                cf_result["sub_matrix"][:, shock_f] *= -1.0
                # Re-run SVD after shock
                sub_matrix = cf_result["sub_matrix"]
                U, S, Vt = np.linalg.svd(sub_matrix, full_matrices=False)
                n_kernels = len(S)
                total = S.sum() + 1e-8
                importance = S / total
                cf_result["U"] = U
                cf_result["S"] = S
                cf_result["Vt"] = Vt
                cf_result["n_kernels"] = n_kernels
                cf_result["importance"] = importance
                cf_result["kernel_activation"] = U * S[np.newaxis, :]
                cf_result["reality_regression"] = importance @ Vt[:n_kernels, :]
                recon = U @ np.diag(S) @ Vt[:n_kernels, :]
                cf_result["reconstruction_error"] = float(np.linalg.norm(sub_matrix - recon))

            st.session_state.counterfactual_result = cf_result
            st.session_state.counterfactual_removed_block = removed_block
        st.success(f"Counterfactual computed. Block '{removed_block}' excluded from analysis.")

    # Display results
    cf_result = st.session_state.get("counterfactual_result")
    if cf_result is None:
        st.caption("Select a block to remove and click 'Run Counterfactual' to begin.")
        return

    removed = cf_result.get("removed_block", "?")
    kept = cf_result.get("kept_blocks", [])

    st.markdown("---")
    st.markdown(f"### Results: '{removed}' block removed")
    st.markdown(
        f"**Remaining blocks:** {', '.join(kept)}  \n"
        f"**Original blocks:** {', '.join(block_names)}"
    )

    # Summary comparison metrics
    final_snap = snapshots[-1]
    orig_error = final_snap["reconstruction_error"]
    cf_error = cf_result["reconstruction_error"]
    orig_kernels = final_snap["n_kernels"]
    cf_kernels = cf_result["n_kernels"]

    # Reality regression difference magnitude
    rr_orig = final_snap["reality_regression"]
    rr_cf = cf_result["reality_regression"]
    max_n = max(len(rr_orig), len(rr_cf))
    rr_orig_p = np.pad(rr_orig, (0, max_n - len(rr_orig)))
    rr_cf_p = np.pad(rr_cf, (0, max_n - len(rr_cf)))
    rr_diff_magnitude = float(np.linalg.norm(rr_cf_p - rr_orig_p))
    rr_cosine = float(
        np.dot(rr_orig_p, rr_cf_p) /
        (np.linalg.norm(rr_orig_p) * np.linalg.norm(rr_cf_p) + 1e-8)
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kernels (original)", str(orig_kernels))
    m1.caption("Kernels before removal")
    m2.metric("Kernels (counterfactual)", str(cf_kernels))
    m2.caption("Kernels after removal")
    m3.metric("RR Cosine Similarity", f"{rr_cosine:.3f}", delta=f"{rr_cosine - 1.0:.3f}")
    m3.caption("1.0 = identical conclusions. Lower = greater divergence.")
    m4.metric("RR Difference Norm", f"{rr_diff_magnitude:.4f}")
    m4.caption("L2 distance between original and counterfactual conclusions.")

    # Governance interpretation
    if rr_cosine >= 0.95:
        st.success(
            f"✅ **Conclusion stable**: Removing '{removed}' causes minimal change "
            f"(cosine={rr_cosine:.3f}). The system's conclusions are not overly dependent "
            "on this data block."
        )
    elif rr_cosine >= 0.80:
        st.warning(
            f"⚠️ **Moderate sensitivity**: Removing '{removed}' meaningfully shifts "
            f"conclusions (cosine={rr_cosine:.3f}). Conclusions citing this block "
            "should note this dependency."
        )
    else:
        st.error(
            f"🔴 **High sensitivity**: Removing '{removed}' radically changes conclusions "
            f"(cosine={rr_cosine:.3f}). The system is highly dependent on this block. "
            "Governance risk: conclusions may collapse if this data source is unavailable or biased."
        )

    st.markdown("---")

    # Main diff visualization
    st.markdown("### Reality Regression Comparison")
    fig_diff = _plot_rr_diff(rr_orig, rr_cf,
                              title=f"Reality Regression Diff ('{removed}' removed)")
    st.plotly_chart(fig_diff, use_container_width=True)
    st.caption(
        "**Top panel:** Original reality regression across all 64 feature dimensions. "
        "**Middle panel:** Counterfactual (after block removal). "
        "**Bottom panel:** Difference (teal = CF increased, red = CF decreased). "
        "Color bands: blue=temporal, orange=semantic, green=structural, red=dynamic."
    )

    # Kernel importance comparison
    st.markdown("### Kernel Importance Comparison")
    fig_ki = _plot_kernel_importance_diff(
        final_snap["importance"],
        cf_result["importance"],
        orig_kernels, cf_kernels,
    )
    st.plotly_chart(fig_ki, use_container_width=True)

    # Region-level impact analysis
    st.markdown("### Region-Level Impact Analysis")
    region_names = ["temporal-pattern", "semantic-embedding", "structural-centrality", "dynamic-agent"]
    region_bounds = [(0, 16), (16, 32), (32, 48), (48, 64)]
    impact_rows = []
    for rname, (lo, hi) in zip(region_names, region_bounds):
        orig_energy = float(np.abs(rr_orig[lo:min(hi, len(rr_orig))]).sum())
        cf_energy_vals = rr_cf[lo:min(hi, len(rr_cf))]
        cf_energy = float(np.abs(cf_energy_vals).sum()) if len(cf_energy_vals) > 0 else 0.0
        delta = cf_energy - orig_energy
        impact_rows.append({
            "Region": rname.replace("-", " ").title(),
            "Original Energy": round(orig_energy, 4),
            "Counterfactual Energy": round(cf_energy, 4),
            "Change": round(delta, 4),
            "% Change": f"{(delta / (orig_energy + 1e-8)) * 100:+.1f}%",
        })
    impact_df = pd.DataFrame(impact_rows)
    st.dataframe(impact_df, use_container_width=True, hide_index=True)
    st.caption(
        "Energy = sum of absolute reality regression weights in each region. "
        "Large changes indicate that the removed block was a primary contributor to that region."
    )

    # Stability comparison
    st.markdown("### Stability Comparison")
    orig_stability = st.session_state.get("ukt_multirun_stability", {})
    cf_stability = cf_result.get("stability", {})
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Original (all blocks):**")
        if orig_stability and orig_stability.get("n_runs", 0) > 0:
            st.metric("Mean Cosine", f"{orig_stability['mean_cosine']:.3f}")
            st.caption(f"Min: {orig_stability['min_cosine']:.3f} | Std: {orig_stability['std_cosine']:.3f}")
        else:
            st.caption("N/A")
    with sc2:
        st.markdown(f"**Counterfactual ('{removed}' removed):**")
        if cf_stability and cf_stability.get("n_runs", 0) > 0:
            orig_mc = orig_stability.get("mean_cosine", 0.0) if orig_stability else 0.0
            delta_stability = cf_stability["mean_cosine"] - orig_mc
            st.metric(
                "Mean Cosine", f"{cf_stability['mean_cosine']:.3f}",
                delta=f"{delta_stability:+.3f}",
            )
            st.caption(f"Min: {cf_stability['min_cosine']:.3f} | Std: {cf_stability['std_cosine']:.3f}")
        else:
            st.caption("N/A")

    # SAE concept comparison
    cf_sae = cf_result.get("sae_result")
    if cf_sae:
        st.markdown("### Concept Discovery Comparison")
        orig_sae = st.session_state.get("sae_result")
        cs1, cs2 = st.columns(2)
        with cs1:
            st.markdown("**Original concepts:**")
            if orig_sae:
                st.metric(
                    "Active Concepts",
                    f"{orig_sae['active_concepts']}/{orig_sae['total_concepts']}",
                )
        with cs2:
            st.markdown(f"**Counterfactual ('{removed}' removed):**")
            orig_active = orig_sae["active_concepts"] if orig_sae else 0
            delta_active = cf_sae["active_concepts"] - orig_active
            st.metric(
                "Active Concepts",
                f"{cf_sae['active_concepts']}/{cf_sae['total_concepts']}",
                delta=str(delta_active),
            )

    # Export counterfactual report
    st.markdown("---")
    st.markdown("### Export Counterfactual Analysis")
    run_id = st.session_state.get("run_id", "UNKNOWN")
    run_ts = st.session_state.get("run_timestamp", "")

    cf_report_lines = [
        f"# Counterfactual Analysis Report",
        f"## Run ID: {run_id} | {run_ts}",
        f"## Scenario: '{removed}' block removed",
        f"",
        f"### Conclusion Change Summary",
        f"- Reality Regression Cosine Similarity: {rr_cosine:.3f}",
        f"- Difference Norm: {rr_diff_magnitude:.4f}",
        f"- Original Kernels: {orig_kernels} | Counterfactual Kernels: {cf_kernels}",
        f"- Original Reconstruction Error: {orig_error:.6f}",
        f"- Counterfactual Reconstruction Error: {cf_error:.6f}",
        f"",
        "### Region-Level Impact",
    ]
    for row in impact_rows:
        cf_report_lines.append(
            f"- {row['Region']}: {row['Original Energy']} → {row['Counterfactual Energy']} "
            f"({row['% Change']})"
        )

    if orig_stability and cf_stability:
        orig_mc = orig_stability.get("mean_cosine")
        cf_mc = cf_stability.get("mean_cosine")
        if orig_mc is not None and cf_mc is not None:
            cf_report_lines += [
                "",
                "### Stability Impact",
                f"- Original mean cosine: {orig_mc:.3f}",
                f"- Counterfactual mean cosine: {cf_mc:.3f}",
            ]

    cf_report_md = "\n".join(cf_report_lines)
    st.download_button(
        "Download Counterfactual Report (Markdown)",
        cf_report_md,
        f"hyperspace_counterfactual_{removed}_{run_id}.md",
        "text/markdown",
        key="cf_download_report",
    )
