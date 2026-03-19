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

from hyperspace.config import (
    CF_STABILITY_HIGH,
    CF_STABILITY_MODERATE,
    FEATURE_NAMES,
    PLOTLY_LAYOUT,
    POLICY_KERNEL_NAMES,
    STRUCTURAL_REGION_BOUNDS,
)
from hyperspace.core.caching import get_or_compute_svd
from hyperspace.models.knowledge_matrix import (
    FEATURE_REGION_LABELS,
    estimate_reality_regression_stability,
)
from hyperspace.models.sparse_ae import train_sparse_ae, map_concepts_to_kernels
from hyperspace.viz import kernel_viz


def _run_counterfactual_ukt(
    snapshots: list[dict],
    removed_block: str,
) -> dict | None:
    """Re-run UKT SVD with one block removed.

    Rebuilds the SharedProjection from the remaining blocks' raw features,
    re-projects them, then decomposes. This ensures the counterfactual
    reflects what would have happened WITHOUT the removed block's influence
    on the entire projection space — not just a row slice.

    Returns a minimal result dict compatible with the main snapshot format,
    or None if not enough blocks remain.
    """
    from hyperspace.models.knowledge_matrix import (
        HYPERSPACE_REGISTRY,
        BLOCK_REGION_MAP,
        _normalize_features,
    )
    from ukt.projection import SharedProjection
    from ukt.kernels import decompose_svd

    block_names = [s["block_name"] for s in snapshots]
    kept_indices = [i for i, name in enumerate(block_names) if name != removed_block]
    if len(kept_indices) < 2:
        return None

    # Get raw features from the final snapshot (which stores all blocks' raw features)
    final_snap = snapshots[-1]
    raw_features = final_snap.get("raw_features")

    if raw_features is None:
        # Fallback for snapshots without raw_features (legacy)
        full_matrix = final_snap["matrix"]
        sub_matrix = full_matrix[kept_indices, :]
    else:
        # Rebuild projection from scratch with only the remaining blocks
        projection = SharedProjection(HYPERSPACE_REGISTRY)
        for idx in kept_indices:
            bn = block_names[idx]
            ri = BLOCK_REGION_MAP.get(bn)
            if ri is not None:
                rname, lo, hi = ri
                normalized = _normalize_features(raw_features[idx])
                projection.observe(rname, normalized[lo:hi])

        # Re-project remaining blocks through the rebuilt projection
        rows = []
        for idx in kept_indices:
            rows.append(projection.project(_normalize_features(raw_features[idx])))
        sub_matrix = np.stack(rows)

    kept_names = [block_names[i] for i in kept_indices]

    # SVD (with caching to avoid recomputation on UI reruns for the same sub-matrix)
    decomposition = get_or_compute_svd(sub_matrix, compute_fn=decompose_svd)
    n_kernels = decomposition.n_kernels

    # SAE on reduced matrix
    sae_result = None
    try:
        sae_result = train_sparse_ae(sub_matrix, hidden_dim=16, epochs=60)
    except Exception:
        st.warning("SAE training unavailable for counterfactual; concept analysis skipped.")

    # Stability
    stability = estimate_reality_regression_stability(sub_matrix, n_runs=6, noise_std=0.01, seed=99)

    return dict(
        removed_block=removed_block,
        kept_blocks=kept_names,
        sub_matrix=sub_matrix,
        U=decomposition.U,
        S=decomposition.S,
        Vt=decomposition.Vt,
        n_kernels=n_kernels,
        importance=decomposition.importance,
        kernel_activation=decomposition.kernel_activation,
        reality_regression=decomposition.reality_regression,
        reconstruction_error=decomposition.reconstruction_error,
        stability=stability,
        sae_result=sae_result,
    )


def _region_color_for_index(idx: int) -> str:
    """Return the configured feature-region color for a feature index."""
    from hyperspace.models.knowledge_matrix import HYPERSPACE_REGISTRY
    # Auto-generate palette from registry order
    _PALETTE = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6",
                "#1abc9c", "#f39c12", "#8e44ad", "#2c3e50", "#d35400"]
    for i, region in enumerate(HYPERSPACE_REGISTRY.ordered_regions):
        if region.start <= idx < region.end:
            return _PALETTE[i % len(_PALETTE)]
    return "#95a5a6"


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

    # Color by configured feature regions
    colors_orig = []
    colors_cf = []
    for i in range(max_n):
        c = _region_color_for_index(i)
        colors_orig.append(c)
        colors_cf.append(c)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Original", "Counterfactual", "Difference (CF − Original)"],
        vertical_spacing=0.10,
    )

    feature_labels = [
        FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
        for i in range(max_n)
    ]

    hover_orig = (
        "<b>%{x}</b><br>Weight: %{y:.4f}<extra></extra>"
    )
    hover_diff = (
        "<b>%{x}</b><br>Diff: %{y:.4f}<extra></extra>"
    )

    fig.add_trace(go.Bar(x=feature_labels, y=rr_orig_padded,
                          marker_color=colors_orig, name="Original",
                          hovertemplate=hover_orig,
                          showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=feature_labels, y=rr_cf_padded,
                          marker_color=colors_cf, name="Counterfactual",
                          hovertemplate=hover_orig,
                          showlegend=False), row=2, col=1)

    diff = rr_cf_padded - rr_orig_padded
    diff_colors = ["#64ffda" if d >= 0 else "#ff6b6b" for d in diff]
    fig.add_trace(go.Bar(x=feature_labels, y=diff,
                          marker_color=diff_colors, name="Difference",
                          hovertemplate=hover_diff,
                          showlegend=False), row=3, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=600,
        margin=dict(l=20, r=20, t=60, b=60),
        xaxis_tickangle=-45,
        xaxis2_tickangle=-45,
        xaxis3_tickangle=-45,
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
            lo, hi = STRUCTURAL_REGION_BOUNDS
            shock_feature = st.slider(
                f"Feature index to flip (structural region: {lo}–{hi}):",
                min_value=lo, max_value=hi, value=(lo + hi) // 2,
                key="cf_shock_feature",
            )
            st.caption(
                f"Feature {shock_feature} will have its sign flipped in the counterfactual matrix."
            )

    if run_cf:
        # Cache key: (removed block, shock config) to avoid recomputing identical scenarios
        shock_on = st.session_state.get("cf_inject_shock", False)
        shock_f_val = st.session_state.get("cf_shock_feature", 35) if shock_on else None
        cf_cache_key = (removed_block, shock_on, shock_f_val)
        cached_key = st.session_state.get("cf_cache_key")
        if cached_key == cf_cache_key and st.session_state.get("counterfactual_result") is not None:
            st.toast("Using cached counterfactual result (same scenario).")
        else:
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
                st.session_state.cf_cache_key = cf_cache_key

                # Store in scenario history (keep last 5)
                if "cf_scenario_history" not in st.session_state:
                    st.session_state.cf_scenario_history = []

                scenario = {
                    "name": f"CF_{len(st.session_state.cf_scenario_history) + 1}: Remove {removed_block}",
                    "removed_block": removed_block,
                    "result": cf_result,
                    "shock_injected": shock_on,
                }
                st.session_state.cf_scenario_history.append(scenario)
                if len(st.session_state.cf_scenario_history) > 5:
                    st.session_state.cf_scenario_history.pop(0)

            st.success(f"Counterfactual computed. Block '{removed_block}' excluded from analysis.")

    # Scenario Management UI
    scenario_history = st.session_state.get("cf_scenario_history", [])
    if scenario_history:
        st.markdown("---")
        st.markdown("### 📊 Scenario Comparison")

        col1, col2 = st.columns([2, 1])
        with col1:
            scenario_names = [s["name"] for s in scenario_history]
            selected_idx = st.selectbox(
                "Select scenario to view:",
                range(len(scenario_names)),
                format_func=lambda i: scenario_names[i],
                key="cf_scenario_select",
            )
            cf_result = scenario_history[selected_idx]["result"]

        with col2:
            st.markdown("#### 📝 Stored Scenarios")
            for i, s in enumerate(scenario_history):
                if st.button(
                    f"#{i+1}: {s['removed_block']}",
                    use_container_width=True,
                    key=f"cf_scenario_btn_{i}"
                ):
                    st.session_state.cf_scenario_select = i
                    st.rerun()
    else:
        # No history yet
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
    if rr_cosine >= CF_STABILITY_HIGH:
        st.success(
            f"✅ **Conclusion stable**: Removing '{removed}' causes minimal change "
            f"(cosine={rr_cosine:.3f}). The system's conclusions are not overly dependent "
            "on this data block."
        )
    elif rr_cosine >= CF_STABILITY_MODERATE:
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
    st.plotly_chart(fig_diff, use_container_width=True, key="cf_rr_diff")
    st.caption(
        "**Top panel:** Original reality regression across all 80 feature dimensions. "
        "**Middle panel:** Counterfactual (after block removal). "
        "**Bottom panel:** Difference (teal = CF increased, red = CF decreased). "
        "Color bands: blue=temporal, orange=semantic, green=structural, red=dynamic, purple=geospatial."
    )

    # Kernel importance comparison
    st.markdown("### Kernel Importance Comparison")
    fig_ki = _plot_kernel_importance_diff(
        final_snap["importance"],
        cf_result["importance"],
        orig_kernels, cf_kernels,
    )
    st.plotly_chart(fig_ki, use_container_width=True, key="cf_kernel_importance")
    st.caption(
        "v3.0 — Kernel importance shift when a block is removed. Large changes "
        "indicate the removed block was critical to that kernel's structure — "
        "operationalising contestability."
    )

    # ── Domain-level impact summary (governance-readable) ────────────
    st.markdown("### Domain-Level Impact")
    from hyperspace.models.knowledge_matrix import HYPERSPACE_REGISTRY
    region_names = [r.name for r in HYPERSPACE_REGISTRY.ordered_regions]
    region_bounds = [(r.start, r.end) for r in HYPERSPACE_REGISTRY.ordered_regions]
    impact_rows = []
    for rname, (lo, hi) in zip(region_names, region_bounds):
        orig_energy = float(np.abs(rr_orig[lo:min(hi, len(rr_orig))]).sum())
        cf_energy_vals = rr_cf[lo:min(hi, len(rr_cf))]
        cf_energy = float(np.abs(cf_energy_vals).sum()) if len(cf_energy_vals) > 0 else 0.0
        delta = cf_energy - orig_energy
        pct = (delta / (orig_energy + 1e-8)) * 100
        impact_rows.append({
            "Data Domain": rname.replace("-", " ").title(),
            "Change": f"{pct:+.0f}%",
            "Interpretation": (
                "Large increase — this domain now dominates" if pct > 15
                else "Large decrease — this domain was previously dominant" if pct < -15
                else "Minor shift — this domain is not highly dependent on the removed block"
            ),
        })

    # Only show rows with notable changes
    notable = [r for r in impact_rows if abs(float(r["Change"].replace("%", ""))) > 5]
    if notable:
        st.caption(
            "Data domains most affected by removing this block "
            "(only domains with >5% change shown):"
        )
        for row in notable:
            change_val = float(row["Change"].replace("%", ""))
            if change_val > 15:
                st.error(f"**{row['Data Domain']}**: {row['Change']} — {row['Interpretation']}")
            elif change_val > 5:
                st.warning(f"**{row['Data Domain']}**: {row['Change']} — {row['Interpretation']}")
            elif change_val < -15:
                st.warning(f"**{row['Data Domain']}**: {row['Change']} — {row['Interpretation']}")
            else:
                st.info(f"**{row['Data Domain']}**: {row['Change']} — {row['Interpretation']}")
    else:
        st.success("No data domain shows a change >5% — the removed block had minimal domain-specific impact.")

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
        "### Domain-Level Impact",
    ]
    for row in impact_rows:
        cf_report_lines.append(
            f"- {row['Data Domain']}: {row['Change']} — {row['Interpretation']}"
        )

    cf_report_md = "\n".join(cf_report_lines)
    st.download_button(
        "Download Counterfactual Report (Markdown)",
        cf_report_md,
        f"hyperspace_counterfactual_{removed}_{run_id}.md",
        "text/markdown",
        key="cf_download_report",
    )
