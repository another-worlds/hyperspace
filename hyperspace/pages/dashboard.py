"""Dashboard: landing page, pipeline orchestration, progress tracking, results display."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.config import (
    DEFAULT_TICKERS, GEOPOLITICAL_NODES, PIPELINE_STEPS, UKT_FEATURE_DIM,
    SCORECARD_THRESHOLDS,
)
from hyperspace.models.knowledge_matrix import (
    UniversalKnowledgeTensor,
    estimate_reality_regression_stability,
)
from hyperspace.viz.charts import source_badge
from hyperspace.viz import kernel_viz
from hyperspace.core.pipeline import PipelineRunner


# --------------------------------------------------------------------------- #
# Internal helpers: governance flag computation                                #
# --------------------------------------------------------------------------- #

def _compute_governance_flags(
    data_sources: dict,
    ukt_snapshots: list,
    sae_result: dict | None,
    graph_result: dict | None,
    timeframe_context: dict,
) -> list[dict]:
    """Auto-detect governance flags from pipeline results.

    Delegates to PipelineRunner._compute_governance_flags to ensure
    consistent flag detection logic across headless and UI pipelines.
    """
    return PipelineRunner._compute_governance_flags(
        data_sources=data_sources,
        snapshots=ukt_snapshots,
        sae_result=sae_result,
        graph_result=graph_result,
        timeframe_context=timeframe_context,
    )


def _compute_scorecard(
    ukt_snapshots: list,
    sae_result: dict | None,
    data_sources: dict,
    stability: dict | None,
    governance_flags: list,
) -> dict:
    """Compute the interpretability scorecard values.

    Returns dict mapping criterion key -> {value, threshold, pass, label, unit, description}
    """
    scorecard: dict[str, dict] = {}

    # Feature traceability: count features with metadata
    if ukt_snapshots:
        final_snap = ukt_snapshots[-1]
        feature_meta = final_snap.get("feature_meta", {})
        traced = sum(
            1 for idx in range(UKT_FEATURE_DIM)
            if idx in feature_meta and feature_meta[idx].get("label")
        )
    else:
        traced = 0

    scorecard["feature_traceability"] = dict(
        value=traced,
        threshold=SCORECARD_THRESHOLDS["feature_traceability"]["threshold"],
        passed=traced >= SCORECARD_THRESHOLDS["feature_traceability"]["threshold"],
        label=SCORECARD_THRESHOLDS["feature_traceability"]["label"],
        unit=SCORECARD_THRESHOLDS["feature_traceability"]["unit"],
        description=SCORECARD_THRESHOLDS["feature_traceability"]["description"],
    )

    # Kernel stability
    mean_cosine = stability.get("mean_cosine", 0.0) if stability else 0.0
    scorecard["kernel_stability"] = dict(
        value=round(mean_cosine, 3),
        threshold=SCORECARD_THRESHOLDS["kernel_stability"]["threshold"],
        passed=mean_cosine >= SCORECARD_THRESHOLDS["kernel_stability"]["threshold"],
        label=SCORECARD_THRESHOLDS["kernel_stability"]["label"],
        unit=SCORECARD_THRESHOLDS["kernel_stability"]["unit"],
        description=SCORECARD_THRESHOLDS["kernel_stability"]["description"],
    )

    # Concept activation rate
    if sae_result:
        total = sae_result.get("total_concepts", 1)
        active = sae_result.get("active_concepts", 0)
        rate = round(100.0 * active / max(total, 1), 1)
    else:
        rate = 0.0
    scorecard["concept_activation_rate"] = dict(
        value=rate,
        threshold=SCORECARD_THRESHOLDS["concept_activation_rate"]["threshold"],
        passed=rate >= SCORECARD_THRESHOLDS["concept_activation_rate"]["threshold"],
        label=SCORECARD_THRESHOLDS["concept_activation_rate"]["label"],
        unit=SCORECARD_THRESHOLDS["concept_activation_rate"]["unit"],
        description=SCORECARD_THRESHOLDS["concept_activation_rate"]["description"],
    )

    # Data source diversity (live count)
    live_count = sum(
        1 for v in data_sources.values()
        if "Live" in v or ("Offline" in v and "Synthetic" not in v)
    )
    scorecard["data_source_diversity"] = dict(
        value=live_count,
        threshold=SCORECARD_THRESHOLDS["data_source_diversity"]["threshold"],
        passed=live_count >= SCORECARD_THRESHOLDS["data_source_diversity"]["threshold"],
        label=SCORECARD_THRESHOLDS["data_source_diversity"]["label"],
        unit=SCORECARD_THRESHOLDS["data_source_diversity"]["unit"],
        description=SCORECARD_THRESHOLDS["data_source_diversity"]["description"],
    )

    # Governance flags count
    flag_count = len(governance_flags)
    scorecard["governance_flags"] = dict(
        value=flag_count,
        threshold=SCORECARD_THRESHOLDS["governance_flags"]["threshold"],
        passed=flag_count == 0,
        label=SCORECARD_THRESHOLDS["governance_flags"]["label"],
        unit=SCORECARD_THRESHOLDS["governance_flags"]["unit"],
        description=SCORECARD_THRESHOLDS["governance_flags"]["description"],
    )

    return scorecard


# --------------------------------------------------------------------------- #
# Landing page (C3: Governance reframe)                                        #
# --------------------------------------------------------------------------- #

def render_landing() -> None:
    """Render the governance-framed dashboard landing page."""
    # Governance header
    st.markdown(
        '<div class="governance-header">'
        '<h2 style="color:#64ffda; margin:0 0 8px 0;">Hyperspace — Accountability Infrastructure for AI Governance</h2>'
        '<p style="color:#a8b2d1; margin:0; font-size:0.95em;">'
        'A demonstration system for the UN Global Dialogue on AI Governance — '
        'February 2026'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Three failure modes framing
    st.markdown("### The Three Accountability Failures This System Addresses")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "**① Opacity**\n\n"
            "Dominant AI systems use billions of parameters with learned concepts "
            "that cannot be labeled, traced, or linked to causal stories. "
            "It is impossible to explain *why* a conclusion was drawn."
        )
    with col2:
        st.markdown(
            "**② Uncontestability**\n\n"
            "Decisions derived from opaque pattern recognition cannot be robustly "
            "validated, challenged, or improved when they fail. There is no mechanism "
            "for due process against an algorithmic conclusion."
        )
    with col3:
        st.markdown(
            "**③ Untraceability**\n\n"
            "Biased and partial knowledge is absorbed into authoritative-sounding "
            "outputs while the pathways of influence remain invisible — machine-generated "
            "meaning without provenance or responsibility."
        )

    st.markdown("---")

    # Three technical guarantees
    st.markdown("### Three Technical Guarantees")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.success(
            "**Full Feature Provenance**\n\n"
            "Every one of 64 input dimensions carries a complete metadata chain: "
            "source, entity, metric, time scope, and block. Any conclusion can be "
            "traced back to its raw data inputs."
        )
    with g2:
        st.success(
            "**Stability-Tested Kernels**\n\n"
            "The Universal Knowledge Tensor runs 8 noisy perturbation tests to verify "
            "that conclusions are robust. A cosine similarity score quantifies how much "
            "conclusions change under small data variations."
        )
    with g3:
        st.success(
            "**Concept-Level Interpretability**\n\n"
            "A Sparse Autoencoder discovers a small set of named, interpretable concepts "
            "from the data. Each concept is mapped to specific kernels and features — "
            "enabling contestation at the level of individual claims."
        )

    st.markdown("---")

    # System Accountability Statement
    with st.expander("📋 System Accountability Statement", expanded=True):
        st.markdown("""
**What this system CAN conclude:**
- Which data domains (financial, informational, geopolitical, agentic) are most active in the current information environment
- Which structural patterns cut across multiple modalities simultaneously
- Whether those patterns are stable under small data perturbations
- Which specific features drive each cross-modal pattern

**What this system CANNOT conclude:**
- Causal relationships between geopolitical events and market movements
- Future outcomes with certainty — all forecasts are probabilistic
- Ground truth about classified or non-public information
- Anything not derivable from the four data domains listed above

**Limitations:**
- All pipeline stages require live network access — the pipeline blocks with an error if any source is unavailable (no silent synthetic fallbacks)
- The geopolitical graph covers 6 actors only — systemic omissions exist
- TFT forecasting runs for 3 epochs on CPU — not production-grade
- All interpretations are generated algorithmically and require human expert review
        """)

    # Pipeline architecture (compact)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipeline Steps", "7", "")
    c2.metric("Feature Dimensions", str(UKT_FEATURE_DIM), "")
    c3.metric("Blocks", "5", "Finance + Clusters + Graph + Spatial + Agents")
    c4.metric("Kernels (max)", "5", "grows with blocks")

    st.markdown("---")

    # Cold-start kernel matrix placeholder
    st.markdown("### Universal Kernel Matrix")
    st.caption("The kernel matrix will populate as each pipeline block completes.")
    placeholder_matrix = np.zeros((1, UKT_FEATURE_DIM))
    import plotly.express as px
    fig = px.imshow(
        placeholder_matrix,
        color_continuous_scale="RdBu_r",
        template="plotly_dark",
        title="UKT: Awaiting pipeline launch...",
    )
    fig.update_layout(height=150, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
    st.plotly_chart(fig, use_container_width=True, key="dashboard_ukt_placeholder")

    st.markdown("---")

    with st.expander("v3.0 Pipeline Architecture"):
        st.code("""
Data Fetch (yfinance + GDELT/RSS + Harvard Dataverse UN Votes + Open-Elevation + Open-Meteo + World Bank)
          |
    [Finance Block]  --> TFT train    --> UKT row 1 [0:16]  --> SVD --> Interpret
          |
    [Cluster Block]  --> BERTopic     --> UKT row 2 [16:32] --> SVD --> Interpret
          |
    [Graph Block]    --> Centrality   --> UKT row 3 [32:48] --> SVD --> Interpret
          |
    [Spatial Raster] --> SVD Kernels  --> UKT row 4 [64:80] --> SVD --> Interpret
          |
    [Agent Sim]      --> Simulate     --> UKT row 5 [48:64] --> SVD --> Interpret
          |
    [Final SAE]      --> Concept discovery on full UKT (80-dim)
          |
    Universal Reality Regression + Governance Accountability Report
        """, language="text")


# --------------------------------------------------------------------------- #
# Pipeline execution                                                            #
# --------------------------------------------------------------------------- #

def run_pipeline() -> None:
    """Execute the full pipeline with step-by-step UKT updates."""
    # D1: Generate run ID at pipeline start
    from hyperspace.state import generate_run_id
    run_id, run_timestamp = generate_run_id()

    ukt = UniversalKnowledgeTensor(feature_dim=UKT_FEATURE_DIM)
    snapshots: list[dict] = []
    data_sources: dict[str, str] = {}

    with st.status("Running Hyperspace Pipeline...", expanded=True) as status:
        # ---- Step 1: Fetch Data ----
        st.write("Fetching real data sources...")
        from hyperspace.data.finance import get_ohlcv
        from hyperspace.data.news import get_text_data
        from hyperspace.data.political import get_political_data

        tickers = st.session_state.get("tickers", DEFAULT_TICKERS) or DEFAULT_TICKERS
        try:
            ohlcv_df, fin_src = get_ohlcv(tickers)
        except RuntimeError as exc:
            status.update(label="Pipeline blocked: finance data unavailable", state="error")
            st.error(str(exc))
            return
        finance_start = None
        finance_end = None
        if "Date" in ohlcv_df.columns and len(ohlcv_df) > 0:
            finance_start = pd.to_datetime(ohlcv_df["Date"]).min().to_pydatetime()
            finance_end = pd.to_datetime(ohlcv_df["Date"]).max().to_pydatetime()

        try:
            docs, docs_src = get_text_data(start_date=finance_start, end_date=finance_end)
        except RuntimeError as exc:
            status.update(label="Pipeline blocked: no live news source", state="error")
            st.error(str(exc))
            return
        min_year = finance_start.year if finance_start else 2000
        max_year = finance_end.year if finance_end else None
        try:
            un_df, agreement, pol_src = get_political_data(min_year=min_year, max_year=max_year)
        except RuntimeError as exc:
            status.update(label="Pipeline blocked: political data unavailable", state="error")
            st.error(str(exc))
            return

        data_sources["Finance"] = fin_src
        data_sources["Clusters"] = docs_src
        data_sources["Graph"] = pol_src
        st.session_state.data_sources = data_sources
        st.session_state.raw_ohlcv = ohlcv_df
        st.session_state.raw_docs = docs
        st.session_state.timeframe_context = {
            "start_date": finance_start.date().isoformat() if finance_start else None,
            "end_date": finance_end.date().isoformat() if finance_end else None,
            "min_year": min_year,
            "max_year": max_year,
        }
        st.write(f"Data fetched: {fin_src} | {docs_src} | {pol_src}")

        # ---- Step 2: Finance Block ----
        st.write("Training Temporal Fusion Transformer (3 epochs, CPU)...")
        from hyperspace.models.tft_forecast import fit_tft

        tft_result = fit_tft(
            tickers=tuple(tickers),
            hidden=32, encoder_len=48, prediction_len=12,
        )
        if tft_result is None:
            status.update(label="Pipeline blocked: TFT fitting failed", state="error")
            st.error(
                "TFT fitting failed. Check the error above — common causes: "
                "missing pytorch-forecasting/lightning packages, or unreachable market data."
            )
            return
        finance_features = tft_result["features_for_ukt"]
        data_sources["Finance"] = tft_result["data_source"]

        snap = ukt.add_block(
            "Finance", finance_features,
            feature_meta=tft_result.get("feature_meta", {}),
            timeframe_context=st.session_state.timeframe_context,
        )
        snapshots.append(snap)
        st.session_state.finance_result = tft_result
        st.write(f"Finance: {(snap.get('report') or '').split(chr(10))[0]}")

        # ---- Step 3: Cluster Block ----
        st.write("Fitting BERTopic on real documents...")
        from hyperspace.models.topic_model import fit_topic_model

        import hashlib
        docs_hash = hashlib.md5("".join(docs[:5]).encode()).hexdigest()[:8]
        cluster_result = fit_topic_model(docs_hash, docs=docs, data_source=docs_src)
        if cluster_result is None:
            status.update(label="Pipeline blocked: BERTopic unavailable", state="error")
            st.error("BERTopic is unavailable; cluster fallback was intentionally removed.")
            return

        cluster_features = cluster_result["features_for_ukt"]
        data_sources["Clusters"] = cluster_result["data_source"]

        snap = ukt.add_block(
            "Clusters", cluster_features,
            feature_meta=cluster_result.get("feature_meta", {}),
            timeframe_context=st.session_state.timeframe_context,
        )
        snapshots.append(snap)
        st.session_state.cluster_result = cluster_result
        st.write(f"Clusters: {(snap.get('report') or '').split(chr(10))[0]}")

        # ---- Step 4: Graph Block ----
        st.write("Analyzing geopolitical graph + centrality...")
        from hyperspace.models.graph_engine import build_geopolitical_graph, analyze_graph

        G, pos = build_geopolitical_graph(agreement_matrix=agreement)
        graph_analysis = analyze_graph(G)
        graph_features = graph_analysis["features_for_ukt"]

        snap = ukt.add_block(
            "Graph", graph_features,
            feature_meta=graph_analysis.get("feature_meta", {}),
            timeframe_context=st.session_state.timeframe_context,
        )
        snapshots.append(snap)
        st.session_state.graph_result = dict(
            G=G, pos=pos, analysis=graph_analysis,
            features_for_ukt=graph_features,
            feature_meta=graph_analysis.get("feature_meta", {}),
            data_source=pol_src,
        )
        st.write(f"Graph: {(snap.get('report') or '').split(chr(10))[0]}")

        # ---- Step 3.5: Spatial Raster Kernelization ----
        st.write("Fetching multimodal spatial rasters (elevation, climate, economics, conflict)...")
        from hyperspace.data.spatial import fetch_all_spatial_data
        from hyperspace.models.spatial_kernels import get_spatial_features

        try:
            raw_spatial = fetch_all_spatial_data()
            spatial_result = get_spatial_features(
                raw_spatial["physical_raster"],
                raw_spatial["country_scalars"],
                raw_spatial["scalar_names"],
                raw_spatial["node_order"],
                st.session_state.timeframe_context,
            )
            data_sources["Spatial"] = raw_spatial["source_label"]
            snap = ukt.add_block(
                "Spatial", spatial_result["features_for_ukt"],
                feature_meta=spatial_result["feature_meta"],
                timeframe_context=st.session_state.timeframe_context,
            )
            snapshots.append(snap)
            st.session_state.spatial_result = spatial_result
            st.write(f"Spatial: {(snap.get('report') or '').split(chr(10))[0]}")
        except RuntimeError as exc:
            status.update(label="Pipeline blocked: spatial data unavailable", state="error")
            st.error(str(exc))
            return

        # ---- Step 5: Agent Simulation ----
        st.write("Running agent simulation (50 steps)...")
        from hyperspace.models.agent_sim import (
            initialize_agents_from_data, run_simulation,
        )

        agents = initialize_agents_from_data(
            graph_analysis, agreement,
            spatial_features=st.session_state.get("spatial_result"),
        )
        agents, log_entries, agent_features, agent_feature_meta = run_simulation(agents, steps=50)

        snap = ukt.add_block(
            "Agents", agent_features,
            feature_meta=agent_feature_meta,
            timeframe_context=st.session_state.timeframe_context,
        )
        snapshots.append(snap)
        data_sources["Agents"] = "agent_simulation (derived from Graph + Spatial)"
        st.session_state.sim_result = dict(
            agents=agents, log=log_entries,
            features_for_ukt=agent_features,
            feature_meta=agent_feature_meta,
            data_source=data_sources["Agents"],
        )
        st.write(f"Agents: {(snap.get('report') or '').split(chr(10))[0]}")

        # ---- Step 6: Final Interpretation ----
        st.write("Running sparse autoencoder for concept discovery...")
        from hyperspace.models.sparse_ae import train_sparse_ae, map_concepts_to_kernels

        final_matrix = ukt.get_final_matrix()
        sae_result = None
        concept_kernel_map = []
        if final_matrix is not None:
            sae_result = train_sparse_ae(final_matrix, hidden_dim=16, epochs=80)
            if sae_result is not None:
                final_snap = ukt.get_latest_snapshot()
                if final_snap:
                    concept_kernel_map = map_concepts_to_kernels(sae_result, final_snap)

        st.session_state.sae_result = sae_result
        st.session_state.concept_kernel_map = concept_kernel_map
        st.session_state.ukt_snapshots = snapshots
        st.session_state.data_sources = data_sources

        # ---- Semantic Canvas: store + generate final narratives ---- #
        st.session_state.semantic_canvas = ukt.canvas

        # ---- Cross-Block Interconnection: UVT + USE ----
        final_matrix = ukt.get_final_matrix()
        if final_matrix is not None and final_matrix.shape[0] >= 2:
            st.write("Computing Universal Variance Tensor (cross-block attention)...")
            from hyperspace.models.cross_block_net import (
                compute_universal_variance_tensor,
                compute_universal_semantic_encoding,
            )
            from hyperspace.models.knowledge_matrix import HYPERSPACE_REGISTRY

            active_block_names = [s["block_name"] for s in snapshots]
            uvt_result = compute_universal_variance_tensor(
                final_matrix, n_heads=4, d_model=32, epochs=120,
                registry=HYPERSPACE_REGISTRY,
                block_names=active_block_names,
            )
            st.session_state.uvt_result = uvt_result

            if uvt_result is not None:
                st.write("Computing Universal Semantic Encoding...")
                use_result = compute_universal_semantic_encoding(
                    final_matrix, uvt_result,
                    canvas=ukt.canvas, semantic_dim=24, epochs=150,
                    registry=HYPERSPACE_REGISTRY,
                    block_names=active_block_names,
                )
                st.session_state.use_result = use_result
                st.write(
                    f"UVT: {len(uvt_result['coupling_labels'])} coupling modes | "
                    f"USE: {use_result['semantic_dim']}-dim encoding"
                    if use_result else "UVT computed, USE unavailable"
                )
            else:
                # UVT failed — clear any stale USE result to avoid mismatched UI
                st.session_state.use_result = None
        else:
            # Not enough blocks — clear both to prevent stale visualizations
            st.session_state.uvt_result = None
            st.session_state.use_result = None

        st.write("Generating semantic narratives via Tiny-LLM...")
        try:
            from hyperspace.models.semantic_narrator import (
                narrate_canvas, narrate_reality_regression,
            )
            from hyperspace.models.sparse_ae import enrich_concepts_with_narratives

            # Full canvas narrative
            canvas_narrative = narrate_canvas(ukt.canvas)
            st.session_state.canvas_narrative = canvas_narrative

            # Reality regression narrative (from final snapshot)
            if snapshots:
                rr_narrative = narrate_reality_regression(snapshots[-1], ukt.canvas)
                st.session_state.reality_narrative = rr_narrative

            # Enrich SAE concepts with semantic narratives
            if sae_result is not None:
                enrich_concepts_with_narratives(sae_result, ukt.canvas)

        except Exception:
            pass  # Narrator unavailable — graceful degradation

        final_matrix = ukt.get_final_matrix()
        if final_matrix is not None:
            st.session_state.ukt_multirun_stability = estimate_reality_regression_stability(
                final_matrix, n_runs=8, noise_std=0.01, seed=42,
            )

        # ---- A3: Compute governance flags ----
        gov_flags = _compute_governance_flags(
            data_sources=data_sources,
            ukt_snapshots=snapshots,
            sae_result=sae_result,
            graph_result=st.session_state.get("graph_result"),
            timeframe_context=st.session_state.get("timeframe_context", {}),
        )
        st.session_state.governance_flags = gov_flags

        # ---- B3: Compute interpretability scorecard ----
        scorecard = _compute_scorecard(
            ukt_snapshots=snapshots,
            sae_result=sae_result,
            data_sources=data_sources,
            stability=st.session_state.get("ukt_multirun_stability"),
            governance_flags=gov_flags,
        )
        st.session_state.interpretability_scorecard = scorecard

        status.update(label="Pipeline complete!", state="complete")
        st.session_state.pipeline_complete = True


# --------------------------------------------------------------------------- #
# Results rendering                                                             #
# --------------------------------------------------------------------------- #

def render_results() -> None:
    """Render the full results dashboard after pipeline completes."""
    snapshots = st.session_state.get("ukt_snapshots", [])
    data_sources = st.session_state.get("data_sources", {})

    if not snapshots:
        st.info("No pipeline results yet. Click 'Launch' to run.")
        return

    # D1: Run ID watermark
    run_id = st.session_state.get("run_id")
    run_ts = st.session_state.get("run_timestamp")
    if run_id:
        st.markdown(
            f'<span class="run-id-watermark">Run ID: {run_id} · {run_ts}</span>',
            unsafe_allow_html=True,
        )

    # D3: Synthetic data banner
    synthetic_blocks = [
        k for k, v in data_sources.items()
        if any(x in v for x in ["Synthetic", "synthetic", "Fallback", "fallback", "Mock", "mock"])
    ]
    if synthetic_blocks:
        st.markdown(
            '<div class="synthetic-banner">'
            '⚠️ <strong>SYNTHETIC DATA ACTIVE</strong> — '
            f'Blocks using simulated data: {", ".join(synthetic_blocks)}. '
            'Conclusions from these blocks are <strong>illustrative only</strong> '
            'and do not represent real-world observations.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Data source badges
    src_html = " ".join(source_badge(v) for v in data_sources.values())
    st.markdown(f"**Data Sources**: {src_html}", unsafe_allow_html=True)

    # Summary metrics from real computations
    final_snap = snapshots[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Active Kernels", str(final_snap["n_kernels"]))

    finance_result = st.session_state.get("finance_result", {})
    if isinstance(finance_result, dict) and "model_params" in finance_result:
        m2.metric("TFT Params", f"{finance_result['model_params']:,}")
    else:
        m2.metric("TFT Params", "N/A (mock)")

    graph_result = st.session_state.get("graph_result", {})
    if isinstance(graph_result, dict) and "analysis" in graph_result:
        density = graph_result["analysis"].get("density", 0.0)
        m3.metric("Graph Density", f"{density:.3f}")
    else:
        m3.metric("Graph Density", "N/A")

    sae_result = st.session_state.get("sae_result")
    if sae_result:
        m4.metric("Active Concepts", f"{sae_result['active_concepts']}/{sae_result['total_concepts']}")
    else:
        m4.metric("Active Concepts", "N/A")

    m5.metric("Recon Error", f"{final_snap['reconstruction_error']:.6f}")

    stability = st.session_state.get("ukt_multirun_stability")
    if isinstance(stability, dict) and stability.get("n_runs", 0) > 0:
        st.caption(
            "UKT multi-run reality-regression stability "
            f"(n={stability['n_runs']}): mean cosine={stability['mean_cosine']:.3f}, "
            f"min cosine={stability['min_cosine']:.3f}, std={stability['std_cosine']:.3f}"
        )

    st.markdown("---")

    # ---- A3: Governance Flags Panel ----
    gov_flags = st.session_state.get("governance_flags", [])
    if gov_flags:
        st.markdown("### ⚠️ Governance Flags")
        st.caption(
            f"{len(gov_flags)} issue(s) detected. These are auto-generated alerts "
            "indicating potential data quality, bias, or coverage concerns."
        )
        for flag in gov_flags:
            severity = flag.get("severity", "warning")
            code = flag.get("code", "GOV-???")
            label = flag.get("label", "Unknown")
            description = flag.get("description", "")
            detail = flag.get("detail", "")
            icon = "⚠️" if severity == "warning" else "ℹ️"
            with st.expander(f'{icon} [{code}] {label}', expanded=True):
                st.markdown(f"**Definition:** {description}")
                if detail:
                    st.markdown(
                        f'<div class="contest-note"><strong>Detected:</strong> {detail}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            '<span class="gov-pass">✓ No governance flags detected</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---- B3: Interpretability Score Card ----
    scorecard = st.session_state.get("interpretability_scorecard", {})
    if scorecard:
        st.markdown("### 📊 Interpretability Accountability Score Card")
        st.caption(
            "Formal pass/fail audit against minimum governance thresholds. "
            "This score card is included in all exported reports."
        )
        sc_rows = []
        for key, item in scorecard.items():
            passed = item.get("passed", False)
            status_icon = "✅ PASS" if passed else "⚠️ WARN"
            sc_rows.append({
                "Dimension": item.get("label", key),
                "Value": f"{item.get('value', 'N/A')}{item.get('unit', '')}",
                "Threshold": f"≥{item.get('threshold', 'N/A')}{item.get('unit', '')}",
                "Status": status_icon,
                "Description": item.get("description", ""),
            })
        sc_df = pd.DataFrame(sc_rows)

        # Style: color Status column
        def _style_status(val: str) -> str:
            if "PASS" in val:
                return "color: #64ffda; font-weight: bold"
            return "color: #ffaa00; font-weight: bold"

        try:
            styled = sc_df.style.map(_style_status, subset=["Status"])
        except AttributeError:
            styled = sc_df.style.applymap(_style_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Overall pass/fail summary
        all_pass = all(item.get("passed", False) for item in scorecard.values())
        pass_count = sum(1 for item in scorecard.values() if item.get("passed", False))
        total_count = len(scorecard)
        if all_pass:
            st.success(f"Overall: {pass_count}/{total_count} criteria PASS — System meets minimum interpretability standards.")
        else:
            st.warning(f"Overall: {pass_count}/{total_count} criteria PASS — Review flagged items before citing conclusions.")

    st.markdown("---")

    # Kernel matrix visualization
    st.markdown("### Universal Kernel Matrix")
    block_names = [s["block_name"] for s in snapshots]
    fig_km = kernel_viz.plot_kernel_matrix(final_snap, block_names)
    st.plotly_chart(fig_km, use_container_width=True, key="dashboard_kernel_matrix")

    # Kernel importance + reality regression
    col1, col2 = st.columns(2)
    with col1:
        fig_imp = kernel_viz.plot_kernel_importance(final_snap)
        st.plotly_chart(fig_imp, use_container_width=True, key="dashboard_kernel_importance")
    with col2:
        fig_rr = kernel_viz.plot_reality_regression(final_snap)
        st.plotly_chart(fig_rr, use_container_width=True, key="dashboard_reality_regression")

    # Kernel evolution
    st.markdown("### Kernel Evolution Across Pipeline Steps")
    fig_evo = kernel_viz.plot_kernel_evolution(snapshots)
    st.plotly_chart(fig_evo, use_container_width=True, key="dashboard_kernel_evolution")

    # Semantic Canvas summary
    canvas_narrative = st.session_state.get("canvas_narrative")
    reality_narrative = st.session_state.get("reality_narrative")
    if canvas_narrative or reality_narrative:
        st.markdown("### Semantic Interpretability — LLM Narratives")
        if canvas_narrative:
            st.success(f"**Cross-Domain Narrative:** {canvas_narrative}")
        if reality_narrative:
            st.info(f"**Reality Assessment:** {reality_narrative}")

    # Semantic interpretation log
    st.markdown("### Semantic Interpretability Report")
    for snap in snapshots:
        with st.expander(f"Step {snap['step']}: {snap['block_name']}", expanded=False):
            st.markdown(snap["report"])
            # Show layer narrative if available
            if snap.get("layer_narrative"):
                st.info(f"**Layer narrative:** {snap['layer_narrative']}")
            for kl in snap["kernel_labels"]:
                st.markdown(
                    f'<span class="concept-badge">{kl["label"]}</span>',
                    unsafe_allow_html=True,
                )
                if kl.get("semantic_narrative"):
                    st.caption(f"Semantic: {kl['semantic_narrative']}")
                elif kl.get("narrative"):
                    st.caption(kl["narrative"])

    # Concept-kernel map (from SAE)
    concept_kernel_map = st.session_state.get("concept_kernel_map", [])
    if concept_kernel_map:
        st.markdown("### Concept-Kernel Correspondence (Sparse Autoencoder)")
        fig_ck = kernel_viz.plot_concept_kernel_map(concept_kernel_map)
        st.plotly_chart(fig_ck, use_container_width=True, key="dashboard_concept_kernel_map")
        st.dataframe(pd.DataFrame(concept_kernel_map), use_container_width=True)

    # Export — D1: stamped with run ID
    st.markdown("### Export")
    run_id = st.session_state.get("run_id", "UNKNOWN")
    run_ts = st.session_state.get("run_timestamp", "")
    exp1, exp2, exp3 = st.columns(3)

    # Build report with scorecard and flags
    report_lines = [
        f"# Hyperspace Governance Report",
        f"## Run ID: {run_id} | {run_ts}",
        "",
        "## Interpretability Score Card",
    ]
    if scorecard:
        for key, item in scorecard.items():
            status_str = "PASS" if item.get("passed") else "WARN"
            report_lines.append(
                f"- {item.get('label')}: {item.get('value')}{item.get('unit','')} "
                f"(threshold ≥{item.get('threshold')}{item.get('unit','')}) — {status_str}"
            )
    if gov_flags:
        report_lines.append("")
        report_lines.append("## Governance Flags")
        for flag in gov_flags:
            report_lines.append(f"- [{flag.get('code')}] {flag.get('label')}: {flag.get('detail', '')}")

    # Semantic narratives section
    canvas_narrative = st.session_state.get("canvas_narrative")
    reality_narrative = st.session_state.get("reality_narrative")
    if canvas_narrative or reality_narrative:
        report_lines.append("")
        report_lines.append("## Semantic Narratives (Tiny-LLM)")
        if canvas_narrative:
            report_lines.append(f"**Cross-Domain Narrative:** {canvas_narrative}")
        if reality_narrative:
            report_lines.append(f"**Reality Assessment:** {reality_narrative}")
        report_lines.append("")
        for snap in snapshots:
            if snap.get("layer_narrative"):
                report_lines.append(
                    f"**Step {snap['step']} ({snap['block_name']}):** "
                    f"{snap['layer_narrative']}"
                )

    report_lines.append("")
    report_lines.append("## Pipeline Reports")
    for snap in snapshots:
        report_lines.append(snap["report"])
    report_md = "\n".join(report_lines)

    exp1.download_button(
        "Download Report (Markdown)", report_md,
        f"hyperspace_report_{run_id}.md", "text/markdown",
        key="dashboard_download_report",
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
            key="dashboard_download_metrics",
        )

    # Score card CSV export
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
            key="dashboard_download_scorecard",
        )
