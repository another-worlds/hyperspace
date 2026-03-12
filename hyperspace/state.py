"""Session state management and pipeline progress tracking."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import streamlit as st


def init_session_state() -> None:
    """Initialize all session state defaults."""
    defaults: dict[str, Any] = dict(
        # Pipeline control
        pipeline_launched=False,
        pipeline_complete=False,
        pipeline_step=None,
        pipeline_progress={},  # step_name -> {status, result, message}
        # Data sources
        data_sources={},  # block_name -> source label string
        # Block results
        finance_result=None,
        cluster_result=None,
        graph_result=None,
        spatial_result=None,
        sim_result=None,
        interpreter_result=None,
        # UKT snapshots (list of dicts, one per block)
        ukt_snapshots=[],
        # Final SAE result
        sae_result=None,
        # Concept-kernel correspondence map (from SAE, list of dicts)
        concept_kernel_map=[],
        # Raw data (cached across reruns)
        raw_ohlcv=None,
        raw_docs=None,
        raw_un_votes=None,
        # Stability result (written by run_pipeline, cleared by reset)
        ukt_multirun_stability=None,
        # Timeframe context (written by run_pipeline, cleared by reset)
        timeframe_context={},
        # Semantic Canvas (written by run_pipeline, cleared by reset)
        semantic_canvas=None,
        canvas_narrative=None,
        reality_narrative=None,
        # ------------------------------------------------------------------ #
        # Governance features
        # ------------------------------------------------------------------ #
        # D1: Unique run identifier + timestamp for non-repudiation
        run_id=None,
        run_timestamp=None,
        # A3: Auto-detected governance flags list[dict]
        # Each dict: {code, label, description, severity}
        governance_flags=[],
        # B3: Interpretability score card dict
        # Keys: feature_traceability, kernel_stability, concept_activation_rate,
        #       data_source_diversity, governance_flags_count
        interpretability_scorecard={},
        alignment_metrics={},
        # Interpretability contract reports
        interpretability_contract={},
        interpretability_contract_summary={
            "total_modules": 0,
            "compliant_modules": 0,
            "noncompliant_modules": 0,
            "compliance_rate": 0.0,
        },
        # B1: Policy language mode toggle
        policy_language_mode=False,
        # A1: Per-kernel contest annotations dict[kernel_id -> str]
        kernel_annotations={},
        # C1: Multi-stakeholder annotations list[dict]
        # Each dict: {kernel_id, role, text, timestamp}
        stakeholder_annotations=[],
        # B2: Counterfactual state
        counterfactual_result=None,
        counterfactual_removed_block=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def update_step(step_name: str, status: str, message: str = "",
                result: Any = None) -> None:
    """Update a pipeline step's status."""
    st.session_state.pipeline_progress[step_name] = dict(
        status=status, message=message, result=result,
    )
    if status == "running":
        st.session_state.pipeline_step = step_name


def step_status(step_name: str) -> str:
    """Get a step's current status (pending/running/done/error)."""
    info = st.session_state.pipeline_progress.get(step_name, {})
    return info.get("status", "pending")


def generate_run_id() -> tuple[str, str]:
    """Generate a new unique run ID and ISO timestamp string.

    Returns:
        (run_id, timestamp_str) — both stored in session state.
    """
    run_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    st.session_state.run_id = run_id
    st.session_state.run_timestamp = timestamp
    return run_id, timestamp


def reset_pipeline() -> None:
    """Reset pipeline state for a fresh run."""
    st.session_state.pipeline_launched = False
    st.session_state.pipeline_complete = False
    st.session_state.pipeline_step = None
    st.session_state.pipeline_progress = {}
    st.session_state.ukt_snapshots = []
    st.session_state.finance_result = None
    st.session_state.cluster_result = None
    st.session_state.graph_result = None
    st.session_state.spatial_result = None
    st.session_state.sim_result = None
    st.session_state.interpreter_result = None
    st.session_state.sae_result = None
    st.session_state.concept_kernel_map = []
    st.session_state.raw_ohlcv = None
    st.session_state.raw_docs = None
    st.session_state.raw_un_votes = None
    st.session_state.data_sources = {}
    st.session_state.ukt_multirun_stability = None
    st.session_state.timeframe_context = {}
    st.session_state.semantic_canvas = None
    st.session_state.canvas_narrative = None
    st.session_state.reality_narrative = None
    # Reset governance state
    st.session_state.run_id = None
    st.session_state.run_timestamp = None
    st.session_state.governance_flags = []
    st.session_state.interpretability_scorecard = {}
    st.session_state.alignment_metrics = {}
    st.session_state.interpretability_contract = {}
    st.session_state.interpretability_contract_summary = {
        "total_modules": 0,
        "compliant_modules": 0,
        "noncompliant_modules": 0,
        "compliance_rate": 0.0,
    }
    st.session_state.kernel_annotations = {}
    st.session_state.stakeholder_annotations = []
    st.session_state.counterfactual_result = None
    st.session_state.counterfactual_removed_block = None
    # Note: policy_language_mode persists across resets (user preference)
