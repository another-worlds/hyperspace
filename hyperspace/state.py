"""Session state management and pipeline progress tracking."""
from __future__ import annotations

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
    st.session_state.sim_result = None
    st.session_state.interpreter_result = None
    st.session_state.sae_result = None
    st.session_state.concept_kernel_map = []
    st.session_state.raw_docs = None
    st.session_state.raw_un_votes = None
    st.session_state.data_sources = {}
