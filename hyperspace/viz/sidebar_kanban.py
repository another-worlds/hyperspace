"""Sidebar kanban-style progress cards for pipeline blocks.

Renders compact status cards in the sidebar showing per-block status,
timing, data sources, and governance context. Integrates with
PipelineProgressTracker via st.session_state.pipeline_tracker.
"""
from __future__ import annotations

import streamlit as st

from hyperspace.viz.pipeline_progress import BlockStatus, PipelineProgressTracker


# Card definitions matching dashboard.py block names exactly
_KANBAN_CARDS = [
    {
        "key": "data_fetch",
        "label": "Data Fetch",
        "context": "Market, news, political, spatial data",
    },
    {
        "key": "model_training",
        "label": "Model Training",
        "context": "TFT + BERTopic models",
    },
    {
        "key": "core_pipeline",
        "label": "Core Pipeline",
        "context": "Graph, agents, interpretation",
    },
    {
        "key": "governance_analysis",
        "label": "Governance",
        "context": "Flags, compliance, audit trail",
    },
]

_STATUS_ICONS = {
    BlockStatus.PENDING: ("○", "pending"),
    BlockStatus.RUNNING: ("◉", "running"),
    BlockStatus.COMPLETED: ("●", "complete"),
    BlockStatus.FAILED: ("✗", "failed"),
    BlockStatus.SKIPPED: ("⊘", "skipped"),
}


def _render_card_html(
    label: str,
    icon: str,
    status_class: str,
    status_text: str,
    context: str,
    source: str = "",
) -> str:
    """Build HTML for a single kanban card."""
    source_line = ""
    if source:
        source_line = f'<span class="kanban-source">{source}</span><br>'

    return (
        f'<div class="kanban-card kanban-card-{status_class}">'
        f'  <div class="kanban-header">'
        f'    <span class="kanban-icon kanban-icon-{status_class}">{icon}</span>'
        f'    <span class="kanban-title">{label}</span>'
        f'  </div>'
        f'  <span class="kanban-status">{status_text}</span><br>'
        f'  {source_line}'
        f'  <span class="kanban-context">{context}</span>'
        f'</div>'
    )


def render_sidebar_kanban() -> None:
    """Render kanban progress cards in the sidebar.

    Reads pipeline state from session_state to determine card status.
    Pre-pipeline: all PENDING. Post-pipeline: shows timing + sources.
    """
    tracker: PipelineProgressTracker | None = st.session_state.get("pipeline_tracker")
    data_sources: dict = st.session_state.get("data_sources", {})
    pipeline_complete: bool = st.session_state.get("pipeline_complete", False)
    run_id = st.session_state.get("run_id")
    run_ts = st.session_state.get("run_timestamp")

    # Run ID header (compact)
    if run_id:
        st.markdown(
            f'<span class="run-id-watermark">Run {run_id} · {run_ts}</span>',
            unsafe_allow_html=True,
        )

    cards_html = []
    for card_def in _KANBAN_CARDS:
        key = card_def["key"]
        label = card_def["label"]
        context = card_def["context"]

        # Determine block status from tracker
        if tracker is not None:
            block = tracker.get_block(key)
            if block is not None:
                icon, status_class = _STATUS_ICONS.get(
                    block.status, ("○", "pending")
                )
                if block.status == BlockStatus.COMPLETED:
                    status_text = f"Complete ({block.duration_sec:.1f}s)"
                elif block.status == BlockStatus.FAILED:
                    status_text = f"Failed — {block.error_msg or 'unknown error'}"
                elif block.status == BlockStatus.RUNNING:
                    status_text = "Running..."
                elif block.status == BlockStatus.SKIPPED:
                    status_text = "Skipped"
                else:
                    status_text = "Pending"

                # Use block governance context if richer than default
                if block.governance_context:
                    context = block.governance_context
            else:
                icon, status_class = "○", "pending"
                status_text = "Pending"
        else:
            icon, status_class = "○", "pending"
            status_text = "Pending"

        # Data source for this block (map block keys to data_sources keys)
        source = ""
        if pipeline_complete and data_sources:
            _SOURCE_MAP = {
                "data_fetch": "Finance",
                "model_training": "Models",
                "core_pipeline": "Graph",
                "governance_analysis": "Governance",
            }
            src_key = _SOURCE_MAP.get(key, "")
            source = data_sources.get(src_key, "")

        cards_html.append(
            _render_card_html(label, icon, status_class, status_text, context, source)
        )

    st.markdown("\n".join(cards_html), unsafe_allow_html=True)

    # Governance flags summary (inline with cards)
    gov_flags = st.session_state.get("governance_flags", [])
    if pipeline_complete:
        if gov_flags:
            st.markdown(
                f'<span class="gov-flag">⚠ {len(gov_flags)} governance flag(s)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="gov-pass">✓ No flags</span>',
                unsafe_allow_html=True,
            )
