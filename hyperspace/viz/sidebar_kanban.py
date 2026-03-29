"""Sidebar kanban-style progress cards for pipeline blocks.

Renders compact status cards in the sidebar showing per-block status,
timing, data sources, and governance context. Integrates with
PipelineProgressTracker via st.session_state.pipeline_tracker.

Cards sync in real-time with the pipeline: PENDING before launch,
RUNNING during execution (with progress bars), COMPLETED/FAILED after.
"""
from __future__ import annotations

import streamlit as st

from hyperspace.viz.pipeline_progress import BlockStatus, PipelineProgressTracker


# Card definitions matching dashboard.py block names exactly
_KANBAN_CARDS = [
    {
        "key": "data_fetch",
        "label": "Data Fetch",
        "icon_label": "01",
        "context": "Market, news, political, spatial data",
    },
    {
        "key": "model_training",
        "label": "Model Training",
        "icon_label": "02",
        "context": "TFT + BERTopic models",
    },
    {
        "key": "core_pipeline",
        "label": "Core Pipeline",
        "icon_label": "03",
        "context": "Graph, agents, interpretation",
    },
    {
        "key": "governance_analysis",
        "label": "Governance",
        "icon_label": "04",
        "context": "Flags, compliance, audit trail",
    },
    {
        "key": "visualization",
        "label": "Visualization",
        "icon_label": "05",
        "context": "Charts, narratives, export",
    },
]

_STATUS_ICONS = {
    BlockStatus.PENDING: ("\u25CB", "pending"),
    BlockStatus.RUNNING: ("\u25C9", "running"),
    BlockStatus.COMPLETED: ("\u25CF", "complete"),
    BlockStatus.FAILED: ("\u2717", "failed"),
    BlockStatus.SKIPPED: ("\u2298", "skipped"),
}

# Source map: block key → data_sources dict key
_SOURCE_MAP = {
    "data_fetch": "Finance",
    "model_training": "Models",
    "core_pipeline": "Graph",
    "governance_analysis": "Governance",
    "visualization": "Visualization",
}


def _render_card_html(
    label: str,
    icon: str,
    status_class: str,
    status_text: str,
    context: str,
    source: str = "",
    timing: str = "",
) -> str:
    """Build HTML for a single kanban card with progress bar."""
    source_line = ""
    if source:
        source_line = f'<span class="kanban-source">{source}</span><br>'

    timing_line = ""
    if timing:
        timing_line = f'<span class="kanban-timing">{timing}</span>'

    # Progress bar fill class
    fill_class = f"fill-{status_class}"

    return (
        f'<div class="kanban-card kanban-card-{status_class}">'
        f'  <div class="kanban-header">'
        f'    <span class="kanban-icon kanban-icon-{status_class}">{icon}</span>'
        f'    <span class="kanban-title">{label}</span>'
        f'    {timing_line}'
        f'  </div>'
        f'  <span class="kanban-status">{status_text}</span><br>'
        f'  {source_line}'
        f'  <span class="kanban-context">{context}</span>'
        f'  <div class="kanban-progress-bar">'
        f'    <div class="kanban-progress-fill {fill_class}"></div>'
        f'  </div>'
        f'</div>'
    )


def render_sidebar_kanban() -> None:
    """Render kanban progress cards in the sidebar.

    Reads pipeline state from session_state to determine card status.
    Pre-pipeline: all PENDING. During pipeline: live RUNNING/COMPLETED.
    Post-pipeline: shows timing + sources + governance flags.
    """
    tracker: PipelineProgressTracker | None = st.session_state.get("pipeline_tracker")
    data_sources: dict = st.session_state.get("data_sources", {})
    pipeline_complete: bool = st.session_state.get("pipeline_complete", False)
    pipeline_launched: bool = st.session_state.get("pipeline_launched", False)
    run_id = st.session_state.get("run_id")
    run_ts = st.session_state.get("run_timestamp")

    # Run ID header (compact)
    if run_id:
        st.markdown(
            f'<span class="run-id-watermark">Run {run_id} \u00b7 {run_ts}</span>',
            unsafe_allow_html=True,
        )

    cards_html = []
    completed_count = 0
    total_count = len(_KANBAN_CARDS)

    for card_def in _KANBAN_CARDS:
        key = card_def["key"]
        label = card_def["label"]
        context = card_def["context"]
        timing = ""

        # Determine block status from tracker
        if tracker is not None:
            block = tracker.get_block(key)
            if block is not None:
                icon, status_class = _STATUS_ICONS.get(
                    block.status, ("\u25CB", "pending")
                )
                if block.status == BlockStatus.COMPLETED:
                    status_text = "Complete"
                    timing = f"{block.duration_sec:.1f}s"
                    completed_count += 1
                elif block.status == BlockStatus.FAILED:
                    status_text = f"Failed \u2014 {block.error_msg or 'unknown'}"
                    completed_count += 1
                elif block.status == BlockStatus.RUNNING:
                    status_text = "Running\u2026"
                elif block.status == BlockStatus.SKIPPED:
                    status_text = "Skipped"
                    completed_count += 1
                else:
                    status_text = "Pending"

                # Use block governance context if available
                if block.governance_context:
                    context = block.governance_context
            else:
                icon, status_class = "\u25CB", "pending"
                status_text = "Pending"
        elif not pipeline_launched:
            icon, status_class = "\u25CB", "pending"
            status_text = "Awaiting launch"
        else:
            icon, status_class = "\u25CB", "pending"
            status_text = "Pending"

        # Data source for this block (post-pipeline)
        source = ""
        if pipeline_complete and data_sources:
            src_key = _SOURCE_MAP.get(key, "")
            source = data_sources.get(src_key, "")

        cards_html.append(
            _render_card_html(
                label, icon, status_class, status_text,
                context, source, timing,
            )
        )

    # Overall progress indicator
    if pipeline_launched and tracker is not None:
        frac = tracker.progress_fraction
        pct = int(frac * 100)
        if pipeline_complete:
            bar_color = "#34d399"
            bar_label = f"Complete \u2014 {tracker.total_duration_sec:.1f}s total"
        else:
            bar_color = "#4da6ff"
            bar_label = f"{pct}% \u2014 {completed_count}/{total_count} blocks"

        st.markdown(
            f'<div style="margin-bottom:8px;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<span style="font-size:0.68rem; color:#8ab4cc; font-weight:600;">'
            f'Pipeline Progress</span>'
            f'<span style="font-size:0.65rem; color:#64ffda; '
            f'font-family:\'JetBrains Mono\',monospace;">{bar_label}</span>'
            f'</div>'
            f'<div style="height:4px; background:#1a3a5c; border-radius:2px; '
            f'margin-top:4px; overflow:hidden;">'
            f'<div style="height:100%; width:{pct}%; background:{bar_color}; '
            f'border-radius:2px; transition:width 0.4s ease;"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("\n".join(cards_html), unsafe_allow_html=True)

    # Governance flags summary (inline with cards)
    gov_flags = st.session_state.get("governance_flags", [])
    if pipeline_complete:
        if gov_flags:
            st.markdown(
                f'<span class="gov-flag">\u26A0 {len(gov_flags)} governance flag(s)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="gov-pass">\u2713 No flags</span>',
                unsafe_allow_html=True,
            )
