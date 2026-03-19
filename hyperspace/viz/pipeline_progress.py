"""Pipeline progress visualization with per-block timing and status tracking.

Replaces the generic st.status() spinner with detailed per-source progress,
timing breakdown, and governance context for each pipeline step.
"""
from __future__ import annotations

import streamlit as st
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time


class BlockStatus(str, Enum):
    """Status of a pipeline block."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BlockProgress:
    """Progress tracking for a single pipeline block."""
    block_name: str
    status: BlockStatus = BlockStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_sec: float = 0.0
    message: str = ""
    error_msg: str | None = None
    governance_context: str = ""  # e.g., "Analyzing market signals and regime changes"

    @property
    def is_running(self) -> bool:
        return self.status == BlockStatus.RUNNING

    @property
    def is_complete(self) -> bool:
        return self.status in (BlockStatus.COMPLETED, BlockStatus.FAILED, BlockStatus.SKIPPED)

    def start(self, message: str = "", context: str = "") -> None:
        """Mark block as started."""
        self.status = BlockStatus.RUNNING
        self.start_time = datetime.now()
        self.message = message
        self.governance_context = context

    def complete(self, message: str = "") -> None:
        """Mark block as completed."""
        self.status = BlockStatus.COMPLETED
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_sec = (self.end_time - self.start_time).total_seconds()
        self.message = message

    def fail(self, error_msg: str) -> None:
        """Mark block as failed."""
        self.status = BlockStatus.FAILED
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_sec = (self.end_time - self.start_time).total_seconds()
        self.error_msg = error_msg

    def skip(self, message: str = "") -> None:
        """Mark block as skipped."""
        self.status = BlockStatus.SKIPPED
        self.end_time = datetime.now()
        self.message = message


@dataclass
class PipelineProgressTracker:
    """Track progress across all pipeline blocks."""
    blocks: dict[str, BlockProgress] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def add_block(
        self,
        block_name: str,
        governance_context: str = "",
    ) -> BlockProgress:
        """Add a block to tracking."""
        block = BlockProgress(block_name, governance_context=governance_context)
        self.blocks[block_name] = block
        return block

    def get_block(self, block_name: str) -> BlockProgress | None:
        """Get a block by name."""
        return self.blocks.get(block_name)

    @property
    def total_duration_sec(self) -> float:
        """Total elapsed time."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def completed_blocks(self) -> list[BlockProgress]:
        """Get all completed blocks."""
        return [b for b in self.blocks.values() if b.is_complete]

    @property
    def progress_fraction(self) -> float:
        """Fraction of blocks completed."""
        if not self.blocks:
            return 0.0
        completed = sum(1 for b in self.blocks.values() if b.is_complete)
        return completed / len(self.blocks)

    def finalize(self) -> None:
        """Mark pipeline as complete."""
        self.end_time = datetime.now()


def render_pipeline_progress(tracker: PipelineProgressTracker) -> None:
    """Render the pipeline progress visualization.

    Args:
        tracker: PipelineProgressTracker instance with block data
    """
    # Progress bar
    progress = tracker.progress_fraction
    col1, col2 = st.columns([4, 1])
    with col1:
        st.progress(progress, text=f"{progress:.0%} complete")
    with col2:
        st.metric("Elapsed", f"{tracker.total_duration_sec:.1f}s")

    # Per-block status
    st.markdown("**Block Status:**")
    for block_name in tracker.blocks.keys():
        block = tracker.blocks[block_name]
        _render_block_status(block)

    # Summary metrics
    if tracker.completed_blocks:
        st.markdown("**Timing Breakdown:**")
        timing_cols = st.columns(min(4, len(tracker.completed_blocks)))
        for i, block in enumerate(tracker.completed_blocks):
            with timing_cols[i % len(timing_cols)]:
                st.metric(
                    block.block_name,
                    f"{block.duration_sec:.1f}s",
                    delta=None,
                    label_visibility="collapsed" if i > 0 else "visible"
                )


def _render_block_status(block: BlockProgress) -> None:
    """Render a single block's status."""
    if block.status == BlockStatus.COMPLETED:
        icon = "✓"
        color = "#34d399"
    elif block.status == BlockStatus.RUNNING:
        icon = "⏳"
        color = "#fbbf24"
    elif block.status == BlockStatus.FAILED:
        icon = "✗"
        color = "#ff6b6b"
    elif block.status == BlockStatus.SKIPPED:
        icon = "⊘"
        color = "#8b8b8b"
    else:
        icon = "○"
        color = "#7a9ab8"

    # Format timing
    timing_str = f"{block.duration_sec:.1f}s" if block.duration_sec > 0 else ""

    # Build status line
    status_line = f"{icon} **{block.block_name}** — {block.message}"
    if timing_str:
        status_line += f" ({timing_str})"
    if block.governance_context:
        status_line += f"  \n*{block.governance_context}*"

    if block.status == BlockStatus.FAILED:
        st.error(status_line)
        if block.error_msg:
            st.caption(f"Error: {block.error_msg}")
    elif block.status == BlockStatus.COMPLETED:
        st.success(status_line)
    elif block.status == BlockStatus.RUNNING:
        st.warning(status_line)
    else:
        st.info(status_line)


def render_timing_summary(tracker: PipelineProgressTracker) -> None:
    """Render a summary of block timings."""
    if not tracker.completed_blocks:
        return

    st.markdown("### Execution Timeline")

    # Create timeline visualization
    blocks_sorted = sorted(tracker.completed_blocks, key=lambda b: b.duration_sec, reverse=True)
    total_time = sum(b.duration_sec for b in blocks_sorted)

    for block in blocks_sorted:
        fraction = block.duration_sec / total_time if total_time > 0 else 0
        bar_width = int(fraction * 40)  # Scale to ~40 chars
        bar = "█" * bar_width + "░" * (40 - bar_width)
        status_emoji = "✓" if block.status == BlockStatus.COMPLETED else "✗"
        st.write(f"{status_emoji} {block.block_name:<20} {bar} {block.duration_sec:>6.1f}s ({fraction*100:>5.1f}%)")

    st.caption(f"**Total time:** {tracker.total_duration_sec:.1f}s")


class StreamlitProgressContext:
    """Context manager for tracking pipeline progress in Streamlit.

    Usage:
        tracker = PipelineProgressTracker()
        with StreamlitProgressContext(tracker, "data_fetch", "Fetching financial data") as block:
            # ... do work ...
            block.complete("Data fetched successfully")
    """

    def __init__(
        self,
        tracker: PipelineProgressTracker,
        block_name: str,
        message: str = "",
        governance_context: str = "",
    ):
        self.tracker = tracker
        self.block_name = block_name
        self.message = message
        self.context = governance_context
        self.block = tracker.get_block(block_name) or tracker.add_block(block_name, governance_context)

    def __enter__(self) -> BlockProgress:
        self.block.start(self.message, self.context)
        return self.block

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.block.fail(str(exc_val))
        else:
            if not self.block.is_complete:
                self.block.complete()
        return False
