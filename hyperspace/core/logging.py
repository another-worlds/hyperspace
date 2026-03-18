"""Structured logging for Hyperspace system events.

Provides audit trails for:
- Governance flag triggers
- SAE training and caching
- Pipeline execution
- Counterfactual runs
- Error conditions

All logs are stored in session state for export in governance reports.
"""
from __future__ import annotations

from enum import Enum
from datetime import datetime
import json
import streamlit as st
from typing import Any


class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEvent:
    """Structured log event."""

    def __init__(
        self,
        event_type: str,
        level: LogLevel = LogLevel.INFO,
        message: str = "",
        context: dict | None = None,
        timestamp: datetime | None = None,
    ):
        """Create a log event.

        Args:
            event_type: Category (e.g., "governance_flag", "sae_training", "pipeline_step")
            level: Severity level
            message: Human-readable description
            context: Structured data (e.g., governance flag code, block name, error)
            timestamp: Event timestamp (auto-filled if None)
        """
        self.event_type = event_type
        self.level = level
        self.message = message
        self.context = context or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "level": self.level.value,
            "message": self.message,
            "context": self.context,
        }

    def __repr__(self) -> str:
        ctx = f" | {json.dumps(self.context)}" if self.context else ""
        return f"[{self.level.value}] {self.event_type}: {self.message}{ctx}"


class StructuredLogger:
    """Central logging sink for Hyperspace events."""

    SESSION_STATE_KEY = "hyperspace_logs"
    MAX_LOGS = 500  # Keep last 500 events per session

    @staticmethod
    def log(
        event_type: str,
        level: LogLevel = LogLevel.INFO,
        message: str = "",
        context: dict | None = None,
    ) -> LogEvent:
        """Record a structured log event.

        Args:
            event_type: Category
            level: Severity level
            message: Description
            context: Structured context data

        Returns:
            The log event
        """
        event = LogEvent(event_type, level, message, context)

        # Store in session state
        if StructuredLogger.SESSION_STATE_KEY not in st.session_state:
            st.session_state[StructuredLogger.SESSION_STATE_KEY] = []

        logs = st.session_state[StructuredLogger.SESSION_STATE_KEY]
        logs.append(event)

        # Keep only recent logs
        if len(logs) > StructuredLogger.MAX_LOGS:
            logs = logs[-StructuredLogger.MAX_LOGS :]
            st.session_state[StructuredLogger.SESSION_STATE_KEY] = logs

        return event

    @staticmethod
    def info(
        event_type: str,
        message: str = "",
        context: dict | None = None,
    ) -> LogEvent:
        """Log an informational event."""
        return StructuredLogger.log(event_type, LogLevel.INFO, message, context)

    @staticmethod
    def warning(
        event_type: str,
        message: str = "",
        context: dict | None = None,
    ) -> LogEvent:
        """Log a warning event."""
        return StructuredLogger.log(event_type, LogLevel.WARNING, message, context)

    @staticmethod
    def error(
        event_type: str,
        message: str = "",
        context: dict | None = None,
    ) -> LogEvent:
        """Log an error event."""
        return StructuredLogger.log(event_type, LogLevel.ERROR, message, context)

    @staticmethod
    def critical(
        event_type: str,
        message: str = "",
        context: dict | None = None,
    ) -> LogEvent:
        """Log a critical event."""
        return StructuredLogger.log(event_type, LogLevel.CRITICAL, message, context)

    @staticmethod
    def get_logs(
        event_type: str | None = None,
        level: LogLevel | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Retrieve logs with optional filtering.

        Args:
            event_type: Filter by event type
            level: Filter by log level
            limit: Return only last N logs

        Returns:
            List of log event dicts
        """
        logs = st.session_state.get(StructuredLogger.SESSION_STATE_KEY, [])

        # Filter
        if event_type:
            logs = [e for e in logs if e.event_type == event_type]
        if level:
            logs = [e for e in logs if e.level == level]

        # Limit
        if limit:
            logs = logs[-limit:]

        return [e.to_dict() for e in logs]

    @staticmethod
    def clear_logs() -> None:
        """Clear all logs from session state."""
        st.session_state[StructuredLogger.SESSION_STATE_KEY] = []

    @staticmethod
    def export_json() -> str:
        """Export all logs as JSON."""
        logs = st.session_state.get(StructuredLogger.SESSION_STATE_KEY, [])
        return json.dumps(
            [e.to_dict() for e in logs],
            indent=2,
            default=str,  # Handle datetime
        )


# Convenience functions
def log_governance_flag(
    flag_code: str,
    label: str,
    description: str,
) -> None:
    """Log detection of a governance flag.

    Args:
        flag_code: GOV-001, GOV-002, etc.
        label: Human-readable label
        description: Detailed explanation
    """
    StructuredLogger.warning(
        "governance_flag",
        f"Flag {flag_code}: {label}",
        {
            "flag_code": flag_code,
            "label": label,
            "description": description,
        }
    )


def log_sae_training(
    cached: bool,
    hidden_dim: int,
    epochs: int,
    duration_sec: float | None = None,
) -> None:
    """Log SAE training event."""
    StructuredLogger.info(
        "sae_training",
        f"SAE training {'(CACHED)' if cached else '(trained)'}",
        {
            "cached": cached,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "duration_sec": duration_sec,
        }
    )


def log_pipeline_step(
    block_name: str,
    status: str,  # "started", "completed", "failed"
    duration_sec: float | None = None,
    error_msg: str | None = None,
) -> None:
    """Log a pipeline block execution."""
    level = LogLevel.ERROR if status == "failed" else LogLevel.INFO
    message = f"Pipeline block '{block_name}': {status}"
    if error_msg:
        message += f" - {error_msg}"

    StructuredLogger.log(
        "pipeline_step",
        level,
        message,
        {
            "block_name": block_name,
            "status": status,
            "duration_sec": duration_sec,
            "error": error_msg,
        }
    )


def log_counterfactual_run(
    removed_block: str,
    shock_injected: bool,
    duration_sec: float | None = None,
) -> None:
    """Log a counterfactual analysis run."""
    StructuredLogger.info(
        "counterfactual_run",
        f"Counterfactual: removed '{removed_block}'",
        {
            "removed_block": removed_block,
            "shock_injected": shock_injected,
            "duration_sec": duration_sec,
        }
    )


def log_error(
    component: str,
    error_msg: str,
    error_type: str | None = None,
) -> None:
    """Log an error condition."""
    StructuredLogger.error(
        "application_error",
        f"{component}: {error_msg}",
        {
            "component": component,
            "error_type": error_type,
            "error_message": error_msg,
        }
    )
