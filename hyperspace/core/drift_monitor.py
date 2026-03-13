"""Temporal drift monitor for kernel regression and representation stability.

Tracks reality regression vectors and kernel stability metrics across multiple
pipeline runs. Computes windowed drift statistics and raises governance alerts
when drift exceeds configurable thresholds.

Usage:
    monitor = DriftMonitor()
    monitor.record_run(run_id, reality_regression, stability, kernel_labels)
    drift = monitor.compute_drift()
    alerts = monitor.check_alert_thresholds(drift)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DriftRecord:
    """Snapshot of one pipeline run for drift tracking."""

    run_id: str
    timestamp: str
    reality_regression: np.ndarray
    kernel_importances: np.ndarray
    mean_cosine_stability: float
    n_kernels: int


@dataclass
class DriftResult:
    """Computed drift statistics between recent windows."""

    regression_cosine: float
    regression_l2: float
    importance_cosine: float
    importance_l2: float
    stability_delta: float
    n_kernel_delta: int
    window_size: int
    n_records: int


@dataclass
class DriftAlert:
    """A single drift alert with code, message, and measured value."""

    code: str
    label: str
    severity: str
    detail: str
    measured: float
    threshold: float


# Default alerting thresholds
DRIFT_THRESHOLDS: dict[str, dict[str, Any]] = {
    "DRIFT-001": {
        "label": "Reality Regression Drift",
        "metric": "regression_cosine",
        "direction": "below",
        "threshold": 0.85,
        "severity": "warning",
        "description": (
            "Cosine similarity between current and prior-window reality "
            "regression vectors dropped below threshold."
        ),
    },
    "DRIFT-002": {
        "label": "Kernel Importance Redistribution",
        "metric": "importance_cosine",
        "direction": "below",
        "threshold": 0.80,
        "severity": "warning",
        "description": (
            "Kernel importance distribution shifted significantly between "
            "consecutive runs."
        ),
    },
    "DRIFT-003": {
        "label": "Stability Degradation",
        "metric": "stability_delta",
        "direction": "below",
        "threshold": -0.10,
        "severity": "warning",
        "description": (
            "Multi-run kernel stability dropped by more than the allowed "
            "delta between consecutive runs."
        ),
    },
}


class DriftMonitor:
    """Windowed drift tracker across pipeline runs.

    Stores a rolling history of DriftRecords and computes drift statistics
    by comparing the most recent record against the prior window.
    """

    def __init__(self, max_history: int = 50) -> None:
        self._history: list[DriftRecord] = []
        self._max_history = max_history

    @property
    def history(self) -> list[DriftRecord]:
        return list(self._history)

    @property
    def n_records(self) -> int:
        return len(self._history)

    def record_run(
        self,
        run_id: str,
        timestamp: str,
        reality_regression: np.ndarray,
        kernel_importances: np.ndarray,
        mean_cosine_stability: float,
        n_kernels: int,
    ) -> None:
        """Append a new run record to the drift history."""
        record = DriftRecord(
            run_id=run_id,
            timestamp=timestamp,
            reality_regression=np.asarray(reality_regression, dtype=np.float64),
            kernel_importances=np.asarray(kernel_importances, dtype=np.float64),
            mean_cosine_stability=float(mean_cosine_stability),
            n_kernels=int(n_kernels),
        )
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def compute_drift(self, window: int = 1) -> DriftResult | None:
        """Compute drift between the latest record and the prior window.

        Args:
            window: Number of prior records to average for the reference.
                1 means compare against the immediately preceding run.

        Returns:
            DriftResult or None if insufficient history.
        """
        if len(self._history) < 2:
            return None

        current = self._history[-1]
        prior_records = self._history[-(1 + window) : -1]
        if not prior_records:
            return None

        # Average prior-window regression vectors
        prior_regressions = np.stack(
            [r.reality_regression for r in prior_records]
        )
        prior_mean_reg = prior_regressions.mean(axis=0)

        # Average prior-window importances
        prior_importances = np.stack(
            [r.kernel_importances for r in prior_records]
        )
        prior_mean_imp = prior_importances.mean(axis=0)

        # Average prior stability
        prior_mean_stability = float(
            np.mean([r.mean_cosine_stability for r in prior_records])
        )
        prior_mean_nk = int(
            np.mean([r.n_kernels for r in prior_records])
        )

        return DriftResult(
            regression_cosine=_cosine_sim(
                current.reality_regression, prior_mean_reg
            ),
            regression_l2=float(
                np.linalg.norm(current.reality_regression - prior_mean_reg)
            ),
            importance_cosine=_cosine_sim(
                current.kernel_importances, prior_mean_imp
            ),
            importance_l2=float(
                np.linalg.norm(current.kernel_importances - prior_mean_imp)
            ),
            stability_delta=current.mean_cosine_stability - prior_mean_stability,
            n_kernel_delta=current.n_kernels - prior_mean_nk,
            window_size=len(prior_records),
            n_records=len(self._history),
        )

    def check_alert_thresholds(
        self,
        drift: DriftResult | None = None,
        thresholds: dict[str, dict[str, Any]] | None = None,
    ) -> list[DriftAlert]:
        """Evaluate drift against configurable alert thresholds.

        Args:
            drift: Pre-computed DriftResult (or None to auto-compute).
            thresholds: Override default DRIFT_THRESHOLDS.

        Returns:
            List of DriftAlert objects for triggered conditions.
        """
        if drift is None:
            drift = self.compute_drift()
        if drift is None:
            return []

        thresholds = thresholds or DRIFT_THRESHOLDS
        alerts: list[DriftAlert] = []

        for code, spec in thresholds.items():
            metric_name = spec["metric"]
            measured = getattr(drift, metric_name, None)
            if measured is None:
                continue

            threshold = spec["threshold"]
            triggered = False
            if spec["direction"] == "below" and measured < threshold:
                triggered = True
            elif spec["direction"] == "above" and measured > threshold:
                triggered = True

            if triggered:
                alerts.append(
                    DriftAlert(
                        code=code,
                        label=spec["label"],
                        severity=spec["severity"],
                        detail=(
                            f"{spec['description']} "
                            f"Measured={measured:.4f}, threshold={threshold}"
                        ),
                        measured=float(measured),
                        threshold=float(threshold),
                    )
                )

        return alerts

    def export_history_rows(self) -> list[dict[str, object]]:
        """Export drift history as flat dicts for CSV/DataFrame export."""
        rows: list[dict[str, object]] = []
        for i, rec in enumerate(self._history):
            row: dict[str, object] = {
                "index": i,
                "run_id": rec.run_id,
                "timestamp": rec.timestamp,
                "mean_cosine_stability": rec.mean_cosine_stability,
                "n_kernels": rec.n_kernels,
                "regression_norm": float(np.linalg.norm(rec.reality_regression)),
            }
            # Pairwise drift vs previous
            if i > 0:
                prev = self._history[i - 1]
                row["regression_cosine_vs_prev"] = _cosine_sim(
                    rec.reality_regression, prev.reality_regression
                )
                row["importance_cosine_vs_prev"] = _cosine_sim(
                    rec.kernel_importances, prev.kernel_importances
                )
            else:
                row["regression_cosine_vs_prev"] = None
                row["importance_cosine_vs_prev"] = None
            rows.append(row)
        return rows


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, with zero-guard."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
