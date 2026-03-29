"""Temporal World-Model Memory — GRU transition encoder (SPEC-5).

Lightweight GRU that learns state transitions over UKT history.  Given the
current reality regression + optional intervention vector, predicts the next
run's reality regression.

Disabled by default (``ENABLE_TEMPORAL_MEMORY=False``).  Online learning after
each pipeline run.  Requires a minimum of 3 runs before predictions are
surfaced, and 5 consecutive high-confidence runs before governance reports
include temporal predictions.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from hyperspace.config import (
    TEMPORAL_ENCODER_HIDDEN_DIM,
    TEMPORAL_GOVERNANCE_CONFIDENCE,
    TEMPORAL_GOVERNANCE_MIN_RUNS,
    TEMPORAL_MIN_RUNS,
    UKT_FEATURE_DIM,
)


class TemporalWorldModel:
    """GRU-based transition encoder for reality regression prediction.

    Pure numpy implementation — no torch dependency at runtime.
    """

    def __init__(
        self,
        input_dim: int = UKT_FEATURE_DIM,
        hidden_dim: int = TEMPORAL_ENCODER_HIDDEN_DIM,
        seed: int = 42,
    ) -> None:
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        # GRU parameters (reset gate, update gate, candidate)
        scale_ih = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (hidden_dim + hidden_dim))

        self.W_ir = rng.normal(0, scale_ih, (input_dim, hidden_dim)).astype(np.float32)
        self.W_hr = rng.normal(0, scale_hh, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_r = np.zeros(hidden_dim, dtype=np.float32)

        self.W_iz = rng.normal(0, scale_ih, (input_dim, hidden_dim)).astype(np.float32)
        self.W_hz = rng.normal(0, scale_hh, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_z = np.zeros(hidden_dim, dtype=np.float32)

        self.W_in = rng.normal(0, scale_ih, (input_dim, hidden_dim)).astype(np.float32)
        self.W_hn = rng.normal(0, scale_hh, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_n = np.zeros(hidden_dim, dtype=np.float32)

        # Output projection: hidden → input_dim (predict next reality regression)
        scale_out = np.sqrt(2.0 / (hidden_dim + input_dim))
        self.W_out = rng.normal(0, scale_out, (hidden_dim, input_dim)).astype(np.float32)
        self.b_out = np.zeros(input_dim, dtype=np.float32)

        # Hidden state
        self._h = np.zeros(hidden_dim, dtype=np.float32)

        # History tracking
        self._states: list[np.ndarray] = []
        self._predictions: list[np.ndarray] = []
        self._errors: list[float] = []
        self._confidence_history: list[float] = []

        self._lr = 0.001

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _gru_step(self, x: np.ndarray) -> np.ndarray:
        """Single GRU step: (input_dim,) → updated hidden state (hidden_dim,)."""
        r = self._sigmoid(x @ self.W_ir + self._h @ self.W_hr + self.b_r)
        z = self._sigmoid(x @ self.W_iz + self._h @ self.W_hz + self.b_z)
        n = np.tanh(x @ self.W_in + (r * self._h) @ self.W_hn + self.b_n)
        self._h = (1 - z) * n + z * self._h
        return self._h.copy()

    def predict_next(
        self,
        current_state: np.ndarray,
        intervention: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Predict next-run reality regression from current state.

        Args:
            current_state: Current reality regression (80,).
            intervention: Optional intervention vector (not used in v1).

        Returns:
            Dict with 'prediction', 'confidence', 'has_sufficient_history',
            'is_governance_ready'.
        """
        if not self.has_sufficient_history():
            return {
                "prediction": None,
                "confidence": 0.0,
                "has_sufficient_history": False,
                "is_governance_ready": False,
            }

        x = current_state.astype(np.float32)
        h = self._gru_step(x)
        prediction = h @ self.W_out + self.b_out

        # Confidence: based on recent prediction error trend
        confidence = self._compute_confidence()

        return {
            "prediction": prediction,
            "confidence": confidence,
            "has_sufficient_history": True,
            "is_governance_ready": self.is_governance_ready(),
        }

    def update(
        self,
        current_state: np.ndarray,
        actual_next_state: np.ndarray | None = None,
    ) -> None:
        """Online learning step: record current state and update from previous prediction.

        Call this at the end of each pipeline run. If we made a prediction last
        run, compute error and do a gradient step.
        """
        self._states.append(current_state.copy())

        if actual_next_state is not None and self._predictions:
            last_pred = self._predictions[-1]
            error = float(np.linalg.norm(actual_next_state - last_pred))
            self._errors.append(error)

            # Simple SGD on output projection (approximation)
            residual = (actual_next_state - last_pred).astype(np.float32)
            # Gradient of MSE w.r.t. W_out: h^T * residual
            if len(self._states) >= 2:
                x_prev = self._states[-2].astype(np.float32)
                h_prev = self._h.copy()
                # Approximate: update output layer only (avoids BPTT complexity)
                self.W_out += self._lr * np.outer(h_prev, residual)
                self.b_out += self._lr * residual

        # Make prediction for next run
        if self.has_sufficient_history():
            x = current_state.astype(np.float32)
            h = self._gru_step(x)
            pred = h @ self.W_out + self.b_out
            self._predictions.append(pred)

            confidence = self._compute_confidence()
            self._confidence_history.append(confidence)

    def has_sufficient_history(self) -> bool:
        """Whether we have enough runs to make predictions."""
        return len(self._states) >= TEMPORAL_MIN_RUNS

    def is_governance_ready(self) -> bool:
        """Whether predictions are reliable enough for governance reports."""
        if len(self._confidence_history) < TEMPORAL_GOVERNANCE_MIN_RUNS:
            return False
        recent = self._confidence_history[-TEMPORAL_GOVERNANCE_MIN_RUNS:]
        return all(c >= TEMPORAL_GOVERNANCE_CONFIDENCE for c in recent)

    def _compute_confidence(self) -> float:
        """Compute prediction confidence based on recent error trend."""
        if not self._errors:
            return 0.0
        recent = self._errors[-5:]
        # Normalise errors: lower error → higher confidence
        mean_error = np.mean(recent)
        # Map error to confidence via sigmoid-like transform
        confidence = float(1.0 / (1.0 + mean_error))
        return min(confidence, 1.0)

    def get_summary(self) -> dict[str, Any]:
        """Return summary for UI display."""
        return {
            "n_runs": len(self._states),
            "has_sufficient_history": self.has_sufficient_history(),
            "is_governance_ready": self.is_governance_ready(),
            "confidence": self._confidence_history[-1] if self._confidence_history else 0.0,
            "mean_error": float(np.mean(self._errors[-5:])) if self._errors else None,
            "n_predictions": len(self._predictions),
        }
