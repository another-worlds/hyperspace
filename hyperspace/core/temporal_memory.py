"""Temporal memory for cross-run kernel persistence and evolution tracking.

Persists UKT kernel decomposition snapshots (importance vectors, kernel
activations, reality regression vectors) across pipeline runs.  Provides
retrieval of historical kernel states for evolution analysis and governance
auditing.

Usage:
    memory = KernelMemory.load(path)       # or KernelMemory()
    memory.store_run(run_id, timestamp, snapshots)
    history = memory.get_kernel_history(block_name="Finance", last_n=5)
    evolution = memory.compute_kernel_evolution()
    memory.save(path)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class KernelSnapshot:
    """Minimal kernel state captured from one pipeline run."""

    run_id: str
    timestamp: str
    block_name: str
    step: int
    n_kernels: int
    importance: np.ndarray          # (n_kernels,)
    reality_regression: np.ndarray  # (feature_dim,)
    reconstruction_error: float
    kernel_activation_row: np.ndarray  # this block's row in kernel_activation matrix


@dataclass
class KernelEvolution:
    """Computed evolution statistics for a single block across runs."""

    block_name: str
    n_runs: int
    importance_trend: list[list[float]]    # per-run importance vectors
    reconstruction_trend: list[float]      # per-run reconstruction errors
    regression_cosines: list[float]        # pairwise cosines between consecutive runs
    mean_importance_stability: float       # mean cosine of consecutive importance vectors


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with zero-guard."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class KernelMemory:
    """Cross-run kernel snapshot store with evolution analysis.

    Stores a rolling window of per-block kernel snapshots and provides
    methods for tracking how kernel structure evolves over time.
    """

    def __init__(self, max_runs: int = 100) -> None:
        self._snapshots: list[KernelSnapshot] = []
        self._max_runs = max_runs
        self._run_ids: list[str] = []   # ordered unique run IDs

    @property
    def n_snapshots(self) -> int:
        return len(self._snapshots)

    @property
    def n_runs(self) -> int:
        return len(self._run_ids)

    @property
    def run_ids(self) -> list[str]:
        return list(self._run_ids)

    def store_run(
        self,
        run_id: str,
        timestamp: str,
        snapshots: list[dict],
    ) -> int:
        """Store kernel snapshots from a pipeline run.

        Args:
            run_id: Unique run identifier.
            timestamp: ISO-style timestamp string.
            snapshots: List of UKT snapshot dicts (as produced by PipelineRunner).

        Returns:
            Number of snapshots stored from this run.
        """
        stored = 0
        for snap in snapshots:
            importance = snap.get("importance")
            rr = snap.get("reality_regression")
            ka = snap.get("kernel_activation")
            block_name = snap.get("block_name", "")
            step = snap.get("step", 0)

            if importance is None or rr is None:
                continue

            importance = np.asarray(importance, dtype=np.float64)
            rr = np.asarray(rr, dtype=np.float64)

            # Extract this block's row from kernel_activation matrix
            ka_row = np.zeros(len(importance))
            if ka is not None:
                ka = np.asarray(ka)
                block_idx = step - 1
                if 0 <= block_idx < ka.shape[0]:
                    ka_row = ka[block_idx].astype(np.float64)

            self._snapshots.append(KernelSnapshot(
                run_id=run_id,
                timestamp=timestamp,
                block_name=block_name,
                step=step,
                n_kernels=int(snap.get("n_kernels", len(importance))),
                importance=importance,
                reality_regression=rr,
                reconstruction_error=float(snap.get("reconstruction_error", 0.0)),
                kernel_activation_row=ka_row,
            ))
            stored += 1

        if run_id not in self._run_ids:
            self._run_ids.append(run_id)

        # Enforce max_runs limit
        if len(self._run_ids) > self._max_runs:
            cutoff_ids = set(self._run_ids[:-self._max_runs])
            self._snapshots = [
                s for s in self._snapshots if s.run_id not in cutoff_ids
            ]
            self._run_ids = self._run_ids[-self._max_runs:]

        return stored

    def get_block_history(
        self,
        block_name: str,
        last_n: int | None = None,
    ) -> list[KernelSnapshot]:
        """Get chronological snapshots for a specific block.

        Args:
            block_name: Block to query (e.g. "Finance", "Clusters").
            last_n: If set, return only the most recent N snapshots.
        """
        history = [s for s in self._snapshots if s.block_name == block_name]
        if last_n is not None:
            history = history[-last_n:]
        return history

    def compute_kernel_evolution(
        self,
        block_name: str | None = None,
    ) -> dict[str, KernelEvolution]:
        """Compute kernel evolution statistics per block.

        Args:
            block_name: If set, compute only for this block. Otherwise all blocks.

        Returns:
            Dict mapping block_name → KernelEvolution.
        """
        block_names = (
            [block_name] if block_name
            else sorted({s.block_name for s in self._snapshots})
        )
        result: dict[str, KernelEvolution] = {}

        for bn in block_names:
            history = self.get_block_history(bn)
            if len(history) < 1:
                continue

            importance_trend = [s.importance.tolist() for s in history]
            reconstruction_trend = [s.reconstruction_error for s in history]

            # Consecutive cosines
            regression_cosines: list[float] = []
            importance_cosines: list[float] = []
            for i in range(1, len(history)):
                regression_cosines.append(
                    _cosine(history[i].reality_regression, history[i - 1].reality_regression)
                )
                importance_cosines.append(
                    _cosine(history[i].importance, history[i - 1].importance)
                )

            mean_imp_stability = (
                float(np.mean(importance_cosines)) if importance_cosines else 1.0
            )

            result[bn] = KernelEvolution(
                block_name=bn,
                n_runs=len(history),
                importance_trend=importance_trend,
                reconstruction_trend=reconstruction_trend,
                regression_cosines=regression_cosines,
                mean_importance_stability=round(mean_imp_stability, 4),
            )

        return result

    def get_evolution_summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary of kernel evolution for governance export."""
        evolutions = self.compute_kernel_evolution()
        summary: dict[str, Any] = {
            "n_runs": self.n_runs,
            "n_snapshots": self.n_snapshots,
            "blocks": {},
        }
        for bn, evo in evolutions.items():
            summary["blocks"][bn] = {
                "n_runs": evo.n_runs,
                "mean_importance_stability": evo.mean_importance_stability,
                "reconstruction_trend": evo.reconstruction_trend[-5:],
                "latest_importance": evo.importance_trend[-1] if evo.importance_trend else [],
            }
        return summary

    # ── Disk serialization ─────────────────────────────────────────────── #

    def save(self, path: str | Path) -> None:
        """Persist kernel memory to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_runs": self._max_runs,
            "run_ids": self._run_ids,
            "snapshots": [_snap_to_dict(s) for s in self._snapshots],
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "KernelMemory":
        """Load kernel memory from a previously saved JSON file."""
        path = Path(path)
        payload = json.loads(path.read_text())
        mem = cls(max_runs=payload.get("max_runs", 100))
        mem._run_ids = payload.get("run_ids", [])
        mem._snapshots = [_snap_from_dict(d) for d in payload.get("snapshots", [])]
        return mem


def _snap_to_dict(s: KernelSnapshot) -> dict[str, Any]:
    return {
        "run_id": s.run_id,
        "timestamp": s.timestamp,
        "block_name": s.block_name,
        "step": s.step,
        "n_kernels": s.n_kernels,
        "importance": s.importance.tolist(),
        "reality_regression": s.reality_regression.tolist(),
        "reconstruction_error": s.reconstruction_error,
        "kernel_activation_row": s.kernel_activation_row.tolist(),
    }


def _snap_from_dict(d: dict[str, Any]) -> KernelSnapshot:
    return KernelSnapshot(
        run_id=d["run_id"],
        timestamp=d["timestamp"],
        block_name=d["block_name"],
        step=d["step"],
        n_kernels=d["n_kernels"],
        importance=np.asarray(d["importance"], dtype=np.float64),
        reality_regression=np.asarray(d["reality_regression"], dtype=np.float64),
        reconstruction_error=float(d["reconstruction_error"]),
        kernel_activation_row=np.asarray(d["kernel_activation_row"], dtype=np.float64),
    )
