"""Kernel Knowledge Library — versioning, distillation, and transfer (SPEC-7).

Three-layer persistence system:
1. **Versioning**: Stable lineage tracking via cosine matching on Vt rows.
2. **Distillation**: Freeze ``KernelTemplate`` from kernels persisting 5+ runs.
3. **Transfer**: Warm-start SharedProjection with template Vt rows.

All data is JSON-serializable for session-state and optional disk persistence.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from hyperspace.config import (
    KERNEL_DISTILLATION_MIN_RUNS,
    KERNEL_LINEAGE_COSINE_THRESHOLD,
    KERNEL_STALE_THRESHOLD_RUNS,
)


# --------------------------------------------------------------------------- #
# Data classes                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class KernelLineage:
    """Tracks a single kernel identity across runs."""
    lineage_id: str
    label: str
    vt_history: list[list[float]] = field(default_factory=list)
    importance_history: list[float] = field(default_factory=list)
    narrative_history: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    first_seen_run: str = ""
    last_seen_run: str = ""
    consecutive_count: int = 0

    def to_dict(self) -> dict:
        return {
            "lineage_id": self.lineage_id,
            "label": self.label,
            "vt_history": self.vt_history,
            "importance_history": self.importance_history,
            "narrative_history": self.narrative_history,
            "run_ids": self.run_ids,
            "first_seen_run": self.first_seen_run,
            "last_seen_run": self.last_seen_run,
            "consecutive_count": self.consecutive_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KernelLineage":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class KernelTemplate:
    """Frozen kernel template distilled from stable lineages."""
    lineage_id: str
    frozen_vt: list[float]
    importance_range: tuple[float, float]
    stability_score: float
    dominant_region: str
    canonical_narrative: str
    first_seen_run: str
    run_count: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "lineage_id": self.lineage_id,
            "frozen_vt": self.frozen_vt,
            "importance_range": list(self.importance_range),
            "stability_score": self.stability_score,
            "dominant_region": self.dominant_region,
            "canonical_narrative": self.canonical_narrative,
            "first_seen_run": self.first_seen_run,
            "run_count": self.run_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KernelTemplate":
        d = dict(d)
        if "importance_range" in d and isinstance(d["importance_range"], list):
            d["importance_range"] = tuple(d["importance_range"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _determine_dominant_region(vt_row: np.ndarray) -> str:
    """Determine dominant region from a Vt row based on feature loadings."""
    # Region boundaries in the 80-dim UKT space
    regions = {
        "temporal-pattern": (0, 16),
        "semantic-embedding": (16, 32),
        "structural-centrality": (32, 48),
        "dynamic-agent": (48, 64),
        "geospatial-kernel": (64, 80),
    }
    best_region = "unknown"
    best_loading = 0.0
    for region_name, (start, end) in regions.items():
        if end <= len(vt_row):
            loading = float(np.sum(np.abs(vt_row[start:end])))
        elif start < len(vt_row):
            loading = float(np.sum(np.abs(vt_row[start:])))
        else:
            continue
        if loading > best_loading:
            best_loading = loading
            best_region = region_name
    return best_region


# --------------------------------------------------------------------------- #
# KernelLibrary                                                                 #
# --------------------------------------------------------------------------- #

class KernelLibrary:
    """Three-layer kernel persistence: versioning, distillation, transfer."""

    def __init__(self) -> None:
        self.lineages: dict[str, KernelLineage] = {}
        self.templates: list[KernelTemplate] = []
        self.archived: list[KernelTemplate] = []
        self._current_run_lineages: list[str] = []
        self._total_runs: int = 0

    # ----- Layer 1: Versioning -----

    def match_or_mint(
        self,
        vt_row: np.ndarray,
        label: str,
        importance: float,
        narrative: str,
        run_id: str,
    ) -> str:
        """Compare vt_row against known lineages. Assign or mint lineage_id."""
        best_id: str | None = None
        best_cos = -1.0

        for lid, lineage in self.lineages.items():
            if not lineage.vt_history:
                continue
            last_vt = np.array(lineage.vt_history[-1])
            cos = _cosine_similarity(vt_row, last_vt)
            if cos > best_cos:
                best_cos = cos
                best_id = lid

        if best_cos >= KERNEL_LINEAGE_COSINE_THRESHOLD and best_id is not None:
            return best_id

        # Mint new lineage
        new_id = str(uuid.uuid4())[:8]
        self.lineages[new_id] = KernelLineage(
            lineage_id=new_id,
            label=label,
            first_seen_run=run_id,
        )
        return new_id

    def record_run(
        self,
        lineage_id: str,
        importance: float,
        narrative: str,
        run_id: str,
        vt_row: np.ndarray | None = None,
    ) -> None:
        """Append run data to a lineage."""
        if lineage_id not in self.lineages:
            return
        lin = self.lineages[lineage_id]
        if vt_row is not None:
            lin.vt_history.append(vt_row.tolist() if hasattr(vt_row, "tolist") else list(vt_row))
        lin.importance_history.append(float(importance))
        lin.narrative_history.append(narrative or "")
        lin.run_ids.append(run_id)
        lin.last_seen_run = run_id
        lin.label = narrative[:80] if narrative else lin.label
        # Track consecutive count
        if lineage_id in self._current_run_lineages:
            lin.consecutive_count += 1
        else:
            lin.consecutive_count = 1
        self._current_run_lineages.append(lineage_id)

    def begin_run(self) -> None:
        """Call at the start of each pipeline run to reset per-run tracking."""
        self._current_run_lineages = []
        self._total_runs += 1

    # ----- Layer 2: Distillation -----

    def distill_eligible(self) -> None:
        """Create KernelTemplate for lineages with enough consecutive runs."""
        existing_template_ids = {t.lineage_id for t in self.templates}

        for lid, lin in self.lineages.items():
            if lid in existing_template_ids:
                continue
            if len(lin.vt_history) < KERNEL_DISTILLATION_MIN_RUNS:
                continue

            # Compute mean Vt from last N runs
            recent_vts = [np.array(v) for v in lin.vt_history[-KERNEL_DISTILLATION_MIN_RUNS:]]
            frozen_vt = np.mean(recent_vts, axis=0)

            # Stability score: mean cosine between consecutive runs
            cosines = []
            for i in range(1, len(recent_vts)):
                cosines.append(_cosine_similarity(recent_vts[i - 1], recent_vts[i]))
            stability = float(np.mean(cosines)) if cosines else 0.0

            imp_hist = lin.importance_history[-KERNEL_DISTILLATION_MIN_RUNS:]
            template = KernelTemplate(
                lineage_id=lid,
                frozen_vt=frozen_vt.tolist(),
                importance_range=(min(imp_hist), max(imp_hist)),
                stability_score=stability,
                dominant_region=_determine_dominant_region(frozen_vt),
                canonical_narrative=lin.narrative_history[-1] if lin.narrative_history else "",
                first_seen_run=lin.first_seen_run,
                run_count=len(lin.run_ids),
                metadata={
                    "distilled_at": datetime.now(timezone.utc).isoformat(),
                    "source_runs": lin.run_ids[-KERNEL_DISTILLATION_MIN_RUNS:],
                },
            )
            self.templates.append(template)

    # ----- Layer 3: Transfer -----

    def get_transfer_bases(self) -> list[np.ndarray]:
        """Return frozen Vt arrays weighted by stability for warm-start."""
        bases = []
        for t in self.templates:
            arr = np.array(t.frozen_vt)
            bases.append(arr * t.stability_score)
        return bases

    def prune_stale(self) -> None:
        """Archive templates not seen for too many runs."""
        if self._total_runs < KERNEL_STALE_THRESHOLD_RUNS:
            return
        stale = []
        active = []
        for t in self.templates:
            lineage = self.lineages.get(t.lineage_id)
            if lineage is None:
                stale.append(t)
                continue
            runs_since_last = self._total_runs - len(lineage.run_ids)
            if runs_since_last >= KERNEL_STALE_THRESHOLD_RUNS:
                stale.append(t)
            else:
                active.append(t)
        self.archived.extend(stale)
        self.templates = active

    # ----- Summary & Stats -----

    def get_library_summary(self) -> list[dict]:
        """Return summary dicts for UI display."""
        summary = []
        template_ids = {t.lineage_id for t in self.templates}
        for lid, lin in self.lineages.items():
            status = "template" if lid in template_ids else "active"
            summary.append({
                "lineage_id": lid,
                "label": lin.label[:60] if lin.label else "",
                "run_count": len(lin.run_ids),
                "status": status,
                "last_seen": lin.last_seen_run,
            })
        return summary

    def get_lineage_ids_for_snapshot(self) -> list[dict]:
        """Return lineage IDs from the current run for snapshot storage."""
        return [
            {"lineage_id": lid, "run_count": len(self.lineages[lid].run_ids)}
            for lid in self._current_run_lineages
            if lid in self.lineages
        ]

    # ----- Persistence -----

    def save(self, path: str) -> None:
        """Save library to JSON file."""
        data = {
            "lineages": {lid: lin.to_dict() for lid, lin in self.lineages.items()},
            "templates": [t.to_dict() for t in self.templates],
            "archived": [t.to_dict() for t in self.archived],
            "total_runs": self._total_runs,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "KernelLibrary":
        """Load library from JSON file."""
        lib = cls()
        try:
            with open(path) as f:
                data = json.load(f)
            lib.lineages = {
                lid: KernelLineage.from_dict(d)
                for lid, d in data.get("lineages", {}).items()
            }
            lib.templates = [KernelTemplate.from_dict(d) for d in data.get("templates", [])]
            lib.archived = [KernelTemplate.from_dict(d) for d in data.get("archived", [])]
            lib._total_runs = data.get("total_runs", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # Start fresh
        return lib
