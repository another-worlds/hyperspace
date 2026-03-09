"""Unified type definitions for the Hyperspace pipeline.

Provides TypedDicts and protocols that enforce consistent interfaces
across all pipeline blocks, data fetchers, and model outputs.
"""
from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


# --------------------------------------------------------------------------- #
# Block result: every model block must return this structure                    #
# --------------------------------------------------------------------------- #

class BlockResult(TypedDict, total=False):
    """Standardised output from any pipeline block.

    Required keys (total=False allows gradual adoption):
        features_for_ukt: 80-dim feature vector for the UKT.
        feature_meta:     Per-index metadata dict for governance traceability.
        data_source:      Human-readable label of the data source used.

    Optional keys are block-specific (model objects, raw outputs, etc).
    """
    features_for_ukt: np.ndarray
    feature_meta: dict[int, dict]
    data_source: str


class SnapshotResult(TypedDict):
    """One UKT snapshot produced by UniversalKnowledgeTensor.add_block()."""
    step: int
    block_name: str
    matrix: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    n_kernels: int
    importance: np.ndarray
    kernel_activation: np.ndarray
    reality_regression: np.ndarray
    kernel_labels: list[dict]
    reconstruction_error: float
    report: str
    feature_meta: dict[int, dict]
    timeframe_context: dict
    stage_sae_result: dict | None
    canvas_entry: Any
    layer_narrative: str | None


class StabilityResult(TypedDict):
    """Result from estimate_reality_regression_stability()."""
    n_runs: int
    mean_cosine: float
    min_cosine: float
    std_cosine: float


class GovernanceFlag(TypedDict):
    """A single auto-detected governance flag."""
    code: str
    label: str
    description: str
    severity: str
    detail: str


class ScorecardEntry(TypedDict):
    """One dimension of the interpretability scorecard."""
    label: str
    value: float
    threshold: float
    unit: str
    passed: bool
    description: str


class PipelineResult(TypedDict, total=False):
    """Full pipeline output — everything needed to render the UI.

    This is the single return type of PipelineRunner.run().
    """
    # Core outputs
    snapshots: list[SnapshotResult]
    final_matrix: np.ndarray | None
    data_sources: dict[str, str]
    timeframe_context: dict

    # Block results
    finance_result: dict | None
    cluster_result: dict | None
    graph_result: dict | None
    spatial_result: dict | None
    sim_result: dict | None

    # Interpretation
    sae_result: dict | None
    concept_kernel_map: list[dict]
    semantic_canvas: Any
    canvas_narrative: str | None
    reality_narrative: str | None

    # Governance
    stability: StabilityResult | None
    governance_flags: list[GovernanceFlag]
    interpretability_scorecard: dict[str, ScorecardEntry]

    # Provenance
    run_id: str
    run_timestamp: str


# --------------------------------------------------------------------------- #
# Validation helpers                                                           #
# --------------------------------------------------------------------------- #

def validate_block_result(result: dict, block_name: str) -> list[str]:
    """Validate that a block result has the required keys.

    Returns list of warning messages (empty = valid).
    """
    warnings = []
    if result is None:
        return [f"{block_name}: result is None"]

    if "features_for_ukt" not in result:
        warnings.append(f"{block_name}: missing 'features_for_ukt'")
    else:
        feat = result["features_for_ukt"]
        if not isinstance(feat, np.ndarray):
            warnings.append(f"{block_name}: features_for_ukt is not ndarray")
        elif feat.shape != (80,):
            warnings.append(
                f"{block_name}: features_for_ukt shape is {feat.shape}, expected (80,)"
            )

    if "feature_meta" not in result:
        warnings.append(f"{block_name}: missing 'feature_meta'")

    if "data_source" not in result:
        warnings.append(f"{block_name}: missing 'data_source'")

    return warnings


def validate_snapshot(snapshot: dict) -> list[str]:
    """Validate that a UKT snapshot has all required keys."""
    required = [
        "step", "block_name", "matrix", "U", "S", "Vt",
        "n_kernels", "importance", "kernel_activation",
        "reality_regression", "kernel_labels", "reconstruction_error",
        "report", "feature_meta",
    ]
    missing = [k for k in required if k not in snapshot]
    if missing:
        return [f"Snapshot missing keys: {missing}"]
    return []
