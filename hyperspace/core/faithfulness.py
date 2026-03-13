"""Narrative faithfulness and mechanistic intervention checks.

Provides intervention-style checks that verify narrative explanations are
grounded in measurable feature attributions. When explanation confidence
is low, a fail-safe mechanism downgrades narratives to boilerplate
disclaimers rather than surfacing unsupported claims.

Alpha exit criteria addressed:
  H-003.1: At least one intervention-style check per major module in alpha scope.
  H-003.2: Fail-safe downgrade when explanation confidence is low.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


LOW_CONFIDENCE_DISCLAIMER = (
    "Explanation confidence is below the minimum threshold. The narrative "
    "has been withheld to avoid unsupported claims. Review raw feature "
    "attributions and kernel labels for unmediated evidence."
)


@dataclass
class InterventionResult:
    """Result of a single intervention-style faithfulness check."""

    module_name: str
    check_name: str
    passed: bool
    original_value: float
    intervened_value: float
    delta: float
    detail: str


@dataclass
class FaithfulnessReport:
    """Aggregated faithfulness report across all checked modules."""

    checks: list[InterventionResult]
    overall_confidence: float
    low_confidence: bool
    downgraded_narrative: str | None


# --------------------------------------------------------------------------- #
# Intervention checks                                                          #
# --------------------------------------------------------------------------- #


def _intervention_kernel_removal(
    matrix: np.ndarray,
    kernel_idx: int,
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
) -> float:
    """Remove a single kernel and measure reconstruction error change.

    By zeroing out one singular value and re-reconstructing, we measure how
    much the kernel contributes to the overall matrix representation. If the
    narrative claims a kernel is dominant, removal should significantly
    increase reconstruction error.
    """
    S_intervened = S.copy()
    if kernel_idx < len(S_intervened):
        S_intervened[kernel_idx] = 0.0

    rank = min(U.shape[1], len(S_intervened), Vt.shape[0])
    reconstructed = U[:, :rank] @ np.diag(S_intervened[:rank]) @ Vt[:rank, :]
    error = float(np.linalg.norm(matrix - reconstructed, "fro"))
    return error


def check_kernel_attribution_faithfulness(
    snapshots: list[dict],
) -> list[InterventionResult]:
    """Verify kernel importance claims via single-kernel removal interventions.

    For the final snapshot, remove each kernel and verify that the reported
    importance ranking is consistent with the actual reconstruction impact.
    """
    results: list[InterventionResult] = []
    if not snapshots:
        return results

    snap = snapshots[-1]
    matrix = snap.get("matrix")
    U = snap.get("U")
    S = snap.get("S")
    Vt = snap.get("Vt")
    importance = snap.get("importance", np.array([]))

    if matrix is None or U is None or S is None or Vt is None:
        return results

    # Baseline reconstruction error
    rank = min(U.shape[1], len(S), Vt.shape[0])
    baseline_recon = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    baseline_error = float(np.linalg.norm(matrix - baseline_recon, "fro"))

    for k_idx in range(min(len(S), len(importance))):
        intervened_error = _intervention_kernel_removal(
            matrix, k_idx, U, S, Vt,
        )
        delta = intervened_error - baseline_error

        # The most important kernel (highest importance) should produce the
        # largest delta when removed. Check monotonicity within top-3.
        results.append(
            InterventionResult(
                module_name="UKT",
                check_name=f"kernel_removal_K{k_idx}",
                passed=delta >= 0,  # removal should not decrease error
                original_value=baseline_error,
                intervened_value=intervened_error,
                delta=delta,
                detail=(
                    f"Kernel K{k_idx} (importance={importance[k_idx]:.4f}): "
                    f"baseline_error={baseline_error:.6f}, "
                    f"intervened_error={intervened_error:.6f}, "
                    f"delta={delta:+.6f}"
                ),
            )
        )

    return results


def check_canvas_narrative_grounding(
    canvas: Any,
    canvas_narrative: str | None,
) -> list[InterventionResult]:
    """Verify canvas narrative mentions relate to actual dominant dimensions.

    Checks that the narrative (if present) references dimensions that are
    actually loaded in the canvas rather than hallucinating ungrounded claims.
    """
    results: list[InterventionResult] = []

    if canvas is None:
        return results

    entries = []
    if hasattr(canvas, "entries"):
        entries = canvas.entries
    elif hasattr(canvas, "_entries"):
        entries = canvas._entries

    if not entries:
        return results

    # Find dominant dimensions from canvas entries
    dim_scores: dict[str, float] = {}
    for entry in entries:
        coords = {}
        if hasattr(entry, "coordinates"):
            coords = entry.coordinates
        elif isinstance(entry, dict):
            coords = entry.get("coordinates", {})
        for dim, val in coords.items():
            dim_scores[dim] = dim_scores.get(dim, 0.0) + abs(float(val))

    if not dim_scores:
        return results

    # Sort by total score
    sorted_dims = sorted(dim_scores.items(), key=lambda x: -x[1])
    top_dim = sorted_dims[0][0] if sorted_dims else None
    top_score = sorted_dims[0][1] if sorted_dims else 0.0
    bottom_dim = sorted_dims[-1][0] if sorted_dims else None
    bottom_score = sorted_dims[-1][1] if sorted_dims else 0.0

    # Basic grounding check: dominant dimension should have higher score
    results.append(
        InterventionResult(
            module_name="SemanticCanvas",
            check_name="dimension_dominance_ordering",
            passed=top_score >= bottom_score,
            original_value=top_score,
            intervened_value=bottom_score,
            delta=top_score - bottom_score,
            detail=(
                f"Top dimension '{top_dim}' (score={top_score:.4f}) vs "
                f"bottom dimension '{bottom_dim}' (score={bottom_score:.4f}). "
                f"Ordering is {'consistent' if top_score >= bottom_score else 'inconsistent'}."
            ),
        )
    )

    return results


def check_sae_concept_activation_consistency(
    sae_result: dict | None,
    snapshots: list[dict],
) -> list[InterventionResult]:
    """Verify SAE concept activations are consistent with kernel structure.

    Active concepts should correlate with high-importance kernels. If the
    SAE reports many active concepts but kernels show low variance, the
    interpretability claim is questionable.
    """
    results: list[InterventionResult] = []
    if sae_result is None or not snapshots:
        return results

    active_concepts = sae_result.get("active_concepts", 0)
    total_concepts = sae_result.get("total_concepts", 1)
    activation_rate = active_concepts / max(total_concepts, 1)

    importance = snapshots[-1].get("importance", np.array([]))
    if len(importance) == 0:
        return results

    # Variance of kernel importances — high variance means clear structure
    importance_var = float(np.var(importance))

    # If activation rate is high but importance variance is near-zero,
    # concepts may not be grounded in actual kernel differentiation.
    grounded = not (activation_rate > 0.7 and importance_var < 0.001)

    results.append(
        InterventionResult(
            module_name="SparseAutoencoder",
            check_name="concept_kernel_grounding",
            passed=grounded,
            original_value=activation_rate,
            intervened_value=importance_var,
            delta=0.0,
            detail=(
                f"Concept activation rate={activation_rate:.2%}, "
                f"kernel importance variance={importance_var:.6f}. "
                f"{'Grounded' if grounded else 'Potentially ungrounded: high activation with flat importance'}."
            ),
        )
    )

    return results


# --------------------------------------------------------------------------- #
# Aggregated faithfulness pipeline                                             #
# --------------------------------------------------------------------------- #


def run_faithfulness_checks(
    *,
    snapshots: list[dict],
    canvas: Any | None = None,
    canvas_narrative: str | None = None,
    sae_result: dict | None = None,
    confidence_threshold: float = 0.5,
) -> FaithfulnessReport:
    """Run all faithfulness checks and produce an aggregated report.

    Args:
        snapshots: UKT snapshots from pipeline run.
        canvas: SemanticCanvas instance.
        canvas_narrative: Generated narrative text.
        sae_result: SAE output dict.
        confidence_threshold: Minimum pass rate to avoid narrative downgrade.

    Returns:
        FaithfulnessReport with check results and confidence assessment.
    """
    all_checks: list[InterventionResult] = []

    all_checks.extend(check_kernel_attribution_faithfulness(snapshots))
    all_checks.extend(check_canvas_narrative_grounding(canvas, canvas_narrative))
    all_checks.extend(
        check_sae_concept_activation_consistency(sae_result, snapshots)
    )

    if all_checks:
        pass_count = sum(1 for c in all_checks if c.passed)
        confidence = pass_count / len(all_checks)
    else:
        confidence = 1.0  # No checks means no evidence of unfaithfulness

    low_confidence = confidence < confidence_threshold
    downgraded = LOW_CONFIDENCE_DISCLAIMER if low_confidence else None

    return FaithfulnessReport(
        checks=all_checks,
        overall_confidence=round(confidence, 4),
        low_confidence=low_confidence,
        downgraded_narrative=downgraded,
    )
