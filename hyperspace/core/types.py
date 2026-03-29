"""Unified type definitions for the Hyperspace pipeline.

Provides TypedDicts and protocols that enforce consistent interfaces
across all pipeline blocks, data fetchers, and model outputs.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

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


class InterpretabilityContractReport(TypedDict):
    """Validation report for one InterpretableModule implementation."""
    status: Literal["compliant", "noncompliant", "not_applicable"]
    compliant: bool
    interface_issues: list[str]
    payload_issues: list[str]
    na_reason: str | None
    na_owner: str | None


class InterpretabilityContractSummary(TypedDict):
    """Aggregated compliance summary across reported modules."""
    total_modules: int
    compliant_modules: int
    na_modules: int
    noncompliant_modules: int
    compliance_rate: float


class AlphaScopeModulePolicy(TypedDict):
    """Interpretability policy metadata for one alpha-scope module."""
    module_name: str
    owner: str
    status: Literal["contract", "not_applicable"]
    rationale: str


class FaithfulnessCheckResult(TypedDict):
    """Result of a single intervention-style faithfulness check."""
    module_name: str
    check_name: str
    passed: bool
    original_value: float
    intervened_value: float
    delta: float
    detail: str


class FaithfulnessReportDict(TypedDict, total=False):
    """Serializable faithfulness report for pipeline output."""
    checks: list[FaithfulnessCheckResult]
    overall_confidence: float
    low_confidence: bool
    downgraded_narrative: str | None


class DriftResultDict(TypedDict, total=False):
    """Serializable drift result for pipeline output."""
    regression_cosine: float
    regression_l2: float
    importance_cosine: float
    importance_l2: float
    stability_delta: float
    n_kernel_delta: int
    window_size: int
    n_records: int
    alerts: list[dict]


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

    # Cross-block interconnection
    uvt_result: dict | None
    use_result: dict | None

    # Governance
    stability: StabilityResult | None
    governance_flags: list[GovernanceFlag]
    interpretability_scorecard: dict[str, ScorecardEntry]
    alignment_metrics: dict[str, Any]

    # Provenance
    run_id: str
    run_timestamp: str

    # Interpretability contract coverage
    interpretability_contract: dict[str, InterpretabilityContractReport]
    interpretability_contract_summary: InterpretabilityContractSummary

    # Faithfulness checks (H-003)
    faithfulness_report: FaithfulnessReportDict | None

    # Temporal drift (H-002)
    drift_result: DriftResultDict | None

    # Kernel evolution (temporal memory)
    kernel_evolution: dict[str, Any] | None

    # Latent space versioning (Phase 3 governance)
    latent_version: dict[str, Any] | None

    # SPEC-5: Temporal world-model prediction
    temporal_prediction: dict[str, Any] | None


@runtime_checkable
class InterpretableModule(Protocol):
    """Intrinsic interpretability contract for core model components.

    Any module that claims interpretability support should expose a consistent
    interface for latent unit export, attribution, modality alignment reports,
    and structured explanations.
    """

    def export_latent_units(self) -> dict[str, Any]:
        """Return named latent units/subspaces/concepts exposed by the module."""

    def export_feature_attributions(self, input_batch: Any | None = None) -> dict[str, Any]:
        """Return feature-level attributions for the latest state or input."""

    def export_alignment_report(self, reference_modalities: list[str] | None = None) -> dict[str, Any]:
        """Return modality alignment diagnostics in a machine-readable form."""

    def explain_prediction(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return structured and narrative explanation for current outputs."""


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




def build_interpretable_report(module: Any, module_name: str) -> InterpretabilityContractReport:
    """Return a structured interpretability compliance report."""
    interface_issues = validate_interpretable_module(module, module_name)
    # Only run payload validation if the interface is compliant. Otherwise,
    # payload checks may raise or record confusing errors that are actually
    # caused by interface issues (e.g., missing or uncallable methods).
    if interface_issues:
        payload_issues: list[str] = []
    else:
        payload_issues = validate_interpretable_payload(module, module_name)
    return InterpretabilityContractReport(
        status=("compliant" if len(interface_issues) == 0 and len(payload_issues) == 0 else "noncompliant"),
        compliant=(len(interface_issues) == 0 and len(payload_issues) == 0),
        interface_issues=interface_issues,
        payload_issues=payload_issues,
        na_reason=None,
        na_owner=None,
    )


def build_not_applicable_report(
    *, module_name: str, rationale: str, owner: str,
) -> InterpretabilityContractReport:
    """Return an explicit N/A report for alpha-scope modules."""
    return InterpretabilityContractReport(
        status="not_applicable",
        compliant=False,
        interface_issues=[],
        payload_issues=[],
        na_reason=f"{module_name}: {rationale}",
        na_owner=owner,
    )


def summarize_interpretable_reports(
    reports: dict[str, InterpretabilityContractReport],
) -> InterpretabilityContractSummary:
    """Summarize module-level interpretability compliance reports."""
    total = len(reports)
    compliant = sum(1 for r in reports.values() if r.get("status") == "compliant")
    na_modules = sum(1 for r in reports.values() if r.get("status") == "not_applicable")
    noncompliant = total - compliant - na_modules
    rate = (compliant / total) if total > 0 else 0.0
    return InterpretabilityContractSummary(
        total_modules=total,
        compliant_modules=compliant,
        na_modules=na_modules,
        noncompliant_modules=noncompliant,
        compliance_rate=round(float(rate), 4),
    )


def validate_alpha_scope_contract_coverage(
    reports: dict[str, InterpretabilityContractReport],
    alpha_scope_modules: list[AlphaScopeModulePolicy],
) -> list[str]:
    """Ensure every alpha-scope module is compliant or explicit N/A."""
    issues: list[str] = []
    expected = {m["module_name"]: m for m in alpha_scope_modules}

    for module_name, module_policy in expected.items():
        report = reports.get(module_name)
        if report is None:
            issues.append(f"{module_name}: missing interpretability contract report")
            continue

        status = report.get("status")
        if status not in {"compliant", "noncompliant", "not_applicable"}:
            issues.append(f"{module_name}: invalid status '{status}'")
            continue

        if status == "not_applicable":
            if not report.get("na_reason"):
                issues.append(f"{module_name}: N/A status missing rationale")
            if not report.get("na_owner"):
                issues.append(f"{module_name}: N/A status missing owner")

        if module_policy["status"] == "contract" and status == "not_applicable":
            issues.append(f"{module_name}: expected contract compliance, got N/A")

        if module_policy["status"] == "not_applicable" and status != "not_applicable":
            issues.append(f"{module_name}: expected explicit N/A policy")

    extra_modules = set(reports) - set(expected)
    if extra_modules:
        issues.append(f"Unexpected modules in interpretability contract: {sorted(extra_modules)}")

    return issues


def validate_interpretable_payload(module: Any, module_name: str) -> list[str]:
    """Validate that interpretability methods return expected payload shapes."""
    issues: list[str] = []

    try:
        latent = module.export_latent_units()
        if not isinstance(latent, dict):
            issues.append(f"{module_name}: export_latent_units must return dict")
    except Exception as exc:  # pragma: no cover - defensive runtime check
        issues.append(f"{module_name}: export_latent_units raised {type(exc).__name__}")

    try:
        attr = module.export_feature_attributions()
        if not isinstance(attr, dict):
            issues.append(f"{module_name}: export_feature_attributions must return dict")
        elif "attributions" not in attr:
            issues.append(f"{module_name}: export_feature_attributions missing 'attributions'")
        elif not isinstance(attr.get("attributions"), list):
            issues.append(f"{module_name}: 'attributions' must be a list")
    except Exception as exc:  # pragma: no cover - defensive runtime check
        issues.append(f"{module_name}: export_feature_attributions raised {type(exc).__name__}")

    try:
        align = module.export_alignment_report()
        if not isinstance(align, dict):
            issues.append(f"{module_name}: export_alignment_report must return dict")
    except Exception as exc:  # pragma: no cover - defensive runtime check
        issues.append(f"{module_name}: export_alignment_report raised {type(exc).__name__}")

    try:
        expl = module.explain_prediction()
        if not isinstance(expl, dict):
            issues.append(f"{module_name}: explain_prediction must return dict")
        elif "summary" not in expl:
            issues.append(f"{module_name}: explain_prediction missing 'summary'")
    except Exception as exc:  # pragma: no cover - defensive runtime check
        issues.append(f"{module_name}: explain_prediction raised {type(exc).__name__}")

    return issues


def validate_interpretable_module(module: Any, module_name: str) -> list[str]:
    """Validate that an object satisfies the InterpretableModule contract."""
    missing: list[str] = []
    required = [
        "export_latent_units",
        "export_feature_attributions",
        "export_alignment_report",
        "explain_prediction",
    ]
    for method_name in required:
        method = getattr(module, method_name, None)
        if method is None or not callable(method):
            missing.append(method_name)

    if missing:
        return [f"{module_name}: missing interpretability methods {missing}"]
    return []
