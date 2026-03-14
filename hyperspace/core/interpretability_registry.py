"""Alpha-scope interpretability contract registry and report builders."""
from __future__ import annotations

from typing import Any

from hyperspace.core.types import (
    AlphaScopeModulePolicy,
    InterpretabilityContractReport,
    build_interpretable_report,
    build_not_applicable_report,
    validate_alpha_scope_contract_coverage,
)

ALPHA_SCOPE_MODULE_POLICIES: list[AlphaScopeModulePolicy] = [
    {
        "module_name": "UniversalKnowledgeTensor",
        "owner": "Core Platform",
        "status": "contract",
        "rationale": "Core latent integration matrix; must expose full interpretability contract.",
    },
    {
        "module_name": "SemanticCanvas",
        "owner": "Core Platform",
        "status": "contract",
        "rationale": "Semantic projection surface powering governance narratives; must be contract-compliant.",
    },
    {
        "module_name": "GraphEngine",
        "owner": "Geopolitics Modeling",
        "status": "not_applicable",
        "rationale": "Current production graph block is function-oriented and does not maintain a persistent model object implementing module endpoints.",
    },
    {
        "module_name": "SpatialKernels",
        "owner": "Spatial Modeling",
        "status": "not_applicable",
        "rationale": "Spatial kernel extraction is a pure feature-construction function with no standalone latent module state to export.",
    },
    {
        "module_name": "AgentSimulation",
        "owner": "Agent Systems",
        "status": "not_applicable",
        "rationale": "Simulation block is procedural and returns trajectory artifacts, not an interpretable module instance.",
    },
    {
        "module_name": "SparseAutoencoder",
        "owner": "Interpretability",
        "status": "not_applicable",
        "rationale": "SAE is currently trained via stateless helper functions; no retained module object exists in the production path.",
    },
    {
        "module_name": "UniversalVarianceTensor",
        "owner": "Cross-Block Modeling",
        "status": "not_applicable",
        "rationale": "UVT execution path returns tensors/metrics from a functional API and does not expose a persistent interpretable module interface.",
    },
    {
        "module_name": "UniversalSemanticEncoding",
        "owner": "Cross-Block Modeling",
        "status": "not_applicable",
        "rationale": "USE is produced as derived outputs from helper functions without a stable module object for endpoint contracts.",
    },
    {
        "module_name": "SharedLatentHead",
        "owner": "Core Platform",
        "status": "not_applicable",
        "rationale": "Shadow-only prototype behind feature flag; does not control production decisions or narratives.",
    },
    {
        "module_name": "DriftMonitor",
        "owner": "Governance",
        "status": "not_applicable",
        "rationale": "Diagnostic monitoring service; emits alerts and history but is not a predictive module with latent state to interpret.",
    },
    {
        "module_name": "KernelMemory",
        "owner": "Governance",
        "status": "not_applicable",
        "rationale": "Persistence layer for kernel snapshot history; provides evolution analytics but no model-level predictions to explain.",
    },
]


def build_alpha_scope_contract_reports(
    module_instances: dict[str, Any],
) -> dict[str, InterpretabilityContractReport]:
    """Build contract/N/A reports for all registered alpha-scope modules."""
    reports: dict[str, InterpretabilityContractReport] = {}

    for policy in ALPHA_SCOPE_MODULE_POLICIES:
        module_name = policy["module_name"]
        if policy["status"] == "contract":
            module_instance = module_instances.get(module_name)
            if module_instance is None:
                reports[module_name] = InterpretabilityContractReport(
                    status="noncompliant",
                    compliant=False,
                    interface_issues=[f"{module_name}: module instance unavailable in pipeline run"],
                    payload_issues=[],
                    na_reason=None,
                    na_owner=None,
                )
            else:
                reports[module_name] = build_interpretable_report(
                    module_instance, module_name,
                )
        else:
            reports[module_name] = build_not_applicable_report(
                module_name=module_name,
                rationale=policy["rationale"],
                owner=policy["owner"],
            )

    return reports


def enforce_alpha_scope_contract_coverage(
    reports: dict[str, InterpretabilityContractReport],
) -> None:
    """Raise if alpha-scope modules are not compliant or explicit N/A."""
    issues = validate_alpha_scope_contract_coverage(
        reports=reports,
        alpha_scope_modules=ALPHA_SCOPE_MODULE_POLICIES,
    )
    if issues:
        raise ValueError(
            "Alpha-scope interpretability contract coverage failure: " + " | ".join(issues),
        )
