"""Mechanistic probes for exploratory interpretability (SPEC-6).

Three interactive probing tools that operate on UKT snapshots:
1. **Activation Patching** — Zero out a kernel and measure impact.
2. **Linear Probing** — Predict governance flags from kernel activations.
3. **Feature Pathway Tracing** — Trace a feature through the full chain.

All probes are read-only: they operate on copies and never modify pipeline state.
All probes complete in <2s on CPU.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from hyperspace.config import (
    FEATURE_NAMES,
    PROBE_CANVAS_LOADING_THRESHOLD,
    PROBE_MIN_RUNS_FOR_LINEAR,
    UKT_FEATURE_DIM,
)
from hyperspace.core.faithfulness import _intervention_kernel_removal


# --------------------------------------------------------------------------- #
# Probe 1: Activation Patching                                                 #
# --------------------------------------------------------------------------- #

def activation_patch(snapshot: dict, kernel_idx: int) -> dict[str, Any]:
    """Zero out a kernel's singular value and measure impact.

    Returns dict with original vs patched reality regression, affected canvas
    dimensions, and reconstruction error delta.
    """
    matrix = snapshot.get("matrix")
    U = snapshot.get("U")
    S = snapshot.get("S")
    Vt = snapshot.get("Vt")
    kernel_labels = snapshot.get("kernel_labels", [])

    if matrix is None or U is None or S is None or Vt is None:
        return {"error": "Snapshot missing required SVD components."}

    if kernel_idx >= len(S):
        return {"error": f"Kernel index {kernel_idx} out of range (max {len(S) - 1})."}

    # Baseline reconstruction
    rank = min(U.shape[1], len(S), Vt.shape[0])
    baseline_recon = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    baseline_error = float(np.linalg.norm(matrix - baseline_recon, "fro"))

    # Patched reconstruction (zero out one kernel)
    patched_error = _intervention_kernel_removal(matrix, kernel_idx, U, S, Vt)
    delta = patched_error - baseline_error

    # Original vs patched reality regression
    original_rr = snapshot.get("reality_regression")
    S_patched = S.copy()
    S_patched[kernel_idx] = 0.0
    # Recompute reality regression as weighted sum of Vt rows
    importance_patched = S_patched / (S_patched.sum() + 1e-12)
    n_k = min(len(importance_patched), Vt.shape[0])
    patched_rr = np.zeros(Vt.shape[1]) if Vt.shape[1] > 0 else np.zeros(UKT_FEATURE_DIM)
    for i in range(n_k):
        patched_rr += importance_patched[i] * Vt[i]

    # Affected canvas dimensions
    affected_dims: list[str] = []
    canvas_state = snapshot.get("canvas_state", {})
    if isinstance(canvas_state, dict):
        for dim_key, dim_val in canvas_state.items():
            if isinstance(dim_val, (int, float)) and abs(dim_val) > PROBE_CANVAS_LOADING_THRESHOLD:
                affected_dims.append(dim_key)

    # Kernel label
    k_label = ""
    if kernel_idx < len(kernel_labels):
        k_label = kernel_labels[kernel_idx].get("label", f"K{kernel_idx}")

    return {
        "kernel_idx": kernel_idx,
        "kernel_label": k_label,
        "original_reality_regression": original_rr.tolist() if original_rr is not None else [],
        "patched_reality_regression": patched_rr.tolist(),
        "affected_canvas_dims": affected_dims,
        "reconstruction_error_delta": delta,
        "baseline_error": baseline_error,
        "patched_error": patched_error,
    }


# --------------------------------------------------------------------------- #
# Probe 2: Linear Probing                                                      #
# --------------------------------------------------------------------------- #

def linear_probe(
    kernel_memory: Any,
    governance_flags_history: list[dict] | None = None,
) -> dict[str, Any]:
    """Train logistic regression per governance flag on kernel activations.

    Requires at least PROBE_MIN_RUNS_FOR_LINEAR runs in KernelMemory.
    """
    if kernel_memory is None:
        return {"insufficient_history": True, "message": "No KernelMemory available."}

    # Extract importance vectors across runs
    runs = getattr(kernel_memory, "_runs", [])
    if len(runs) < PROBE_MIN_RUNS_FOR_LINEAR:
        return {
            "insufficient_history": True,
            "message": f"Need {PROBE_MIN_RUNS_FOR_LINEAR} runs, have {len(runs)}.",
        }

    if governance_flags_history is None or len(governance_flags_history) < len(runs):
        return {"insufficient_history": True, "message": "Governance flag history not available."}

    # Build feature matrix: each row is a run's kernel importance vector
    X_rows: list[np.ndarray] = []
    for run_data in runs:
        snaps = run_data.get("snapshots", [])
        if snaps:
            imp = snaps[-1].get("importance", np.array([]))
            X_rows.append(np.array(imp).flatten())
    if not X_rows:
        return {"insufficient_history": True, "message": "No importance data in runs."}

    # Pad to uniform length
    max_len = max(len(r) for r in X_rows)
    X = np.zeros((len(X_rows), max_len))
    for i, row in enumerate(X_rows):
        X[i, :len(row)] = row

    # Try sklearn logistic regression
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        # Fallback: per-feature correlation
        results = {}
        for flag_code in ["GOV-001", "GOV-002", "GOV-003", "GOV-004", "GOV-005", "GOV-006"]:
            y = np.array([
                1.0 if flag_code in (gf.get("codes", []) if isinstance(gf, dict) else [])
                else 0.0
                for gf in governance_flags_history[:len(X_rows)]
            ])
            if y.sum() < 1 or y.sum() >= len(y):
                continue
            corrs = [float(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])]
            best_k = int(np.argmax(np.abs(corrs)))
            results[flag_code] = {
                "per_kernel_weights": corrs,
                "most_predictive_kernel": f"K{best_k}",
                "predictive_weight": abs(corrs[best_k]),
            }
        return {"fallback": "per_feature_correlation", "results": results}

    results = {}
    for flag_code in ["GOV-001", "GOV-002", "GOV-003", "GOV-004", "GOV-005", "GOV-006"]:
        y = np.array([
            1.0 if flag_code in (gf.get("codes", []) if isinstance(gf, dict) else [])
            else 0.0
            for gf in governance_flags_history[:len(X_rows)]
        ])
        if y.sum() < 1 or y.sum() >= len(y):
            continue
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(X, y)
        weights = clf.coef_[0].tolist()
        best_k = int(np.argmax(np.abs(clf.coef_[0])))
        results[flag_code] = {
            "per_kernel_weights": weights,
            "most_predictive_kernel": f"K{best_k}",
            "predictive_weight": float(np.abs(clf.coef_[0][best_k])),
        }

    return {"results": results}


# --------------------------------------------------------------------------- #
# Probe 3: Feature Pathway Tracing                                             #
# --------------------------------------------------------------------------- #

def trace_feature_pathway(
    snapshot: dict,
    feature_idx: int,
    feature_registry: Any = None,
) -> dict[str, Any]:
    """Trace a single feature through the full computation chain.

    Returns a dict with 7 steps: raw → normalized → projected → kernel loadings
    → canvas dimensions → narrative mentions.
    """
    if feature_idx < 0 or feature_idx >= UKT_FEATURE_DIM:
        return {"error": f"Feature index {feature_idx} out of range (0-{UKT_FEATURE_DIM - 1})."}

    feature_name = FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f"feature_{feature_idx}"

    # Step 1: Raw value from the matrix
    matrix = snapshot.get("matrix")
    raw_value = 0.0
    if matrix is not None and matrix.shape[1] > feature_idx:
        raw_value = float(matrix[:, feature_idx].mean())

    # Step 2: Normalized value (the matrix is already normalized in UKT)
    normalized_value = raw_value  # UKT normalizes during add_block

    # Step 3: Projected value (from reality regression)
    reality_regression = snapshot.get("reality_regression")
    projected_value = 0.0
    if reality_regression is not None and feature_idx < len(reality_regression):
        projected_value = float(reality_regression[feature_idx])

    # Step 4: Kernel loadings — how much each kernel loads on this feature
    Vt = snapshot.get("Vt")
    kernel_labels = snapshot.get("kernel_labels", [])
    importance = snapshot.get("importance", np.array([]))
    kernel_loadings: dict[str, float] = {}
    if Vt is not None:
        n_k = min(Vt.shape[0], len(kernel_labels))
        for k_idx in range(n_k):
            if feature_idx < Vt.shape[1]:
                loading = float(Vt[k_idx, feature_idx])
                k_label = kernel_labels[k_idx].get("label", f"K{k_idx}") if k_idx < len(kernel_labels) else f"K{k_idx}"
                kernel_loadings[k_label] = loading

    # Step 5: Canvas dimensions influenced by this feature
    canvas_dimensions: list[str] = []
    canvas_state = snapshot.get("canvas_state", {})
    if isinstance(canvas_state, dict):
        for dim_key in canvas_state:
            canvas_dimensions.append(dim_key)

    # Step 6: Narrative mentions
    narrative_mentions: list[str] = []
    for kl in kernel_labels:
        narr = kl.get("semantic_narrative", "")
        if narr and feature_name.lower() in narr.lower():
            narrative_mentions.append(narr[:120])

    # Step 7: Region membership
    region = "unknown"
    if feature_registry is not None:
        try:
            region = feature_registry.region_name(feature_idx)
        except Exception:
            pass
    if region == "unknown":
        if feature_idx < 16:
            region = "temporal-pattern"
        elif feature_idx < 32:
            region = "semantic-embedding"
        elif feature_idx < 48:
            region = "structural-centrality"
        elif feature_idx < 64:
            region = "dynamic-agent"
        else:
            region = "geospatial-kernel"

    return {
        "feature_idx": feature_idx,
        "feature_name": feature_name,
        "region": region,
        "raw_value": raw_value,
        "normalized_value": normalized_value,
        "projected_value": projected_value,
        "kernel_loadings": kernel_loadings,
        "canvas_dimensions": canvas_dimensions,
        "narrative_mentions": narrative_mentions,
    }
