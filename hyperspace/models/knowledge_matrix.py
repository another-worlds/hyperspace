"""Universal Knowledge Tensor (UKT): incremental SVD, reality regression, semantic interpretability.

The UKT grows after each pipeline block. At every step it:
  1. Accepts a feature vector from the block
  2. Normalizes it to [0, 1] per-region for cross-block comparability
  3. Appends it as a new row (blocks-so-far x feature_dim)
  4. Decomposes via SVD to get current kernel structure
  5. Computes "universal reality regression" (weighted kernel basis)
  6. Generates data-grounded, human-readable semantic labels for active kernels
"""
from __future__ import annotations

import numpy as np

from hyperspace.config import (
    FEATURE_NAMES,
    REGION_DESCRIPTIONS,
    UKT_FEATURE_DIM,
)


def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate a 1D array to target length."""
    arr = arr.flatten()
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)))


# Feature provenance labels: indices 0-15 = temporal, 16-31 = embedding,
# 32-47 = structural, 48-63 = agent/dynamic
FEATURE_REGION_LABELS: dict[tuple[int, int], str] = {
    (0, 16): "temporal-pattern",
    (16, 32): "semantic-embedding",
    (32, 48): "structural-centrality",
    (48, 64): "dynamic-agent",
}


def _normalize_features(arr: np.ndarray) -> np.ndarray:
    """Normalize each region of a feature vector to [0, 1] range."""
    out = arr.copy()
    for (lo, hi) in FEATURE_REGION_LABELS.keys():
        region = out[lo:hi]
        rng = region.max() - region.min()
        if rng > 1e-8:
            out[lo:hi] = (region - region.min()) / rng
    return out




def estimate_reality_regression_stability(
    matrix: np.ndarray,
    n_runs: int = 8,
    noise_std: float = 0.01,
    seed: int = 42,
) -> dict:
    """Estimate cross-run stability of reality regression under small perturbations.

    This approximates whether UKT regression is data-modality agnostic by checking
    if the leading reality-regression direction is stable across repeated noisy runs.
    """
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {"n_runs": 0, "mean_cosine": 0.0, "min_cosine": 0.0, "std_cosine": 0.0}

    rng = np.random.default_rng(seed)
    regressions = []
    for _ in range(n_runs):
        noisy = matrix + rng.normal(0.0, noise_std, size=matrix.shape)
        U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
        importance = S / (S.sum() + 1e-8)
        rr = importance @ Vt[:len(S), :]
        rr = rr / (np.linalg.norm(rr) + 1e-8)
        regressions.append(rr)

    cosines = []
    for i in range(len(regressions)):
        for j in range(i + 1, len(regressions)):
            cosines.append(float(np.dot(regressions[i], regressions[j])))

    if not cosines:
        return {"n_runs": n_runs, "mean_cosine": 0.0, "min_cosine": 0.0, "std_cosine": 0.0}

    arr = np.array(cosines, dtype=float)
    return {
        "n_runs": n_runs,
        "mean_cosine": float(arr.mean()),
        "min_cosine": float(arr.min()),
        "std_cosine": float(arr.std()),
    }

def _feature_name(idx: int, feature_meta: dict[int, dict] | None = None) -> str:
    """Return the name for a feature index, preferring metadata-derived labels."""
    if feature_meta and idx in feature_meta and feature_meta[idx].get("label"):
        return str(feature_meta[idx]["label"])
    if idx < len(FEATURE_NAMES):
        return FEATURE_NAMES[idx]
    return f"feature_{idx}"


def _region_for_index(idx: int) -> str:
    """Return the region label for a feature index."""
    for (lo, hi), label in FEATURE_REGION_LABELS.items():
        if lo <= idx < hi:
            return label
    return "unknown"


def _describe_top_features(
    vt_row: np.ndarray,
    feature_meta: dict[int, dict] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Return descriptions for the top-N loaded features."""
    indices = np.argsort(np.abs(vt_row))[-top_n:][::-1]
    descriptions = []
    for idx in indices:
        meta = feature_meta.get(int(idx), {}) if feature_meta else {}
        descriptions.append(dict(
            index=int(idx),
            name=_feature_name(int(idx), feature_meta),
            region=_region_for_index(int(idx)),
            loading=float(vt_row[int(idx)]),
            abs_loading=float(abs(vt_row[int(idx)])),
            entity=meta.get("entity"),
            metric=meta.get("metric"),
            source=meta.get("source"),
            time_scope=meta.get("time_scope"),
        ))
    return descriptions


def _generate_kernel_narrative(
    k_idx: int,
    dominant_block: str,
    dominant_region: str,
    importance: float,
    top_features: list[dict],
    block_names: list[str],
    u_col: np.ndarray,
    timeframe_context: dict | None,
) -> str:
    """Generate a data-grounded narrative for a kernel."""
    block_contributions = []
    for i, bn in enumerate(block_names):
        contrib = abs(float(u_col[i]))
        if contrib > 0.1:
            block_contributions.append(f"{bn} ({contrib:.2f})")

    region_desc = REGION_DESCRIPTIONS.get(dominant_region, dominant_region)
    feature_strs = [f"{f['name']} ({f['loading']:+.3f})" for f in top_features[:3]]

    evidence_lines = []
    temporal_features = [f for f in top_features if f["region"] == "temporal-pattern"]
    if temporal_features:
        evidence_lines.append(
            "Temporal step attribution: " + ", ".join(
                f"{f['name']}={f['loading']:+.3f}" for f in temporal_features[:3]
            )
        )

    semantic_features = [f for f in top_features if f["region"] == "semantic-embedding"]
    structural_features = [f for f in top_features if f["region"] == "structural-centrality"]
    if structural_features:
        evidence_lines.append(
            "Structural attribution: " + ", ".join(
                f"{f['name']}={f['loading']:+.3f}" for f in structural_features[:3]
            )
        )

    if temporal_features and semantic_features:
        evidence_lines.append(
            "Emergent temporal↔semantic coupling: "
            f"{temporal_features[0]['name']} with {semantic_features[0]['name']} indicates "
            "attention dynamics being transposed into semantic latent space."
        )

    timeframe_line = ""
    if timeframe_context:
        start = timeframe_context.get("start_date")
        end = timeframe_context.get("end_date")
        if start and end:
            timeframe_line = f"Time alignment window: {start} to {end}."

    lines = [
        f"Kernel K{k_idx} explains {importance:.1%} of total variance.",
        f"Primary driver: {dominant_block} block; dominant latent region: {dominant_region.replace('-', ' ')}.",
        f"Top evidence features: {', '.join(feature_strs)}.",
        f"Block contributions: {', '.join(block_contributions) if block_contributions else 'weak/mixed'}.",
        f"Region context: {region_desc}",
    ]
    if timeframe_line:
        lines.append(timeframe_line)
    lines.extend(evidence_lines)
    return "\n".join(lines)


def _label_kernel(
    k_idx: int,
    vt_row: np.ndarray,
    u_col: np.ndarray,
    block_names: list[str],
    importance: float,
    feature_meta: dict[int, dict] | None,
    timeframe_context: dict | None,
) -> dict:
    """Generate a semantic label for a single kernel."""
    region_scores = {}
    for (lo, hi), label in FEATURE_REGION_LABELS.items():
        region_scores[label] = float(np.abs(vt_row[lo:hi]).sum())
    dominant_region = max(region_scores, key=region_scores.get)

    dominant_block_idx = int(np.argmax(np.abs(u_col)))
    dominant_block = (block_names[dominant_block_idx]
                      if dominant_block_idx < len(block_names) else "unknown")

    top_features = _describe_top_features(vt_row, feature_meta=feature_meta, top_n=5)
    top_feat_name = top_features[0]["name"] if top_features else "?"
    short_label = (
        f"K{k_idx}: {dominant_block} — {dominant_region.replace('-', ' ')} "
        f"({importance:.1%} var, lead: {top_feat_name})"
    )

    narrative = _generate_kernel_narrative(
        k_idx, dominant_block, dominant_region, importance,
        top_features, block_names, u_col, timeframe_context,
    )

    return dict(
        kernel_id=f"K{k_idx}",
        dominant_block=dominant_block,
        dominant_region=dominant_region,
        importance=float(importance),
        top_features=top_features,
        top_feature_indices=[f["index"] for f in top_features],
        label=short_label,
        narrative=narrative,
        region_scores={k: round(v, 4) for k, v in region_scores.items()},
    )


class UniversalKnowledgeTensor:
    """Incrementally built cross-block knowledge tensor with SVD decomposition."""

    def __init__(self, feature_dim: int = UKT_FEATURE_DIM):
        self.feature_dim = feature_dim
        self.block_names: list[str] = []
        self.rows: list[np.ndarray] = []
        self.snapshots: list[dict] = []
        self.global_feature_meta: dict[int, dict] = {}

    def add_block(
        self,
        name: str,
        features: np.ndarray,
        feature_meta: dict[int, dict] | None = None,
        timeframe_context: dict | None = None,
    ) -> dict:
        """Add a block's feature vector, normalize, decompose, interpret."""
        self.block_names.append(name)
        if feature_meta:
            self.global_feature_meta.update(feature_meta)
        raw = _pad_or_truncate(features, self.feature_dim)
        normalized = _normalize_features(raw)
        self.rows.append(normalized)

        matrix = np.stack(self.rows)
        try:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback: create identity-like decomposition
            n = matrix.shape[0]
            U = np.eye(n)
            S = np.ones(n) * 0.01
            Vt = np.zeros((n, matrix.shape[1]))
            for i in range(min(n, matrix.shape[1])):
                Vt[i, i] = 1.0
        n_kernels = len(S)

        total = S.sum() + 1e-8
        importance = S / total
        kernel_activation = U * S[np.newaxis, :]
        reality_regression = importance @ Vt[:n_kernels, :]

        kernel_labels = []
        for k in range(n_kernels):
            label = _label_kernel(
                k,
                Vt[k],
                U[:, k],
                self.block_names,
                importance[k],
                self.global_feature_meta,
                timeframe_context,
            )
            kernel_labels.append(label)

        recon = U[:, :n_kernels] @ np.diag(S[:n_kernels]) @ Vt[:n_kernels, :]
        recon_error = float(np.linalg.norm(matrix - recon))

        report_lines = [
            f"=== Step {len(self.rows)}: Added '{name}' block ===",
            f"Active kernels: {n_kernels}",
            f"Reconstruction error: {recon_error:.6f}",
            f"Reality regression norm: {np.linalg.norm(reality_regression):.4f}",
            "",
        ]
        for kl in kernel_labels:
            report_lines.append(f"--- {kl['label']} ---")
            report_lines.append(kl["narrative"])
            report_lines.append("")

        if len(self.rows) > 1:
            report_lines.append("--- Cross-Block Coherence ---")
            for k in range(n_kernels):
                contribs = []
                for i, bn in enumerate(self.block_names):
                    val = float(kernel_activation[i, k])
                    if abs(val) > 0.01:
                        contribs.append(f"{bn}={val:+.3f}")
                report_lines.append(f"K{k} activation across blocks: {', '.join(contribs)}")
            report_lines.append("")

        rr_top = np.argsort(np.abs(reality_regression))[-5:][::-1]
        report_lines.append("--- Reality Regression (top features) ---")
        for idx in rr_top:
            report_lines.append(
                f"  {_feature_name(int(idx), self.global_feature_meta)} [{_region_for_index(int(idx))}]: "
                f"{reality_regression[int(idx)]:+.4f}"
            )

        snapshot = dict(
            step=len(self.rows),
            block_name=name,
            matrix=matrix.copy(),
            U=U.copy(),
            S=S.copy(),
            Vt=Vt.copy(),
            n_kernels=n_kernels,
            importance=importance.copy(),
            kernel_activation=kernel_activation.copy(),
            reality_regression=reality_regression.copy(),
            kernel_labels=kernel_labels,
            reconstruction_error=recon_error,
            report="\n".join(report_lines),
            feature_meta=self.global_feature_meta.copy(),
            timeframe_context=timeframe_context or {},
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_final_matrix(self) -> np.ndarray | None:
        if not self.rows:
            return None
        return np.stack(self.rows)

    def get_latest_snapshot(self) -> dict | None:
        return self.snapshots[-1] if self.snapshots else None
