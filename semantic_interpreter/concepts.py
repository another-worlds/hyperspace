"""Concept-Kernel mapping: connects SAE-discovered concepts to UKT SVD kernels.

This is where the two frameworks meet. The UKT discovers emergent kernels
(via SVD). The Semantic Interpreter discovers sparse concepts (via SAE).
This module maps between them — showing which human-interpretable concept
corresponds to which statistical kernel.
"""
from __future__ import annotations

import numpy as np


def map_concepts_to_kernels(
    sae_result: dict,
    kernel_snapshot: dict,
) -> list[dict]:
    """Map SAE concepts to UKT kernels via cosine projection.

    Projects concept vectors into kernel space (Vt) and finds the best-matching
    kernel for each concept.

    Args:
        sae_result: Output from train_global_sae() with concept_vectors.
        kernel_snapshot: UKT snapshot with Vt and importance arrays.

    Returns:
        List of dicts with concept-kernel correspondences, coherence scores.
    """
    try:
        concept_vectors = sae_result["concept_vectors"]  # (n_concepts, input_dim)
        Vt = kernel_snapshot["Vt"]                        # (n_kernels, feature_dim)
        importance = kernel_snapshot["importance"]

        # Cosine similarity: concepts @ kernels^T → (n_concepts, n_kernels)
        similarity = concept_vectors @ Vt.T

        rows = []
        for c in range(concept_vectors.shape[0]):
            best_kernel = int(np.argmax(np.abs(similarity[c])))
            coherence = float(np.abs(similarity[c, best_kernel]))
            mean_act = float(sae_result["mean_activation"][c])
            rows.append(dict(
                concept=f"C{c:02d}",
                best_kernel=f"K{best_kernel}",
                kernel_importance=float(importance[best_kernel]) if best_kernel < len(importance) else 0.0,
                coherence=round(coherence, 4),
                mean_activation=round(mean_act, 4),
                active=mean_act > sae_result["mean_activation"].mean(),
            ))
        return rows
    except Exception:
        return []


def compute_concept_kernel_matrix(
    sae_result: dict,
    kernel_snapshot: dict,
) -> np.ndarray | None:
    """Compute the full concept-kernel similarity matrix.

    Returns:
        (n_concepts, n_kernels) cosine similarity matrix, or None on failure.
    """
    try:
        concept_vectors = sae_result["concept_vectors"]
        Vt = kernel_snapshot["Vt"]
        return concept_vectors @ Vt.T
    except Exception:
        return None
