"""Sparse Autoencoder for unsupervised concept discovery on the UKT.

Hyperspace-specific wrapper around the standalone semantic_interpreter framework.
Preserves the existing API (train_sparse_ae, map_concepts_to_kernels,
enrich_concepts_with_narratives, SparseAutoencoder) while delegating
core logic to the standalone package.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Re-export from standalone framework
from semantic_interpreter.sae import GlobalSAE as SparseAutoencoder
from semantic_interpreter.concepts import map_concepts_to_kernels


def train_sparse_ae(
    combined_features: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 0.005,
) -> dict | None:
    """Train sparse AE on combined block features.

    Delegates to the standalone semantic_interpreter framework, adding
    Hyperspace-specific concept labeling with feature region awareness.
    """
    from semantic_interpreter.sae import train_global_sae
    from hyperspace.models.knowledge_matrix import HYPERSPACE_REGISTRY

    return train_global_sae(
        combined_features,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        registry=HYPERSPACE_REGISTRY,
    )


def enrich_concepts_with_narratives(
    sae_result: dict,
    canvas: Any = None,
) -> dict:
    """Add Tiny-LLM semantic narratives to active SAE concepts.

    Modifies sae_result in-place, adding 'semantic_narrative' to each
    active concept label.
    """
    if sae_result is None or canvas is None:
        return sae_result

    try:
        from hyperspace.models.semantic_narrator import narrate_concept
        for cl in sae_result.get("concept_labels", []):
            if cl.get("active"):
                narrative = narrate_concept(cl, canvas)
                if narrative:
                    cl["semantic_narrative"] = narrative
    except Exception:
        pass

    return sae_result
