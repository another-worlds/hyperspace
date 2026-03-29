"""Semantic Narrator: embedded Tiny-LLM translator from machine neuron
clusters to human-readable semantics.

This module serves as the bridge between the UKT's latent space (machine
neuron clusters, kernel decompositions, sparse autoencoder concepts) and
human-interpretable semantic narratives. The Tiny-LLM (arnir0/Tiny-LLM,
13M parameter Llama-based model, MIT license) acts as an embedded translator
that unravels the latent space into natural language.

Architecture:
    UKT Kernels → Feature Attributions → Semantic Canvas Coordinates
    → LLM Narrator (Tiny-LLM) → Human-Readable Narrative

The LLM receives structured prompts containing:
  - Kernel importances and dominant feature loadings
  - Canvas coordinates (the semantic projection of latent activations)
  - Region provenance and cross-domain coupling signals

And produces continuations that translate these machine representations
into plain-language interpretations.

Falls back to deterministic TemplateNarrator when the model cannot load.
Cached via st.cache_resource for single-load in Streamlit contexts.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from semantic_interpreter.narrator import LLMNarrator

if TYPE_CHECKING:
    from hyperspace.models.semantic_canvas import SemanticCanvas, CanvasEntry

# --------------------------------------------------------------------------- #
# Shared narrator instance (embedded LLM translator)                          #
# --------------------------------------------------------------------------- #

_narrator: LLMNarrator | None = None


def _get_narrator() -> LLMNarrator:
    """Get or create the shared LLM narrator instance.

    The narrator is the embedded translator from machine neuron clusters
    (UKT kernels, SAE concepts, canvas coordinates) to human semantics.
    """
    global _narrator
    if _narrator is None:
        # Try to use Streamlit caching for model persistence
        cache_fn = None
        try:
            import streamlit as st
            cache_fn = st.cache_resource
        except Exception:
            pass  # Non-Streamlit context; caching disabled
        _narrator = LLMNarrator(
            model_name="arnir0/Tiny-LLM",
            max_new_tokens=60,
            temperature=0.7,
            cache_fn=cache_fn,
            generation_timeout=30.0,
        )
    return _narrator


def is_llm_available() -> bool:
    """Check whether the embedded Tiny-LLM loaded successfully."""
    narrator = _get_narrator()
    model, tokenizer = narrator._load_model()
    return model is not None and tokenizer is not None


def get_llm_health_status() -> dict[str, object]:
    """Return health status for diagnostics and UI display."""
    narrator = _get_narrator()
    model_loaded = narrator._model is not None
    if not model_loaded and not narrator._model_load_failed:
        # attempt at least one load
        narrator._load_model()
        model_loaded = narrator._model is not None

    return {
        "model_name": narrator.model_name,
        "model_loaded": model_loaded,
        "model_load_failed": narrator._model_load_failed,
        "timeout_count": narrator._timeout_count,
        "timeout_threshold": narrator._timeout_threshold,
        "generation_timeout": narrator._generation_timeout,
        "permanently_disabled": narrator._timeout_count >= narrator._timeout_threshold,
    }


# --------------------------------------------------------------------------- #
# Public API — latent-space-to-semantics translation                          #
# --------------------------------------------------------------------------- #

def narrate_canvas(canvas: "SemanticCanvas", kernel_labels: list[dict] | None = None) -> str | None:
    """Translate the accumulated semantic canvas into a narrative.

    The canvas contains all projected coordinates from every pipeline layer.
    The LLM reads the structured canvas state and produces a holistic
    interpretation of the cross-domain latent patterns.
    """
    return _get_narrator().narrate_canvas(canvas, kernel_labels=kernel_labels)


def narrate_layer(entry: "CanvasEntry", canvas: "SemanticCanvas") -> str | None:
    """Translate a single pipeline layer's latent projection into narrative.

    Takes the canvas entry (projected coordinates, active concepts, dominant
    dimensions) and generates a human-readable interpretation of what this
    layer's neuron activations mean semantically.
    """
    return _get_narrator().narrate_layer(entry, canvas)


def narrate_kernel(kernel_label: dict, canvas: "SemanticCanvas") -> str | None:
    """Translate a UKT kernel (machine neuron cluster) into semantic narrative.

    A kernel is a singular vector from the SVD of the Universal Knowledge
    Tensor — a latent pattern discovered across all data domains. This
    function translates the kernel's feature loadings, variance explained,
    and dominant region into natural language.
    """
    return _get_narrator().narrate_kernel(kernel_label, canvas)


def narrate_reality_regression(
    snapshot: dict,
    canvas: "SemanticCanvas",
) -> str | None:
    """Translate the reality regression vector into semantic narrative.

    The reality regression is the UKT's unified prediction vector —
    summarizing all kernels into a single direction of maximum explained
    variance. This function converts the vector's feature attributions
    and region energies into an interpretable assessment.
    """
    from hyperspace.models.knowledge_matrix import (
        _feature_name, _region_for_index, HYPERSPACE_REGISTRY,
    )

    region_bounds = {
        r.name: (r.start, r.end)
        for r in HYPERSPACE_REGISTRY.ordered_regions
    }

    return _get_narrator().narrate_reality_regression(
        snapshot, canvas,
        feature_name_fn=_feature_name,
        region_for_index_fn=_region_for_index,
        region_bounds=region_bounds,
    )


def narrate_concept(concept_label: dict, canvas: "SemanticCanvas") -> str | None:
    """Translate an SAE-discovered sparse concept into semantic narrative.

    SAE concepts are learned basis vectors in the latent space — each one
    captures a specific pattern in the feature activations. This function
    translates the concept's region, activation level, and feature loadings
    into interpretable language.
    """
    return _get_narrator().narrate_concept(concept_label, canvas)
