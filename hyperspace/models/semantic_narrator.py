"""Semantic Narrator: Hyperspace-specific wrapper around the standalone
semantic_interpreter narrator framework.

Preserves the existing API (narrate_canvas, narrate_layer, narrate_kernel,
narrate_reality_regression, narrate_concept) while delegating to the
standalone LLMNarrator backend.

Uses arnir0/Tiny-LLM (10M parameter Llama-based model, MIT license) for
text generation. Cached via st.cache_resource for single load.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from semantic_interpreter.narrator import LLMNarrator, TemplateNarrator

if TYPE_CHECKING:
    from hyperspace.models.semantic_canvas import SemanticCanvas, CanvasEntry

# --------------------------------------------------------------------------- #
# Shared narrator instance                                                     #
# --------------------------------------------------------------------------- #

_narrator: LLMNarrator | None = None


def _get_narrator() -> LLMNarrator:
    """Get or create the shared LLM narrator instance."""
    global _narrator
    if _narrator is None:
        # Try to use Streamlit caching
        cache_fn = None
        try:
            import streamlit as st
            cache_fn = st.cache_resource(show_spinner="Loading Tiny-LLM narrator model...")
        except Exception:
            pass
        _narrator = LLMNarrator(
            model_name="arnir0/Tiny-LLM",
            max_new_tokens=150,
            temperature=0.7,
            cache_fn=cache_fn,
        )
    return _narrator


# --------------------------------------------------------------------------- #
# Public API (backward-compatible with existing Hyperspace usage)              #
# --------------------------------------------------------------------------- #

def narrate_canvas(canvas: "SemanticCanvas") -> str | None:
    """Generate a full narrative from the accumulated semantic canvas."""
    return _get_narrator().narrate_canvas(canvas)


def narrate_layer(entry: "CanvasEntry", canvas: "SemanticCanvas") -> str | None:
    """Generate a narrative for a single pipeline layer's contribution."""
    return _get_narrator().narrate_layer(entry, canvas)


def narrate_kernel(kernel_label: dict, canvas: "SemanticCanvas") -> str | None:
    """Generate a semantic narrative for a single UKT kernel."""
    return _get_narrator().narrate_kernel(kernel_label, canvas)


def narrate_reality_regression(
    snapshot: dict,
    canvas: "SemanticCanvas",
) -> str | None:
    """Generate a narrative for the reality regression vector."""
    from hyperspace.models.knowledge_matrix import (
        _feature_name, _region_for_index, FEATURE_REGION_LABELS,
    )

    region_bounds = {label: (lo, hi) for (lo, hi), label in FEATURE_REGION_LABELS.items()}

    return _get_narrator().narrate_reality_regression(
        snapshot, canvas,
        feature_name_fn=_feature_name,
        region_for_index_fn=_region_for_index,
        region_bounds=region_bounds,
    )


def narrate_concept(concept_label: dict, canvas: "SemanticCanvas") -> str | None:
    """Generate a semantic narrative for an SAE-discovered concept."""
    return _get_narrator().narrate_concept(concept_label, canvas)
