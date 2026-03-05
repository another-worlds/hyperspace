"""Semantic Narrator: Tiny-LLM based narrative generation from the Semantic Canvas.

Loads arnir0/Tiny-LLM (10M parameter Llama-based model, MIT license) and translates
structured semantic canvas coordinates into coherent natural-language explanations.

The narrator consumes the semantic canvas — a unified interpretive space where
each pipeline layer's internal representations have been projected onto named
semantic dimensions — and produces text that describes the reasoning trajectory
implied by those representations.

Design notes:
    - Tiny-LLM is a completion model (not instruction-tuned), so prompts are
      structured as analytical text for the model to continue.
    - We cache the model via @st.cache_resource for single load.
    - All generation is CPU-only with max_new_tokens=150 for speed.
    - Graceful fallback: if the model can't be loaded, returns None and the
      system falls back to algorithmic interpretations.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hyperspace.models.semantic_canvas import SemanticCanvas, CanvasEntry

# --------------------------------------------------------------------------- #
# Model loading (cached)                                                       #
# --------------------------------------------------------------------------- #

_MODEL_NAME = "arnir0/Tiny-LLM"


def _load_narrator_model():
    """Load the Tiny-LLM model and tokenizer. Cached externally via st.cache_resource."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME)
        model.eval()
        return model, tokenizer
    except Exception as exc:
        import warnings
        warnings.warn(f"Semantic narrator: failed to load {_MODEL_NAME}: {exc}")
        return None, None


def get_narrator_model():
    """Get the cached narrator model. Uses st.cache_resource if available."""
    try:
        import streamlit as st

        @st.cache_resource(show_spinner="Loading Tiny-LLM narrator model...")
        def _cached_load():
            return _load_narrator_model()

        return _cached_load()
    except Exception:
        return _load_narrator_model()


# --------------------------------------------------------------------------- #
# Text generation                                                              #
# --------------------------------------------------------------------------- #

def _generate(prompt: str, max_new_tokens: int = 150,
              temperature: float = 0.7, top_k: int = 50,
              top_p: float = 0.92) -> str | None:
    """Generate text continuation using the Tiny-LLM.

    Returns the generated continuation (without the prompt), or None on failure.
    """
    model, tokenizer = get_narrator_model()
    if model is None or tokenizer is None:
        return None

    try:
        import torch
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True,
                                  max_length=800)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated continuation
        continuation = full_text[len(tokenizer.decode(inputs[0],
                                                       skip_special_tokens=True)):]
        return _postprocess(continuation)
    except Exception:
        return None


def _postprocess(text: str) -> str:
    """Clean up generated text: truncate at last complete sentence."""
    text = text.strip()
    if not text:
        return ""

    # Remove any partial sentence at the end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1 and not sentences[-1].rstrip().endswith(('.', '!', '?')):
        sentences = sentences[:-1]
    result = ' '.join(sentences)

    # Limit to ~3 sentences max for conciseness
    final_sentences = re.split(r'(?<=[.!?])\s+', result)
    if len(final_sentences) > 3:
        result = ' '.join(final_sentences[:3])

    return result.strip()


# --------------------------------------------------------------------------- #
# Narrative generation functions                                               #
# --------------------------------------------------------------------------- #

def narrate_canvas(canvas: "SemanticCanvas") -> str | None:
    """Generate a full narrative from the accumulated semantic canvas.

    This is the primary narrative generation function. It takes the entire
    canvas state and produces a coherent multi-sentence description of
    the system's findings across all pipeline layers.

    Returns:
        Generated narrative text, or None if generation fails.
    """
    prompt = canvas.format_for_narrator()
    return _generate(prompt, max_new_tokens=200, temperature=0.7)


def narrate_layer(entry: "CanvasEntry", canvas: "SemanticCanvas") -> str | None:
    """Generate a narrative for a single pipeline layer's contribution.

    Describes what this specific layer found and how it shifted the
    semantic canvas.

    Returns:
        Generated narrative text, or None if generation fails.
    """
    from hyperspace.models.semantic_canvas import CANVAS_DIMENSIONS

    coords = entry.coordinates
    parts = []
    for i, dim in enumerate(CANVAS_DIMENSIONS):
        if coords[i] > 0.3:
            strength = "strongly" if coords[i] > 0.7 else "moderately"
            parts.append(f"{strength} activates {dim['label'].lower()}")

    canvas_state = canvas.get_accumulated_state()
    prior_blocks = [e.block_name for e in canvas_state["entries"]
                    if e.step < entry.step]
    prior_context = ""
    if prior_blocks:
        prior_context = (
            f" Building on prior analysis of {', '.join(prior_blocks)}, "
        )

    dim_description = "; ".join(parts) if parts else "shows weak signals across all dimensions"

    prompt = (
        f"Cross-domain analytical report, layer {entry.step} of 5:\n\n"
        f"The {entry.block_name} analysis stage processed its data and "
        f"{dim_description}.{prior_context}\n"
        f"{entry.active_concepts} sparse concepts were discovered by the "
        f"stage's autoencoder.\n\n"
        f"The {entry.block_name} layer reveals that"
    )
    return _generate(prompt, max_new_tokens=120, temperature=0.7)


def narrate_kernel(
    kernel_label: dict,
    canvas: "SemanticCanvas",
) -> str | None:
    """Generate a semantic narrative for a single UKT kernel.

    Translates the abstract kernel (SVD component) into coherent text
    describing the concept it represents.

    Args:
        kernel_label: Dict from _label_kernel() with importance, region, features.
        canvas: The current semantic canvas state.

    Returns:
        Generated narrative text, or None if generation fails.
    """
    importance = kernel_label.get("importance", 0)
    region = kernel_label.get("dominant_region", "unknown").replace("-", " ")
    block = kernel_label.get("dominant_block", "unknown")
    features = kernel_label.get("top_features", [])

    feat_desc = ", ".join(
        f"{f['name']} (loading {f['loading']:+.3f})"
        for f in features[:3]
    )

    # Get canvas context for richer prompt
    state = canvas.get_accumulated_state()
    dominant_dims = state.get("dominant_narrative", [])
    canvas_context = ""
    if dominant_dims:
        dim_strs = [f"{d['label']} ({d['value']:.2f})" for d in dominant_dims[:2]]
        canvas_context = (
            f" The semantic canvas shows dominant signals in {' and '.join(dim_strs)}."
        )

    prompt = (
        f"Universal Knowledge Tensor analysis — Kernel interpretation:\n\n"
        f"Kernel K{kernel_label.get('kernel_id', '?')} explains "
        f"{importance:.1%} of the total observed variance across all data domains. "
        f"It is primarily driven by {region} patterns from the {block} data stream. "
        f"The strongest feature loadings are: {feat_desc}.{canvas_context}\n\n"
        f"In plain language, this kernel represents"
    )
    return _generate(prompt, max_new_tokens=120, temperature=0.7)


def narrate_reality_regression(
    snapshot: dict,
    canvas: "SemanticCanvas",
) -> str | None:
    """Generate a narrative for the reality regression vector.

    The reality regression is the system's single best summary of the
    current state across all data domains. This function translates
    that 80-dimensional vector into coherent text.

    Args:
        snapshot: The UKT snapshot containing reality_regression, kernel_labels, etc.
        canvas: The semantic canvas with accumulated state.

    Returns:
        Generated narrative text, or None if generation fails.
    """
    rr = snapshot.get("reality_regression")
    if rr is None:
        return None

    from hyperspace.models.knowledge_matrix import (
        _feature_name, _region_for_index, FEATURE_REGION_LABELS,
    )

    # Top features in reality regression
    top_idx = np.argsort(np.abs(rr))[-5:][::-1]
    top_feats = [
        f"{_feature_name(int(i))} [{_region_for_index(int(i))}] = {rr[int(i)]:+.4f}"
        for i in top_idx
    ]

    # Region energy distribution
    region_lines = []
    for (lo, hi), label in FEATURE_REGION_LABELS.items():
        energy = float(np.abs(rr[lo:hi]).sum())
        if energy > 0.01:
            region_lines.append(f"{label.replace('-', ' ')}: {energy:.3f}")

    # Canvas summary
    canvas_text = ""
    state = canvas.get_accumulated_state()
    dominant = state.get("dominant_narrative", [])
    if dominant:
        dim_names = [d["label"] for d in dominant]
        canvas_text = (
            f" The semantic canvas converged on these dominant themes: "
            f"{', '.join(dim_names)}."
        )

    n_kernels = snapshot.get("n_kernels", 0)
    recon_err = snapshot.get("reconstruction_error", 0)

    prompt = (
        f"Final reality assessment — Universal Knowledge Tensor synthesis:\n\n"
        f"After integrating {n_kernels} data domains (financial, informational, "
        f"geopolitical, spatial, agent-based), the system computed a unified "
        f"reality regression vector that captures the most important patterns.\n\n"
        f"Top contributing features: {'; '.join(top_feats[:3])}.\n"
        f"Region energy: {'; '.join(region_lines)}.\n"
        f"Reconstruction quality: {recon_err:.4f} error across {n_kernels} kernels."
        f"{canvas_text}\n\n"
        f"The overall assessment of the current state is that"
    )
    return _generate(prompt, max_new_tokens=200, temperature=0.7)


def narrate_concept(concept_label: dict,
                    canvas: "SemanticCanvas") -> str | None:
    """Generate a semantic narrative for an SAE-discovered concept.

    Args:
        concept_label: Dict from train_sparse_ae() concept_labels.
        canvas: The semantic canvas.

    Returns:
        Generated narrative text, or None if generation fails.
    """
    concept_id = concept_label.get("concept_id", "C??")
    region = concept_label.get("dominant_region", "unknown").replace("-", " ")
    activation = concept_label.get("mean_activation", 0)
    features = concept_label.get("top_features", [])

    feat_desc = ", ".join(
        f"{f['name']} ({f['loading']:+.3f})"
        for f in features[:3]
    )

    prompt = (
        f"Sparse Autoencoder concept analysis:\n\n"
        f"Concept {concept_id} is an abstract pattern discovered by the autoencoder "
        f"within the {region} region of the knowledge tensor. "
        f"It has a mean activation of {activation:.4f} and loads most strongly on: "
        f"{feat_desc}.\n\n"
        f"This concept captures the idea that"
    )
    return _generate(prompt, max_new_tokens=100, temperature=0.7)
