"""Narrative generation backends for the Semantic Interpreter.

Translates structured semantic coordinates into human-readable text.
Supports multiple backends:
  - TemplateNarrator: deterministic, no model required
  - LLMNarrator: uses a small language model for richer narratives
  - Custom backends: implement NarratorBackend protocol

The narrator is pluggable — use templates for speed, LLMs for richness,
or write your own backend for domain-specific language.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from semantic_interpreter.canvas import SemanticCanvas, CanvasEntry


# --------------------------------------------------------------------------- #
# Narrator protocol                                                            #
# --------------------------------------------------------------------------- #

class NarratorBackend(ABC):
    """Base class for narrative generation backends."""

    @abstractmethod
    def narrate_canvas(self, canvas: "SemanticCanvas") -> str | None:
        """Generate a full narrative from the accumulated canvas."""

    @abstractmethod
    def narrate_layer(self, entry: "CanvasEntry", canvas: "SemanticCanvas") -> str | None:
        """Generate a narrative for a single layer's contribution."""

    @abstractmethod
    def narrate_kernel(self, kernel_label: dict, canvas: "SemanticCanvas") -> str | None:
        """Generate a narrative for a single UKT kernel."""

    @abstractmethod
    def narrate_concept(self, concept_label: dict, canvas: "SemanticCanvas") -> str | None:
        """Generate a narrative for an SAE-discovered concept."""

    def narrate_reality_regression(
        self, snapshot: dict, canvas: "SemanticCanvas",
        feature_name_fn: Any = None,
        region_for_index_fn: Any = None,
        region_bounds: dict | None = None,
    ) -> str | None:
        """Generate a narrative for the reality regression vector.

        Args:
            snapshot: UKT snapshot with reality_regression, kernel_labels, etc.
            canvas: The semantic canvas with accumulated state.
            feature_name_fn: Callable(idx) -> str for feature names.
            region_for_index_fn: Callable(idx) -> str for region names.
            region_bounds: {name: (start, end)} for region energy computation.
        """
        return None


# --------------------------------------------------------------------------- #
# Template-based narrator (no model needed)                                    #
# --------------------------------------------------------------------------- #

class TemplateNarrator(NarratorBackend):
    """Deterministic template-based narrator. Fast, no model required.

    Generates structured analytical text from the semantic coordinates.
    """

    def narrate_canvas(self, canvas: "SemanticCanvas") -> str | None:
        state = canvas.get_accumulated_state()
        dominant = state.get("dominant_narrative", [])
        if not dominant:
            return "No dominant patterns detected across the analysis layers."

        parts = []
        for d in dominant:
            strength = "strongly" if d["value"] > 0.7 else "moderately"
            parts.append(f"The analysis {strength} indicates {d['label'].lower()} "
                         f"(score: {d['value']:.2f}).")

        n_layers = len(state.get("entries", []))
        parts.append(f"This assessment integrates signals from {n_layers} processing layers.")
        return " ".join(parts)

    def narrate_layer(self, entry: "CanvasEntry", canvas: "SemanticCanvas") -> str | None:
        parts = [f"The {entry.block_name} layer"]
        strong_dims = []
        for i, dim_key in enumerate(
            [d.key for d in canvas.dimensions] if hasattr(canvas, 'dimensions') else []
        ):
            if i < len(entry.coordinates) and entry.coordinates[i] > 0.3:
                strength = "strongly" if entry.coordinates[i] > 0.7 else "moderately"
                dim = canvas.dimensions[i]
                strong_dims.append(f"{strength} activates {dim.label.lower()}")
        if strong_dims:
            parts.append(", ".join(strong_dims))
        else:
            parts.append("shows weak signals across all dimensions")
        parts.append(f"with {entry.active_concepts} active concepts discovered.")
        return " ".join(parts)

    def narrate_kernel(self, kernel_label: dict, canvas: "SemanticCanvas") -> str | None:
        importance = kernel_label.get("importance", 0)
        region = kernel_label.get("dominant_region", "unknown").replace("-", " ")
        block = kernel_label.get("dominant_block", "unknown")
        features = kernel_label.get("top_features", [])

        feat_desc = ", ".join(f['name'] for f in features[:3])
        return (
            f"Kernel {kernel_label.get('kernel_id', '?')} explains "
            f"{importance:.1%} of variance, driven primarily by {region} "
            f"patterns from {block}. Key features: {feat_desc}."
        )

    def narrate_concept(self, concept_label: dict, canvas: "SemanticCanvas") -> str | None:
        cid = concept_label.get("concept_id", "C??")
        region = concept_label.get("dominant_region", "unknown").replace("-", " ")
        activation = concept_label.get("mean_activation", 0)
        features = concept_label.get("top_features", [])
        feat_desc = ", ".join(f['name'] for f in features[:3])
        return (
            f"Concept {cid} encodes {region} patterns "
            f"(activation: {activation:.4f}), loading on: {feat_desc}."
        )

    def narrate_reality_regression(
        self, snapshot: dict, canvas: "SemanticCanvas",
        feature_name_fn=None, region_for_index_fn=None,
        region_bounds=None,
    ) -> str | None:
        rr = snapshot.get("reality_regression")
        if rr is None:
            return None

        top_idx = np.argsort(np.abs(rr))[-3:][::-1]
        parts = []
        for idx in top_idx:
            name = feature_name_fn(int(idx)) if feature_name_fn else f"feature_{idx}"
            parts.append(f"{name} ({rr[int(idx)]:+.4f})")

        return (
            f"The reality regression is most influenced by: {', '.join(parts)}. "
            f"This vector summarizes {snapshot.get('n_kernels', 0)} kernels into "
            f"a single direction of maximum explained variance."
        )


# --------------------------------------------------------------------------- #
# LLM-based narrator                                                           #
# --------------------------------------------------------------------------- #

class LLMNarrator(NarratorBackend):
    """Narrator using a small language model for richer text generation.

    Loads a causal LM (default: arnir0/Tiny-LLM) and generates continuations
    from structured prompts. Falls back to TemplateNarrator on failure.

    Args:
        model_name: HuggingFace model name/path.
        max_new_tokens: Maximum tokens to generate per call.
        temperature: Sampling temperature.
        cache_fn: Optional caching function (e.g., st.cache_resource).
    """

    def __init__(
        self,
        model_name: str = "arnir0/Tiny-LLM",
        max_new_tokens: int = 60,
        temperature: float = 0.7,
        cache_fn: Any = None,
        generation_timeout: float = 15.0,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._cache_fn = cache_fn
        self._generation_timeout = generation_timeout
        self._model = None
        self._tokenizer = None
        self._fallback = TemplateNarrator()
        self._timeout_count = 0       # Consecutive timeouts
        self._timeout_threshold = 3   # Permanently disable after N consecutive

    def _load_model(self):
        """Load model and tokenizer."""
        if self._model is not None:
            return self._model, self._tokenizer

        def _do_load():
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
                model.eval()
                return model, tokenizer
            except Exception:
                return None, None

        if self._cache_fn:
            result = self._cache_fn(_do_load)()
        else:
            result = _do_load()

        if isinstance(result, tuple):
            self._model, self._tokenizer = result
        return self._model, self._tokenizer

    def _generate(self, prompt: str, max_tokens: int | None = None) -> str | None:
        """Generate text continuation with timeout protection.

        Uses a thread-based timeout to prevent blocking the pipeline when
        CPU inference is too slow. Falls back to None (triggering template
        narrator) on timeout.
        """
        if self._timeout_count >= self._timeout_threshold:
            return None  # Too many consecutive timeouts; skip LLM

        model, tokenizer = self._load_model()
        if model is None or tokenizer is None:
            return None

        try:
            import torch
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

            inputs = tokenizer.encode(prompt, return_tensors="pt",
                                      truncation=True, max_length=400)

            def _run_generation():
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=max_tokens or self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                return outputs

            # Use daemon thread so it won't block process exit, and
            # cancel the future on timeout so the executor __exit__
            # doesn't wait for the worker to finish.
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_run_generation)
            executor.shutdown(wait=False)
            try:
                outputs = future.result(timeout=self._generation_timeout)
                self._timeout_count = 0  # Reset on success
            except FuturesTimeout:
                future.cancel()
                self._timeout_count += 1
                return None

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = full_text[len(tokenizer.decode(inputs[0],
                                                          skip_special_tokens=True)):]
            return _postprocess(continuation)
        except Exception:
            return None

    def narrate_canvas(self, canvas: "SemanticCanvas") -> str | None:
        prompt = canvas.format_for_narrator()
        result = self._generate(prompt, max_tokens=80)
        return result or self._fallback.narrate_canvas(canvas)

    def narrate_layer(self, entry: "CanvasEntry", canvas: "SemanticCanvas") -> str | None:
        coords = entry.coordinates
        parts = []
        for i, dim in enumerate(canvas.dimensions):
            if i < len(coords) and coords[i] > 0.3:
                strength = "strongly" if coords[i] > 0.7 else "moderately"
                parts.append(f"{strength} activates {dim.label.lower()}")

        state = canvas.get_accumulated_state()
        prior_blocks = [e.block_name for e in state["entries"] if e.step < entry.step]
        prior_context = ""
        if prior_blocks:
            prior_context = f" Building on prior analysis of {', '.join(prior_blocks)}, "

        dim_description = "; ".join(parts) if parts else "shows weak signals across all dimensions"

        prompt = (
            f"Cross-domain analytical report, layer {entry.step}:\n\n"
            f"The {entry.block_name} analysis stage processed its data and "
            f"{dim_description}.{prior_context}\n"
            f"{entry.active_concepts} sparse concepts were discovered.\n\n"
            f"The {entry.block_name} layer reveals that"
        )
        result = self._generate(prompt, max_tokens=60)
        return result or self._fallback.narrate_layer(entry, canvas)

    def narrate_kernel(self, kernel_label: dict, canvas: "SemanticCanvas") -> str | None:
        importance = kernel_label.get("importance", 0)
        region = kernel_label.get("dominant_region", "unknown").replace("-", " ")
        block = kernel_label.get("dominant_block", "unknown")
        features = kernel_label.get("top_features", [])

        feat_desc = ", ".join(
            f"{f['name']} (loading {f['loading']:+.3f})" for f in features[:3]
        )

        state = canvas.get_accumulated_state()
        dominant_dims = state.get("dominant_narrative", [])
        canvas_context = ""
        if dominant_dims:
            dim_strs = [f"{d['label']} ({d['value']:.2f})" for d in dominant_dims[:2]]
            canvas_context = f" The semantic canvas shows dominant signals in {' and '.join(dim_strs)}."

        prompt = (
            f"Neural network kernel interpretation:\n\n"
            f"Kernel {kernel_label.get('kernel_id', '?')} explains "
            f"{importance:.1%} of the total variance. "
            f"Driven by {region} patterns from {block}. "
            f"Top loadings: {feat_desc}.{canvas_context}\n\n"
            f"In plain language, this kernel represents"
        )
        result = self._generate(prompt, max_tokens=60)
        return result or self._fallback.narrate_kernel(kernel_label, canvas)

    def narrate_concept(self, concept_label: dict, canvas: "SemanticCanvas") -> str | None:
        cid = concept_label.get("concept_id", "C??")
        region = concept_label.get("dominant_region", "unknown").replace("-", " ")
        activation = concept_label.get("mean_activation", 0)
        features = concept_label.get("top_features", [])

        feat_desc = ", ".join(
            f"{f['name']} ({f['loading']:+.3f})" for f in features[:3]
        )

        prompt = (
            f"Sparse Autoencoder concept analysis:\n\n"
            f"Concept {cid} is a pattern discovered in the {region} region. "
            f"Mean activation: {activation:.4f}. Loadings: {feat_desc}.\n\n"
            f"This concept captures the idea that"
        )
        result = self._generate(prompt, max_tokens=50)
        return result or self._fallback.narrate_concept(concept_label, canvas)

    def narrate_reality_regression(
        self, snapshot: dict, canvas: "SemanticCanvas",
        feature_name_fn=None, region_for_index_fn=None,
        region_bounds=None,
    ) -> str | None:
        rr = snapshot.get("reality_regression")
        if rr is None:
            return None

        top_idx = np.argsort(np.abs(rr))[-5:][::-1]
        top_feats = []
        for i in top_idx:
            name = feature_name_fn(int(i)) if feature_name_fn else f"feature_{i}"
            rname = region_for_index_fn(int(i)) if region_for_index_fn else "unknown"
            top_feats.append(f"{name} [{rname}] = {rr[int(i)]:+.4f}")

        region_lines = []
        if region_bounds:
            for rname, (lo, hi) in region_bounds.items():
                energy = float(np.abs(rr[lo:hi]).sum())
                if energy > 0.01:
                    region_lines.append(f"{rname.replace('-', ' ')}: {energy:.3f}")

        canvas_text = ""
        state = canvas.get_accumulated_state()
        dominant = state.get("dominant_narrative", [])
        if dominant:
            dim_names = [d["label"] for d in dominant]
            canvas_text = f" Dominant themes: {', '.join(dim_names)}."

        n_kernels = snapshot.get("n_kernels", 0)

        prompt = (
            f"Final reality assessment — feature synthesis:\n\n"
            f"After integrating {n_kernels} data domains, the system computed a "
            f"unified reality regression vector.\n\n"
            f"Top features: {'; '.join(top_feats[:3])}.\n"
            f"Region energy: {'; '.join(region_lines)}."
            f"{canvas_text}\n\n"
            f"The overall assessment is that"
        )
        result = self._generate(prompt, max_tokens=80)
        return result or self._fallback.narrate_reality_regression(
            snapshot, canvas, feature_name_fn, region_for_index_fn, region_bounds,
        )


# --------------------------------------------------------------------------- #
# Utility                                                                      #
# --------------------------------------------------------------------------- #

def _postprocess(text: str) -> str:
    """Clean up generated text: truncate at last complete sentence."""
    text = text.strip()
    if not text:
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1 and not sentences[-1].rstrip().endswith(('.', '!', '?')):
        sentences = sentences[:-1]
    result = ' '.join(sentences)

    final_sentences = re.split(r'(?<=[.!?])\s+', result)
    if len(final_sentences) > 3:
        result = ' '.join(final_sentences[:3])

    return result.strip()


def get_narrator(
    backend: str = "template",
    model_name: str = "arnir0/Tiny-LLM",
    cache_fn: Any = None,
) -> NarratorBackend:
    """Factory function to get a narrator backend.

    Args:
        backend: "template" for fast deterministic, "llm" for model-based.
        model_name: Model name for LLM backend.
        cache_fn: Optional caching function for model loading.

    Returns:
        A NarratorBackend instance.
    """
    if backend == "llm":
        return LLMNarrator(model_name=model_name, cache_fn=cache_fn)
    return TemplateNarrator()
