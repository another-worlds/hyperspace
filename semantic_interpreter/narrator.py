"""Narrative generation backends for the Semantic Interpreter.

Translates structured semantic coordinates into human-readable text.
Supports multiple backends:
  - TemplateNarrator: deterministic, no model required
  - LLMNarrator: uses a small language model for richer narratives
  - Custom backends: implement NarratorBackend protocol

The narrator is pluggable — use templates for speed, LLMs for richness,
or write your own backend for domain-specific language.

GOVERNANCE WARNING (VISION.md invariants 5, 9, 10):
  TemplateNarrator is a provenance formatter, NOT an interpreter. It is
  permitted to list measured quantities (loadings, activations, variance
  shares) in sentence form. It is NOT permitted to emit interpretive verbs
  ("encodes", "indicates", "represents", "drives") or single-block labels
  for cross-block concepts. When LLMNarrator is unavailable, the governance
  pipeline must raise a hard failure — TemplateNarrator is not a fallback
  for interpretation. See VISION.md invariants 5, 9, 10 and CLAUDE.md
  Grammar Rules for Interpretive Text.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from semantic_interpreter.canvas import SemanticCanvas, CanvasEntry


# --------------------------------------------------------------------------- #
# Few-shot exemplars for instruction-style prompts                             #
# Gold-standard examples that teach the model register, structure, and depth.  #
# --------------------------------------------------------------------------- #

_EXEMPLARS: dict[str, str] = {
    "canvas": (
        "### Example\n"
        "Data: K0 (45.2% var, Finance 0.61, Graph 0.27), equity_volatility_7d_zscore "
        "(+0.41), trade_centrality (+0.33). C02 active across temporal-pattern and "
        "structural-centrality. 5 layers.\n\n"
        "Summary: Financial volatility and geopolitical network centrality jointly drive "
        "45% of the signal — when markets stress, the most connected actors shift "
        "position. This cross-domain coupling means market instability is structurally "
        "linked to alliance dynamics, not isolated.\n\n"
    ),
    "kernel": (
        "### Example\n"
        "Data: K1 22.8% var. topic_diversity_entropy (+0.52), sentiment_mean_30d "
        "(-0.31), geo_event_density (+0.19).\n\n"
        "Interpretation: News diversity and sentiment move inversely — broadening topics "
        "coincide with dropping sentiment and concentrated geopolitical events, linking "
        "media fragmentation to on-the-ground activity.\n\n"
    ),
    "concept": (
        "### Example\n"
        "Data: C05 activation 0.73. equity_volatility_7d_zscore (+0.41), "
        "bond_spread_delta (-0.28). Cross-block: temporal (+0.38), semantic (+0.22), "
        "structural (-0.11).\n\n"
        "Explanation: This concept links market stress to news sentiment and network "
        "structure across three domains — a genuine cross-domain signal, not a "
        "single-source artifact.\n\n"
    ),
    "reality": (
        "### Example\n"
        "Data: equity_volatility_7d_zscore [temporal] +0.041, trade_centrality "
        "[structural] +0.030. Region energy: temporal 0.089, structural 0.065.\n\n"
        "Assessment: Temporal market features dominate but structural network dynamics "
        "reinforce the signal. The negative topic-entropy loading means narrowing news "
        "coverage accompanies instability — monitor equity-volatility / trade-centrality "
        "coupling as an early warning.\n\n"
    ),
}


# --------------------------------------------------------------------------- #
# Narrator protocol                                                            #
# --------------------------------------------------------------------------- #

class NarratorBackend(ABC):
    """Base class for narrative generation backends."""

    @abstractmethod
    def narrate_canvas(
        self,
        canvas: "SemanticCanvas",
        kernel_labels: list[dict] | None = None,
    ) -> str | None:
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

    def narrate_canvas(
        self,
        canvas: "SemanticCanvas",
        kernel_labels: list[dict] | None = None,
    ) -> str | None:
        state = canvas.get_accumulated_state()
        dominant = state.get("dominant_narrative", [])
        if not dominant and not kernel_labels:
            return "No dominant patterns detected across the analysis layers."

        parts = []
        # Kernel-aware narrative should reference emergent structure first
        if kernel_labels:
            sorted_kernels = sorted(
                kernel_labels,
                key=lambda k: k.get("importance", 0),
                reverse=True,
            )
            top_kernels = sorted_kernels[:3]
            kernel_parts = []
            for kl in top_kernels:
                cid = kl.get("kernel_id", "?")
                imp = kl.get("importance", 0.0)
                contrib = kl.get("contributing_blocks", [])
                contrib_desc = ", ".join(f"{b} ({v:.2f})" for b, v in contrib[:2])
                kernel_parts.append(
                    f"{cid} ({imp:.1%} var, {contrib_desc if contrib_desc else 'single-block'})"
                )
            parts.append(
                f"Emergent kernels driving the analysis: {', '.join(kernel_parts)}."
            )

        if dominant:
            for d in dominant:
                # Emit provenance only, no interpretive verbs (CLAUDE.md Grammar Rules #4)
                parts.append(
                    f"Concept {d['label'].lower()} active "
                    f"(score: {d['value']:.2f})."
                )

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
        # Provenance only: variance explained and feature loadings (Grammar Rule #4)
        return (
            f"Kernel {kernel_label.get('kernel_id', '?')}: "
            f"{importance:.1%} variance explained. "
            f"Top loadings from {block}: {feat_desc}."
        )

    def narrate_concept(self, concept_label: dict, canvas: "SemanticCanvas") -> str | None:
        cid = concept_label.get("concept_id", "C??")
        activation = concept_label.get("mean_activation", 0)
        features = concept_label.get("top_features", [])
        
        # Check if cross-block signature is available (new format)
        cross_block_sig = concept_label.get("cross_block_signature", {})
        if cross_block_sig:
            # Show cross-block signature instead of single dominant region
            sig_parts = []
            for region, contrib in cross_block_sig.items():
                if abs(contrib) > 0.01:  # Only show meaningful contributions
                    sig_parts.append(f"{region} ({contrib:+.2f})")
            sig_desc = ", ".join(sig_parts) if sig_parts else "no significant loadings"
            feat_desc = ", ".join(f['name'] for f in features[:2])
            # Provenance only: activation, loadings, no interpretive verbs
            return (
                f"Concept {cid}: activation {activation:.4f}. "
                f"Cross-block signature: {sig_desc}. "
                f"Top features: {feat_desc}."
            )
        else:
            # Fallback for old format (until all concepts updated)
            region = concept_label.get("dominant_region_hint", "unknown").replace("-", " ")
            feat_desc = ", ".join(f['name'] for f in features[:3])
            return (
                f"Concept {cid}: activation {activation:.4f}, "
                f"loading on: {feat_desc} (provenance: {region})."
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

        kernel_labels = snapshot.get("kernel_labels", [])
        kernel_part = ""
        if kernel_labels:
            top_k = sorted(kernel_labels, key=lambda k: k.get("importance", 0), reverse=True)[:3]
            kernel_part = " Top kernels: " + ", ".join(
                f"{k.get('kernel_id','?')} ({k.get('importance',0.0):.1%})"
                for k in top_k
            ) + "."

        return (
            f"The reality regression is most influenced by: {', '.join(parts)}. "
            f"This vector summarizes {snapshot.get('n_kernels', 0)} kernels into "
            f"a single direction of maximum explained variance.{kernel_part}"
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
        max_new_tokens: int = 3000,
        temperature: float = 0.7,
        cache_fn: Any = None,
        generation_timeout: float = 30.0,
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
        self._model_load_failed = False  # Avoid repeated load retries on persistent failure

    def _load_model(self):
        """Load model and tokenizer."""
        if self._model is not None:
            return self._model, self._tokenizer

        if self._model_load_failed:
            logger.warning("Skipping model load: previous load has failed permanently.")
            return None, None

        def _do_load():
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
                model.eval()
                return model, tokenizer
            except Exception as exc:
                logger.error(
                    "Failed to load LLM model %s: %s",
                    self.model_name,
                    exc,
                    exc_info=True,
                )
                self._model_load_failed = True
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
            logger.warning(
                "LLM generation disabled after %d consecutive timeouts",
                self._timeout_count,
            )
            return None  # Too many consecutive timeouts; skip LLM

        model, tokenizer = self._load_model()
        if model is None or tokenizer is None:
            logger.warning("LLM model not available; using fallback narrative.")
            return None

        try:
            import torch
            import threading
            import queue

            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=768,
                               return_attention_mask=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Respect model's context window: leave room for generation
            model_max_len = getattr(model.config, "max_position_embeddings", 1024)
            input_len = input_ids.shape[1]
            budget = max(model_max_len - input_len, 32)
            actual_max_tokens = min(max_tokens or self.max_new_tokens, budget)

            result_queue: queue.Queue = queue.Queue()

            def _run_generation():
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=actual_max_tokens,
                            do_sample=True,
                            temperature=self.temperature,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    result_queue.put((True, outputs))
                except Exception as e:
                    result_queue.put((False, e))

            thread = threading.Thread(target=_run_generation, daemon=True)
            thread.start()

            try:
                success, data = result_queue.get(timeout=self._generation_timeout)
                if not success:
                    raise data
                outputs = data
                self._timeout_count = 0  # Reset on success
            except queue.Empty:
                self._timeout_count += 1
                logger.warning(
                    "LLM generation timeout (%ss) for %s (count %d/%d)",
                    self._generation_timeout,
                    self.model_name,
                    self._timeout_count,
                    self._timeout_threshold,
                )
                return None

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = full_text[len(tokenizer.decode(input_ids[0],
                                                          skip_special_tokens=True)):]
            return _postprocess(continuation)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc, exc_info=True)
            return None

    def _refine(self, draft: str, context: str = "") -> str:
        """Second-pass refinement: tighten a draft narrative.

        Asks the model to improve an existing draft by making it more
        specific, removing hedging, and grounding claims in evidence.
        Returns the original draft if refinement fails or produces garbage.
        """
        prompt = (
            f"### Draft narrative\n{draft}\n\n"
            f"{context}"
            f"### Instructions\n"
            f"Rewrite the draft above to be more specific and direct. "
            f"Replace vague phrases with concrete feature names and percentages. "
            f"Remove hedging words (may, could, might, possibly, seems). "
            f"Keep the same meaning but make every sentence testable against the data. "
            f"Preserve factual claims from the draft — do not invent new findings.\n\n"
            f"Revised:"
        )
        result = self._generate(prompt)
        if result and self._passes_quality_gate(result):
            return result
        return draft

    def _passes_quality_gate(self, text: str) -> bool:
        """Check if generated text passes the quality gate.

        Returns True if text is coherent enough to display.
        """
        alpha_chars = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_chars / max(len(text), 1)
        tokens = text.split()
        real_words = sum(1 for t in tokens if sum(c.isalpha() for c in t) >= 3)
        word_ratio = real_words / max(len(tokens), 1)
        return alpha_ratio >= 0.5 and word_ratio >= 0.5

    def narrate_canvas(
        self,
        canvas: "SemanticCanvas",
        kernel_labels: list[dict] | None = None,
    ) -> str | None:
        # --- Phase 1: Generate intermediate concept & kernel narratives ---
        # This gives the model its own intermediate reasoning to synthesize from,
        # rather than just raw numbers (VISION invariant 9: cross-item reasoning).
        intermediate_parts = []

        # Concept mini-narratives
        concept_labels = []
        for entry in canvas.entries:
            if hasattr(entry, 'concept_activations'):
                # Pull from session state if available
                break
        # Try to get concept labels from canvas dimensions
        state = canvas.get_accumulated_state()
        active_dims = [
            d for d in state.get("dominant_narrative", [])
            if d.get("value", 0) > 0.05
        ]
        if active_dims:
            for d in active_dims[:5]:
                intermediate_parts.append(
                    f"- {d['label']} (score {d['value']:.2f}): {d.get('desc', 'no description')}"
                )

        # Kernel mini-narratives (generate each, then feed into synthesis)
        kernel_summaries = []
        if kernel_labels:
            top_kernels = sorted(
                kernel_labels,
                key=lambda k: k.get("importance", 0),
                reverse=True,
            )[:3]
            for kl in top_kernels:
                mini = self.narrate_kernel(kl, canvas)
                if mini and self._passes_quality_gate(mini):
                    kernel_summaries.append(mini)
                else:
                    # Fallback: structured provenance
                    contrib = kl.get("contributing_blocks", [])
                    contrib_desc = ", ".join(f"{b} ({v:.2f})" for b, v in contrib[:2])
                    feats = ", ".join(f['name'] for f in kl.get('top_features', [])[:3])
                    kernel_summaries.append(
                        f"{kl.get('kernel_id', '?')}: {kl.get('importance', 0):.1%} variance, "
                        f"blocks: {contrib_desc or 'single-block'}, features: {feats}."
                    )

        # --- Phase 2: Assemble synthesis prompt with intermediate reasoning ---
        prompt = canvas.format_for_narrator(kernel_labels=kernel_labels)

        if kernel_summaries:
            prompt += "\n\n### Kernel interpretations (generated above)\n"
            for i, ks in enumerate(kernel_summaries):
                prompt += f"{i+1}. {ks}\n"

        if intermediate_parts:
            prompt += "\n\n### Active concept signals\n"
            prompt += "\n".join(intermediate_parts)

        # Few-shot exemplar
        prompt += "\n\n" + _EXEMPLARS["canvas"]

        prompt += (
            "### Instructions\n"
            "Now write a 3-5 sentence analytical summary that synthesizes the kernel "
            "interpretations and concept signals above. Explain how the different data "
            "domains interact — do not describe each kernel separately. Address a "
            "non-technical governance audience. Use specific feature names and percentages. "
            "Be direct about what the data shows and its policy implications.\n\n"
            "Summary:"
        )

        result = self._generate(prompt)
        if result is None:
            logger.info("narrate_canvas: LLM not available, using template fallback")
            return self._fallback.narrate_canvas(canvas, kernel_labels=kernel_labels)

        if not self._passes_quality_gate(result):
            logger.warning(
                "narrate_canvas: LLM output failed quality gate, using template fallback",
            )
            return self._fallback.narrate_canvas(canvas, kernel_labels=kernel_labels)

        # --- Phase 3: Refinement pass ---
        result = self._refine(
            result,
            context=(
                f"The narrative is about a cross-domain intelligence analysis with "
                f"{len(canvas.entries)} data sources and "
                f"{len(kernel_summaries)} emergent kernels.\n\n"
            ),
        )

        return result

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
            f"### Instructions\n"
            f"Write 2-3 sentences explaining what the {entry.block_name} layer contributed "
            f"to the overall analysis. Be specific about signal strengths and which "
            f"semantic dimensions were activated. Address a governance audience.\n\n"
            f"Analysis:"
        )
        result = self._generate(prompt)
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
            f"Kernel analysis data:\n\n"
            f"Kernel {kernel_label.get('kernel_id', '?')} explains "
            f"{importance:.1%} of the total variance. "
            f"Driven by {region} patterns from {block}. "
            f"Top loadings: {feat_desc}.{canvas_context}\n\n"
            f"{_EXEMPLARS['kernel']}"
            f"### Instructions\n"
            f"Write 2-4 sentences explaining what this kernel means for a governance audience. "
            f"Name the specific data domains that interact in this kernel and explain "
            f"what their co-variance implies. Use the feature names directly. "
            f"Do not say 'this kernel represents' — say what it reveals about the data.\n\n"
            f"Interpretation:"
        )
        result = self._generate(prompt)
        if result and self._passes_quality_gate(result):
            result = self._refine(result)
            return result
        return self._fallback.narrate_kernel(kernel_label, canvas)

    def narrate_concept(self, concept_label: dict, canvas: "SemanticCanvas") -> str | None:
        cid = concept_label.get("concept_id", "C??")
        region = concept_label.get("dominant_region_hint", concept_label.get("dominant_region", "unknown")).replace("-", " ")
        activation = concept_label.get("mean_activation", 0)
        features = concept_label.get("top_features", [])

        feat_desc = ", ".join(
            f"{f['name']} ({f['loading']:+.3f})" for f in features[:3]
        )

        # Include cross-block signature if available
        cross_sig = concept_label.get("cross_block_signature", {})
        sig_desc = ""
        if cross_sig:
            sig_parts = [f"{r} ({v:+.2f})" for r, v in cross_sig.items() if abs(v) > 0.01]
            if sig_parts:
                sig_desc = f" Cross-block signature: {', '.join(sig_parts)}."

        prompt = (
            f"SAE concept data:\n\n"
            f"Concept {cid} — mean activation: {activation:.4f}. "
            f"Primary provenance: {region}. Top loadings: {feat_desc}.{sig_desc}\n\n"
            f"{_EXEMPLARS['concept']}"
            f"### Instructions\n"
            f"Write 2-3 sentences explaining what pattern this concept captures. "
            f"If it spans multiple data domains, explain the cross-domain interaction. "
            f"Use specific feature names. Address a non-technical audience.\n\n"
            f"Explanation:"
        )
        result = self._generate(prompt)
        if result and self._passes_quality_gate(result):
            result = self._refine(result)
            return result
        return self._fallback.narrate_concept(concept_label, canvas)

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

        kernel_infos = ""
        kernel_labels = snapshot.get("kernel_labels", [])
        if kernel_labels:
            top_k = sorted(kernel_labels, key=lambda k: k.get("importance", 0), reverse=True)[:3]
            kernel_infos = "\n\nKernel summary: " + ", ".join(
                f"{k.get('kernel_id','?')} ({k.get('importance',0.0):.1%})"
                for k in top_k
            ) + "."

        prompt = (
            f"Reality regression data:\n\n"
            f"After integrating {n_kernels} data domains, the system computed a "
            f"unified reality regression vector.\n\n"
            f"Top features: {'; '.join(top_feats[:3])}.\n"
            f"Region energy: {'; '.join(region_lines)}."
            f"{canvas_text}{kernel_infos}\n\n"
            f"{_EXEMPLARS['reality']}"
            f"### Instructions\n"
            f"Write 3-5 sentences synthesizing this reality assessment for a "
            f"governance audience. Explain which features dominate, which data domains "
            f"drive the conclusions, and what this means in practical terms. "
            f"Cite specific percentages and feature names. Be direct and concrete.\n\n"
            f"Assessment:"
        )
        result = self._generate(prompt)
        if result is None:
            logger.info("narrate_reality_regression: LLM not available, using template fallback")
            return self._fallback.narrate_reality_regression(
                snapshot, canvas, feature_name_fn, region_for_index_fn, region_bounds,
            )
        if self._passes_quality_gate(result):
            result = self._refine(result)
            return result
        return self._fallback.narrate_reality_regression(
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
    if len(final_sentences) > 10:
        result = ' '.join(final_sentences[:10])

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
