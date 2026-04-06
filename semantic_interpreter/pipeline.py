"""End-to-end InterpretationPipeline for the Semantic Interpreter framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semantic_interpreter.canvas import SemanticCanvas, build_emergent_canvas
from semantic_interpreter.concepts import map_concepts_to_kernels
from semantic_interpreter.narrator import get_narrator
from semantic_interpreter.sae import train_global_sae


@dataclass
class InterpretationConfig:
    sae_hidden_dim: int = 32
    sae_epochs: int = 50
    sae_lr: float = 0.005
    narrator_backend: str = "template"  # "template" | "llm"
    narrator_model_name: str = "arnir0/Tiny-LLM"
    narrator_max_tokens: int = 60
    narrator_timeout: float = 30.0


@dataclass
class InterpretationResult:
    sae_result: dict | None
    canvas: SemanticCanvas
    concept_kernel_map: list[dict]
    narratives: dict[str, str | None]


class InterpretationPipeline:
    """Unified interpreter class for emergent neural interpretation."""

    def __init__(self, config: InterpretationConfig | None = None):
        self.config = config or InterpretationConfig()

    def interpret(
        self,
        feature_matrix: np.ndarray,
        block_names: list[str],
        kernel_snapshot: dict | None = None,
        registry: object | None = None,
    ) -> InterpretationResult:
        """Interpret an arbitrary feature matrix with the semantic interpreter."""
        sae_result = train_global_sae(
            feature_matrix,
            hidden_dim=self.config.sae_hidden_dim,
            epochs=self.config.sae_epochs,
            lr=self.config.sae_lr,
            registry=registry,
        )

        canvas = build_emergent_canvas(sae_result or {}, block_names) if sae_result else SemanticCanvas()

        concept_kernel_map = []
        if sae_result and kernel_snapshot is not None:
            concept_kernel_map = map_concepts_to_kernels(sae_result, kernel_snapshot)

        narrator = get_narrator(
            self.config.narrator_backend,
            model_name=self.config.narrator_model_name,
            cache_fn=None,
        )

        canvas_narrative = None
        reality_narrative = None

        try:
            canvas_narrative = narrator.narrate_canvas(canvas, kernel_labels=kernel_snapshot.get("kernel_labels") if kernel_snapshot else None)
        except Exception:
            canvas_narrative = None

        if kernel_snapshot is not None:
            try:
                reality_narrative = narrator.narrate_reality_regression(kernel_snapshot, canvas)
            except Exception:
                reality_narrative = None

        return InterpretationResult(
            sae_result=sae_result,
            canvas=canvas,
            concept_kernel_map=concept_kernel_map,
            narratives={
                "canvas_narrative": canvas_narrative,
                "reality_narrative": reality_narrative,
            },
        )
