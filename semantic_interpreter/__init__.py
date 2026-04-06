"""Semantic Interpreter: standalone framework for interpreting neural network
hidden layer features through concept discovery and semantic projection.

Neural networks generate features. This framework interprets them.

The Semantic Interpreter provides three layers of interpretation:

1. **Concept Discovery (SAE)**: Sparse Autoencoders decompose hidden-layer
   activations into a small number of interpretable "concepts." Most concepts
   are dormant for any given input — only the relevant ones activate. This
   sparsity is what makes interpretation possible.

2. **Semantic Canvas**: Discovered concepts are projected onto named semantic
   dimensions — human-readable axes like "momentum", "volatility", "cohesion".
   The canvas accumulates across layers/sources, building a trajectory of
   meaning through the network.

3. **Narrative Generation**: Structured semantic coordinates are translated
   into natural-language explanations via pluggable narrator backends (tiny
   LLMs, templates, or external APIs).

Usage:
    from semantic_interpreter import (
        SemanticCanvas, SemanticDimension,
        StageSAE, train_stage_sae, train_global_sae,
        map_concepts_to_kernels,
    )

    # 1. Define your semantic space
    dims = [
        SemanticDimension("momentum", "Market Momentum", "Direction of price movement"),
        SemanticDimension("volatility", "Volatility", "Degree of price fluctuation"),
    ]

    # 2. Define how feature regions map to semantic dimensions
    region_mapping = {
        "encoder_attention": [(0, 1.0), (1, 0.8)],  # maps to momentum + volatility
    }

    # 3. Create canvas and project features
    canvas = SemanticCanvas(dims, region_mapping)
    entry = canvas.project_block("encoder", 1, "encoder_attention", features, sae_result)

    # 4. Discover concepts via SAE
    sae_result = train_stage_sae(features, concept_dim=8)

    # 5. Get narrative interpretation
    print(canvas.format_for_narrator())
"""

from semantic_interpreter.canvas import (
    SemanticCanvas,
    SemanticDimension,
    CanvasEntry,
    build_emergent_canvas,
)
from semantic_interpreter.sae import (
    StageSAE,
    GlobalSAE,
    train_stage_sae,
    train_global_sae,
)
from semantic_interpreter.concepts import map_concepts_to_kernels
from semantic_interpreter.narrator import (
    NarratorBackend,
    TemplateNarrator,
    LLMNarrator,
    get_narrator,
)
from semantic_interpreter.pipeline import (
    InterpretationConfig,
    InterpretationResult,
    InterpretationPipeline,
)

__all__ = [
    "SemanticCanvas",
    "SemanticDimension",
    "CanvasEntry",
    "build_emergent_canvas",
    "StageSAE",
    "GlobalSAE",
    "train_stage_sae",
    "train_global_sae",
    "map_concepts_to_kernels",
    "NarratorBackend",
    "TemplateNarrator",
    "LLMNarrator",
    "get_narrator",
    "InterpretationConfig",
    "InterpretationResult",
    "InterpretationPipeline",
]
