"""UKT: Universal Knowledge Tensor — standalone framework for neural network
feature extraction, standardization, and kernel-based interpretation.

Neural networks are complex, uncertain feature generators. They operate on
simple optimization but become powerful at scale. What they lack is
standardization: every hidden layer produces opaque activations with no
common language across architectures.

UKT solves this by:
  1. Extracting features from any neural network via hook-based extractors
  2. Registering named feature regions so activations carry semantic meaning
  3. Normalizing features to a common [0,1] scale per region
  4. Decomposing the combined feature matrix via SVD into interpretable kernels
  5. Computing a "reality regression" — a single weighted direction summarizing
     all discovered structure
  6. Labeling each kernel with data-grounded narratives

Usage:
    from ukt import UniversalKnowledgeTensor, FeatureRegionRegistry, HookExtractor

    # 1. Define your feature space
    registry = FeatureRegionRegistry()
    registry.register("encoder_attention", 0, 16, "Attention patterns from encoder")
    registry.register("hidden_repr", 16, 48, "Hidden layer representations")

    # 2. Create the UKT
    tensor = UniversalKnowledgeTensor(registry)

    # 3. Extract features from your model
    extractor = HookExtractor(your_model)
    extractor.attach("encoder.self_attn", region="encoder_attention")
    extractor.attach("encoder.ffn", region="hidden_repr")
    output = your_model(input_data)
    features = extractor.collect(registry)

    # 4. Add to UKT and get kernel interpretation
    snapshot = tensor.add_block("my_encoder", features)
    print(snapshot["kernel_labels"])
    print(snapshot["reality_regression"])

Configuration bundle usage:
    from ukt import UKTConfig, SharedProjection, create_ukt

    config = UKTConfig(
        registry=my_registry,
        projection=SharedProjection(block_names=["enc", "dec"]),
    )
    tensor = create_ukt(config)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ukt.registry import FeatureRegionRegistry
from ukt.tensor import UniversalKnowledgeTensor
from ukt.extractors import HookExtractor
from ukt.kernels import decompose_svd, label_kernel, generate_kernel_narrative
from ukt.projection import SharedProjection
from ukt.stability import estimate_regression_stability
from ukt.interfaces import (
    ModelInterface,
    LayerSpec,
    LLMInterface,
    CNNInterface,
    DeepLinearInterface,
    unified_registry,
    extract_all,
)


# --------------------------------------------------------------------------- #
# Configuration bundle                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class UKTConfig:
    """Configuration bundle for :func:`create_ukt`.

    Bundles all optional constructor kwargs so callers can define a UKT setup
    once and reuse or pass it around without threading individual kwargs.

    Attributes:
        registry: Feature region registry.  If None, a default 80-dim single-
            region registry is created by the constructor.
        projection: Optional :class:`SharedProjection` for adaptive cross-block
            feature mixing.
        normalizer: Optional callable ``(arr, registry) -> arr`` that replaces
            the default per-region [0, 1] normalization.
        on_snapshot: Optional callable ``(snapshot) -> snapshot | None`` called
            after each snapshot is assembled.  Use to inject domain-specific
            fields without subclassing.
    """
    registry: FeatureRegionRegistry | None = None
    projection: SharedProjection | None = None
    normalizer: Callable | None = None
    on_snapshot: Callable | None = None


def create_ukt(config: UKTConfig | None = None) -> UniversalKnowledgeTensor:
    """Create a :class:`UniversalKnowledgeTensor` from a :class:`UKTConfig`.

    Args:
        config: Configuration bundle.  If None, defaults to
            ``UKTConfig()`` which produces a plain UKT with a single 80-dim
            region and no projection or hooks.

    Returns:
        Configured :class:`UniversalKnowledgeTensor` instance.
    """
    cfg = config or UKTConfig()
    return UniversalKnowledgeTensor(
        cfg.registry,
        projection=cfg.projection,
        normalizer=cfg.normalizer,
        on_snapshot=cfg.on_snapshot,
    )


__all__ = [
    # Core
    "UniversalKnowledgeTensor",
    "FeatureRegionRegistry",
    "HookExtractor",
    # Config / factory
    "UKTConfig",
    "create_ukt",
    # SVD / kernel utilities
    "decompose_svd",
    "label_kernel",
    "generate_kernel_narrative",
    # Cross-block projection
    "SharedProjection",
    # Stability
    "estimate_regression_stability",
    # Model interfaces
    "ModelInterface",
    "LayerSpec",
    "LLMInterface",
    "CNNInterface",
    "DeepLinearInterface",
    "unified_registry",
    "extract_all",
]
