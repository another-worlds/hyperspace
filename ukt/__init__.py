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
"""

from ukt.registry import FeatureRegionRegistry
from ukt.tensor import UniversalKnowledgeTensor
from ukt.extractors import HookExtractor
from ukt.kernels import decompose_svd, label_kernel, generate_kernel_narrative
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

__all__ = [
    "UniversalKnowledgeTensor",
    "FeatureRegionRegistry",
    "HookExtractor",
    "decompose_svd",
    "label_kernel",
    "generate_kernel_narrative",
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
