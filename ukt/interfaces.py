"""Standardized model interfaces for classic architecture families.

Provides architecture-aware wrappers that know how to extract features from
LLMs (transformer-based), CNNs, and Deep Linear Networks, mapping them into
a common UKT feature space.  This enables **intermodel compatibility**: swap
the underlying architecture and the UKT analysis remains comparable.

Each interface:
  1. Builds a FeatureRegionRegistry tailored to the architecture
  2. Auto-attaches hooks via HookExtractor with appropriate reducers
  3. Provides ``extract(input_data) -> features`` for one-call extraction
  4. Produces architecture-specific feature metadata for provenance

Usage:
    from ukt.interfaces import LLMInterface, CNNInterface, DeepLinearInterface

    # Wrap any transformer
    llm = LLMInterface(my_transformer, layer_specs={...})
    snapshot = llm.extract_and_add(ukt, input_data, block_name="gpt")

    # Wrap any CNN
    cnn = CNNInterface(my_cnn, layer_specs={...})
    snapshot = cnn.extract_and_add(ukt, input_data, block_name="resnet")

    # Compare kernels across architectures
    print(ukt.get_latest_snapshot()["kernel_labels"])
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from ukt.registry import FeatureRegionRegistry
from ukt.extractors import HookExtractor, ManualExtractor
from ukt.utils import _pad_or_truncate


# ------------------------------------------------------------------ #
# Layer specification                                                 #
# ------------------------------------------------------------------ #

@dataclass
class LayerSpec:
    """Describes one layer to hook for feature extraction.

    Args:
        layer_path: Dot-separated path to the nn.Module
                    (e.g. "encoder.layers.0.self_attn").
        region:     Name of the UKT feature region this maps to.
        reducer:    Reducer name or callable (see extractors.BUILTIN_REDUCERS).
        output_index: Which tuple element to capture (None = first tensor).
        meta:       Extra metadata attached to every feature from this layer.
    """
    layer_path: str
    region: str
    reducer: str | Callable = "mean"
    output_index: int | None = None
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Abstract base                                                       #
# ------------------------------------------------------------------ #

class ModelInterface(abc.ABC):
    """Abstract interface that any classic model type must implement.

    Subclasses define *how* to configure the registry and extract features
    for a specific architecture family.  The UKT itself stays agnostic —
    only the interface knows the architecture details.
    """

    def __init__(self, model: Any, *, feature_dim: int = 80) -> None:
        self._model = model
        self._feature_dim = feature_dim
        self._registry: FeatureRegionRegistry | None = None
        self._extractor: HookExtractor | ManualExtractor | None = None
        self._layer_specs: list[LayerSpec] = []

    # -- public API ------------------------------------------------- #

    @property
    def registry(self) -> FeatureRegionRegistry:
        """Lazily build and return the feature region registry."""
        if self._registry is None:
            self._registry = self._build_registry()
        return self._registry

    @property
    def model_type(self) -> str:
        """Short identifier for the architecture family."""
        return self._model_type()

    def extract(self, input_data: Any) -> np.ndarray:
        """Run forward pass and return a UKT-ready feature vector.

        Args:
            input_data: Whatever the underlying model expects as input.

        Returns:
            (feature_dim,) numpy array, ready for ``ukt.add_block()``.
        """
        if self._extractor is None:
            self._setup_extractor()
        return self._extract_impl(input_data)

    def extract_and_add(
        self,
        ukt: Any,
        input_data: Any,
        block_name: str | None = None,
        timeframe_context: dict | None = None,
    ) -> dict:
        """Extract features and add them to a UKT in one call.

        Args:
            ukt:  UniversalKnowledgeTensor instance.
            input_data: Model input.
            block_name: Block name for the UKT (defaults to model_type).
            timeframe_context: Optional temporal context dict.

        Returns:
            UKT snapshot dict.
        """
        features = self.extract(input_data)
        meta = self._build_feature_meta()
        return ukt.add_block(
            name=block_name or self.model_type,
            features=features,
            feature_meta=meta,
            timeframe_context=timeframe_context,
        )

    def feature_meta(self) -> dict[int, dict]:
        """Return per-feature metadata for provenance tracking."""
        return self._build_feature_meta()

    def detach(self) -> None:
        """Remove all hooks from the model."""
        if isinstance(self._extractor, HookExtractor):
            self._extractor.detach_all()
            self._extractor = None

    # -- subclass contract ------------------------------------------ #

    @abc.abstractmethod
    def _model_type(self) -> str: ...

    @abc.abstractmethod
    def _build_registry(self) -> FeatureRegionRegistry: ...

    @abc.abstractmethod
    def _default_layer_specs(self) -> list[LayerSpec]: ...

    @abc.abstractmethod
    def _extract_impl(self, input_data: Any) -> np.ndarray: ...

    # -- shared helpers --------------------------------------------- #

    def _setup_extractor(self) -> None:
        """Attach hooks based on layer specs."""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for model interfaces")
        specs = self._layer_specs or self._default_layer_specs()
        extractor = HookExtractor(self._model)
        for spec in specs:
            extractor.attach(
                spec.layer_path,
                region=spec.region,
                reducer=spec.reducer,
                output_index=spec.output_index,
            )
        self._extractor = extractor

    def _collect_after_forward(self) -> np.ndarray:
        """Collect features from extractor after a forward pass."""
        assert isinstance(self._extractor, HookExtractor)
        features = self._extractor.collect(self.registry)
        self._extractor.reset()
        return features

    def _build_feature_meta(self) -> dict[int, dict]:
        """Build per-feature metadata from layer specs + registry."""
        meta: dict[int, dict] = {}
        reg = self.registry
        specs = self._layer_specs or self._default_layer_specs()
        spec_by_region = {s.region: s for s in specs}

        for region in reg.ordered_regions:
            spec = spec_by_region.get(region.name)
            for i in range(region.start, region.end):
                entry = {
                    "label": reg.feature_name(i),
                    "block": self.model_type,
                    "source": self.model_type,
                    "region": region.name,
                }
                if spec and spec.meta:
                    entry.update(spec.meta)
                meta[i] = entry
        return meta


# ------------------------------------------------------------------ #
# LLM / Transformer interface                                        #
# ------------------------------------------------------------------ #

class LLMInterface(ModelInterface):
    """Interface for transformer-based language models.

    Extracts three canonical feature families:
      - **attention_patterns**: Attention weight distributions (reduced via
        ``attention_weights`` reducer) — captures what the model attends to.
      - **hidden_repr**: Hidden-layer / FFN activations (``mean`` reducer) —
        the model's internal representations.
      - **embedding_space**: Token embedding or final logit layer
        (``mean`` reducer) — the model's semantic space.

    Args:
        model:          Any nn.Module with transformer-like sub-modules.
        layer_specs:    Optional list of LayerSpec overrides.  If None,
                        ``_default_layer_specs()`` auto-discovers layers.
        attention_dim:  Feature slots for attention patterns (default 20).
        hidden_dim:     Feature slots for hidden representations (default 40).
        embedding_dim:  Feature slots for embedding space (default 20).

    Example:
        llm = LLMInterface(
            my_gpt,
            layer_specs=[
                LayerSpec("transformer.h.0.attn", "attention_patterns",
                          reducer="attention_weights"),
                LayerSpec("transformer.h.0.mlp", "hidden_repr"),
                LayerSpec("transformer.wte", "embedding_space"),
            ],
        )
    """

    def __init__(
        self,
        model: Any,
        *,
        layer_specs: list[LayerSpec] | None = None,
        attention_dim: int = 20,
        hidden_dim: int = 40,
        embedding_dim: int = 20,
    ) -> None:
        total = attention_dim + hidden_dim + embedding_dim
        super().__init__(model, feature_dim=total)
        self._attn_dim = attention_dim
        self._hidden_dim = hidden_dim
        self._embed_dim = embedding_dim
        if layer_specs:
            self._layer_specs = layer_specs

    def _model_type(self) -> str:
        return "llm"

    def _build_registry(self) -> FeatureRegionRegistry:
        reg = FeatureRegionRegistry()
        a = 0
        reg.register(
            "attention_patterns", a, a + self._attn_dim,
            "Attention weight distributions across heads/layers",
        )
        b = a + self._attn_dim
        reg.register(
            "hidden_repr", b, b + self._hidden_dim,
            "Feed-forward / hidden-layer activations",
        )
        c = b + self._hidden_dim
        reg.register(
            "embedding_space", c, c + self._embed_dim,
            "Token embedding or output projection space",
        )
        return reg

    def _default_layer_specs(self) -> list[LayerSpec]:
        """Auto-discover transformer layers by common naming conventions."""
        if not _HAS_TORCH:
            return []
        specs: list[LayerSpec] = []
        named = dict(self._model.named_modules())

        # Look for attention layers
        for path, mod in named.items():
            cls_name = type(mod).__name__.lower()
            if any(k in cls_name for k in ("attention", "attn", "mha")):
                specs.append(LayerSpec(
                    path, "attention_patterns",
                    reducer="attention_weights",
                    meta={"metric": "attention", "layer": path},
                ))
                break  # take the first one found

        # Look for FFN / MLP layers
        for path, mod in named.items():
            cls_name = type(mod).__name__.lower()
            if any(k in cls_name for k in ("mlp", "ffn", "feedforward", "linear")):
                if "attn" not in path and "attention" not in path:
                    specs.append(LayerSpec(
                        path, "hidden_repr",
                        reducer="mean",
                        meta={"metric": "hidden_activation", "layer": path},
                    ))
                    break

        # Look for embedding layer
        for path, mod in named.items():
            cls_name = type(mod).__name__.lower()
            if any(k in cls_name for k in ("embedding", "embed", "wte", "wpe")):
                specs.append(LayerSpec(
                    path, "embedding_space",
                    reducer="mean",
                    meta={"metric": "embedding", "layer": path},
                ))
                break

        return specs

    def _extract_impl(self, input_data: Any) -> np.ndarray:
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required")
        with torch.no_grad():
            self._model.eval()
            self._model(input_data)
        return self._collect_after_forward()


# ------------------------------------------------------------------ #
# CNN interface                                                       #
# ------------------------------------------------------------------ #

class CNNInterface(ModelInterface):
    """Interface for convolutional neural networks.

    Extracts three canonical feature families:
      - **conv_features**: Early convolutional feature maps (``max_pool``
        reducer) — edge / texture detectors.
      - **deep_conv_features**: Deeper conv layers (``mean`` reducer) —
        higher-level spatial patterns.
      - **classifier_features**: Fully-connected / classifier head
        (``mean`` reducer) — task-level representations.

    Args:
        model:              Any nn.Module with convolutional architecture.
        layer_specs:        Optional LayerSpec overrides.
        conv_dim:           Feature slots for early conv maps (default 24).
        deep_conv_dim:      Feature slots for deep conv maps (default 32).
        classifier_dim:     Feature slots for classifier head (default 24).

    Example:
        cnn = CNNInterface(
            my_resnet,
            layer_specs=[
                LayerSpec("features.0", "conv_features", reducer="max_pool"),
                LayerSpec("features.6", "deep_conv_features", reducer="mean"),
                LayerSpec("classifier.0", "classifier_features"),
            ],
        )
    """

    def __init__(
        self,
        model: Any,
        *,
        layer_specs: list[LayerSpec] | None = None,
        conv_dim: int = 24,
        deep_conv_dim: int = 32,
        classifier_dim: int = 24,
    ) -> None:
        total = conv_dim + deep_conv_dim + classifier_dim
        super().__init__(model, feature_dim=total)
        self._conv_dim = conv_dim
        self._deep_dim = deep_conv_dim
        self._cls_dim = classifier_dim
        if layer_specs:
            self._layer_specs = layer_specs

    def _model_type(self) -> str:
        return "cnn"

    def _build_registry(self) -> FeatureRegionRegistry:
        reg = FeatureRegionRegistry()
        a = 0
        reg.register(
            "conv_features", a, a + self._conv_dim,
            "Early convolutional feature maps (edges, textures)",
        )
        b = a + self._conv_dim
        reg.register(
            "deep_conv_features", b, b + self._deep_dim,
            "Deeper convolutional layers (objects, spatial patterns)",
        )
        c = b + self._deep_dim
        reg.register(
            "classifier_features", c, c + self._cls_dim,
            "Fully-connected classifier representations",
        )
        return reg

    def _default_layer_specs(self) -> list[LayerSpec]:
        if not _HAS_TORCH:
            return []
        specs: list[LayerSpec] = []
        named = dict(self._model.named_modules())

        conv_layers = [
            (p, m) for p, m in named.items()
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        ]
        if conv_layers:
            # First conv → early features
            specs.append(LayerSpec(
                conv_layers[0][0], "conv_features",
                reducer="max_pool",
                meta={"metric": "conv_activation", "layer": conv_layers[0][0]},
            ))
            # Last conv → deep features
            if len(conv_layers) > 1:
                specs.append(LayerSpec(
                    conv_layers[-1][0], "deep_conv_features",
                    reducer="mean",
                    meta={"metric": "deep_conv_activation",
                           "layer": conv_layers[-1][0]},
                ))

        # Look for classifier / FC head
        linear_layers = [
            (p, m) for p, m in named.items()
            if isinstance(m, nn.Linear)
        ]
        if linear_layers:
            specs.append(LayerSpec(
                linear_layers[-1][0], "classifier_features",
                reducer="mean",
                meta={"metric": "classifier_activation",
                       "layer": linear_layers[-1][0]},
            ))

        return specs

    def _extract_impl(self, input_data: Any) -> np.ndarray:
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required")
        with torch.no_grad():
            self._model.eval()
            self._model(input_data)
        return self._collect_after_forward()


# ------------------------------------------------------------------ #
# Deep Linear Network interface                                       #
# ------------------------------------------------------------------ #

class DeepLinearInterface(ModelInterface):
    """Interface for deep linear networks (stacked linear layers, no non-linearity).

    Deep linear networks are important theoretically — they exhibit rich
    learning dynamics (saddle points, progressive rank increase) despite
    computing a linear function.  This interface extracts per-layer
    activations and the SVD structure of weight matrices.

    Feature families:
      - **layer_activations**: Mean activations at each linear layer.
      - **weight_spectrum**: Top singular values of each weight matrix,
        capturing effective rank and learned structure.

    Args:
        model:              Any nn.Module that is a stack of nn.Linear layers.
        layer_specs:        Optional LayerSpec overrides.
        activation_dim:     Feature slots for layer activations (default 40).
        spectrum_dim:       Feature slots for weight singular values (default 40).

    Example:
        dln = DeepLinearInterface(my_deep_linear)
        snapshot = dln.extract_and_add(ukt, input_data, block_name="dln")
    """

    def __init__(
        self,
        model: Any,
        *,
        layer_specs: list[LayerSpec] | None = None,
        activation_dim: int = 40,
        spectrum_dim: int = 40,
    ) -> None:
        total = activation_dim + spectrum_dim
        super().__init__(model, feature_dim=total)
        self._act_dim = activation_dim
        self._spec_dim = spectrum_dim
        if layer_specs:
            self._layer_specs = layer_specs

    def _model_type(self) -> str:
        return "deep_linear"

    def _build_registry(self) -> FeatureRegionRegistry:
        reg = FeatureRegionRegistry()
        reg.register(
            "layer_activations", 0, self._act_dim,
            "Per-layer mean activations across the linear stack",
        )
        reg.register(
            "weight_spectrum", self._act_dim, self._act_dim + self._spec_dim,
            "Top singular values of weight matrices (effective rank)",
        )
        return reg

    def _default_layer_specs(self) -> list[LayerSpec]:
        if not _HAS_TORCH:
            return []
        specs: list[LayerSpec] = []
        named = dict(self._model.named_modules())

        linear_layers = [
            (p, m) for p, m in named.items()
            if isinstance(m, nn.Linear)
        ]
        # Attach hooks to all linear layers → activations region
        for path, _ in linear_layers:
            specs.append(LayerSpec(
                path, "layer_activations",
                reducer="mean",
                meta={"metric": "linear_activation", "layer": path},
            ))

        return specs

    def _setup_extractor(self) -> None:
        """Override base: activation aggregation is handled directly in _extract_impl."""
        # Set a sentinel so the base class extract() does not call this again.
        # Actual per-layer hook management happens inside _extract_impl.
        self._extractor = object()

    def detach(self) -> None:
        """Reset extractor so it can be re-initialized on the next extract() call."""
        self._extractor = None

    def _extract_impl(self, input_data: Any) -> np.ndarray:
        """Extract mean activations across ALL linear layers + weight spectra.

        All linear layer activations are aggregated (mean) into the single
        ``layer_activations`` region, avoiding the overwrite problem that
        arises when multiple hooks share the same region key in HookExtractor.
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required")

        reg = self.registry
        act_region = reg.regions["layer_activations"]
        features = np.zeros(reg.total_dim, dtype=np.float64)

        # Collect activations from ALL linear layers into a list, then mean-pool
        all_activations: list[np.ndarray] = []
        temp_hooks: list = []

        def _create_activation_hook():
            def hook_fn(module, input, output):
                t = output.detach().cpu().float()
                while t.dim() > 1:
                    t = t.mean(dim=0)
                all_activations.append(t.numpy())
            return hook_fn

        try:
            with torch.no_grad():
                self._model.eval()
                for _, mod in self._model.named_modules():
                    if isinstance(mod, nn.Linear):
                        temp_hooks.append(mod.register_forward_hook(_create_activation_hook()))
                self._model(input_data)
        finally:
            for h in temp_hooks:
                h.remove()

        if all_activations:
            mean_act = np.mean(
                [_pad_or_truncate(a, act_region.dim) for a in all_activations], axis=0,
            )
            features[act_region.start:act_region.end] = mean_act

        # Weight spectrum via SVD on weight matrices
        spectrum = self._extract_weight_spectrum()
        spec_region = reg.regions["weight_spectrum"]
        padded = _pad_or_truncate(spectrum, spec_region.dim)
        features[spec_region.start:spec_region.end] = padded

        return features

    def _extract_weight_spectrum(self) -> np.ndarray:
        """Compute top singular values from all weight matrices."""
        if not _HAS_TORCH:
            return np.zeros(self._spec_dim)

        linear_count = self._count_linear()
        singular_values: list[float] = []
        for _, mod in self._model.named_modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight.detach().cpu().float()
                svs = torch.linalg.svdvals(w)
                # Take top-k singular values (proportional allocation)
                k = max(1, self._spec_dim // max(1, linear_count))
                singular_values.extend(svs[:k].numpy().tolist())

        arr = np.array(singular_values, dtype=np.float64)
        # Normalize to [0, 1]
        if arr.max() > 1e-8:
            arr = arr / arr.max()
        return arr

    def _count_linear(self) -> int:
        """Count nn.Linear layers in the model."""
        return sum(
            1 for _, m in self._model.named_modules()
            if isinstance(m, nn.Linear)
        )


# ------------------------------------------------------------------ #
# Cross-model compatibility helpers                                   #
# ------------------------------------------------------------------ #

def unified_registry(
    *interfaces: ModelInterface,
    prefix_regions: bool = True,
) -> FeatureRegionRegistry:
    """Build a combined registry spanning multiple model interfaces.

    Concatenates the feature regions of each interface into a single
    registry, enabling side-by-side comparison in a shared UKT.

    Args:
        interfaces:     ModelInterface instances to combine.
        prefix_regions: If True, prefix region names with the model type
                        to avoid collisions (e.g. "llm/attention_patterns").

    Returns:
        A FeatureRegionRegistry covering all interfaces.
    """
    combined = FeatureRegionRegistry()
    offset = 0
    for iface in interfaces:
        reg = iface.registry
        for region in reg.ordered_regions:
            name = (
                f"{iface.model_type}/{region.name}"
                if prefix_regions else region.name
            )
            combined.register(
                name,
                offset + region.start,
                offset + region.end,
                description=region.description,
                feature_names=[
                    f"{iface.model_type}/{fn}" for fn in region.feature_names
                ] if region.feature_names else [],
                metadata={**region.metadata, "source_model": iface.model_type},
            )
        offset += reg.total_dim
    return combined


def extract_all(
    interfaces: list[ModelInterface],
    inputs: list[Any],
    ukt: Any,
    block_prefix: str = "",
    timeframe_context: dict | None = None,
) -> list[dict]:
    """Extract features from multiple models and add them all to one UKT.

    Args:
        interfaces:         List of ModelInterface instances.
        inputs:             Corresponding inputs for each model.
        ukt:                UniversalKnowledgeTensor to populate.
        block_prefix:       Optional prefix for block names.
        timeframe_context:  Shared temporal context.

    Returns:
        List of UKT snapshots, one per model.
    """
    snapshots = []
    for iface, inp in zip(interfaces, inputs):
        name = f"{block_prefix}{iface.model_type}" if block_prefix else iface.model_type
        snap = iface.extract_and_add(
            ukt, inp,
            block_name=name,
            timeframe_context=timeframe_context,
        )
        snapshots.append(snap)
    return snapshots
