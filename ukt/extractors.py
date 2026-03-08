"""Hook-based feature extractors for any PyTorch neural network.

The extractor attaches forward hooks to arbitrary layers of a neural network,
captures their activations, and maps them into UKT feature regions. This
makes UKT work with ANY architecture — transformers, CNNs, RNNs, MLPs,
graph neural networks — without modifying the network's code.

Usage:
    extractor = HookExtractor(model)
    extractor.attach("encoder.layer.0.self_attn", region="attention_pattern",
                     reducer="mean_head")
    extractor.attach("encoder.layer.0.ffn", region="hidden_repr")

    output = model(input_data)  # hooks fire automatically
    features = extractor.collect(registry)  # (total_dim,) numpy array
    extractor.reset()  # clear for next forward pass
"""
from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from ukt.registry import FeatureRegionRegistry


# --------------------------------------------------------------------------- #
# Built-in reducers: transform raw activations into fixed-size feature vectors #
# --------------------------------------------------------------------------- #

def _flatten_reducer(tensor) -> np.ndarray:
    """Flatten the activation tensor to 1D."""
    return tensor.detach().cpu().numpy().flatten()


def _mean_reducer(tensor) -> np.ndarray:
    """Global mean across all dimensions except the last."""
    t = tensor.detach().cpu().float()
    while t.dim() > 1:
        t = t.mean(dim=0)
    return t.numpy()


def _mean_head_reducer(tensor) -> np.ndarray:
    """Mean over batch and heads, keep per-position features.

    Works for multi-head attention outputs: (batch, heads, seq, dim) or
    (batch, seq, dim).
    """
    t = tensor.detach().cpu().float()
    while t.dim() > 2:
        t = t.mean(dim=0)
    # Now (seq, dim) — mean over sequence to get (dim,)
    return t.mean(dim=0).numpy()


def _max_pool_reducer(tensor) -> np.ndarray:
    """Max-pool across all spatial/temporal dimensions."""
    t = tensor.detach().cpu().float()
    while t.dim() > 1:
        t = t.max(dim=0).values
    return t.numpy()


def _attention_weights_reducer(tensor) -> np.ndarray:
    """Extract mean attention weights across heads.

    Expects: (batch, heads, seq_q, seq_k) or (batch, seq_q, seq_k).
    Returns: mean attention per query position → (seq_q,).
    """
    t = tensor.detach().cpu().float()
    while t.dim() > 2:
        t = t.mean(dim=0)
    # (seq_q, seq_k) → mean over keys → (seq_q,)
    return t.mean(dim=-1).numpy()


def _variance_reducer(tensor) -> np.ndarray:
    """Per-feature variance across batch/spatial dims — captures activation spread."""
    t = tensor.detach().cpu().float()
    while t.dim() > 1:
        t = t.var(dim=0)
    return t.numpy()


BUILTIN_REDUCERS: dict[str, Callable] = {
    "flatten": _flatten_reducer,
    "mean": _mean_reducer,
    "mean_head": _mean_head_reducer,
    "max_pool": _max_pool_reducer,
    "attention_weights": _attention_weights_reducer,
    "variance": _variance_reducer,
}


# --------------------------------------------------------------------------- #
# Feature extractor                                                            #
# --------------------------------------------------------------------------- #

def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate a 1D array to target length."""
    arr = arr.flatten()
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)))


class HookExtractor:
    """Attaches forward hooks to a PyTorch model and collects activations.

    Maps raw activations into UKT feature regions via configurable reducers.
    Works with any nn.Module — no architecture assumptions.

    Args:
        model: Any PyTorch nn.Module.
    """

    def __init__(self, model: "nn.Module") -> None:
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for HookExtractor")
        self._model = model
        self._hooks: list = []
        self._captured: dict[str, np.ndarray] = {}
        self._configs: list[dict] = []

    def attach(
        self,
        layer_name: str,
        region: str,
        reducer: str | Callable = "mean",
        output_index: int | None = None,
    ) -> "HookExtractor":
        """Attach a forward hook to a named layer.

        Args:
            layer_name: Dot-separated path to the module (e.g., "encoder.layer.0.ffn").
            region: Name of the UKT feature region this maps to.
            reducer: How to reduce the raw activation tensor to a 1D vector.
                     Can be a string key from BUILTIN_REDUCERS or a custom callable.
            output_index: If the layer returns a tuple, which element to capture.
                          None means capture the full output (must be a tensor).

        Returns:
            Self, for chaining.
        """
        # Resolve the layer module
        module = self._model
        for part in layer_name.split("."):
            if part.isdigit():
                module = list(module.children())[int(part)]
            else:
                module = getattr(module, part)

        # Resolve reducer
        if isinstance(reducer, str):
            if reducer not in BUILTIN_REDUCERS:
                raise ValueError(
                    f"Unknown reducer '{reducer}'. Available: {list(BUILTIN_REDUCERS)}"
                )
            reduce_fn = BUILTIN_REDUCERS[reducer]
        else:
            reduce_fn = reducer

        config = {
            "layer_name": layer_name,
            "region": region,
            "reducer": reduce_fn,
            "output_index": output_index,
        }
        self._configs.append(config)

        def make_hook(cfg):
            def hook_fn(module, input, output):
                idx = cfg["output_index"]
                if idx is not None:
                    tensor = output[idx]
                elif isinstance(output, tuple):
                    tensor = output[0]
                else:
                    tensor = output
                self._captured[cfg["region"]] = cfg["reducer"](tensor)
            return hook_fn

        handle = module.register_forward_hook(make_hook(config))
        self._hooks.append(handle)
        return self

    def collect(self, registry: FeatureRegionRegistry) -> np.ndarray:
        """Collect captured activations into a UKT feature vector.

        Maps each captured region into the corresponding feature indices
        defined by the registry. Regions without captured data are left as zeros.

        Args:
            registry: The feature region registry defining the feature space.

        Returns:
            (total_dim,) numpy array with captured features placed in their regions.
        """
        total_dim = registry.total_dim
        features = np.zeros(total_dim, dtype=np.float64)

        for region_name, activation in self._captured.items():
            if region_name not in registry:
                continue
            region = registry.regions[region_name]
            region_dim = region.dim
            padded = _pad_or_truncate(activation, region_dim)
            features[region.start:region.end] = padded

        return features

    def collect_raw(self) -> dict[str, np.ndarray]:
        """Return raw captured activations without mapping to regions."""
        return dict(self._captured)

    def reset(self) -> None:
        """Clear captured activations for the next forward pass."""
        self._captured.clear()

    def detach_all(self) -> None:
        """Remove all hooks from the model."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._configs.clear()
        self._captured.clear()

    def __del__(self) -> None:
        self.detach_all()


class ManualExtractor:
    """For non-PyTorch models or custom feature pipelines.

    Accepts features directly as numpy arrays and maps them into UKT regions.

    Usage:
        extractor = ManualExtractor()
        extractor.set_features("attention", attention_weights)
        extractor.set_features("hidden", hidden_activations)
        features = extractor.collect(registry)
    """

    def __init__(self) -> None:
        self._features: dict[str, np.ndarray] = {}

    def set_features(self, region: str, features: np.ndarray) -> "ManualExtractor":
        """Set features for a named region.

        Args:
            region: Region name (must match registry).
            features: 1D numpy array of feature values.

        Returns:
            Self, for chaining.
        """
        self._features[region] = features.flatten()
        return self

    def collect(self, registry: FeatureRegionRegistry) -> np.ndarray:
        """Collect features into a UKT feature vector."""
        total_dim = registry.total_dim
        result = np.zeros(total_dim, dtype=np.float64)

        for region_name, activation in self._features.items():
            if region_name not in registry:
                continue
            region = registry.regions[region_name]
            padded = _pad_or_truncate(activation, region.dim)
            result[region.start:region.end] = padded

        return result

    def reset(self) -> None:
        """Clear stored features."""
        self._features.clear()
