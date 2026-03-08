"""Feature Region Registry: defines named, typed regions within the UKT feature vector.

The registry replaces hardcoded feature indices with a declarative system.
Any neural network can register its feature regions — attention heads, hidden
layers, embedding spaces — and the UKT will normalize, decompose, and
interpret them automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureRegion:
    """A named region within the UKT feature vector."""
    name: str
    start: int
    end: int
    description: str = ""
    feature_names: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.end - self.start

    def feature_name(self, local_idx: int) -> str:
        """Get name for a feature at local index within this region."""
        if local_idx < len(self.feature_names):
            return self.feature_names[local_idx]
        return f"{self.name}_{local_idx}"


class FeatureRegionRegistry:
    """Registry of named feature regions within the UKT feature space.

    Allows any neural network to declare what its feature regions represent,
    enabling cross-network standardized interpretation.

    Example:
        registry = FeatureRegionRegistry()
        registry.register("attention", 0, 16, "Self-attention weights",
                          feature_names=["attn_head_0", "attn_head_1", ...])
        registry.register("hidden", 16, 48, "FFN hidden activations")
    """

    def __init__(self) -> None:
        self._regions: dict[str, FeatureRegion] = {}
        self._region_order: list[str] = []

    def register(
        self,
        name: str,
        start: int,
        end: int,
        description: str = "",
        feature_names: list[str] | None = None,
        metadata: dict | None = None,
    ) -> "FeatureRegionRegistry":
        """Register a named feature region.

        Args:
            name: Unique name for this region (e.g., "encoder_attention").
            start: Start index in the feature vector (inclusive).
            end: End index in the feature vector (exclusive).
            description: Human-readable description of what this region encodes.
            feature_names: Optional per-feature names within the region.
            metadata: Optional metadata dict (source, model layer, etc.).

        Returns:
            Self, for chaining.
        """
        if name in self._regions:
            raise ValueError(f"Region '{name}' already registered")
        if start >= end:
            raise ValueError(f"Invalid region bounds: [{start}, {end})")
        for existing in self._regions.values():
            if start < existing.end and end > existing.start:
                raise ValueError(
                    f"Region '{name}' [{start}:{end}) overlaps with "
                    f"'{existing.name}' [{existing.start}:{existing.end})"
                )

        self._regions[name] = FeatureRegion(
            name=name,
            start=start,
            end=end,
            description=description,
            feature_names=feature_names or [],
            metadata=metadata or {},
        )
        self._region_order.append(name)
        return self

    @property
    def total_dim(self) -> int:
        """Total feature dimension (max end index across all regions)."""
        if not self._regions:
            return 0
        return max(r.end for r in self._regions.values())

    @property
    def regions(self) -> dict[str, FeatureRegion]:
        return dict(self._regions)

    @property
    def ordered_regions(self) -> list[FeatureRegion]:
        """Regions in registration order."""
        return [self._regions[n] for n in self._region_order]

    def region_for_index(self, idx: int) -> FeatureRegion | None:
        """Find the region containing a given feature index."""
        for region in self._regions.values():
            if region.start <= idx < region.end:
                return region
        return None

    def feature_name(self, idx: int) -> str:
        """Get the human-readable name for a feature index."""
        region = self.region_for_index(idx)
        if region is not None:
            local = idx - region.start
            return region.feature_name(local)
        return f"feature_{idx}"

    def region_bounds(self) -> dict[str, tuple[int, int]]:
        """Return {name: (start, end)} for all regions."""
        return {name: (r.start, r.end) for name, r in self._regions.items()}

    def __len__(self) -> int:
        return len(self._regions)

    def __contains__(self, name: str) -> bool:
        return name in self._regions

    def __repr__(self) -> str:
        parts = [f"{r.name}[{r.start}:{r.end}]" for r in self.ordered_regions]
        return f"FeatureRegionRegistry({', '.join(parts)}, total_dim={self.total_dim})"
