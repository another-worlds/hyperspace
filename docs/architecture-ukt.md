# UKT Framework Architecture

## Overview

The Universal Knowledge Tensor (`ukt/`) is a **network-agnostic** framework
for standardizing, mixing, and decomposing feature vectors from heterogeneous
data sources into interpretable emergent kernels. It is designed to work with
any neural network or data pipeline — the Hyperspace-specific wiring lives in
`hyperspace/models/knowledge_matrix.py`.

The framework's central claim: **kernels are not predetermined**. They emerge
from the data through adaptive cross-region projection and SVD decomposition.

---

## Module Map

```
ukt/
├── __init__.py           # Public exports
├── registry.py           # Feature region definitions
├── extractors.py         # Hook-based and manual feature capture
├── projection.py         # Adaptive cross-region coupling matrix
├── kernels.py            # SVD decomposition + kernel labeling + narratives
├── stability.py          # Bootstrap robustness estimation
├── interfaces.py         # Standardized model wrappers (LLM, CNN, DeepLinear)
├── tensor.py             # Standalone UKT class (no Hyperspace dependencies)
└── utils.py              # Pad/truncate, helpers
```

---

## Data Flow

```
Raw features from N sources
        │
        ▼
┌─────────────────────────┐
│  Feature Region Registry │  Define named, non-overlapping regions
│  (ukt/registry.py)       │  in the shared feature vector
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Per-Region Normalize    │  Scale each region to [0,1] independently
│  [0,1] min-max           │  BEFORE projection (critical ordering)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  SharedProjection        │  Adaptive mixing matrix P creates
│  (ukt/projection.py)    │  cross-region feature coupling
│                          │
│  P = I + Σ coupling      │  Strength: energy ratio (data-driven)
│      blocks              │  Direction: rank-1 outer product (data-driven)
│                          │  Blend: α = strength (confidence-weighted)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Projected Matrix        │  (n_sources × feature_dim) with cross-region
│  np.stack(P @ x_i)      │  correlations baked in
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  SVD Decomposition       │  U · S · Vt
│  (ukt/kernels.py)        │
│                          │  Kernels = rows of Vt
│                          │  Importance = S / Σ(S)
│                          │  Activation = U · diag(S)
│                          │  Reality regression = importance @ Vt
└──────────┬──────────────┘
           │
           ├──► Kernel Labels (region analysis + top features)
           ├──► Kernel Narratives (complexity-adaptive text)
           ├──► Reality Regression (single best summary direction)
           └──► Reconstruction Error (faithfulness signal)
```

---

## 1. Feature Region Registry (`ukt/registry.py`)

### Purpose

Defines named, non-overlapping regions in the shared feature vector. Every
feature index belongs to exactly one region. The registry is the structural
prior — it says which features belong to which domain — but does NOT determine
what patterns SVD discovers.

### API

```python
class FeatureRegion:
    name: str           # e.g., "temporal-pattern"
    start: int          # Inclusive start index
    end: int            # Exclusive end index
    description: str    # Human-readable semantics
    feature_names: list[str]  # Per-feature names
    metadata: dict      # Arbitrary region metadata

    @property
    def dim(self) -> int  # end - start

class FeatureRegionRegistry:
    def register(name, start, end, description, feature_names, metadata) -> self
    def region_for_index(idx) -> FeatureRegion
    def feature_name(idx) -> str

    @property
    def total_dim -> int          # Max end across all regions
    def ordered_regions -> list   # In registration order
    def region_bounds() -> dict   # {name: (start, end)}
```

### Hyperspace Configuration

| Region | Indices | Block | Description |
|--------|---------|-------|-------------|
| temporal-pattern | 0–15 | Finance / TFT | Encoder attention weights, regime shifts, volatility |
| semantic-embedding | 16–31 | Clusters / BERTopic | Topic distributions, embedding statistics |
| structural-centrality | 32–47 | Graph Engine | Degree, betweenness, eigenvector, PageRank |
| dynamic-agent | 48–63 | Agent Simulation | Density, clustering, community count, alliance eigenvalues |
| geospatial-kernel | 64–79 | Spatial Raster | Spatial SVD importances, per-country loadings |

---

## 2. Feature Extraction (`ukt/extractors.py`)

### HookExtractor — PyTorch Models

Attaches forward hooks to model layers and captures activations during
inference. Built-in reducers compress high-dimensional activations to
region-sized vectors:

| Reducer | Purpose |
|---------|---------|
| `flatten` | Raw activation vector |
| `mean` | Global mean across spatial/temporal dims |
| `mean_head` | Mean over batch and attention heads |
| `max_pool` | Max across all non-feature dims |
| `attention_weights` | Mean attention per query position |
| `variance` | Per-feature variance |

```python
extractor = HookExtractor(model)
extractor.attach("encoder.layer.0.self_attn", region="temporal-pattern", reducer="attention_weights")
output = model(input_data)          # Hooks fire
features = extractor.collect(registry)  # (80,) array
extractor.reset()
```

### ManualExtractor — Non-PyTorch Pipelines

For data sources that don't use PyTorch (e.g., NetworkX graph metrics,
BERTopic outputs):

```python
extractor = ManualExtractor()
extractor.set_features("structural-centrality", centrality_vector)
features = extractor.collect(registry)  # (80,) array
```

---

## 3. Shared Projection (`ukt/projection.py`)

### Purpose

The Shared Projection is the mechanism that makes emergent cross-domain
kernels mathematically possible. Without it, each block writes to its own
16-dim slice, SVD recovers the block structure, and "kernels" are just
renamed layers.

The projection creates a mixing matrix P such that `projected = P @ features`
introduces cross-region correlations that SVD can discover.

### Emergence Invariants

These four invariants guarantee that kernels are genuinely emergent:

**1. Normalize before project.**
Per-region [0,1] normalization happens BEFORE projection. Normalizing after
would re-isolate regions and erase the cross-region energy ratios that the
projection encoded.

**2. Open topology.**
All region pairs can couple by default. The topology set is:
```python
{(src, tgt) for src in regions for tgt in regions if src != tgt}
```
No hardcoded restrictions on which domains can influence each other.

**3. Data-driven coupling.**
Coupling strength and direction come entirely from the data:

```python
# Strength: energy ratio (how comparable are the two regions?)
strength = min(E_src, E_tgt) / (max(E_src, E_tgt) + 1e-8)

# Direction: rank-1 outer product of normalized feature vectors
v_src = f_src / ||f_src||
v_tgt = f_tgt / ||f_tgt||
data_block = outer(v_tgt, v_src)  # (tgt_dim, src_dim)

# Blend: confidence tracks coupling strength
alpha = strength
block = alpha * data_block + (1 - alpha) * fallback_orthogonal
```

Strong signals get data-driven coupling. Weak signals get safe orthogonal
fallback. No hardcoded weights.

**4. Full replay after each block.**
When block N arrives:
1. `observe(region, features)` stores block N's features
2. `_rebuild()` recomputes P from ALL active blocks
3. ALL blocks 1..N are re-projected through the new P
4. SVD runs on the fully re-projected matrix

No block retains stale coordinates from an earlier projection epoch.

### API

```python
class SharedProjection:
    def __init__(registry, topology=None, seed=2025)
    def observe(region_name, features)   # Store features, rebuild P
    def project(features) -> np.ndarray  # P @ features
    def back_project(projected) -> np.ndarray  # P_inv @ projected

    @property
    def active_couplings -> list[CouplingInfo]
        # source_region, target_region, weight, data_driven=True
```

---

## 4. SVD Decomposition (`ukt/kernels.py`)

### Kernel Discovery

```python
def decompose_svd(matrix) -> KernelDecomposition:
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    importance = S / S.sum()
    kernel_activation = U * S[np.newaxis, :]
    reality_regression = importance @ Vt
    reconstruction_error = ||matrix - U @ diag(S) @ Vt||_F
```

| Output | Shape | Meaning |
|--------|-------|---------|
| `U` | (n_blocks, n_kernels) | How much each block loads on each kernel |
| `S` | (n_kernels,) | Singular values (kernel strength) |
| `Vt` | (n_kernels, feature_dim) | Feature loadings per kernel |
| `importance` | (n_kernels,) | Normalized importance (sums to 1) |
| `kernel_activation` | (n_blocks, n_kernels) | Block × kernel activation strength |
| `reality_regression` | (feature_dim,) | Importance-weighted kernel sum |
| `reconstruction_error` | float | How well kernels explain the data |

### Kernel Labeling

`label_kernel(k_idx, decomposition, registry, block_names, ...)` produces a
structured label for each kernel:

1. **Region scores**: Sum absolute feature loadings per region
2. **Contributing regions**: Regions with >15% of total loading
3. **Block contributions**: Blocks with |U[block, kernel]| > 0.1
4. **Top features**: Top-5 features by absolute loading with names and provenance

**Label format** depends on pattern complexity:
- **3+ regions**: `K2: Finance+Clusters+Graph — temporal × semantic × structural (18.5% var)`
- **2 regions**: `K1: Finance+Clusters — temporal × semantic (24.2% var)`
- **1 region**: `K0: Finance — temporal-pattern (31.5% var)`

### Kernel Narratives

`generate_kernel_narrative(...)` produces data-grounded text that adapts to
kernel complexity:

**Multi-domain (3+ regions):**
> Emergent cross-domain pattern spanning 3 regions: temporal pattern,
> semantic embedding, structural centrality. Contributing blocks:
> Finance (0.45), Clusters (0.38), Graph (0.32). Key evidence features:
> attn_head_2 (+0.523), embedding_dist (−0.401), centrality_betweenness (+0.387).

**Cross-domain coupling (2 regions):**
> Cross-domain coupling between temporal pattern and semantic embedding.
> Coupling signature: timeseries_volatility (+0.612) co-varies with
> semantic_distance (−0.489).

**Single-domain:**
> Single-domain pattern: Finance block, temporal pattern region.
> Top features: volatility_index (+0.754), price_change (+0.521).

Every claim is backed by a number from the SVD decomposition.

---

## 5. Stability Estimation (`ukt/stability.py`)

```python
def estimate_regression_stability(matrix, n_runs=8, noise_std=0.01, seed=42):
```

**Algorithm**:
1. Add Gaussian noise (σ=0.01) to the matrix N times
2. Recompute SVD and reality regression for each noisy matrix
3. Compute all pairwise cosine similarities between regressions

**Output**: `{n_runs, mean_cosine, min_cosine, std_cosine}`

**Interpretation**:
- `mean_cosine >= 0.75`: Stable — conclusions robust to perturbation
- `mean_cosine < 0.50`: Unstable — dominated by noise artifacts

---

## 6. Model Interfaces (`ukt/interfaces.py`)

Standardized wrappers for common neural network architectures:

| Interface | Regions | Purpose |
|-----------|---------|---------|
| `LLMInterface` | attention_patterns (20) + hidden_repr (40) + embedding_space (20) | Transformer models |
| `CNNInterface` | conv_features (24) + deep_conv_features (32) + classifier_features (24) | Convolutional networks |
| `DeepLinearInterface` | layer_activations (40) + weight_spectrum (40) | Linear networks |

All implement the `ModelInterface` protocol:
```python
class ModelInterface(abc.ABC):
    def extract(input_data) -> np.ndarray         # (feature_dim,) vector
    def extract_and_add(ukt, input_data, ...) -> dict  # Extract + add to UKT
    def detach() -> None                           # Remove hooks

    @property
    def registry -> FeatureRegionRegistry
    def model_type -> str  # "llm", "cnn", "deep_linear"
```

Helper functions:
- `unified_registry(*interfaces)`: Combines registries for multi-model UKTs
- `extract_all(interfaces, inputs, ukt, ...)`: Bulk extraction + UKT addition

---

## 7. Hyperspace Wrapper (`hyperspace/models/knowledge_matrix.py`)

The `UniversalKnowledgeTensor` class wraps the standalone `ukt` package with
Hyperspace-specific wiring:

- Pre-configured 80-dim registry with 5 regions
- Semantic Canvas integration (reset + replay per block)
- Feature evidence attachment
- Kernel narrative generation via Tiny-LLM
- Snapshot storage with raw features for counterfactual rebuilds
- `export_latent_units()`, `export_feature_attributions()`,
  `explain_prediction()` for interpretability contract compliance

### Key Method: `add_block(name, features, feature_meta, timeframe_context)`

Performs the full pipeline per block:
1. Pad/truncate to 80 dims
2. Store raw features
3. Normalize and observe in SharedProjection
4. Re-project all blocks through updated P
5. Stack into matrix, run SVD
6. Label all kernels
7. Reset and replay Semantic Canvas
8. Generate narratives
9. Return snapshot dict

---

## Centralized Configuration

All thresholds are defined in `hyperspace/config.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `UKT_FEATURE_DIM` | 80 | Feature vector dimensionality |
| `KERNEL_CONTRIBUTING_REGION_THRESHOLD` | 0.15 | Region must contribute >15% of loading to be listed |
| `KERNEL_BLOCK_CONTRIBUTION_MIN` | 0.10 | Block must have |U[b,k]| > 0.1 to be cited |
| `KERNEL_NARRATOR_IMPORTANCE_MIN` | 0.15 | Kernel must explain >15% variance for detailed narrative |

---

## Design Principles

1. **Network-agnostic core.** The `ukt/` package has no Hyperspace imports.
   It works with any feature vector from any source.

2. **Emergence by construction.** The four invariants (normalize-before-project,
   open topology, data-driven coupling, full replay) are not optional — they
   are the mechanism that makes cross-domain kernel discovery possible.

3. **Incremental computation.** Each `add_block()` call produces a complete
   snapshot. The system handles streaming multi-source data without requiring
   all sources to be available simultaneously.

4. **Inspectable intermediates.** Every transformation step (raw features,
   normalized features, projection matrix, projected matrix, SVD components)
   is stored and available for auditing.

5. **Graceful degradation.** SVD handles degenerate matrices. Zero-energy
   regions decouple automatically. Missing blocks produce valid (if less
   informative) kernels.

---

*Document version: Alpha 1.0 — March 2026*
