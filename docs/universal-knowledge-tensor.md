# Universal Knowledge Tensor (UKT)

## Overview

The Universal Knowledge Tensor is the central data structure in Hyperspace. It
is an incrementally built matrix where each row is a normalized feature vector
from one pipeline block. After every block addition, the UKT decomposes the
matrix via SVD to extract latent "kernels" (principal directions of variance),
computes a reality regression vector (the system's single best summary of the
current state), and generates human-readable semantic labels grounded in
concrete feature evidence.

```
Pipeline Block
    │
    ▼
80-dim feature vector
    │
    ▼
Per-region [0,1] normalization
    │
    ▼
Append row to UKT matrix  (n_blocks × 80)
    │
    ▼
SVD:  U · S · Vt
    │
    ├──► Kernel importance = S / sum(S)
    ├──► Kernel activation = U · diag(S)         (n_blocks × n_kernels)
    ├──► Reality regression = importance · Vt     (80,)
    ├──► Reconstruction error = ‖matrix − U·S·Vt‖
    ├──► Semantic kernel labels + narratives
    └──► Per-stage SAE → Semantic Canvas projection
```

---

## Feature Space Layout

The 80-dimensional feature space is divided into 5 regions of 16 features each.
Every region is owned by exactly one pipeline block, and features within a
region are semantically coherent.

| Index Range | Region Key | Populated By | Description |
|-------------|-----------|-------------|-------------|
| 0–15 | `temporal-pattern` | Finance / TFT | Encoder attention weights across time horizons, regime shift indicators, volatility signals |
| 16–31 | `semantic-embedding` | Clusters / BERTopic | Topic distribution shares (top-8 topics), topic embedding statistics (norm, std, percentiles) |
| 32–47 | `structural-centrality` | Graph Engine | Node centrality measures (degree, betweenness, eigenvector, PageRank) for each geopolitical actor |
| 48–63 | `dynamic-agent` | Agent Simulation | Graph density, clustering, community count, per-agent resource shares, alliance matrix eigenvalues |
| 64–79 | `geospatial-kernel` | Spatial Raster | Spatial SVD kernel importances, per-country node loadings, summary scalars (elevation, temperature, conflict, economic) |

### Named Features

All 80 features have human-readable names defined in `hyperspace/config.py:FEATURE_NAMES`.
Examples:

```
[0]  attention_recent_1d          — attention weight on most-recent encoder step
[16] topic_share_dominant         — fraction of docs in largest topic
[32] centrality_node_0            — flattened node centrality
[48] graph_density                — geopolitical graph density
[64] spatial_kernel_importance_0  — normalized singular value from spatial SVD
```

Feature names are used in kernel labels, narratives, and provenance reports.
When a pipeline block provides feature metadata (source, entity, metric, time
scope), that metadata takes precedence over the default names.

---

## Core Class: `UniversalKnowledgeTensor`

**File:** `hyperspace/models/knowledge_matrix.py`

### Constructor

```python
UniversalKnowledgeTensor(feature_dim: int = 80)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `feature_dim` | `int` | Dimensionality (default 80, from `UKT_FEATURE_DIM`) |
| `block_names` | `list[str]` | Names of added blocks in order |
| `rows` | `list[np.ndarray]` | Normalized feature vectors |
| `snapshots` | `list[dict]` | Full snapshot after each step |
| `global_feature_meta` | `dict[int, dict]` | Accumulated metadata for all feature indices |
| `canvas` | `SemanticCanvas` | Semantic canvas subsystem instance |

### `add_block()`

```python
def add_block(
    self,
    name: str,
    features: np.ndarray,
    feature_meta: dict[int, dict] | None = None,
    timeframe_context: dict | None = None,
) -> dict
```

This is the main entry point. Each call executes the full UKT update pipeline:

1. **Pad/truncate** the feature vector to `feature_dim`
2. **Normalize** each 16-dim region independently to [0, 1] via min-max scaling
3. **Append** as a new row
4. **SVD decompose** the full matrix: `U, S, Vt = np.linalg.svd(matrix)`
5. **Compute kernels**: importance, activation, reality regression
6. **Label kernels**: determine dominant block, region, top features, narrative
7. **Run per-stage SAE** on the block's feature region (8 concepts, 60 epochs)
8. **Project onto Semantic Canvas** via `canvas.project_block()`
9. **Generate Tiny-LLM narratives** for the layer and high-importance kernels
10. **Build human-readable report** and store as snapshot

Returns a snapshot dict (see below).

### `get_final_matrix()`

```python
def get_final_matrix(self) -> np.ndarray | None
```

Returns the full `(n_blocks, 80)` matrix, or `None` if no blocks have been added.

### `get_latest_snapshot()`

```python
def get_latest_snapshot(self) -> dict | None
```

Returns the most recent snapshot dict.

---

## Snapshot Structure

Each call to `add_block()` produces a snapshot dict containing all computed
state at that pipeline step:

```python
{
    # Identity
    "step": int,                    # 1-indexed pipeline step number
    "block_name": str,              # e.g. "Finance", "Graph"

    # SVD decomposition
    "matrix": np.ndarray,           # (n_blocks, 80) — full UKT matrix
    "U": np.ndarray,                # (n_blocks, n_kernels) — left singular vectors
    "S": np.ndarray,                # (n_kernels,) — singular values
    "Vt": np.ndarray,               # (n_kernels, 80) — right singular vectors

    # Derived quantities
    "n_kernels": int,               # min(n_blocks, 80)
    "importance": np.ndarray,       # (n_kernels,) — S / sum(S)
    "kernel_activation": np.ndarray,# (n_blocks, n_kernels) — U * S
    "reality_regression": np.ndarray,# (80,) — importance @ Vt
    "reconstruction_error": float,  # ‖matrix − U·S·Vt‖

    # Semantic labels
    "kernel_labels": list[dict],    # one per kernel (see below)
    "report": str,                  # full human-readable text report

    # Metadata
    "feature_meta": dict[int, dict],# accumulated feature metadata
    "timeframe_context": dict,      # date range, year bounds

    # Semantic subsystem
    "stage_sae_result": dict | None,# per-stage SAE output
    "canvas_entry": CanvasEntry | None,  # canvas projection
    "layer_narrative": str | None,  # Tiny-LLM narrative for this layer
}
```

### Kernel Label Structure

Each kernel label describes one SVD component:

```python
{
    "kernel_id": "K0",              # human-readable ID
    "dominant_block": "Finance",    # block with highest |U| loading
    "dominant_region": "temporal-pattern",  # region with highest |Vt| energy
    "importance": 0.45,             # fraction of total variance explained
    "top_features": [               # top-5 features by |loading|
        {
            "index": 12,
            "name": "attention_trend_strength",
            "region": "temporal-pattern",
            "loading": 0.342,
            "abs_loading": 0.342,
            "entity": "SPY",        # from feature_meta (may be None)
            "metric": "attention",
            "source": "TFT",
            "time_scope": "2025-2026",
        },
        ...
    ],
    "top_feature_indices": [12, 8, 3, ...],
    "label": "K0: Finance — temporal pattern (45.0% var, lead: attention_trend_strength)",
    "narrative": "Kernel K0 explains 45.0% of total variance...",  # multi-line
    "region_scores": {              # |Vt| energy per region
        "temporal-pattern": 1.234,
        "semantic-embedding": 0.456,
        ...
    },
    "semantic_narrative": "...",    # Tiny-LLM text (if importance > 15%)
}
```

---

## Key Computations

### SVD Decomposition

```
matrix  =  U  ·  diag(S)  ·  Vt
(n×80)    (n×k)   (k×k)    (k×80)
```

- **U columns** — how much each block contributes to each kernel
- **S values** — how much variance each kernel captures
- **Vt rows** — which features define each kernel

### Reality Regression

```python
reality_regression = importance @ Vt[:n_kernels, :]
# Shape: (80,) — weighted importance of each feature across all kernels
```

This is the system's single best summary vector. Each feature's weight
reflects its contribution to the overall cross-domain pattern, weighted by
the importance of the kernel it belongs to. The top features in this vector
are the most explanatory features across the entire pipeline.

### Kernel Importance

```python
importance = S / (S.sum() + 1e-8)
```

Normalized singular values. The first kernel typically explains the most
variance. Importance values are reported as percentages in kernel labels.

### Kernel Activation

```python
kernel_activation = U * S[np.newaxis, :]
# Shape: (n_blocks, n_kernels)
```

How strongly each block activates each kernel. This matrix is visualized as
the "Universal Kernel Matrix" heatmap in the dashboard and interpreter tab.

### Reconstruction Error

```python
recon = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
error = np.linalg.norm(matrix - recon)
```

Measures how much information the kernel decomposition preserves. Lower is
better. Reported in snapshots and dashboard metrics.

---

## Per-Region Normalization

Before being added to the UKT, each feature vector is normalized per-region:

```python
for (lo, hi) in [(0,16), (16,32), (32,48), (48,64), (64,80)]:
    region = features[lo:hi]
    features[lo:hi] = (region - region.min()) / (region.max() - region.min())
```

This ensures cross-block comparability: a finance attention weight and a
centrality score both live in [0, 1] after normalization, preventing one
region from dominating the SVD purely by scale.

---

## Reality Regression Stability

**Function:** `estimate_reality_regression_stability()`

Tests whether the reality regression direction is robust to small data
perturbations, approximating modality-agnosticism.

**Algorithm:**
1. Run `n_runs` iterations (default 8)
2. Each run: add Gaussian noise (std=0.01) → SVD → compute reality regression → normalize
3. Compute pairwise cosine similarities between all regression vectors
4. Return mean, min, and std of cosine similarities

**Interpretation:**
- `mean_cosine > 0.9` — stable, robust conclusions
- `mean_cosine < 0.5` — unstable, conclusions shift with minor data changes

```python
from hyperspace.models.knowledge_matrix import estimate_reality_regression_stability

stability = estimate_reality_regression_stability(matrix, n_runs=8, noise_std=0.01)
# Returns: {"n_runs": 8, "mean_cosine": 0.95, "min_cosine": 0.91, "std_cosine": 0.02}
```

---

## Feature Provenance

Every feature index can be traced back to its source through multiple layers:

1. **Feature name** — `FEATURE_NAMES[idx]` (e.g. `"attention_recent_1d"`)
2. **Region** — `FEATURE_REGION_LABELS` (e.g. `"temporal-pattern"`)
3. **Block** — `BLOCK_REGION_MAP` (e.g. `"Finance"`)
4. **Metadata** — `feature_meta[idx]` provides source, entity, metric, time scope
5. **Region description** — `REGION_DESCRIPTIONS[region]` provides semantic context

This chain enables full accountability: any value in the reality regression
vector can be traced from its weight → feature name → region → block → data
source → raw data.

**Helper functions:**

```python
from hyperspace.models.knowledge_matrix import _feature_name, _region_for_index

_feature_name(12)         # "attention_trend_strength"
_region_for_index(12)     # "temporal-pattern"
_feature_name(32)         # "centrality_node_0"
_region_for_index(32)     # "structural-centrality"
```

---

## Kernel Labeling and Narratives

Each kernel receives a data-grounded label and narrative generated by
`_label_kernel()` and `_generate_kernel_narrative()`.

### Label Format

```
K0: Finance — temporal pattern (45.0% var, lead: attention_trend_strength)
```

Components: kernel ID, dominant block, dominant region, variance explained,
leading feature.

### Narrative Content

The narrative includes:
- Variance explained and primary driver
- Top 3 evidence features with their loadings
- Block contributions (from U column)
- Region context (from `REGION_DESCRIPTIONS`)
- Cross-region coupling if temporal and semantic features co-occur
- Time alignment window if available

Example:
```
Kernel K0 explains 45.0% of total variance.
Primary driver: Finance block; dominant latent region: temporal pattern.
Top evidence features: attention_trend_strength (+0.342), attention_mid_month (+0.287), attention_recent_1d (+0.201).
Block contributions: Finance (0.89), Clusters (0.31).
Region context: Temporal attention patterns from the TFT encoder...
Temporal step attribution: attention_trend_strength=+0.342, attention_mid_month=+0.287, attention_recent_1d=+0.201
```

For kernels with importance > 15%, a Tiny-LLM semantic narrative is also
generated (see [Semantic Interpretability](./semantic-interpretability.md)).

---

## Global SAE Concept Discovery

**File:** `hyperspace/models/sparse_ae.py`

After the full UKT matrix is built, a global Sparse Autoencoder discovers
abstract concepts that span multiple blocks.

### `train_sparse_ae()`

```python
def train_sparse_ae(
    combined_features: np.ndarray,  # (n_blocks, 80) — the full UKT matrix
    hidden_dim: int = 16,           # number of concepts to discover
    epochs: int = 80,
    lr: float = 0.005,
) -> dict | None
```

Training uses noise-augmented data (±0.05 std) and L1 sparsity penalty
(weight=0.01). Each discovered concept is labeled with:
- Dominant region (by encoder weight energy)
- Top 3 feature loadings
- Active/dormant status (above/below mean activation)
- Full narrative grounded in feature evidence

**Return dict:**

```python
{
    "concept_vectors": np.ndarray,       # (16, 80) encoder weights
    "concept_activations": np.ndarray,   # (n_blocks, 16) per-block activations
    "mean_activation": np.ndarray,       # (16,) average activation per concept
    "active_concepts": int,              # count of above-mean concepts
    "total_concepts": int,               # hidden_dim
    "final_loss": float,
    "loss_history": list[float],
    "concept_labels": list[dict],        # one per concept with label, narrative, etc.
}
```

### `map_concepts_to_kernels()`

```python
def map_concepts_to_kernels(sae_result: dict, kernel_snapshot: dict) -> list[dict]
```

Projects concept vectors into kernel space via `concepts @ Vt.T` and finds
the best-matching kernel for each concept. Returns a correspondence table:

```python
[
    {"concept": "C00", "best_kernel": "K0", "kernel_importance": 0.45,
     "coherence": 0.89, "mean_activation": 0.12, "active": True},
    ...
]
```

### `enrich_concepts_with_narratives()`

```python
def enrich_concepts_with_narratives(sae_result: dict, canvas: SemanticCanvas) -> dict
```

Calls `narrate_concept()` for each active concept and adds a
`semantic_narrative` field to its concept label. Modifies `sae_result`
in-place.

---

## Visualization

**File:** `hyperspace/viz/kernel_viz.py`

| Function | Renders |
|----------|---------|
| `plot_kernel_matrix(snapshot, block_names)` | Heatmap of kernel activations across blocks (RdBu colorscale) |
| `plot_kernel_importance(snapshot)` | Bar chart of kernel importance (variance explained) |
| `plot_reality_regression(snapshot)` | Bar chart of reality regression weights per feature |
| `plot_kernel_evolution(snapshots)` | Line chart tracking kernel importance across pipeline steps |
| `plot_concept_kernel_map(concept_kernel_map)` | Scatter plot of concept-kernel correspondence |

All plots use `plotly_dark` template and the shared `PLOTLY_LAYOUT` config.

---

## Example: Manual UKT Construction

```python
import numpy as np
from hyperspace.models.knowledge_matrix import (
    UniversalKnowledgeTensor,
    estimate_reality_regression_stability,
)

ukt = UniversalKnowledgeTensor(feature_dim=80)

# Add blocks (features would come from actual pipeline models)
snap1 = ukt.add_block("Finance", finance_features,
                       feature_meta=tft_result["feature_meta"])
snap2 = ukt.add_block("Clusters", cluster_features,
                       feature_meta=cluster_result["feature_meta"])
snap3 = ukt.add_block("Graph", graph_features,
                       feature_meta=graph_analysis["feature_meta"])
snap4 = ukt.add_block("Spatial", spatial_features,
                       feature_meta=spatial_result["feature_meta"])
snap5 = ukt.add_block("Agents", agent_features,
                       feature_meta=agent_feature_meta)

# Access results
final = ukt.get_latest_snapshot()
print(f"Kernels: {final['n_kernels']}")
print(f"Top kernel: {final['kernel_labels'][0]['label']}")
print(f"Recon error: {final['reconstruction_error']:.6f}")

# Reality regression: which features matter most?
rr = final["reality_regression"]
top_idx = np.argsort(np.abs(rr))[-5:][::-1]
for i in top_idx:
    print(f"  {i}: {rr[i]:+.4f}")

# Stability test
matrix = ukt.get_final_matrix()
stability = estimate_reality_regression_stability(matrix)
print(f"Stability: mean_cosine={stability['mean_cosine']:.3f}")

# Semantic canvas
canvas_state = ukt.canvas.get_accumulated_state()
for d in canvas_state["dominant_narrative"]:
    print(f"  {d['label']}: {d['value']:.2f}")
```
