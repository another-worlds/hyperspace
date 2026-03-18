# Semantic Interpretability Framework Architecture

## Overview

The Semantic Interpretability framework translates emergent UKT structure into
human-readable coordinates, natural-language narratives, and contestable
counterfactual analyses. It spans two packages:

- **`semantic_interpreter/`** — Standalone interpretability components (canvas,
  SAE, narrator, concept mapping)
- **`hyperspace/models/`** — Hyperspace-specific wrappers with registry-aware
  canvas dimensions and Streamlit integration
- **`hyperspace/pages/`** — UI rendering (interpreter tab, counterfactual tab)

The framework's central principle: **the canvas is an interpretive lens, not a
learning system.** Its dimensions are fixed by domain knowledge; its
coordinates are data-driven. The UKT discovers structure; the canvas provides
vocabulary for communicating it.

---

## Module Map

```
semantic_interpreter/
├── __init__.py        # Public exports
├── canvas.py          # Core SemanticCanvas + CanvasEntry + projection logic
├── sae.py             # StageSAE + GlobalSAE + training functions
├── narrator.py        # NarratorBackend protocol + Template + LLM narrators
└── concepts.py        # Concept-to-kernel mapping

hyperspace/models/
├── semantic_canvas.py # Registry-aware canvas with REGION_SEMANTIC_SPEC
├── semantic_narrator.py  # Hyperspace narrator singleton + wrapper functions
└── sparse_ae.py       # Hyperspace SAE wrapper

hyperspace/pages/
├── interpreter_tab.py    # Canvas radar, SAE heatmap, narrative rendering
└── counterfactual_tab.py # Block ablation + contestability analysis
```

---

## Three-Layer Translation Architecture

```
Machine Latent Space (80-dim projected features)
         │
         ▼
   ┌──────────────┐
   │ Semantic      │  12 named dimensions
   │ Canvas        │  Data-driven coordinates (entropy, concentration, energy)
   │               │  Fixed interpretive frame — stable across runs
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ Sparse        │  Concept discovery via L1-regularized autoencoders
   │ Autoencoders  │  Maps latent patterns to region-grounded concepts
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ Narrator      │  Tiny-LLM (13M params) with template fallback
   │               │  Canvas coordinates + feature evidence → natural language
   └──────┬───────┘
          │
          ▼
   Human-Readable Narrative
```

---

## 1. Semantic Canvas (`semantic_interpreter/canvas.py`, `hyperspace/models/semantic_canvas.py`)

### Purpose

The canvas projects high-dimensional feature vectors onto a fixed set of named
semantic dimensions. A stakeholder sees "strong market momentum, moderate
alliance polarity" instead of a floating-point vector.

### Canvas Dimensions

Defined in `REGION_SEMANTIC_SPEC` (registry-aware — adding a new region
automatically extends the canvas):

| Region | Dimension | Description |
|--------|-----------|-------------|
| temporal-pattern | `market_momentum` | Strength and direction of short-term financial momentum |
| temporal-pattern | `temporal_memory` | How far back the system looks |
| temporal-pattern | `volatility_regime` | Market stability vs. turbulence |
| semantic-embedding | `information_focus` | Single narrative vs. fragmented information landscape |
| semantic-embedding | `narrative_diversity` | Breadth of distinct informational themes |
| structural-centrality | `alliance_polarity` | Unipolar vs. multipolar geopolitical structure |
| structural-centrality | `network_cohesion` | Density and clustering of relationship graph |
| structural-centrality | `power_concentration` | How concentrated influence is among actors |
| dynamic-agent | `cooperation_signal` | Net positive alignment and cooperative dynamics |
| dynamic-agent | `competition_signal` | Net negative alignment, rivalry |
| geospatial-kernel | `geographic_coupling` | Co-variance of physical and socioeconomic factors |
| geospatial-kernel | `systemic_stress` | Aggregate pressure across conflict, economic, political dims |

### Projection Mechanism: `project_block()`

```python
def project_block(block_name, step, region_name, features, sae_result=None) -> CanvasEntry
```

**Step 1: Feature analysis**
```python
energy = np.sum(np.abs(features))           # Total activation strength
concentration = max_act / (mean_act + 1e-8)  # How peaked the distribution is
probs = abs_feat / energy
entropy = -Σ(probs * log(probs)) / log(dim)  # Normalized Shannon entropy [0,1]
```

**Step 2: Coordinate computation** for each canvas dimension `i`:

- **Primary dimensions** (weight ≥ 0.99):
  `coords[i] += weight × mean_act × concentration`
  Primary axes respond to strong, focused signals.

- **Coupling dimensions** (weight < 0.99):
  `coords[i] += weight × mean_act × (1.0 + entropy)`
  Coupling axes respond to broad, distributed patterns.

**Step 3: Normalization** to [0,1] range.

**Step 4: Dominant dimensions** — top-2 by coordinate value.

**Step 5: Output** — `CanvasEntry` with coordinates, dominant dimensions,
interpretation text, and feature evidence.

### Accumulation

The canvas tracks cumulative state across all blocks:
```python
self.cumulative = self.cumulative + coords  # Running sum
```

This allows cross-block synthesis: the accumulated canvas represents the
system's total semantic position after all blocks have contributed.

### Canvas ↔ UKT Integration

In `UniversalKnowledgeTensor.add_block()`:
1. Canvas is **reset** after each new block
2. **All blocks** (1..N) are replayed through the canvas with current projection
3. Feature evidence is attached to each canvas entry
4. This ensures canvas entries are coherent — all computed under the same
   projection epoch

---

## 2. Sparse Autoencoders (`semantic_interpreter/sae.py`)

### Purpose

SAEs discover sparse, interpretable concepts from the feature matrix. Each
concept is a direction in feature space that activates for specific patterns.
L1 regularization ensures only a few concepts activate per input, making them
human-interpretable.

### Two Variants

#### StageSAE (per-block, lightweight)
```python
class StageSAE(nn.Module):
    encoder: Linear(input_dim=16, hidden_dim=8) + ReLU
    decoder: Linear(8, 16)
    loss = MSE(x_hat, x) + 0.02 × mean(|h|)
```

- Input: (n, 16) — one block's region features
- Epochs: 60
- Data augmentation: original + 3 noise copies (σ ∈ {0.03, 0.06, 0.1})
- Output: concept vectors (8×16), activations (n×8), active mask

**Note:** Per-block SAE was identified as low-utility in the audit (training on
1 vector + 3 noise copies learns little). Currently passed as `sae_result=None`
to `project_block()`. Retained for future use with larger per-block datasets.

#### GlobalSAE (full matrix, larger)
```python
class GlobalSAE(nn.Module):
    encoder: Linear(input_dim=80, hidden_dim=32) + ReLU
    decoder: Linear(32, 80)
    loss = MSE(x_hat, x) + 0.02 × mean(|h|)
```

- Input: (n_blocks, 80) — full projected matrix
- Epochs: 50
- Data augmentation: original + ±noise
- **Concept labeling**: Maps each concept to its dominant feature region via
  registry, identifies top-3 loadings, generates per-concept narrative

This is the meaningful SAE — it operates on the full cross-domain matrix where
patterns span multiple regions.

### Concept-Kernel Mapping (`semantic_interpreter/concepts.py`)

```python
def map_concepts_to_kernels(sae_result, snapshot) -> dict
```

Aligns SAE concepts with SVD kernels by computing cosine similarity between
concept vectors and kernel basis vectors (Vt rows). Each concept is mapped to
its most aligned kernel, creating a bridge between the two decomposition methods.

---

## 3. Narrator (`semantic_interpreter/narrator.py`, `hyperspace/models/semantic_narrator.py`)

### Purpose

Translates structured canvas coordinates, kernel labels, and SAE concepts into
natural-language text that governance reviewers and domain experts can read.

### Two Backends

#### TemplateNarrator (deterministic, always available)

Produces structured text from templates. Used as fallback when LLM is
unavailable or when faithfulness checks downgrade confidence.

Methods:
| Method | Input | Output |
|--------|-------|--------|
| `narrate_canvas()` | Accumulated canvas state | Bulleted dimension summary |
| `narrate_layer()` | Block name, canvas entry | "{block} layer {strength} activates {dimensions}" |
| `narrate_kernel()` | Kernel label, canvas context | "Kernel {id} explains {importance}% variance..." |
| `narrate_concept()` | SAE concept, region | "Concept {id} encodes {region} patterns..." |
| `narrate_reality_regression()` | Final snapshot, canvas | "Reality regression most influenced by: {features}" |

#### LLMNarrator (generative, CPU-based)

- **Model**: arnir0/Tiny-LLM (HuggingFace, 13M params, MIT license)
- **Cached**: `@st.cache_resource` — single load per session
- **Config**: max_new_tokens=60, temperature=0.7, 15s timeout
- **CPU-only** inference

**Prompt construction**: Each narrative type has a structured prompt that
provides canvas coordinates, feature evidence, and kernel data, ending with
a completion trigger:

```
Canvas narrative:   "...the key finding is that"
Layer narrative:    "...the {block} layer reveals that"
Kernel narrative:   "...this kernel represents"
Concept narrative:  "...this concept captures the idea that"
Reality narrative:  "...the overall assessment is that"
```

**Post-processing**: Truncates at last complete sentence, caps at 3 sentences,
cleans whitespace.

### Faithfulness Guarantee

Narratives are subject to intervention-based faithfulness checks
(`hyperspace/core/faithfulness.py`). If checks fail:
1. `faithfulness.low_confidence` is set to `True`
2. `canvas_narrative` and `reality_narrative` are replaced with the
   conservative `downgraded_narrative`
3. The system never presents explanations that fail mechanistic validation

---

## 4. Counterfactual Analysis (`hyperspace/pages/counterfactual_tab.py`)

### Purpose

Operationalizes **contestability** — the ability for a stakeholder to ask
"What would the system conclude without this data source?" and receive a
mathematically rigorous answer.

### Mechanism

Unlike naive approaches that slice rows from an already-projected matrix,
Hyperspace rebuilds the **entire pipeline** from raw features:

```
1. User selects block to remove
2. Load raw_features from snapshot (pre-projection)
3. Create new SharedProjection from remaining blocks only
   └── For each kept block:
       normalize → observe in new projection
4. Re-project remaining blocks through new P
5. Stack into reduced matrix → new SVD
6. Compare kernels: original vs. counterfactual
```

This is correct because the projection matrix P depends on ALL active blocks.
Removing a block changes P, which changes how ALL remaining blocks are projected,
which changes what SVD discovers.

### Comparison Metrics

| Metric | Computation | Interpretation |
|--------|-------------|----------------|
| RR Cosine Similarity | dot(rr_orig, rr_cf) / norms | 1.0 = identical conclusions |
| RR L2 Distance | ‖rr_orig − rr_cf‖₂ | Magnitude of conclusion shift |
| Kernel Count Delta | n_kernels_orig − n_kernels_cf | Structural complexity change |
| Per-Region Energy Shift | % change in region energy | Which domains are most affected |

### Governance Interpretation

| Cosine | Interpretation |
|--------|---------------|
| ≥ 0.95 | Conclusions stable — minimal dependency on removed block |
| 0.80–0.95 | Moderate sensitivity — meaningful but not critical influence |
| < 0.80 | High sensitivity — system highly dependent on this block (governance risk) |

### Optional: Structural Shock

Flips the sign of features in the structural-centrality region (indices 32–47)
to simulate geopolitical inversion. Tests whether conclusions are robust to
fundamental structural changes.

---

## 5. Visualization (`hyperspace/pages/interpreter_tab.py`)

### Canvas Radar Chart

Polar plot with:
- Teal-filled shape: accumulated canvas coordinates (all blocks combined)
- Dotted traces: per-block canvas entries
- Labeled axes: the 12 semantic dimension names

### SAE Concept Heatmap

- Rows: pipeline blocks
- Columns: SAE concepts
- Cell color: activation strength
- Expandable per-concept narratives

### Reality Regression Bar Chart

- 80 bars (one per feature)
- Color-coded by region (blue=temporal, orange=semantic, green=structural,
  red=dynamic, purple=spatial)
- Height = importance in the reality regression

### Advanced Diagnostics

- Canvas trajectory heatmap: how cumulative coordinates evolve as blocks are added
- Concept-kernel correspondence matrix: alignment between SAE and SVD
- SAE loss curve: training convergence
- Dormant concept list: concepts that never activate (potential overhead)

### Known UI Rendering Gaps

The following issues affect how the interpretability framework is **presented**
to users, even though the backend produces correct data:

| Issue | Location | Impact |
|-------|----------|--------|
| **Feature indices instead of names** | `kernel_viz.plot_reality_regression()` | 80-bar chart shows indices 0–79, not `FEATURE_NAMES`. Breaks interpretability for non-technical users. |
| **Radar chart label overlap** | `interpreter_tab.py:88–118` | 12 polar axis labels crowd each other at default Plotly layout size. |
| **Concept label truncation** | `interpreter_tab.py:205–214` | SAE concept labels truncated to 18 chars in heatmap y-axis. |
| **Advanced Diagnostics unnavigable** | `interpreter_tab.py:316–508` | ~200 lines of charts/tables in one collapsed expander. No sub-navigation or table of contents. |
| **SAE not cached** | `interpreter_tab.py:48` | `train_sparse_ae(matrix, hidden_dim=16, epochs=100)` re-runs on every "Run Interpretability Scan" click. Should cache by matrix hash. |
| **Kernel expander threshold hardcoded** | `interpreter_tab.py:282` | `importance > 0.2` determines auto-expand. Should be `KERNEL_EXPANDER_THRESHOLD` in config. |
| **Per-head attention too small** | `interpreter_tab.py:436–452` | 4 attention heatmaps in a row are narrow and hard to read. |

See `STRATEGY_UI.md` §5 "Tab 5: Semantic Interpreter" for full analysis.

---

## Design Principles

1. **Fixed lens, data-driven coordinates.** The canvas dimensions are defined by
   domain knowledge and remain stable across runs. What changes is where the
   data lands on those dimensions. This prevents the failure mode of explanation
   systems that generate different "important features" on each run.

2. **Registry-driven extensibility.** Adding a new region to
   `REGION_SEMANTIC_SPEC` automatically adds canvas dimensions. No hardcoded
   dimension counts anywhere.

3. **Layered abstraction.** Features → Kernels → Canvas → Narratives. Each
   layer adds interpretability while preserving the underlying data.

4. **Graceful degradation.** LLM narrator falls back to templates. SAE handles
   degenerate inputs. Canvas handles empty entries. The system never crashes on
   missing or malformed data.

5. **Faithfulness enforcement.** Narratives that fail mechanistic intervention
   checks are automatically downgraded. The system prefers honest silence over
   plausible-sounding falsehoods.

6. **Contestability by construction.** Counterfactual analysis rebuilds the
   entire pipeline from raw features, including the projection matrix. This
   is the only correct approach — slicing an already-projected matrix would
   produce misleading results because P depends on all active blocks.

7. **Sparsity for interpretability.** SAE L1 regularization ensures only a few
   concepts activate per input. Dense activations would be as opaque as the
   original features.

---

## Key Data Types

### CanvasEntry
```python
CanvasEntry:
    coordinates: np.ndarray      # (n_dims,) semantic position
    concept_activations: np.ndarray  # Raw SAE concept activations
    active_concepts: int         # Count of above-mean concepts
    dominant_dimensions: list[str]   # Top-2 dimension keys
    interpretation: str          # Algorithmic text summary
    feature_evidence: list[dict] # Top-3 features with canvas dim links
```

### SemanticDimension
```python
SemanticDimension:
    key: str                     # e.g., "market_momentum"
    label: str                   # Human-readable name
    description: str             # Full description
    region: str                  # Owning region name
    weight: float                # Projection weight (1.0 = primary, <1.0 = coupling)
    canvas_index: int            # Position in coordinate vector
```

### Accumulated Canvas State
```python
state = canvas.get_accumulated_state()
# {
#     "coordinates": np.ndarray,        # Cumulative (n_dims,)
#     "dominant_narrative": list[str],   # Top dimensions
#     "entries": list[CanvasEntry],      # Per-block entries
#     "dimension_keys": list[str],       # Ordered dimension names
# }
```

---

*Document version: Alpha 1.0 — March 2026*
