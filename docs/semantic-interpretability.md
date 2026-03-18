# Semantic Interpretability Subsystem

## Overview

The semantic interpretability subsystem adds three layers of human-readable
explanation to the Hyperspace pipeline. Instead of leaving model internals as
opaque matrices and floating-point vectors, the subsystem translates every
pipeline stage into named semantic dimensions and natural-language narratives.

**Architecture:**

```
Pipeline Stage
    │
    ▼
Hidden Activations (16-dim feature vector per block)
    │
    ▼
Per-Stage SAE ── discovers sparse concepts from each block
    │
    ▼
Semantic Canvas ── projects concepts onto 12 named dimensions
    │
    ▼
Tiny-LLM Narrator ── generates plain-English explanations
```

---

## Components

### 1. Per-Stage Sparse Autoencoders

**File:** `hyperspace/models/semantic_canvas.py` — `StageSAE`, `train_stage_sae()`

Each pipeline block (Finance, Clusters, Graph, Spatial, Agents) receives its
own lightweight SAE that runs in parallel with the main pipeline. The SAE
observes the block's 16-dimensional feature region and discovers 8 sparse
concepts.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 16 | Feature vector size per block region |
| `concept_dim` | 8 | Number of sparse concepts to discover |
| `sparsity_weight` | 0.02 | L1 penalty on concept activations |
| `epochs` | 60 | Training iterations |
| `lr` | 0.008 | Learning rate |

Training uses noise-augmented data (3 noise scales: 0.03, 0.06, 0.1) to
produce richer gradients from the single-sample input.

**Usage:**

```python
from hyperspace.models.semantic_canvas import train_stage_sae

# features: (1, 16) or (n, 16) numpy array from one pipeline block
result = train_stage_sae(features, concept_dim=8, epochs=60)

# result keys:
#   concept_vectors   — (8, 16) learned encoder weights
#   activations       — (n, 8) concept activation values
#   mean_activation   — (8,) average activation per concept
#   active_mask       — (8,) boolean: above-mean activation
#   active_count      — int: number of active concepts
#   model             — the trained StageSAE nn.Module
```

### 2. Semantic Canvas

**File:** `hyperspace/models/semantic_canvas.py` — `SemanticCanvas`

The canvas is a 12-dimensional coordinate system with named axes. Each axis
represents a human-interpretable property of the data environment:

| Index | Key | Label | Description |
|-------|-----|-------|-------------|
| 0 | `market_momentum` | Market Momentum | Short-term financial momentum signals |
| 1 | `temporal_memory` | Temporal Memory Depth | How far back the system looks |
| 2 | `volatility_regime` | Volatility Regime | Stability vs turbulence |
| 3 | `information_focus` | Information Focus | Single narrative vs fragmented |
| 4 | `narrative_diversity` | Narrative Diversity | Breadth of informational themes |
| 5 | `alliance_polarity` | Alliance Polarity | Unipolar vs multipolar structure |
| 6 | `network_cohesion` | Network Cohesion | Graph density and clustering |
| 7 | `power_concentration` | Power Concentration | Resource concentration among actors |
| 8 | `geographic_coupling` | Geographic Coupling | Physical-socioeconomic co-variance |
| 9 | `systemic_stress` | Systemic Stress | Aggregate cross-domain pressure |
| 10 | `cooperation_signal` | Cooperation Signal | Positive alignment dynamics |
| 11 | `competition_signal` | Competition Signal | Rivalry and zero-sum dynamics |

**Region-to-canvas mapping:**

Each UKT feature region projects onto specific canvas axes:

- **temporal-pattern** [0:16] → Market Momentum, Temporal Memory, Volatility Regime
- **semantic-embedding** [16:32] → Information Focus, Narrative Diversity
- **structural-centrality** [32:48] → Alliance Polarity, Network Cohesion, Power Concentration
- **dynamic-agent** [48:64] → Power Concentration, Cooperation Signal, Competition Signal
- **geospatial-kernel** [64:80] → Geographic Coupling, Systemic Stress

**Usage:**

```python
from hyperspace.models.semantic_canvas import SemanticCanvas

canvas = SemanticCanvas()

# Project a block onto the canvas
entry = canvas.project_block(
    block_name="Finance",
    step=1,
    region_name="temporal-pattern",
    features=finance_features,      # (16,) numpy array
    sae_result=stage_sae_result,    # from train_stage_sae()
)

# entry.coordinates     — (12,) canvas position
# entry.dominant_dimensions — e.g. ["market_momentum", "temporal_memory"]
# entry.interpretation  — algorithmic summary string

# After all blocks have been projected:
state = canvas.get_accumulated_state()
# state["coordinates"]        — (12,) cumulative, normalized to [0,1]
# state["trajectory"]         — list of per-step cumulative states
# state["dominant_narrative"] — top-3 dimensions with labels and values
```

**Integration with UKT:**

The canvas is automatically maintained by `UniversalKnowledgeTensor.add_block()`.
Each call to `add_block()` runs `train_stage_sae()` on the block's feature
region and calls `canvas.project_block()` to accumulate the result. The
snapshot returned by `add_block()` includes `layer_narrative` (from the
Tiny-LLM narrator) and the full canvas entry.

### 3. Tiny-LLM Narrator

**File:** `hyperspace/models/semantic_narrator.py`

The narrator loads [arnir0/Tiny-LLM](https://huggingface.co/arnir0/Tiny-LLM)
(~10M parameters, MIT license) via HuggingFace Transformers and generates
natural-language narratives from structured canvas data.

**Model details:**
- Llama-based completion model (not instruction-tuned)
- Loaded once via `@st.cache_resource`
- CPU-only inference, `max_new_tokens` capped at 150–200
- Prompts are structured as analytical text for the model to continue
- Output is post-processed: truncated at last complete sentence, capped at 3
  sentences

**Graceful degradation:** If the model cannot be loaded (network unavailable,
dependency missing), all narration functions return `None` and the system falls
back to algorithmic interpretations from the canvas.

**Narration functions:**

| Function | Input | Purpose |
|----------|-------|---------|
| `narrate_canvas(canvas)` | Full `SemanticCanvas` | Cross-domain synthesis narrative |
| `narrate_layer(entry, canvas)` | Single `CanvasEntry` | Per-layer contribution narrative |
| `narrate_kernel(kernel_label, canvas)` | Kernel dict from SVD | Plain-language kernel description |
| `narrate_reality_regression(snapshot, canvas)` | UKT snapshot | Reality assessment summary |
| `narrate_concept(concept_label, canvas)` | SAE concept dict | Individual concept explanation |

**Usage:**

```python
from hyperspace.models.semantic_narrator import narrate_canvas, narrate_layer

# Full pipeline narrative
narrative = narrate_canvas(canvas)
# Returns: "the system detected strong market momentum driven by..."

# Single layer narrative
layer_text = narrate_layer(canvas_entry, canvas)
# Returns: "the Finance layer reveals that short-term momentum..."
```

---

## Pipeline Integration

The semantic subsystem is wired into the pipeline at two points:

### During pipeline execution (`dashboard.py: run_pipeline()`)

1. **`ukt.add_block()`** — Each call internally runs a per-stage SAE on the
   block's feature region and projects the result onto the semantic canvas.
   The snapshot includes a `layer_narrative` generated by the narrator.

2. **Post-pipeline narration** — After all blocks complete, the dashboard calls:
   - `narrate_canvas()` → stored in `st.session_state.canvas_narrative`
   - `narrate_reality_regression()` → stored in `st.session_state.reality_narrative`
   - `enrich_concepts_with_narratives()` → adds `semantic_narrative` to each
     SAE concept label

### In the Semantic Interpreter tab (`interpreter_tab.py`)

The tab renders:

1. **Semantic Canvas Radar Chart** — Polar chart showing accumulated canvas
   coordinates, with per-layer traces overlaid as dotted lines.

2. **Canvas Evolution Heatmap** — `imshow` plot showing how the cumulative
   canvas changes as each pipeline layer is added.

3. **Dominant Semantic Themes** — Bulleted list of the top-3 canvas dimensions
   with their descriptions.

4. **Layer-by-Layer Narratives** — Expandable sections per pipeline step showing
   the algorithmic interpretation and Tiny-LLM narrative.

5. **Overall Reality Narrative** — Full-canvas Tiny-LLM output.

6. **Per-kernel semantic narratives** — Each discovered kernel shows its
   Tiny-LLM interpretation alongside the standard feature-loading table.

7. **Per-concept semantic narratives** — Each SAE concept shows its Tiny-LLM
   interpretation within its expander.

**Known UI gaps** (see `STRATEGY_UI.md` and `docs/architecture-semantic-interpretability.md`):
- Reality Regression chart shows feature indices instead of `FEATURE_NAMES` (ERR-009)
- SAE training not cached between clicks (H-007)
- Concept labels truncated to 18 chars in heatmap
- Radar chart dimension labels overlap at default size
- Advanced Diagnostics expander has no internal navigation (~200 lines)

---

## Session State Keys

| Key | Type | Description |
|-----|------|-------------|
| `semantic_canvas` | `SemanticCanvas` | Canvas instance with all projected entries |
| `canvas_narrative` | `str \| None` | Tiny-LLM cross-domain narrative |
| `reality_narrative` | `str \| None` | Tiny-LLM reality regression narrative |

These are populated by `run_pipeline()` and consumed by `render_results()` and
the Semantic Interpreter tab.

---

## Adding a New Pipeline Block

To integrate a new block with the semantic subsystem:

1. **Choose a UKT feature region** — The block's 16-dim features must map to
   one of the 5 regions (temporal-pattern, semantic-embedding, etc.).

2. **Update `REGION_TO_CANVAS`** in `semantic_canvas.py` — Add canvas dimension
   mappings for the new region if needed.

3. **Call `ukt.add_block()`** — The per-stage SAE and canvas projection are
   handled automatically by `UniversalKnowledgeTensor.add_block()`.

No changes to the narrator are needed — it reads from the canvas structure
which is populated by the projection step.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.1 | SAE training (CPU) |
| `transformers` | ≥4.36 | Tiny-LLM model loading and inference |
| `numpy` | ≥1.24 | Feature vectors, canvas coordinates |
| `plotly` | ≥5.18 | Radar chart, heatmap visualizations |
| `streamlit` | ≥1.31 | `@st.cache_resource` for model caching |
