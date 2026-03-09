# Integration Testing — Hyperspace v3.0

## Overview

The integration test suite validates the full Hyperspace pipeline end-to-end using **synthetic data only** — no external APIs, no Streamlit session state, no GPU required. All 74 tests run in ~60 seconds on CPU.

```bash
# Run all integration tests
python -m pytest tests/test_full_pipeline_integration.py -v

# Run a specific test class
python -m pytest tests/test_full_pipeline_integration.py::TestPipelineRunner -v
```

## Test Architecture

```
hyperspace/
├── core/
│   ├── __init__.py           # Core package
│   ├── types.py              # Unified type system (BlockResult, PipelineResult, etc.)
│   └── pipeline.py           # Streamlit-free PipelineRunner orchestrator
tests/
├── conftest.py               # Shared fixtures (synthetic data generators)
└── test_full_pipeline_integration.py   # 74 integration tests across 13 test classes
```

### Core Architecture (`hyperspace/core/`)

The `core` package provides the integration backbone:

**`types.py`** — Unified type definitions:
- `BlockResult`: Standard output from any pipeline block (`features_for_ukt`, `feature_meta`, `data_source`)
- `SnapshotResult`: UKT snapshot with all SVD decomposition fields
- `StabilityResult`: Multi-run stability estimation output
- `GovernanceFlag`: Auto-detected governance issue
- `ScorecardEntry`: One dimension of the interpretability scorecard
- `PipelineResult`: Full pipeline output (all block results + governance + narratives)
- `validate_block_result()`: Validates any block output against the BlockResult contract
- `validate_snapshot()`: Validates UKT snapshot completeness

**`pipeline.py`** — `PipelineRunner` class:
- Orchestrates all 5 blocks through UKT without Streamlit dependency
- Accepts pre-fetched data (finance, clusters, spatial, agreement matrix)
- Returns a `PipelineResult` with everything needed for rendering
- Computes governance flags and interpretability scorecard
- Wires semantic narrator functions (graceful degradation)
- Supports progress callbacks for UI integration

### Fixtures (`conftest.py`)

All fixtures produce deterministic synthetic data via `np.random.default_rng(42)`:

| Fixture | Description |
|---------|-------------|
| `rng` | Seeded random generator for reproducibility |
| `synthetic_finance_features` | 80-dim vector mimicking TFT output (slots 0–31) |
| `synthetic_cluster_features` | 80-dim vector mimicking BERTopic output (slots 16–31) |
| `synthetic_agreement_matrix` | 6x6 pairwise agreement matrix for geopolitical nodes |
| `synthetic_spatial_data` | Physical raster (4x6x9) + country scalars (10x6) |
| `timeframe_context` | Date range metadata dict |

## Test Classes & Coverage

### 1. `TestUKTCore` (7 tests)
Validates the Universal Knowledge Tensor accumulation and SVD decomposition.

- Single block addition produces correct snapshot shape and metadata
- Multi-block accumulation grows the matrix correctly
- Reality regression vector has correct dimensionality (80-dim)
- Feature normalization maps all regions to [0, 1]
- `_pad_or_truncate` handles short and long vectors
- Kernel labels contain all required fields
- `get_final_matrix()` returns `None` when empty, correct shape after blocks

### 2. `TestGraphEngine` (4 tests)
Validates geopolitical graph construction and centrality analysis.

- Graph construction without agreement matrix uses static edge list (6 nodes, 15 edges)
- Agreement matrix blending adjusts edge weights (40% original + 60% agreement)
- Centrality analysis produces degree, betweenness, eigenvector, PageRank + community detection
- Graph features integrate into UKT structural-centrality region (slots 32–47)

### 3. `TestSpatialKernels` (4 tests)
Validates SVD spatial kernelization across 10 data layers.

- `build_full_feature_matrix` produces (14, 6) row-normalized matrix
- `kernelize_spatial` populates geospatial region (slots 64–79), kernel importances sum to 1
- `get_spatial_features` returns complete result dict with metadata
- Spatial features integrate correctly into UKT

### 4. `TestAgentSimulation` (5 tests)
Validates multi-agent bounded-rational simulation.

- Default initialization creates 6 agents matching geopolitical nodes
- Graph-driven initialization uses centrality for agent parameters
- Spatial-enriched initialization adjusts capability multipliers (0.5–2.0)
- Simulation produces correct 80-dim feature vector with resource shares summing to ~1
- Agent features integrate into UKT dynamic-agent region (slots 48–63)

### 5. `TestSparseAutoencoder` (4 tests)
Validates SAE concept discovery on UKT matrices.

- SAE training on 5x80 matrix produces 16 concept vectors with activations
- Concept labels contain required fields (`concept_id`, `dominant_region`, `active`, `narrative`)
- Concept-kernel mapping projects SAE concepts into SVD kernel space
- Single-row input correctly returns `None` (insufficient data)

### 6. `TestSemanticCanvas` (7 tests)
Validates the 12-dimensional semantic interpretation space.

- Canvas initializes with zero cumulative state
- Block projection creates `CanvasEntry` with coordinates, dominant dimensions, interpretation
- Multi-block accumulation grows canvas entries and cumulative coordinates
- `get_accumulated_state()` returns dimensions, coordinates, entries, trajectory
- `format_for_narrator()` produces text containing all block names
- Stage SAE training produces concept vectors and active masks
- All UKT regions have canvas dimension mappings (`REGION_TO_CANVAS`)

### 7. `TestUKTCanvasIntegration` (2 tests)
Validates that UKT `add_block` automatically drives the Semantic Canvas.

- Each `add_block` call creates a canvas entry and stage SAE result
- Canvas grows to 5 entries after full 5-block pipeline

### 8. `TestRealityRegressionStability` (3 tests)
Validates multi-run stability estimation (8 noisy SVD reruns).

- Basic stability returns mean/min/std cosine similarity with n_runs=4
- Single-row matrix returns n_runs=0 (insufficient for stability test)
- Same seed produces identical stability results (determinism)

### 9. `TestFullPipelineIntegration` (14 tests)
**End-to-end integration**: all 5 blocks through UKT -> SAE -> governance.

Runs the complete pipeline via `_run_full_pipeline()`:
```
Finance -> Clusters -> Graph -> Spatial -> Agents -> SAE -> Concept Mapping -> Stability
```

Validates: matrix growth, block names, kernel count, reconstruction error, importance normalization, SAE concepts, concept-kernel map, stability, canvas layers, narrator output, feature meta, reports, cross-block coherence.

### 10. `TestGovernanceValidation` (2 tests)
Validates governance framework configuration and traceability.

- All 5 scorecard dimensions defined with thresholds and labels
- Feature traceability exceeds 50% of threshold after multi-block pipeline

### 11. `TestTypeValidation` (11 tests)
Validates the unified type system and contract enforcement.

- Valid BlockResult passes validation
- None result detected
- Missing `features_for_ukt`, wrong shape, missing `data_source` all caught
- UKT snapshot validation (complete and incomplete)
- Graph, Spatial, and Agent outputs conform to BlockResult contract

### 12. `TestPipelineRunner` (10 tests)
Validates the Streamlit-free `PipelineRunner` orchestrator.

- Full pipeline completes with 5 snapshots
- All 5 blocks tracked in `data_sources` (including Agents)
- Governance flags have required keys (code, label, description, severity)
- Scorecard has all 5 dimensions with value/threshold/passed
- Run ID generated (8-char UUID)
- Semantic canvas populated with 5 entries
- Progress callback receives step notifications
- Pipeline works without optional blocks (graph + agents only)
- Concept-kernel map populated
- Stability estimation works through runner

### 13. `TestCounterfactualIntegration` (2 tests)
Validates block removal and diff analysis.

- Removing one block produces valid sub-matrix SVD
- Counterfactual reality regression differs from original (non-zero diff)

## Data Flow Tested

```
Synthetic Data
    |
    +-> Finance Features (0-31)  -----> UKT.add_block("Finance")
    |                                        |
    +-> Cluster Features (16-31) -----> UKT.add_block("Clusters")
    |                                        |
    +-> Graph Centrality (32-47) -----> UKT.add_block("Graph")
    |                                        |
    +-> Spatial SVD Kernels (64-79) --> UKT.add_block("Spatial")
    |                                        |
    +-> Agent Simulation (48-63) -----> UKT.add_block("Agents")
                                             |
                                     +-------+-------+
                                     |  Final Matrix  |
                                     |   (5 x 80)     |
                                     +-------+-------+
                                             |
                           +-----------------+------------------+
                           |                 |                  |
                    Sparse AE          SVD Stability      Semantic Canvas
                  (16 concepts)      (4 noisy reruns)     (12 dimensions)
                           |                                    |
                    Concept-Kernel                        Narrator Format
                      Mapping                           (all 5 blocks)
                           |
                    +------+------+
                    |  Governance  |
                    |  Flags (5)   |
                    |  Scorecard   |
                    +-------------+
```

## Software Architecture

```
hyperspace/
├── core/                    # Integration backbone (NEW)
│   ├── types.py             #   Unified TypedDicts + validation
│   └── pipeline.py          #   Streamlit-free PipelineRunner
├── config.py                # Constants, UKT layout, CSS
├── state.py                 # Session state management
├── models/                  # ML/core algorithms
│   ├── knowledge_matrix.py  #   UKT: incremental SVD + semantic canvas
│   ├── tft_forecast.py      #   TemporalFusionTransformer
│   ├── topic_model.py       #   BERTopic clustering
│   ├── graph_engine.py      #   NetworkX geopolitical graph
│   ├── spatial_kernels.py   #   SVD spatial kernelization
│   ├── agent_sim.py         #   Multi-agent simulation
│   ├── sparse_ae.py         #   Sparse Autoencoder
│   ├── semantic_canvas.py   #   12-dim interpretive space
│   └── semantic_narrator.py #   Tiny-LLM narrative generation
├── data/                    # Data fetchers (32+ keyless APIs)
│   ├── finance.py           #   10 finance sources
│   ├── news.py              #   11 news sources
│   ├── political.py         #   10 political sources
│   └── spatial.py           #   12 spatial sources
├── pages/                   # Streamlit UI tabs
│   ├── dashboard.py         #   Landing + pipeline orchestration
│   ├── mission_control_tab.py
│   ├── finance_tab.py
│   ├── clusters_tab.py
│   ├── politics_tab.py
│   ├── agents_tab.py
│   ├── interpreter_tab.py
│   ├── pipeline_tab.py
│   ├── counterfactual_tab.py
│   ├── governance.py        #   Governance UI helpers
│   └── _report_section.py   #   Shared interpretability report
└── viz/                     # Visualization modules
    ├── charts.py            #   Reusable Plotly builders
    └── kernel_viz.py        #   Kernel matrix/importance plots
```

## Key Design Decisions

1. **Synthetic-only**: Tests never hit external APIs, ensuring CI reliability and speed
2. **Deterministic seeds**: `rng(42)` and `rng(99)` ensure reproducible results across runs
3. **No Streamlit dependency**: Tests import model/engine modules and `PipelineRunner` directly
4. **Structural assertions**: Tests verify shapes, ranges, and field presence rather than exact values
5. **Progressive complexity**: Individual block tests -> pairwise integration -> full pipeline -> PipelineRunner
6. **Contract enforcement**: `validate_block_result()` ensures all blocks conform to the unified interface
7. **Governance testing**: Flags and scorecard computed and validated end-to-end
