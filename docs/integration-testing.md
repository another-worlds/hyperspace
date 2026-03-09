# Integration Testing — Hyperspace v3.0

## Overview

The integration test suite validates the full Hyperspace pipeline end-to-end using **synthetic data only** — no external APIs, no Streamlit session state, no GPU required. All 52 tests run in ~14 seconds on CPU.

```bash
# Run all integration tests
python -m pytest tests/test_full_pipeline_integration.py -v

# Run a specific test class
python -m pytest tests/test_full_pipeline_integration.py::TestFullPipelineIntegration -v
```

## Test Architecture

```
tests/
├── conftest.py                         # Shared fixtures (synthetic data generators)
└── test_full_pipeline_integration.py   # 52 integration tests across 10 test classes
```

### Fixtures (`conftest.py`)

All fixtures produce deterministic synthetic data via `np.random.default_rng(42)`:

| Fixture | Description |
|---------|-------------|
| `rng` | Seeded random generator for reproducibility |
| `synthetic_finance_features` | 80-dim vector mimicking TFT output (slots 0–31) |
| `synthetic_cluster_features` | 80-dim vector mimicking BERTopic output (slots 16–31) |
| `synthetic_agreement_matrix` | 6×6 pairwise agreement matrix for geopolitical nodes |
| `synthetic_spatial_data` | Physical raster (4×6×9) + country scalars (10×6) |
| `timeframe_context` | Date range metadata dict |

## Test Classes & Coverage

### 1. `TestUKTCore` (7 tests)
Validates the Universal Knowledge Tensor accumulation and SVD decomposition.

- Single block addition produces correct snapshot shape and metadata
- Multi-block accumulation grows the matrix correctly
- Reality regression vector has correct dimensionality (80-dim)
- Feature normalization maps all regions to [0, 1]
- `_pad_or_truncate` handles short and long vectors
- Kernel labels contain all required fields (`kernel_id`, `dominant_block`, `dominant_region`, `importance`, `top_features`, `label`, `narrative`)
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

- SAE training on 5×80 matrix produces 16 concept vectors with activations
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
**End-to-end integration**: all 5 blocks through UKT → SAE → governance.

This class runs the complete pipeline via `_run_full_pipeline()`:

```
Finance → Clusters → Graph → Spatial → Agents → SAE → Concept Mapping → Stability
```

Validates:
- Pipeline completes with 5 snapshots and (5, 80) final matrix
- UKT matrix grows from (1, 80) to (5, 80) incrementally
- Block names match expected order
- Kernel count equals block count (5)
- Reconstruction error < 0.001
- Importance vector sums to 1.0
- SAE discovers 16 concepts with active count > 0
- Concept-kernel map has 16 entries with valid kernel references
- Stability mean cosine > 0.1 (synthetic data threshold)
- Canvas contains all 5 layers in correct order
- Canvas narrator output references all block names
- Feature metadata accumulated from multiple blocks (>20 entries)
- All snapshots contain reports (>50 characters)
- Cross-block coherence appears in reports after step 1

### 10. `TestGovernanceValidation` (2 tests)
Validates governance framework configuration and traceability.

- All 5 scorecard dimensions defined with thresholds and labels
- Feature traceability exceeds 50% of threshold after multi-block pipeline

## Data Flow Tested

```
Synthetic Data
    │
    ├─→ Finance Features (0–31) ─────→ UKT.add_block("Finance")
    │                                        │
    ├─→ Cluster Features (16–31) ────→ UKT.add_block("Clusters")
    │                                        │
    ├─→ Graph Centrality (32–47) ────→ UKT.add_block("Graph")
    │                                        │
    ├─→ Spatial SVD Kernels (64–79) ─→ UKT.add_block("Spatial")
    │                                        │
    └─→ Agent Simulation (48–63) ────→ UKT.add_block("Agents")
                                             │
                                     ┌───────┴───────┐
                                     │  Final Matrix  │
                                     │   (5 × 80)     │
                                     └───────┬───────┘
                                             │
                           ┌─────────────────┼─────────────────┐
                           │                 │                 │
                    Sparse AE          SVD Stability     Semantic Canvas
                  (16 concepts)      (4 noisy reruns)    (12 dimensions)
                           │                                   │
                    Concept-Kernel                      Narrator Format
                      Mapping                          (all 5 blocks)
```

## Key Design Decisions

1. **Synthetic-only**: Tests never hit external APIs, ensuring CI reliability and speed
2. **Deterministic seeds**: `rng(42)` and `rng(99)` ensure reproducible results across runs
3. **No Streamlit dependency**: Tests import model/engine modules directly, bypassing UI layer
4. **Structural assertions**: Tests verify shapes, ranges, and field presence rather than exact values (which vary with random seeds)
5. **Progressive complexity**: Individual block tests → pairwise integration → full 5-block pipeline
