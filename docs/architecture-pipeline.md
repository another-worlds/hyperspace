# Pipeline Architecture

## Overview

The Hyperspace pipeline (`hyperspace/core/pipeline.py`) is a sequential
orchestrator that executes five heterogeneous model blocks, feeds their outputs
into the Universal Knowledge Tensor, and produces a fully typed
`PipelineResult` with 26 keys covering raw data, interpretations, governance
checks, and audit trails.

The pipeline is **Streamlit-free** — it accepts pre-fetched data, performs all
computation in pure Python/NumPy/PyTorch, and returns a result dict that the
UI layer renders.

---

## Block Execution Order

```
Block 1: Finance (TFT)
    │  Input: finance_result dict with features_for_ukt (80-dim)
    │  Region: temporal-pattern [0:16]
    │  Source: Encoder attention weights, regime shifts, volatility
    ▼
Block 2: Clusters (BERTopic)
    │  Input: cluster_result dict with features_for_ukt (80-dim)
    │  Region: semantic-embedding [16:32]
    │  Source: Topic distributions, embedding statistics
    ▼
Block 3: Graph Engine (NetworkX)
    │  Input: agreement_matrix (country voting patterns)
    │  Region: structural-centrality [32:48]
    │  Source: Degree, betweenness, eigenvector, PageRank centrality
    ▼
Block 4: Spatial Raster [OPTIONAL]
    │  Input: spatial_data dict (physical_raster, country_scalars)
    │  Region: geospatial-kernel [64:80]
    │  Source: Spatial SVD kernel importances, per-country loadings
    ▼
Block 5: Agent Simulation
       Input: graph_analysis + agreement_matrix + spatial_result
       Region: dynamic-agent [48:64]
       Source: Graph density, clustering, community count, alliance eigenvalues
```

Each block produces an 80-dimensional feature vector. Only 16 dimensions
(the block's assigned region) carry meaningful signal; the rest are zero.
The UKT's shared projection creates cross-region coupling so that SVD
discovers patterns spanning multiple blocks.

---

## Data Flow: Block → UKT → Snapshot

For each block:

```
1. validate_block_result(result)          — Check required keys exist
2. features = result["features_for_ukt"]  — Extract 80-dim vector
3. snap = ukt.add_block(name, features, meta, context)
   └── Inside add_block():
       a. Pad/truncate to 80 dims
       b. Store raw features
       c. Normalize per-region to [0,1]
       d. Observe in SharedProjection → rebuild P
       e. Re-project ALL blocks through updated P
       f. Stack into matrix → SVD
       g. Label all kernels
       h. Reset + replay Semantic Canvas for all blocks
       i. Generate kernel narratives
       j. Return snapshot dict
4. snapshots.append(snap)
```

---

## Post-Block Analysis Stages

After all blocks have been added to the UKT, the pipeline runs six analysis
stages in sequence:

### Stage 1: Global SAE Concept Discovery

```python
sae_result = train_sparse_ae(final_matrix, hidden_dim=16, epochs=80)
concept_kernel_map = map_concepts_to_kernels(sae_result, final_snap)
```

Trains a sparse autoencoder on the full (n_blocks × 80) projected matrix to
discover cross-block latent concepts. Maps each concept to its most aligned
kernel.

### Stage 2: Cross-Block Interconnection [OPTIONAL]

```python
uvt_result = compute_universal_variance_tensor(final_matrix, ...)
use_result = compute_universal_semantic_encoding(final_matrix, uvt_result, ...)
```

Disabled by default (`compute_cross_block=False`). When enabled, trains:
- **UVT**: Universal Variance Tensor — attention-based variance decomposition
- **USE**: Universal Semantic Encoding — canvas-aligned semantic encoding

### Stage 3: Semantic Narratives

```python
canvas_narrative = narrate_canvas(ukt.canvas)
reality_narrative = narrate_reality_regression(snapshots[-1], ukt.canvas)
enrich_concepts_with_narratives(sae_result, ukt.canvas)
```

Generates natural-language summaries using the Tiny-LLM narrator (13M params,
CPU-only) with template fallback. Narratives are grounded in canvas coordinates
and feature evidence.

### Stage 4: Stability Analysis

```python
stability = estimate_reality_regression_stability(
    final_matrix, n_runs=8, noise_std=0.01, seed=42
)
```

Bootstrap perturbation test: adds Gaussian noise 8 times, recomputes SVD,
measures pairwise cosine similarity of resulting reality regressions.
`mean_cosine >= 0.75` = stable.

### Stage 5: Faithfulness Checks

```python
faithfulness = run_faithfulness_checks(
    snapshots, canvas, canvas_narrative, sae_result
)
if faithfulness.low_confidence:
    canvas_narrative = faithfulness.downgraded_narrative
```

Intervention-style mechanistic checks (H-003 spec). If explanations fail
faithfulness tests, narratives are automatically downgraded to conservative
fallback text. This is a **fail-safe** — the system never presents
unfaithful explanations.

### Stage 6: Governance & Scoring

```python
governance_flags = _compute_governance_flags(...)   # 5 auto-detected codes
scorecard = _compute_scorecard(...)                  # 9-dimension compliance
```

**Governance flags** (auto-detected):

| Code | Condition |
|------|-----------|
| GOV-001 | Modality Imbalance — >50% features from one block |
| GOV-002 | Temporal Coverage Gap — Finance ≠ Clusters date ranges |
| GOV-003 | Geopolitical Centrality Skew — one actor >2× mean centrality |
| GOV-004 | Low Concept Coverage — <40% SAE concepts active |
| GOV-005 | Synthetic Data Active — any block using fallback/mock data |

**Interpretability scorecard** (9 metrics):

| Metric | Pass Threshold |
|--------|---------------|
| Feature Traceability | ≥90% of 80 features traced |
| Kernel Stability | mean cosine ≥0.75 |
| Concept Activation Rate | ≥30% SAE concepts active |
| Data Source Diversity | ≥2 live sources |
| Legacy Retrieval@1 | ≥0.15 |
| Shared-Latent Retrieval@1 | ≥0.20 |
| Shared-Latent Probe Cosine | ≥0.10 |
| Faithfulness Confidence | ≥0.50 |
| Governance Flags | 0 (any flag = WARN) |

---

## Optional Subsystems

### Shared-Latent Shadow Alignment

Feature-flagged (`FEATURE_FLAGS["shared_latent_shadow"]`). Computes alignment
metrics comparing legacy UKT path with shared-latent encoding. Always emits
legacy metrics for governance side-by-side reporting.

### Interpretability Contract

```python
contract = build_alpha_scope_contract_reports({
    "UniversalKnowledgeTensor": ukt,
    "SemanticCanvas": ukt.canvas,
})
enforce_alpha_scope_contract_coverage(contract)
```

Verifies that core components implement the Alpha-scope interpretability
interface: `export_latent_units()`, `export_feature_attributions()`,
`explain_prediction()`.

### Kernel Memory Persistence

Tracks kernel evolution across multiple pipeline runs. Stores run snapshots
and computes evolution summaries for concept drift detection.

### Latent Space Versioning

Phase 3 governance audit trail. Versions the latent space configuration
(feature dimensions, region layout, kernel count, SAE state) per run.
Builds concept audit records for regulatory compliance.

### Drift Detection

H-002 Temporal Coherence spec. Monitors reality regression cosine similarity,
kernel importance distribution, and stability metrics across runs. Raises
alerts when drift exceeds configurable thresholds.

---

## PipelineResult Structure

```python
PipelineResult(
    # Core data
    snapshots: list[dict]              # Per-block UKT snapshots
    final_matrix: np.ndarray           # (n_blocks, 80) projected features
    data_sources: dict                 # Provenance tracking

    # Block results
    finance_result, cluster_result, graph_result,
    spatial_result, sim_result

    # Interpretation
    sae_result: dict                   # SAE concepts
    concept_kernel_map: dict           # Concept → kernel alignment
    semantic_canvas: SemanticCanvas    # Canvas state
    canvas_narrative: str              # LLM-generated synthesis
    reality_narrative: str             # Final assessment

    # Cross-block (optional)
    uvt_result, use_result

    # Governance
    stability: dict                    # Bootstrap robustness
    governance_flags: list             # Auto-detected issues
    interpretability_scorecard: dict   # 9-dim compliance
    alignment_metrics: dict            # Legacy vs. shared-latent
    faithfulness_report: dict          # Intervention checks

    # Audit trail
    run_id: str                        # UUID
    run_timestamp: str                 # ISO datetime
    interpretability_contract: dict
    drift_result: dict
    kernel_evolution: dict
    latent_version: dict
)
```

---

## Error Handling Strategy

1. **Validation on entry**: `validate_block_result()` checks each block's output
2. **Silent failures with warnings**: Narrator wrapped in try/except; failures
   logged to `_warnings`
3. **Graceful skipping**: Optional blocks (Spatial, cross-block) skip silently
   if input is `None` or feature flag disabled
4. **Faithfulness-driven downgrade**: If mechanistic checks fail, narrative
   automatically replaced with conservative fallback
5. **No external dependencies**: All data generated in-app; no API keys required

---

## Key Files

| File | Role |
|------|------|
| `hyperspace/core/pipeline.py` | Main orchestrator (783 lines) |
| `hyperspace/core/types.py` | `PipelineResult` TypedDict definition |
| `hyperspace/core/faithfulness.py` | Intervention-based faithfulness checks |
| `hyperspace/core/drift_monitor.py` | Temporal coherence monitoring |
| `hyperspace/core/latent_versioning.py` | Audit trail versioning |
| `hyperspace/core/interpretability_registry.py` | Contract enforcement |
| `hyperspace/config.py` | Constants, thresholds, feature flags |

---

*Document version: Alpha 1.0 — March 2026*
