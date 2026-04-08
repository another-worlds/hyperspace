# Architecture — Hyperspace

## System Overview

Hyperspace is a Python 3.11+ application built on Streamlit, using Plotly for visualization, PyTorch-based models for analysis, and a custom UKT framework for cross-domain kernel discovery.

```
Data Sources (32+ keyless APIs)
       │
       ├── Finance (10 APIs)   → TFT forecasting        ┐
       ├── News    (11 srcs)   → BERTopic clustering    │
       ├── Political (10 src)  → Graph engine (NetworkX)├→ Monolithic UKT
       ├── Spatial (12 srcs)   → Spatial kernels        │   (80-dim feature space)
       └── Agent simulation    → Resource/alliance sim  ┘   [Blocks couple freely]
                                        │
                            Universal Knowledge Tensor
                              (block-agnostic discovery)
                                        │
                        ┌───────────────┼───────────────┐
                        │               │               │
                   Semantic         Governance      Counterfactual
                   Canvas +         Flags +         Block Ablation
                   Narratives       Scorecard       + Diff Analysis
                        │               │               │
                        └───────────────┼───────────────┘
                                        │
                                   Streamlit UI
                                   (8-tab layout)
```

---

## Component Map

### Core Packages

| Package | Responsibility |
|---------|---------------|
| `hyperspace/core/` | Pipeline orchestration, governance flag computation, structured logging, caching |
| `hyperspace/data/` | Data fetching (finance, news, political, spatial) with parallel execution |
| `hyperspace/models/` | ML models (TFT, BERTopic, Graph, Spatial, Agents), knowledge matrix, SAE wrappers |
| `hyperspace/pages/` | Streamlit tab implementations (8 tabs) |
| `hyperspace/viz/` | Visualization utilities, pipeline progress tracking |
| `hyperspace/config.py` | Centralized configuration: UKT layout, thresholds, CSS, node graph |
| `hyperspace/state.py` | Streamlit session state management |

### Framework Packages

| Package | Responsibility |
|---------|---------------|
| `ukt/` | Universal Knowledge Tensor: feature registry, projection, SVD kernelization, stability estimation, parallel execution |
| `semantic_interpreter/` | Semantic Canvas, Sparse Autoencoders (SAE), Narrator (tiny-LLM + template fallback), concept extraction |

### Entry Point

| File | Role |
|------|------|
| `app.py` | Streamlit entry point, tab routing, global sidebar controls |

---

## Data Flow

### Pipeline Execution (per block)

```
1. Validate block result (required keys)
2. Extract native feature vector from block model
3. UKT.add_block():
   a. Pad/truncate to 80 dims (monolithic space)
   b. Store raw features
   c. Global min-max normalization (all 80 dims at once)
   d. Observe full 80-dim normalized vector in SharedProjection → rebuild P
      - Projection couples blocks via rank-1 outer products across full space
      - Coupling strength: data-driven (energy ratio)
      - Coupling direction: normalized outer product v_tgt ⊗ v_src
      - Fallback: random orthogonal basis when coupling weak
   e. Re-project ALL blocks through updated P
   f. Stack into matrix → SVD decomposition
   g. Label all kernels (block scores via block_feature_ranges + source_block provenance)
   h. Generate kernel narratives
   i. Return snapshot
4. Append snapshot to results
5. After all blocks: run global SAE on final matrix → `build_emergent_canvas(sae_result)` → canvas dims = active concepts
```

#### Critical Invariant Implementations

The pipeline execution above implements several VISION critical invariants directly in the mathematical substrate:

**Normalize-before-project (VISION invariant 1):** `_normalize_features()` in `knowledge_matrix.py` applies symmetric min-max scaling across all 80 dimensions to [-1, 1] range. This is global normalization — no per-region boundaries are consulted. The entire 80-dim vector is scaled by its global min/max, ensuring coupling reflects genuine cross-block structure rather than per-block energy differences.

**Open topology (VISION invariant 2):** `SharedProjection` in `ukt/projection.py` defaults to `topology=None`, which generates all directed block pairs `(src, tgt)` where `src != tgt`. No hardcoded adjacency restricts which blocks can couple. Coupling strength is computed purely from energy ratios: `strength = min(||f_src||, ||f_tgt||) / max(||f_src||, ||f_tgt||)`. If two blocks have no structural similarity, coupling naturally approaches zero — no topology gate required.

**Energy-based graceful degradation (VISION philosophy 5):** When a modality is absent or has near-zero energy, the energy-ratio coupling strength → 0 automatically. The coupling direction blends toward a pre-computed random orthogonal fallback: `block = α × data_block + (1-α) × fallback`, where `α = strength`. This means missing modalities decouple through pure mathematics — no special-case `if missing:` code paths exist or are needed.

**Full replay after each block (VISION invariant 4):** After each block's projection matrix rebuild, ALL previously observed blocks are re-projected through the updated matrix (step 3e). No block retains stale coordinates from an earlier projection epoch. The Semantic Canvas is **not** built per-block — it is built once after the global SAE runs on the finished matrix, so canvas axes are fully emergent SAE concepts rather than pre-projection approximations.

**Regions are provenance metadata only (VISION invariant 6):** `FeatureRegionRegistry` in `ukt/registry.py` maps block names to feature index ranges purely for naming, traceability, and per-block canvas slicing — done after projection, never during. The feature space is monolithic: projection couples blocks across the full 80-dim space without consulting region boundaries.

### Post-Block Analysis

After all 5 blocks are added, the pipeline runs:
1. SAE concept discovery on final matrix → `build_emergent_canvas(sae_result, block_names)` — canvas dimensions = active SAE concepts, one entry per block from concept_activations rows
2. Cross-block networks: UVT coupling matrix + USE semantic encoding
3. Semantic narratives: canvas narrative, reality regression narrative, per-concept narratives
4. Stability estimation (Monte Carlo: 8 noise runs, σ=0.01, bootstrap confidence intervals)
5. **Faithfulness verification** (6 intervention-based checks — VISION invariant 5)
6. Governance flag computation (GOV-001 through GOV-006)
7. Governance scorecard generation (9 metrics against centralized thresholds)
8. Kernel memory recording + latent versioning + drift monitoring
9. Export-ready report assembly

#### Faithful Narratives Verification (VISION invariant 5)

`faithfulness.py` implements 6 intervention checks that verify narrative claims are grounded in actual computation:

| Check | Method | Validates |
|-------|--------|-----------|
| Kernel attribution | Zero out singular value, measure reconstruction error delta | Reported importance matches actual impact |
| Feature region masking | Mask each of 5 regions, recompute reality regression | Highest-energy region produces largest delta |
| Importance-delta monotonicity | Compare top-3 importance ranking vs top-3 delta ranking | Reported ordering matches measured ordering |
| Canvas narrative grounding | Compare dominant canvas dimensions vs actual energy | Narrative references reflect real data |
| SAE concept consistency | Check activation rate vs importance variance | Concepts are grounded in kernel differentiation |
| Concept ablation | Verify dominant concept region matches dominant kernel | Concepts are mechanistically aligned |

Overall confidence = passed_checks / total_checks. If confidence < 0.5, canvas and reality narratives are downgraded to a boilerplate disclaimer — the system admits when its explanations may not be faithful.

### Three-Layer Semantic Translation

```
Machine Latent Space (80-dim projected features)
         │
    Sparse Autoencoders →  Concept discovery via L1-regularized autoencoders
         │
    Semantic Canvas    →  Emergent dimensions (one per active SAE concept),
         │                data-driven coordinates
    Narrator            →  Tiny-LLM (13M params) + template fallback
         │
    Human-Readable Narrative
```

Canvas dimensions are NOT predefined. After the global SAE runs on the full
projected matrix, each active concept becomes a canvas axis. The axis label
is derived from the concept's dominant region and top feature loadings. The
number of axes varies per run — typically 10–20 depending on how many
concepts the SAE activates. Coupling between blocks is measured by
SharedProjection's energy-based coupling weights, not by fixed constants.

### Governance Narrative Contract (VISION invariants 5, 9, 10)

A narrator implementation is "governance-grade" if and only if its output contains all five of the following fields. Any output missing a field must not be presented to governance audiences.

| Field | Source | Content |
|-------|--------|---------|
| **anchor** | Pipeline run metadata | `{event_id, timestamp, decision_or_forecast_reference}` — which decision, forecast, or event this narrative explains |
| **direction** | Reality regression / counterfactual engine | `{source: "reality_regression" \| "counterfactual", vector, sign}` — direction of influence, not just co-variance |
| **temporal** | KernelMemory / DriftMonitor | `{activation_time, delta_from_prior_run}` — when the pattern activated and whether it changed |
| **cross_item_synthesis** | LLM narrator (cannot be templated) | `List[{concept_a, concept_b, relation}]` — how active concepts relate to each other and to kernels |
| **feature_evidence** | FeatureRegionRegistry | `List[{feature_name, block, loading}]` — world-grounded names resolved from registry, not bare indices |

When the LLM narrator is unavailable, the system must emit the provenance fields (anchor, temporal, feature_evidence) as a raw provenance table and explicitly withhold the interpretive fields (direction, cross_item_synthesis) with a governance error. It must not substitute template-generated interpretive text. See VISION.md invariant 9.

### SAE Concept-Labeling Data Contract

SAE concept labels produced by `sae.py:label_concepts()` must preserve the **full signed cross-block signature** as structured data: `{block_name: signed_contribution}` for every registered block region. The `dominant_region` field, if present, is a non-authoritative provenance hint — it records which block region has the largest sum of absolute loadings — and must **never** be used as the concept's display name or as the subject of an interpretive sentence.

Downstream code that treats `dominant_region` as a semantic label is in violation of this contract (VISION invariant 10). A concept whose top-3 feature loadings span multiple blocks with mixed sign is a **cross-block pattern**, not a member of whichever block won the argmax.

**Current status**: `sae.py:252-275` violates this contract. It emits `"primarily encodes {dominant_region} information"` — a single-block interpretive claim manufactured by argmax. This is logged in CLAUDE.md as a known bug.

---

## UKT Feature Layout

| Slots | Region Name | Block | Source |
|-------|-------------|-------|--------|
| 0–15 | temporal-pattern | TFT | Encoder attention weights, regime shifts, volatility |
| 16–23 | semantic-embedding | BERTopic | Topic distributions, embedding statistics |
| 24–26 | (TFT extended) | TFT | Decoder importance |
| 27–31 | (macro) | TFT | GDP growth, inflation, FX, CPI, market-cap/GDP |
| 32–47 | structural-centrality | Graph Engine | Degree, betweenness, eigenvector, PageRank centrality |
| 48–63 | dynamic-agent | Agent Sim | Resource shares, alliance eigenvalues, graph metrics |
| 64–79 | geospatial-kernel | Spatial | SVD kernel importances, per-country loadings |

---

## Data Sources (all keyless)

### Finance — 10 APIs
yfinance (OHLCV), Stooq (fallback), ECB eurofxref (FX rates), IMF DataMapper (GDP), US Treasury OData (yield curve), CoinGecko (crypto market cap), Open.er-api (USD FX), World Bank (CPI, market-cap/GDP, FDI, M2), BIS WS_CBPOL (policy rates), BLS CPI v1 (US consumer prices)

### News — 11 Sources
RSS feeds (BBC, Reuters, Al Jazeera, Guardian, France24), HN Algolia, UN News RSS (3 feeds), GDELT DOC 2.0, Wikipedia events API, Reddit /r/worldnews, GDELT GKG 2.0

### Political — 10 Sources
Harvard Dataverse (alliance data), World Bank WGI (governance indicators), and derived graph metrics

### Spatial — 12 Sources
Open-Elevation, Open-Meteo, USGS earthquake feeds, NOAA climate data, and derived raster/kernel metrics

---

## UI Structure — 8 Tabs

| Tab | Name | Core Feature |
|-----|------|-------------|
| 0 | Mission Control | System overview, metrics, pipeline progress bar and stop/pause/continue controls, governance scorecard, status |
| 1 | Finance-Neural Block | TFT forecasting metrics: trading simulation, trading metrics, 1d kernel narrative explanation |
| 2 | Informational Cluster Mapping | BERTopic multilingual clustering, news anchor clustering and emerging alliance narrative explanation |
| 3 | Politics-Military Block | Graph engine, centrality, kernelization, alliance clusterization and narrative explanation |
| 4 | Agentic Simulation | Multi-agent resource/alliance sim, kernelization of the resource transfer processes, narrative explanation of the kernels |
| 5 | Semantic Interpreter | Kernel narrative explanation, semantic canvas 3d map with relationships |
| 6 | Hyperspace Pipeline | End-to-end orchestration |
| 7 | Counterfactual | Block removal + diff analysis (contestability) |

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| CPU-only inference | Accessibility — no GPU required for deployment |
| Synthetic data fallback | All components work without network access; graceful degradation |
| Streamlit + Plotly | Rapid prototyping with interactive dark-themed visualization |
| SVD over learned decomposition | Interpretable, deterministic, no training required for kernel discovery |
| Global min-max normalization before projection | All 80 dims normalized together to [-1, 1]; coupling reflects genuine cross-block interaction, not per-block energy differences (VISION invariant 1) |
| Centralized thresholds in `config.py` | Auditability — all governance magic numbers in one inspectable location |
| Parallel data fetching (4 workers) | 4-6x speedup for initial data load |
| Parallel model training (TFT + BERTopic) | Concurrent execution reduces wall time |
| SAE caching by input hash | 5-10s savings per iteration; deterministic/stochastic toggle |
| Structured event logging | Governance-compliant audit trails |
| Counterfactual scenario memory | Store last 5 runs for side-by-side comparison |
| Hovertemplates on every chart | Human-readable hover labels for non-technical stakeholders |
| Per-tab retrain buttons | Users can invalidate cached results without full pipeline reset |

---

## Architecture Specifications (Pending Implementation)

### SPEC-3: InterpretableModule Contract for All Blocks

**VISION requirement**: "Every model block implements a standardized interpretability contract exposing latent units, feature attributions, alignment reports, and structured explanations."

**Goal**: Bring all 5 pipeline blocks into compliance with the `InterpretableModule` protocol defined in `hyperspace/core/types.py`, so every model block can be interrogated through a uniform interface.

**Current state**: `InterpretableModule` protocol exists with 4 methods (`export_latent_units`, `export_feature_attributions`, `export_alignment_report`, `explain_prediction`). Only `UniversalKnowledgeTensor` and `SemanticCanvas` implement it. All other blocks are marked `not_applicable` in `interpretability_registry.py`.

**Implementation approach — Adapter pattern**:

Wrap existing block outputs in thin adapter classes that expose the standard interface. No changes to the underlying model logic — adapters translate existing outputs into the contract format.

| Block | `export_latent_units` | `export_feature_attributions` | `export_alignment_report` | `explain_prediction` |
|-------|----------------------|-------------------------------|---------------------------|---------------------|
| **TFT** | Attention weights per lag (16 temporal units) | Encoder/decoder importance vectors | Cross-ticker correlation as alignment proxy | Quantile forecasts + confidence interval + regime assessment |
| **BERTopic** | Topic embeddings (N topics × embedding dim) | Topic-to-document probability weights | Topic coherence scores as alignment proxy | Dominant topics + keywords + outlier ratio |
| **Graph** | Community assignments + adjacency structure | Per-node centrality scores (degree, betweenness, eigenvector, PageRank) | Inter-community coupling strengths | Dominant actor + alliance structure + graph density |
| **Agent Sim** | Agent state vectors (resources, alliance strengths) | Resource flow sensitivity per agent (∂resources/∂flow_rate) | Alliance matrix eigenvalue spectrum | Equilibrium type (unipolar/multipolar) + Gini coefficient |
| **Spatial** | Spatial SVD kernel components | Per-country loadings on each kernel | Cross-region correlation as alignment proxy | Dominant spatial pattern + geographic clustering |

**Pipeline integration**:
- After each block completes, call all 4 `export_*` methods on its adapter
- Store results in the block snapshot under `interpretability_export` key
- New scorecard metric: `block_contract_coverage` = (blocks with full export) / (total blocks)

**Registry update**:
- All 5 block policies in `interpretability_registry.py` change from `not_applicable` → `contract`
- Adapter registration in a new `hyperspace/models/interpretability_adapters.py`

**Files affected**: new `hyperspace/models/interpretability_adapters.py`, `hyperspace/core/interpretability_registry.py`, `hyperspace/core/pipeline.py`, `hyperspace/core/types.py` (snapshot extension)

**Constraints**:
- Adapters must not modify underlying model behavior
- All exports must be JSON-serializable for governance report inclusion
- Adapter methods must complete in < 100ms (no retraining)
- Must not break existing pipeline flow — adapter failures are caught and logged, not fatal

---

### SPEC-4: Cross-Modal Contrastive Alignment

**VISION requirement**: "Modality encoders project diverse data types into a shared latent space through learned alignment, not hand-partitioned concatenation" + "Cross-modal contrastive objectives pull semantically related representations together across modalities, discovering genuine shared structure."

**Goal**: Add a learned cross-modal alignment path that progressively complements the existing SVD path, enabling genuine shared structure discovery beyond linear projection.

**Current state**: Hand-partitioned 80-dim space with fixed modality slices. `SharedProjection` in `ukt/projection.py` couples blocks via energy-gated rank-1 outer products (data-driven but linear). No contrastive learning objective exists.

**Implementation approach — Dual-path architecture**:

```
Block Features (5 blocks × native dims)
         │
    ┌────┴────┐
    │         │
Legacy Path   Learned Path (SPEC-4)
    │         │
80-dim SVD    Per-block MLP encoders → shared 32-dim latent
    │         │
    │     InfoNCE contrastive loss (cross-block pairs)
    │         │
    └────┬────┘
         │
   Blended UKT = (1-α) × SVD_kernels + α × contrastive_kernels
```

**Per-block encoders**: Small MLPs (2 layers, 64 hidden, ReLU, dropout=0.1) mapping each block's native feature vector (variable-length) into a shared 32-dim latent space. All CPU-trainable, ~5K parameters per encoder.

**Contrastive objective**: InfoNCE loss over block pairs within a single pipeline run:
- Positive pairs: `(encoder_i(block_i), encoder_j(block_j))` for all `i ≠ j` in the same run
- Negatives: feature vectors with shuffled block assignments + Gaussian noise augmentation
- Temperature parameter `τ` (default 0.07) in `config.py`

**Alignment score**: Mean pairwise cosine similarity between encoded block pairs — reported as a governance metric in the scorecard. Higher = stronger cross-modal structure discovered.

**Dual-state UKT backend**: `knowledge_matrix.py` runs both SVD (legacy) and learned (contrastive) decomposition. A `CONTRASTIVE_WEIGHT` config parameter (0.0–1.0, default 0.0) blends the two kernel sets. At 0.0 the system is pure SVD (fully backward compatible). Gradual increase as contrastive path proves stable across runs.

**Feature flag**: `ENABLE_CONTRASTIVE_ALIGNMENT` in `config.py`, default `False`.

**Files affected**: new `hyperspace/models/contrastive_encoder.py`, `hyperspace/models/knowledge_matrix.py`, `hyperspace/config.py`, `hyperspace/core/pipeline.py`

**Constraints**:
- CPU-only training (< 2s per pipeline run for all 5 encoders)
- Legacy SVD path must remain fully functional and primary at `CONTRASTIVE_WEIGHT=0.0`
- Encoder weights cached in session state and optionally persisted to disk
- Must not affect governance flag logic — contrastive path is additive, not replacing
- Contrastive kernels must be fully inspectable (encoder weights, latent vectors, loss curve)

---

### SPEC-5: Temporal World-Model Memory

**VISION requirement**: "Temporal world-model memory encodes transitions and interventions over time, not just static co-variance snapshots."

**Goal**: Learn a state-transition function over UKT history that enables temporal prediction and intervention-aware what-if analysis.

**Current state**: `KernelMemory` in `temporal_memory.py` stores per-run snapshots and computes kernel evolution (importance trends, cosine similarities). `DriftMonitor` in `drift_monitor.py` tracks regression/importance drift across runs. `LatentVersionTrail` fingerprints config. All are passive recording systems — no transition modeling.

**Implementation approach — Lightweight transition encoder**:

```
Run History: [UKT_1, UKT_2, ..., UKT_t]
Intervention History: [I_1, I_2, ..., I_t]
         │
    GRU Encoder (hidden=32)
    Input: concat(reality_regression_t, intervention_vector_t)
    Output: predicted_reality_regression_{t+1}
         │
    Prediction Confidence: cosine(predicted, actual) over recent window
```

**Transition encoder**: Lightweight GRU (input=80+intervention_dim, hidden=32, output=80) that learns `f(UKT_t, I_t) → UKT_{t+1}` from cross-run `KernelMemory` history. Trained online after each pipeline run.

**Intervention encoding**: Each pipeline run produces a structured intervention vector encoding:
- Data source availability (5 binary flags: which blocks used live vs synthetic data)
- Parameter settings (normalized slider values from sidebar)
- Governance flag counts per category
- Total intervention vector: ~20 dims, concatenated with 80-dim reality regression

**Temporal prediction**: Given current UKT state + hypothetical intervention vector, predict next-run reality regression. Report prediction confidence via cosine similarity between predicted and actual values over the most recent 5 runs.

**Memory window**: Rolling window of last 50 runs (matches `DriftMonitor` capacity). GRU trains on the full window each time (50 × 100-dim sequences — trivial for CPU).

**Counterfactual tab integration**: New "Temporal What-If" mode alongside existing block ablation:
- User selects a hypothetical intervention (e.g., "what if Finance used synthetic data next run?")
- System predicts the resulting reality regression change
- Displayed as a temporal diff chart analogous to existing block-ablation diff

**Training**: Online learning after each pipeline run. Loss: MSE on predicted vs actual reality regression. Adam optimizer, lr=0.001. < 1s per run on CPU.

**Feature flag**: `ENABLE_TEMPORAL_MEMORY` in `config.py`, default `False`. Requires minimum 3 runs in `KernelMemory` before predictions are attempted.

**Files affected**: new `hyperspace/models/temporal_encoder.py`, `hyperspace/core/temporal_memory.py` (extend), `hyperspace/config.py`, `hyperspace/core/pipeline.py`, `hyperspace/pages/counterfactual_tab.py`

**Constraints**:
- CPU-only (GRU is tiny: ~10K parameters)
- Must not delay pipeline execution (training runs asynchronously after results are displayed)
- Predictions carry a confidence score and a disclaimer when confidence < 0.7
- Model weights persisted to JSON alongside `KernelMemory` snapshots
- No prediction is surfaced in governance reports until confidence > 0.8 over 5+ consecutive runs

---

### SPEC-6: Mechanistic Probes

**VISION requirement**: "Mechanistic probes provide causal/feature pathway explanations through attention-pattern analysis, linear probing, and intervention-based attribution."

**Goal**: Expose interactive, user-driven probing tools that trace causal pathways through the UKT — going beyond the existing pass/fail faithfulness checks into exploratory mechanistic interpretability.

**Current state**: `faithfulness.py` has 6 intervention checks (kernel ablation, feature region masking, importance-delta monotonicity, canvas grounding, SAE concept consistency, concept ablation). These are automated validation checks, not interactive tools. Users cannot select a specific feature or kernel to probe.

**Implementation approach — Three probe types**:

**1. Activation Patching (interactive kernel ablation)**

Extend `check_kernel_attribution_faithfulness` into an interactive tool:
- User selects a kernel (K0–Kn) from a dropdown
- System zeros that kernel's singular value, recomputes reality regression, canvas coordinates, and narrative
- Displays: original vs patched reality regression (bar chart), affected canvas dimensions (radar overlay), narrative diff
- Already partially implemented — extend to accept user-selected kernel index and return structured results

**2. Linear Probing (governance flag prediction)**

Train lightweight linear classifiers on kernel activations to predict governance outcomes:
- For each governance flag (GOV-001 through GOV-006), fit `sklearn.linear_model.LogisticRegression` on `kernel_activation` matrix rows → flag presence
- Reports per-kernel predictive weight: "Kernel K3 is 87% predictive of GOV-001 (modality imbalance)"
- Requires cross-run data from `KernelMemory` (minimum 10 runs with varied flag outcomes)
- Falls back to per-feature correlation analysis when insufficient history exists

**3. Feature Pathway Tracing**

Given a feature index, trace its full contribution path through the system:
```
Feature idx → feature_name (from registry)
           → raw_value (from snapshot)
           → normalized_value (after global min-max)
           → projected_value (after SharedProjection × P)
           → kernel_loadings (Vt[:, feature_idx] for all kernels)
           → canvas_dimensions (which canvas dims this feature influences)
           → narrative_mentions (grep kernel narratives for feature name)
```
Returns a structured `FeaturePathway` dict. UI renders as an expandable trace with values at each step.

**UI integration**: New "Mechanistic Probes" expander in Semantic Interpreter tab (Tab 5), below the existing "Advanced Diagnostics" section:
- Probe type selector (activation patching / linear probing / pathway tracing)
- Target selector (kernel index / feature index)
- Results panel with interactive Plotly charts

**Governance integration**: Probe results stored in `st.session_state["probe_results"]` and included in exported governance reports when present.

**Files affected**: new `hyperspace/core/mechanistic_probes.py`, `hyperspace/core/faithfulness.py` (refactor shared logic), `hyperspace/pages/interpreter_tab.py`, `hyperspace/config.py`

**Constraints**:
- All probes must complete in < 2s on CPU
- Linear probing requires `sklearn` (already a transitive dependency via `sentence-transformers`)
- Probe results are informational, not authoritative — they carry "exploratory" labels in governance reports
- Feature pathway tracing works on the latest snapshot only (no cross-run tracing)
- Activation patching operates on a copy of the decomposition — never modifies cached state

---

### SPEC-7: Knowledge Persistence & Transfer

**VISION requirement**: "Knowledge persistence allows kernels to be versioned, distilled, and transferred as reusable analytical modules across runs."

**Goal**: Give kernels stable identities across runs, distill persistent kernels into reusable templates, and optionally warm-start new runs with prior knowledge.

**Current state**: `KernelMemory` in `temporal_memory.py` stores per-run snapshots (importance, reality regression, kernel activation per block). `LatentVersionTrail` in `latent_versioning.py` fingerprints config and tracks vocabulary drift. Neither versions individual kernels, distills them, or enables transfer.

**Implementation approach — Three-layer persistence**:

**1. Kernel Versioning (lineage tracking)**

Each kernel gets a stable identity across runs via cosine matching on Vt rows (feature loadings):
- After each run, compare each kernel's Vt row against the previous run's kernels
- If `cosine(K_new.Vt, K_old.Vt) > KERNEL_LINEAGE_COSINE_THRESHOLD` (default 0.85, in `config.py`), assign the same `kernel_lineage_id`
- Otherwise, mint a new lineage ID (UUID)
- Track: `kernel_lineage_id`, `first_seen_run`, `last_seen_run`, `run_count`, `importance_trend[]`, `narrative_history[]`

**2. Kernel Distillation (template extraction)**

After a kernel persists for `KERNEL_DISTILLATION_MIN_RUNS` consecutive runs (default 5, in `config.py`), distill it into a `KernelTemplate`:
```python
@dataclass
class KernelTemplate:
    lineage_id: str
    frozen_vt: np.ndarray           # (80,) feature loadings
    importance_range: tuple[float, float]  # (min, max) across observed runs
    stability_score: float           # mean cosine across consecutive runs
    dominant_region: str             # "temporal-pattern", "semantic-embedding", etc.
    canonical_narrative: str         # most recent faithful narrative
    metadata: dict                   # run history, feature evidence
```
- Templates are JSON-serializable (numpy arrays → nested lists)
- Stored in `kernel_library.json` alongside `KernelMemory` data

**3. Kernel Transfer (warm-start initialization)**

When starting a new pipeline run, optionally seed the `SharedProjection` fallback basis with prior `KernelTemplate` loadings:
- For each active template, its `frozen_vt` is used as a fallback coupling direction instead of a random orthogonal basis
- The SVD still runs fresh on actual data — templates only influence the fallback component of the projection
- Transfer is gated by a `KERNEL_TRANSFER_ENABLED` flag (default `False`) and a minimum library size (3+ templates)
- Transfer strength is weighted by template stability score

**Kernel Library UI**:
- Mission Control: new "Knowledge Base" expander showing active templates, their lineage age, and stability scores
- Pipeline tab: kernel lineage annotations on the kernel evolution chart (color-code by lineage ID)
- Export: `kernel_library.json` included in governance report downloads

**Files affected**: new `hyperspace/core/kernel_library.py`, `hyperspace/core/temporal_memory.py` (extend with lineage tracking), `hyperspace/models/knowledge_matrix.py` (template-aware fallback), `ukt/projection.py` (accept template basis), `hyperspace/config.py`, `hyperspace/pages/mission_control_tab.py`, `hyperspace/pages/pipeline_tab.py`

**Constraints**:
- Lineage matching is O(n_kernels²) per run — negligible for typical 3–8 kernels
- Templates are read-only after distillation (immutable artifacts)
- Transfer must not override data-driven coupling — only affects the `(1-α) × fallback` term
- Library pruning: templates not seen for 20+ runs are archived (not deleted)
- All template operations are logged via structured logging for governance audit

---

## Known Issues & Technical Debt

### Architectural Gaps

| Gap | VISION Requirement | Current State | Resolution |
|-----|-------------------|---------------|------------|
| Hand-partitioned latent space | Learned joint alignment | Fixed 80-dim slices per modality | **SPEC-4** (contrastive encoders + dual-path) |
| No cross-modal contrastive objective | Contrastive objectives pull related representations together | SVD after concatenation only | **SPEC-4** (InfoNCE loss) |
| No temporal world-model memory | Encode transitions and interventions over time | Passive recording only (KernelMemory, DriftMonitor) | **SPEC-5** (GRU transition encoder) |
| Weak knowledge reuse | Kernels versioned, distilled, transferred as reusable modules | Per-run snapshots, no versioning or transfer | **SPEC-7** (kernel library) |
| Uneven interpretability upstream | Every block implements standardized contract | Only UKT + Canvas comply; 5 blocks marked N/A | **SPEC-3** (adapter pattern) |
| No mechanistic probes | Causal/feature pathway explanations | Pass/fail faithfulness checks only | **SPEC-6** (interactive probes) |

### UI Gaps (Phase 3 remaining)

- Cross-tab navigation links and breadcrumb trail not yet implemented
- Advanced diagnostics section (~200 lines) needs reorganization into sub-tabs

### Resolved (Phase 0-3)

- WCAG contrast compliance (fixed, including sidebar subtitle)
- Governance-first Mission Control hierarchy (implemented)
- Computation caching for SAE and expensive ops (implemented, extended in Phase 4 to all charts)
- Centralized governance thresholds, including drift monitoring (completed)
- Pipeline progress visualization with per-block timing (implemented)
- Parallel data fetching, model training, kernel labeling, stability estimation (implemented)
- Hovertemplates on all Plotly charts with human-readable field names (Phase 3)
- Mission Control native Streamlit charts replaced with interactive Plotly (Phase 3)
- Per-tab retrain/rebuild buttons for cache invalidation (Phase 3)
- Agent simulation parameter-hash caching (Phase 3)
- Figure caching across all tabs via `get_or_compute_figure` (Phase 4)
- Sidebar kanban 5th Visualization card (Phase 4)
- ARCHITECTURE normalization contradiction fixed (Phase 4)
- Critical invariant implementations documented (Phase 4)

---

## Roadmap

### Phase 4 — Near-Term (UI Completion + Contract Adoption)

- Cross-tab navigation with contextual links between related outputs
- Advanced diagnostics reorganization into collapsible sub-tabs
- **SPEC-3**: InterpretableModule adapters for all 5 pipeline blocks

### Phase 5 — Medium-Term (Learned Alignment + Probes)

- **SPEC-4**: Cross-modal contrastive encoders + dual-path UKT backend
- **SPEC-6**: Mechanistic probes (activation patching, linear probing, pathway tracing)
- **SPEC-7**: Kernel versioning, distillation, and library

### Phase 6 — Long-Term (World Model + Advanced Decomposition)

- **SPEC-5**: Temporal world-model memory (GRU transition encoder + temporal what-if)
- Tensorized factorization (CP/Tucker) for modality-aware decomposition
- Full migration from legacy 80-dim SVD to learned contrastive alignment (SPEC-4 α → 1.0)
