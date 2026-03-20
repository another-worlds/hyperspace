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
   h. Reset + replay Semantic Canvas for all blocks
   i. Generate kernel narratives
   j. Return snapshot
4. Append snapshot to results
```

### Post-Block Analysis

After all 5 blocks are added, the pipeline runs:
1. Governance flag computation (GOV-001 through GOV-005)
2. Governance scorecard generation
3. Stability estimation (bootstrap confidence intervals)
4. Cross-block kernel narrative synthesis
5. Export-ready report assembly

### Three-Layer Semantic Translation

```
Machine Latent Space (80-dim projected features)
         │
    Semantic Canvas    →  12 named dimensions, data-driven coordinates
         │
    Sparse Autoencoders →  Concept discovery via L1-regularized autoencoders
         │
    Narrator            →  Tiny-LLM (13M params) + template fallback
         │
    Human-Readable Narrative
```

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
| 0 | Mission Control | Executive summary, governance scorecard, data provenance, diagnostics, export |
| 1 | Finance-Neural Block | TFT forecasting, correlation heatmaps, confidence summary |
| 2 | Informational Cluster Mapping | BERTopic multilingual clustering, topic distributions |
| 3 | Politics-Military Block | Graph engine, centrality analysis, geographic visualization |
| 4 | Agentic Simulation | Multi-agent resource/alliance simulation |
| 5 | Semantic Interpreter | Concept bottleneck, kernel visualization, semantic canvas |
| 6 | Hyperspace Pipeline | End-to-end orchestration, timing, governance context |
| 7 | Counterfactual | Block removal + diff analysis for contestability |

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| CPU-only inference | Accessibility — no GPU required for deployment |
| Synthetic data fallback | All components work without network access; graceful degradation |
| Streamlit + Plotly | Rapid prototyping with interactive dark-themed visualization |
| SVD over learned decomposition | Interpretable, deterministic, no training required for kernel discovery |
| Per-region normalization before projection | Preserves cross-region energy ratios; prevents scale-dominant modalities |
| Centralized thresholds in `config.py` | Auditability — all governance magic numbers in one inspectable location |
| Parallel data fetching (4 workers) | 4-6x speedup for initial data load |
| Parallel model training (TFT + BERTopic) | Concurrent execution reduces wall time |
| SAE caching by input hash | 5-10s savings per iteration; deterministic/stochastic toggle |
| Structured event logging | Governance-compliant audit trails |
| Counterfactual scenario memory | Store last 5 runs for side-by-side comparison |

---

## Known Issues & Technical Debt

### Architectural Gaps (from vision assessment)

- **Hand-partitioned latent space.** The 80-dim space uses fixed modality-owned slices rather than learned joint alignment. This enforces modality boundaries instead of discovering shared structure.
- **No cross-modal contrastive objective.** SVD is applied after concatenation; there is no paired objective pulling semantically related cross-modal representations together.
- **No temporal world-model memory.** UKT snapshotting is per pipeline run; it lacks sequence modeling over transitions and interventions.
- **Weak knowledge reuse.** Kernels summarize variance per run but are not versioned, distilled, or transferred as reusable modules.
- **Uneven interpretability coverage upstream.** UKT and semantic layers are interpretable, but several upstream model blocks lack standardized concept interfaces.

### UI Gaps (Phase 2 remaining)

- Cross-tab navigation links and breadcrumb trail not yet implemented
- Advanced diagnostics section (~200 lines) needs reorganization into sub-tabs
- Per-tab "Retrain" button for cache invalidation not yet exposed
- Policy language mode (jargon-to-plain-English toggle) not yet built

### Resolved (Phase 0-2)

- WCAG contrast compliance (fixed)
- Governance-first Mission Control hierarchy (implemented)
- Computation caching for SAE and expensive ops (implemented)
- Centralized governance thresholds (completed)
- Pipeline progress visualization with per-block timing (implemented)
- Parallel data fetching, model training, kernel labeling, stability estimation (implemented)

---

## Roadmap

### Near-Term (UI Completion)

- Cross-tab navigation with contextual links between related outputs
- Advanced diagnostics reorganization into collapsible sub-tabs
- Per-tab cache control ("Retrain" buttons)
- Policy language mode for non-technical stakeholders

### Medium-Term (Architectural Evolution)

- `InterpretableModule` protocol for all upstream blocks
- Standardized concept bottlenecks with confidence/calibration metadata
- Modality encoders + shared projector heads with contrastive alignment
- Dual-state UKT backend (legacy 80-d matrix + shared latent tensor)

### Long-Term (Vision Realization)

- Temporal world-model memory over UKT state transitions
- Intervention-based interpretability (activation patching, causal attribution)
- Cross-run kernel persistence, drift tracking, and knowledge transfer
- Tensorized factorization (CP/Tucker) for modality-aware decomposition
