# Hyperspace — Predictive Polymath System v3.0

A multimodal geopolitical intelligence platform that fuses real-time financial,
news, political, and spatial data into a unified 80-dimensional
**Universal Knowledge Tensor (UKT)**, then applies interpretable ML models across
six analytical blocks.

---

## Architecture

```
Live APIs (32+ sources)
       │
       ├── Finance (10 APIs)  →  TFT forecasting      → UKT slots  0-31
       ├── News    (11 srcs)  →  BERTopic clustering   → UKT slots 16-23
       ├── Political (10 src) →  Graph engine (NetworkX)→ UKT slots 32-47
       ├── Spatial (12 srcs)  →  SVD spatial kernels   → UKT slots 64-79
       └── Agent simulation   →  Resource/alliance sim  → UKT slots 48-63
                                          │
                              Universal Knowledge Tensor (80-dim)
                                          │
                              Semantic Interpreter + Governance layer
```

### UKT Feature Layout

| Slots | Block | Source |
|-------|-------|--------|
| 0–15  | TFT attention weights | yfinance + macro APIs |
| 16–23 | BERTopic topic shares | RSS / HN / UN News |
| 24–26 | TFT decoder importance | pytorch-forecasting |
| 27–31 | Macro static reals (GDP growth, inflation, FX, CPI, mkt-cap/GDP) | IMF / ECB / WB / BLS |
| 32–47 | Graph centrality (degree, betweenness, eigenvector, PageRank) | Harvard Dataverse + WB WGI |
| 48–63 | Agent sim (resource shares, alliance eigenvalues) | Internal simulation |
| 64–79 | Spatial SVD kernel loadings | Open-Elevation / Open-Meteo / USGS / NOAA |

---

## Data Sources (all keyless)

### Finance — 10 APIs
| # | Source | Data |
|---|--------|------|
| 1 | yfinance | OHLCV (country ETFs) |
| 2 | Stooq.com | OHLCV fallback |
| 3 | ECB eurofxref-daily XML | EUR cross-rates |
| 4 | IMF DataMapper | GDP growth per country |
| 5 | US Treasury OData XML | Yield curve (9 maturities) |
| 6 | CoinGecko /global | Crypto market cap |
| 7 | Open.er-api | USD FX rates |
| 8 | World Bank financial | CPI, market-cap/GDP, FDI, M2 |
| 9 | BIS WS_CBPOL (monthly) | Central bank policy rates |
| 10 | BLS CPI v1 public API | US consumer price index |

### News — 11 sources
RSS feeds (BBC, Reuters, Al Jazeera, Guardian, France24), HN Algolia,
UN News RSS (3 feeds), GDELT DOC 2.0, Wikipedia events API,
Reddit /r/worldnews, GDELT GKG 2.0

### Political — 10 sources
Harvard Dataverse UN voting records, World Bank WGI (5 indicators:
voice/accountability, govt effectiveness, rule of law, corruption control,
regulatory quality), IMF WEO (debt-to-GDP, unemployment), GDELT political
event volumes, OWID Democracy Index (EIU)

### Spatial — 12 sources
Open-Elevation, Open-Meteo Archive (climate), World Bank (GDP PPP, debt,
military spend, tertiary enrollment, political stability, homicide rate),
USGS Earthquake Hazards, NASA EONET, Open-Meteo Air Quality (PM2.5),
NOAA Tides & Currents (sea-level proxy)

---

## Models

| Model | File | Role |
|-------|------|------|
| TemporalFusionTransformer | `models/tft_forecast.py` | OHLCV + macro → quantile forecast |
| BERTopic (multilingual) | `models/topic_model.py` | News corpus → topic distribution |
| NetworkX graph | `models/graph_engine.py` | UN votes → centrality features |
| SVD spatial kernels | `models/spatial_kernels.py` | Raster (14×6) → kernel loadings |
| Multi-agent sim | `models/agent_sim.py` | Resource + alliance dynamics |
| Sparse autoencoder | `models/sparse_ae.py` | UKT → concept bottleneck |

---

## Tabs (Streamlit UI)

| Tab | Name | Core Feature |
|-----|------|-------------|
| 0 | Mission Control | System overview, metrics, governance scorecard, status |
| 1 | Finance-Neural Block | TFT forecasting, correlation heatmaps |
| 2 | Informational Cluster Mapping | BERTopic multilingual clustering |
| 3 | Politics-Military Block | Graph engine, centrality, kernelization |
| 4 | Agentic Simulation | Multi-agent resource/alliance sim |
| 5 | Semantic Interpreter | Concept bottleneck, kernel viz, semantic canvas |
| 6 | Hyperspace Pipeline | End-to-end orchestration |
| 7 | Counterfactual | Block removal + diff analysis (contestability) |

---

## Semantic Interpreter

The Semantic Interpreter (Tab 5) transforms opaque model internals into
human-readable explanations through three layers:

1. **Per-Stage Sparse Autoencoders** — Each pipeline block (Finance, Clusters,
   Graph, Spatial, Agents) gets a lightweight SAE that discovers 8 sparse
   concepts from its 16-dimensional feature region.

2. **Semantic Canvas** — A 12-dimensional coordinate system with named axes
   (Market Momentum, Volatility Regime, Alliance Polarity, Network Cohesion,
   etc.). Each block projects its discovered concepts onto the canvas, building
   a cumulative interpretive picture across all data domains.

3. **Tiny-LLM Narrator** — A ~10M-parameter language model
   ([arnir0/Tiny-LLM](https://huggingface.co/arnir0/Tiny-LLM)) generates
   plain-English narratives from structured canvas data. Runs CPU-only with
   graceful fallback to algorithmic summaries if unavailable.

### Visualizations

- **Semantic Canvas Radar Chart** — Polar plot of accumulated canvas coordinates
  with per-layer overlays
- **Canvas Evolution Heatmap** — Cumulative semantic state after each pipeline step
- **Kernel Matrix & Importance** — Cross-block activation patterns and SVD
  kernel variance explained
- **Concept Activation Heatmap** — Per-block SAE concept activations
- **Concept-Kernel Correspondence** — Maps SAE concepts to SVD kernels, closing
  the interpretability loop: raw data → features → kernels → concepts → narratives

### Governance Integration

- **Contestability** — Every kernel and concept includes a "Contest This" button
  backed by multi-run stability scores
- **Stakeholder Annotations** — Multi-stakeholder annotation widgets on kernels
  and concepts
- **Policy Language Mode** — Toggle to plain-English briefing language for
  non-technical delegates

See [`docs/semantic-interpretability.md`](docs/semantic-interpretability.md) for
full API documentation and integration guide.

---

## UI Status

The backend pipeline, UKT framework, and governance layer are fully functional.
The **UI rendering layer** has known gaps documented in `STRATEGY_UI.md`:

| Area | Status | Key Issue |
|------|--------|-----------|
| Backend pipeline | **Complete** | 290 tests passing |
| Governance logic | **Complete** | Flags, scorecard, faithfulness, drift |
| Feature interpretability in charts | **Broken** | Reality Regression shows indices 0–79, not feature names |
| Dark theme accessibility | **Failing** | Caption/label/tab text fails WCAG AA contrast |
| Mission Control hierarchy | **Needs work** | 15+ flat sections, no executive summary |
| Computation caching | **Incomplete** | SAE, counterfactual SVD, UVT, USE uncached |
| Pipeline progress UX | **Minimal** | Single spinner for 15–30s, no per-block status |

See `docs/alpha-1.0-issue-tracker.md` (H-004 through H-007) and
`STRATEGY_UI.md` for full details and implementation plans.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Runs fully CPU-only. No API keys required. All 32+ data sources are public/open.

---

## Project Layout

```
app.py                        # Streamlit entry point
hyperspace/
  config.py                   # UKT layout, node graph, CSS, constants
  state.py                    # Session state management
  data/
    finance.py                # 10 keyless finance APIs + TFT dataset builder
    news.py                   # 11 news sources + text aggregator
    political.py              # 10 political sources + agreement matrix
    spatial.py                # 12 spatial sources + raster builder
  models/
    tft_forecast.py           # TFT fit + UKT feature extraction
    topic_model.py            # BERTopic fit + topic feature extraction
    graph_engine.py           # Geopolitical graph + centrality analysis
    spatial_kernels.py        # SVD kernelization of spatial raster
    agent_sim.py              # Multi-agent resource/alliance simulation
    sparse_ae.py              # Sparse autoencoder (concept bottleneck)
    knowledge_matrix.py       # UKT assembly + provenance tracking
  pages/                      # One module per Streamlit tab
requirements.txt
```

---

## Governance Features

- **Provenance tracing** — every UKT feature carries source metadata (API, block, metric)
- **Jurisdiction badges** — data sources labelled by regulatory jurisdiction
- **Governance scorecard** — feature traceability ≥90%, kernel stability ≥0.75
- **Policy language mode** — toggle to plain-English briefing language for non-technical delegates
- **Run IDs** — each pipeline execution tagged with a unique identifier and timestamp
