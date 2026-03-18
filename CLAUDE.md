# Hyperspace – Predictive Polymath System v3.0 Prototype

## Tech Stack & Strict Rules

### Core Stack
- **Python** 3.11+
- **UI Framework**: Streamlit (layout="wide", initial_sidebar_state="expanded")
- **Visualization**: Plotly (interactive, dark template="plotly_dark", hover, zoom)
- **ML / Core**:
  - `pytorch-forecasting` – TemporalFusionTransformer (tiny synthetic)
  - `bertopic` – multilingual topic modeling
  - `networkx` + `plotly` – interactive multi-relational graphs
  - `torch`, `pytorch-lightning` (minimal, CPU only)
- **Data**: synthetic in-app generation via numpy/pandas
- **Caching**: `@st.cache_resource` for models & expensive ops

### Style Guide
- PEP 8 compliant
- Type hints where natural
- Extensive docstrings and inline comments
- Functions over classes where possible (Streamlit idiom)
- All data generated in-app (pandas DataFrames) – no external APIs required

### UI Principles
- Dark modern theme via CSS injection (cards, gradients, badges)
- `st.tabs` for 8 core tabs
- Sidebar controls + global "Run Full Pipeline" button
- Metric cards, expanders, progress bars, success/error toasts
- Interactive Plotly everywhere with captions explaining v3.0 connection
- Educational: every major output has tooltip/expander linking to expert vision

### Constraints
- Total app < 1200 lines preferred
- Load time per tab < 5s on CPU
- No heavy GPU usage – all CPU inference
- Graceful `try/except` with `st.error` + fallback mock data
- No external API keys required for core functionality

### Architecture Tabs
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

### Known UI Issues & Constraints

See `STRATEGY_UI.md` for the full strategic audit. Key constraints:

- **WCAG contrast failures**: Caption text (`#2d4a66`), metric labels (`#3d5673`),
  inactive tabs (`#3d5673`), and expander summaries (`#567090`) all fail WCAG AA
  minimum contrast (4.5:1) against the `#070d1a` background. Fix before any
  user-facing deployment.
- **Feature names not shown**: Reality Regression chart (interpreter_tab.py,
  counterfactual_tab.py) displays feature indices 0–79 instead of
  `FEATURE_NAMES`. This breaks the core interpretability promise.
- **Uncached heavy computation**: SAE training (100 epochs), counterfactual
  baseline SVD, UVT (120 epochs), and USE (150 epochs) all re-run on every
  button click with no session-state caching.
- **Hardcoded thresholds**: 10 governance/display thresholds are scattered across
  tab modules instead of centralized in `config.py`. See STRATEGY_UI.md §6
  "Hardcoded Thresholds to Centralize" for the full list.
- **Mission Control content hierarchy**: 15+ sections in flat scroll with no
  grouping or executive summary. Policy users must scroll past technical
  diagnostics to find governance outputs.
- **Pipeline progress**: Single `st.status` for 15-30s with no per-block timing,
  no progress bar, no per-source API status.

### UI Development Principles

When modifying UI code, follow these rules in addition to the style guide:

1. **Feature names over indices** — Always use `FEATURE_NAMES[idx]` or
   `registry.feature_name(idx)` in hover templates and axis labels
2. **Cache expensive results** — Any computation >1s must be cached in
   `st.session_state` keyed by input hash, or via `@st.cache_data`/`@st.cache_resource`
3. **Centralize thresholds** — All magic numbers used in governance/display logic
   belong in `config.py` with descriptive constant names
4. **Contrast compliance** — All text elements must meet WCAG AA (4.5:1 for body,
   3:1 for large text) against the dark background
5. **Progressive disclosure** — Technical diagnostics collapsed by default;
   governance outputs prominent and early in the page
6. **Hover templates** — Every Plotly chart must have a `hovertemplate` with
   human-readable field names, not just default Plotly hover

### Development Commands
```bash
pip install -r requirements.txt
streamlit run app.py
```
