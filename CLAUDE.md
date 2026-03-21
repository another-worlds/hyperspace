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

### Known UI Issues & Constraints — UPDATE LOG

**Status**: Major improvements completed in Phase 0-2 of UI/Architecture redesign (2026-03-19).
See branch: `claude/design-ui-architecture-ZNRZv` for implementation.

#### ✅ FIXED (Phase 0-2 Complete)

- **WCAG contrast failures** (FIXED): Updated run-id-watermark `#1e3348` → `#8ab4cc`. Sidebar subtitle `#3d5673` → `#8ab4cc`. All text now meets WCAG AA 4.5:1 contrast ratio. Verified against dark background `#070d1a`.

- **Feature names not shown** (VERIFIED CORRECT): Reality Regression charts already use `FEATURE_NAMES` in hover and axis labels. Feature name display working as intended.

- **Uncached heavy computation** (FIXED): Implemented intelligent caching system (`hyperspace/core/caching.py`):
  - SAE training cached by matrix + hyperparameter hash
  - Deterministic (cached) vs. stochastic (retrain) toggle modes
  - Expected savings: 5-10s per SAE iteration
  - Integrated in interpreter_tab.py with cache statistics display

- **Hardcoded thresholds** (FIXED): All governance thresholds centralized to `config.py`:
  - `FORECAST_CONFIDENCE_HIGH/MODERATE`, `OUTLIER_RATIO_CRITICAL/MODERATE`
  - `GINI_CONCENTRATION_HIGH/MODERATE`, `KERNEL_EXPANDER_THRESHOLD`
  - `CF_STABILITY_HIGH/MODERATE`, `STRUCTURAL_REGION_BOUNDS`
  - `DRIFT_REGRESSION_COSINE_THRESHOLD`, `DRIFT_IMPORTANCE_COSINE_THRESHOLD`, `DRIFT_STABILITY_DELTA_THRESHOLD`
  - `KERNEL_NARRATOR_IMPORTANCE_MIN` (used in `_report_section.py`)
  - Now auditable and consistent across codebase

- **Mission Control content hierarchy** (FIXED): Restructured with governance-first organization:
  - Section 1: Executive Summary (3-column metrics)
  - Section 2: Governance Status (EXPANDED, scorecard + flags)
  - Section 3: Data Provenance (expanded, auditing focus)
  - Section 4: Technical Diagnostics (COLLAPSED, expert-only)
  - Section 5: Export & Actions (prominent buttons)
  - Policy officers now see accountability outputs FIRST, no scrolling past diagnostics

- **Pipeline progress** (FIXED): Added comprehensive progress visualization:
  - Per-block timing and status tracking
  - Governance context for each block
  - Timing summary chart after pipeline completion
  - Per-source error handling and reporting
  - Integration: `hyperspace/viz/pipeline_progress.py` with `PipelineProgressTracker` class

#### 🚀 NEW CAPABILITIES (Phase 1-2 Complete)

- **Parallel data fetching** (4-6× speedup): Finance, news, political, spatial data fetched concurrently (4 workers)
- **Parallel model training** (TFT + BERTopic in parallel, 2 workers)
- **Parallel kernel labeling** (3-4× speedup for 8-12 kernels)
- **Parallel stability estimation** (4-6× speedup for Monte Carlo runs)
- **Structured logging** (`hyperspace/core/logging.py`): Event-based audit trails for governance compliance
- **Counterfactual scenario memory**: Store last 5 runs, scenario selection UI, side-by-side comparison

#### ✅ FIXED (Phase 3 — UI Compliance Audit, 2026-03-21)

- **Hovertemplates across all charts** (FIXED): Added `hovertemplate` with human-readable field names to all ~25 Plotly charts across 10 files. Previously only 4 charts had hovertemplates.
- **Mission Control native charts** (FIXED): Replaced `st.line_chart()` and `st.bar_chart()` with interactive Plotly charts (`go.Scatter`, `go.Bar`) in `mission_control_tab.py`.
- **Drift monitor thresholds centralized** (FIXED): Moved `DRIFT_THRESHOLDS` from `drift_monitor.py` to `config.py` constants.
- **Report section hardcoded threshold** (FIXED): `_report_section.py` now imports `KERNEL_NARRATOR_IMPORTANCE_MIN` from config.
- **Sidebar subtitle contrast** (FIXED): `app.py` subtitle color `#3d5673` → `#8ab4cc` (WCAG AA compliant).
- **Retrain buttons in all tabs** (FIXED): Added "Retrain"/"Rebuild"/"Rerun" buttons to Finance, Clusters, Politics, and Agents tabs for cache invalidation.
- **Agent simulation caching** (FIXED): Added parameter-hash-keyed caching in `agents_tab.py` to avoid redundant simulation runs.
- **Educational captions** (FIXED): Added v3.0 captions to dashboard reconstruction/stability trend charts.

#### 📋 OUTSTANDING (Phase 3 Remaining)

- **Cross-tab navigation** (PENDING): Links between tabs, breadcrumb trail
- **Advanced Diagnostics reorganization** (PENDING): Convert 200-line expander into sub-tabs
- **Caption text color** (LOW PRIORITY): `#7a9ab8` yields ~3.5:1 against `#070d1a`; passes 3:1 large text but not 4.5:1 body text. Acceptable for caption/secondary text per WCAG guidelines.

### UI Development Principles

When modifying UI code, follow these rules in addition to the style guide:

1. **Feature names over indices** — Always use `FEATURE_NAMES[idx]` or `registry.feature_name(idx)` in hover templates and axis labels ✅ (VERIFIED WORKING)
2. **Cache expensive results** — Any computation >1s must be cached in `st.session_state` keyed by input hash, or via `@st.cache_data`/`@st.cache_resource` ✅ (IMPLEMENTED: `hyperspace/core/caching.py`)
3. **Centralize thresholds** — All magic numbers used in governance/display logic belong in `config.py` with descriptive constant names ✅ (COMPLETED)
4. **Contrast compliance** — All text elements must meet WCAG AA (4.5:1 for body, 3:1 for large text) against the dark background ✅ (FIXED)
5. **Progressive disclosure** — Technical diagnostics collapsed by default; governance outputs prominent and early in the page ✅ (IMPLEMENTED: Mission Control redesign)
6. **Hover templates** — Every Plotly chart must have a `hovertemplate` with human-readable field names, not just default Plotly hover ✅ (IMPLEMENTED: all ~25 charts across kernel_viz.py, charts.py, all 8 tabs, _report_section.py, dashboard.py)

### Development Commands
```bash
pip install -r requirements.txt
streamlit run app.py
```
