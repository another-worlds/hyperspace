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

### Development Commands
```bash
pip install -r requirements.txt
streamlit run app.py
```
