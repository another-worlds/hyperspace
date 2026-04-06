"""Constants, CSS, geopolitical structures, and news snippets."""
from __future__ import annotations

DARK_CSS = """
<style>
/* ─── Typography: Inter + JetBrains Mono ────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ─── Clean presentation — remove Streamlit chrome ──────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; border: none; }

/* ─── Dividers ──────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #182135 !important;
    margin: 18px 0 !important;
}

/* ─── Sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060b17 0%, #090e1e 100%);
    border-right: 1px solid #152030;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ─── Metric cards ──────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0b1525 0%, #0f1e3a 100%);
    border: 1px solid #1a2e50;
    border-top: 2px solid #64ffda;
    border-radius: 10px;
    padding: 18px 16px 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35), 0 0 0 1px rgba(100,255,218,0.04);
    transition: box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 6px 28px rgba(0,0,0,0.5), 0 0 0 1px rgba(100,255,218,0.08);
}
div[data-testid="stMetric"] label {
    color: #8ea8c2 !important;
    font-size: 0.71em !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stMetricValue"] {
    color: #dce8f0 !important;
    font-weight: 700 !important;
    font-size: 1.65em !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stMetricDelta"] { color: #64ffda !important; }

/* ─── Tab bar ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #152030;
    gap: 0;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #8ea8c2 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.84em !important;
    padding: 10px 18px !important;
    letter-spacing: 0.025em !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #8ab4cc !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    border-bottom: 2px solid #64ffda !important;
    color: #dce8f0 !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

/* ─── Primary button ────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0d3f6b 0%, #1a5898 100%);
    border: 1px solid rgba(100,255,218,0.28);
    color: #dce8f0;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9em;
    letter-spacing: 0.05em;
    border-radius: 8px;
    padding: 10px 28px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4), 0 0 0 1px rgba(100,255,218,0.08);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1a5898 0%, #2471c0 100%);
    border-color: rgba(100,255,218,0.55);
    box-shadow: 0 6px 24px rgba(0,0,0,0.5), 0 0 18px rgba(100,255,218,0.18);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}

/* ─── Expanders ─────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #152030 !important;
    border-radius: 8px !important;
    background: #070d1a !important;
}
[data-testid="stExpander"] summary {
    color: #8ea8c2 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88em !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"]:hover summary { color: #8ba8c8 !important; }

/* ─── Captions ──────────────────────────────────────────────────────── */
.stCaption p, .stCaption {
    color: #8ab4cc !important;
    font-size: 0.79em !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.55 !important;
}

/* ─── Concept badge ─────────────────────────────────────────────────── */
.concept-badge {
    display: inline-block; padding: 4px 14px; margin: 2px;
    border-radius: 20px; font-size: 0.80em; font-weight: 600;
    background: linear-gradient(90deg, #0c2d50, #112a50);
    color: #7dd3fc;
    border: 1px solid #1e4070;
    letter-spacing: 0.03em;
    font-family: 'Inter', sans-serif;
}

/* ─── Log entry ─────────────────────────────────────────────────────── */
.log-entry {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.79em;
    padding: 5px 10px; margin: 2px 0;
    border-left: 3px solid #64ffda;
    background: #06111f; color: #8ab4cc;
    border-radius: 0 4px 4px 0;
}

/* ─── Source badge ──────────────────────────────────────────────────── */
.source-badge {
    display: inline-block; padding: 2px 10px; margin: 2px;
    border-radius: 12px; font-size: 0.74em; font-weight: 600;
    background: #041d12; color: #64ffda; border: 1px solid #0a3d28;
    font-family: 'Inter', sans-serif;
}
.source-badge.fallback {
    background: #200d0d; color: #f87171; border: 1px solid #4a1515;
}

/* ─── Governance flag / pass badges ────────────────────────────────── */
.gov-flag {
    display: inline-block; padding: 3px 12px; margin: 3px 2px;
    border-radius: 6px; font-size: 0.77em; font-weight: 700;
    background: #1c0e00; color: #fbbf24; border: 1px solid #6b3800;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}
.gov-pass {
    display: inline-block; padding: 3px 12px; margin: 3px 2px;
    border-radius: 6px; font-size: 0.77em; font-weight: 700;
    background: #031610; color: #34d399; border: 1px solid #065038;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}

/* ─── Jurisdiction badge ────────────────────────────────────────────── */
.jurisdiction-badge {
    display: inline-block; padding: 2px 10px; margin: 2px;
    border-radius: 12px; font-size: 0.71em; font-weight: 600;
    background: #0d1232; color: #818cf8; border: 1px solid #252480;
    font-family: 'Inter', sans-serif;
}

/* ─── Synthetic data banner ─────────────────────────────────────────── */
.synthetic-banner {
    background: linear-gradient(90deg, #120800, #1e0f00);
    border: 1px solid #6b3800;
    border-left: 4px solid #fbbf24;
    border-radius: 8px;
    padding: 12px 18px; margin: 8px 0;
    color: #fcd34d; font-weight: 600; font-size: 0.88em;
    font-family: 'Inter', sans-serif;
}

/* ─── Governance section header ─────────────────────────────────────── */
.governance-header {
    background: linear-gradient(135deg, #050c1a 0%, #09152a 55%, #060f20 100%);
    border: 1px solid #1a2e50;
    border-top: 3px solid #64ffda;
    border-radius: 12px;
    padding: 28px 32px; margin: 8px 0 20px;
}

/* ─── Run ID watermark ──────────────────────────────────────────────── */
.run-id-watermark {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.67em; color: #8ab4cc; padding: 2px 6px;
    letter-spacing: 0.06em;
}

/* ─── Contestability note ───────────────────────────────────────────── */
.contest-note {
    background: #050f1e; border-left: 3px solid #fbbf24;
    padding: 8px 14px; margin: 4px 0; font-size: 0.84em;
    color: #8ab4cc; border-radius: 0 6px 6px 0;
}

/* ─── Annotation tag ────────────────────────────────────────────────── */
.annotation-tag {
    display: inline-block; padding: 2px 10px; margin: 2px;
    border-radius: 12px; font-size: 0.71em; font-weight: 600;
    background: #130e2c; color: #a5b4fc; border: 1px solid #2e28a0;
    font-family: 'Inter', sans-serif;
}

/* ─── Landing page cards ────────────────────────────────────────────── */
.accountability-card {
    background: linear-gradient(145deg, #0b1525 0%, #0e1e3a 100%);
    border: 1px solid #1a2e50;
    border-radius: 10px;
    padding: 24px 22px 20px;
    height: 100%;
}
.accountability-card .card-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9em; font-weight: 700;
    color: #1a3050; line-height: 1;
    margin-bottom: 10px;
}
.accountability-card h4 {
    color: #f87171; margin: 0 0 10px;
    font-size: 0.93em; font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
}
.accountability-card p {
    color: #8ab4cc; font-size: 0.87em; line-height: 1.65;
    margin: 0; font-family: 'Inter', sans-serif;
}
.guarantee-card {
    background: linear-gradient(145deg, #031610 0%, #052018 100%);
    border: 1px solid #0a3d28;
    border-top: 2px solid #34d399;
    border-radius: 10px;
    padding: 24px 22px 20px;
    height: 100%;
}
.guarantee-card h4 {
    color: #34d399; margin: 0 0 10px;
    font-size: 0.92em; font-weight: 700;
    letter-spacing: 0.03em;
    font-family: 'Inter', sans-serif;
}
.guarantee-card p {
    color: #7aab98; font-size: 0.87em; line-height: 1.65;
    margin: 0; font-family: 'Inter', sans-serif;
}

/* ─── Sidebar Kanban Progress Cards ────────────────────────────────── */
.kanban-card {
    background: #0d1b2a;
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 10px 12px;
    margin: 6px 0;
    font-family: 'Inter', sans-serif;
}
.kanban-card-pending  { border-left: 3px solid #4a5568; }
.kanban-card-running  { border-left: 3px solid #4da6ff; }
.kanban-card-complete { border-left: 3px solid #34d399; }
.kanban-card-failed   { border-left: 3px solid #f56565; }
.kanban-card-skipped  { border-left: 3px solid #718096; }
.kanban-header {
    display: flex; align-items: center; gap: 6px; margin-bottom: 4px;
}
.kanban-icon { font-size: 0.9em; }
.kanban-icon-pending  { color: #4a5568; }
.kanban-icon-running  { color: #4da6ff; }
.kanban-icon-complete { color: #34d399; }
.kanban-icon-failed   { color: #f56565; }
.kanban-icon-skipped  { color: #718096; }
.kanban-title {
    font-size: 0.82em; font-weight: 700; color: #dce8f0;
    letter-spacing: 0.02em;
}
.kanban-status {
    font-size: 0.73em; color: #8ab4cc;
    font-family: 'JetBrains Mono', monospace;
}
.kanban-context {
    font-size: 0.70em; color: #8ab4cc;
    display: block; margin-top: 2px; line-height: 1.4;
}
.kanban-source {
    font-size: 0.70em; color: #64ffda;
    font-family: 'JetBrains Mono', monospace;
}
@keyframes kanban-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.kanban-card-running .kanban-icon { animation: kanban-pulse 2s ease-in-out infinite; }

/* ─── Country Selector Cards ─────────────────────────────────────── */
.country-grid { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.country-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: #0d1b2a; border: 1px solid #1a3a5c; border-radius: 8px;
    padding: 5px 10px; cursor: default; transition: all 0.15s ease;
    font-size: 0.78rem; color: #8ab4cc;
}
.country-chip.selected {
    background: #112940; border-color: #64ffda;
    box-shadow: 0 0 6px rgba(100, 255, 218, 0.15);
}
.country-chip .chip-flag { font-size: 1.1rem; }
.country-chip .chip-name { font-weight: 600; color: #dce8f0; }
.country-chip .chip-bloc {
    font-size: 0.65rem; color: #4da6ff; font-weight: 500;
    background: #0a1929; padding: 1px 5px; border-radius: 4px;
}
.country-chip .chip-etf {
    font-size: 0.65rem; color: #64ffda;
    font-family: 'JetBrains Mono', monospace;
}

/* ─── Sidebar Section Headers ────────────────────────────────────── */
.sidebar-section-header {
    font-size: 0.72rem; font-weight: 700; color: #4da6ff;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 8px 0 4px; padding-bottom: 4px;
    border-bottom: 1px solid #1a3a5c;
}

/* ─── Enhanced Kanban Cards ──────────────────────────────────────── */
.kanban-progress-bar {
    height: 3px; background: #1a3a5c; border-radius: 2px;
    margin-top: 6px; overflow: hidden;
}
.kanban-progress-fill {
    height: 100%; border-radius: 2px; transition: width 0.4s ease;
}
.kanban-progress-fill.fill-complete { background: #34d399; width: 100%; }
.kanban-progress-fill.fill-running  { background: #4da6ff; animation: progress-pulse 1.5s ease-in-out infinite; }
.kanban-progress-fill.fill-failed   { background: #f56565; width: 100%; }
.kanban-progress-fill.fill-pending  { background: #4a5568; width: 0%; }
.kanban-progress-fill.fill-skipped  { background: #718096; width: 100%; }
@keyframes progress-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

.kanban-timing {
    font-size: 0.65rem; color: #64ffda;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}

/* ─── Cross-Tab Navigation Pills ─────────────────────────────────── */
.nav-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem; }
.nav-pill {
    display: inline-block;
    background: #0d1b2a;
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    color: #8ab4cc;
    line-height: 1.4;
}
.nav-pill-tab { font-weight: 600; color: #64ffda; }
</style>
"""

# Plotly dark layout defaults
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#070d1a",
    plot_bgcolor="#070d1a",
    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        size=12,
        color="#8ab4cc",
    ),
    hoverlabel=dict(
        bgcolor="#0d1e38",
        bordercolor="#1a2e50",
        font=dict(
            family="Inter, sans-serif",
            size=12,
            color="#dce8f0",
        ),
    ),
    xaxis=dict(gridcolor="#0d1e30", gridwidth=0.5, zerolinecolor="#182840"),
    yaxis=dict(gridcolor="#0d1e30", gridwidth=0.5, zerolinecolor="#182840"),
)

GEOPOLITICAL_NODES: dict[str, dict] = {
    "USA":     dict(influence=0.95, lat=38.9, lon=-77.0,  color="#3498db", bloc="NATO"),
    "Russia":  dict(influence=0.75, lat=55.8, lon=37.6,   color="#e74c3c", bloc="CIS"),
    "China":   dict(influence=0.90, lat=39.9, lon=116.4,  color="#e67e22", bloc="SCO"),
    "Britain": dict(influence=0.70, lat=51.5, lon=-0.1,   color="#2980b9", bloc="NATO"),
    "India":   dict(influence=0.72, lat=28.6, lon=77.2,   color="#2ecc71", bloc="NAM"),
    "Brazil":  dict(influence=0.55, lat=-15.8, lon=-47.9, color="#9b59b6", bloc="BRICS"),
}

GEOPOLITICAL_EDGES: list[tuple] = [
    ("USA", "Britain",    0.92,  "alliance",       "Five Eyes + NATO special relationship"),
    ("USA", "India",      0.58,  "alignment",      "Quad partnership + defense convergence"),
    ("USA", "China",     -0.45,  "competition",    "Trade war + tech decoupling + Taiwan"),
    ("USA", "Russia",    -0.72,  "competition",    "Sanctions + NATO expansion + proxy conflicts"),
    ("USA", "Brazil",     0.35,  "alignment",      "Trade + regional security cooperation"),
    ("Russia", "China",   0.78,  "alignment",      "Strategic partnership + energy + SCO axis"),
    ("Russia", "India",   0.40,  "alignment",      "Legacy defence + S-400 + energy trade"),
    ("Russia", "Brazil",  0.20,  "alignment",      "BRICS cooperation + commodity trade"),
    ("Russia", "Britain", -0.65, "competition",    "Post-2022 sanctions + diplomatic rupture"),
    ("China", "India",   -0.25,  "competition",    "Border disputes + LAC tensions"),
    ("China", "Britain", -0.30,  "competition",    "Hong Kong fallout + tech restrictions"),
    ("China", "Brazil",   0.60,  "financial_flow", "BRI + commodity imports + infrastructure"),
    ("Britain", "India",  0.52,  "alignment",      "Commonwealth ties + FTA negotiations"),
    ("India", "Brazil",   0.38,  "alignment",      "IBSA + BRICS + South-South cooperation"),
    ("Brazil", "Britain", 0.25,  "alignment",      "Trade + climate finance partnership"),
]

# UKT feature dimension — computed from registry at import time.
# Kept as a module-level constant for backward compatibility, but the
# canonical source of truth is HYPERSPACE_REGISTRY.total_dim.
UKT_FEATURE_DIM: int = 80

# All available tickers for finance block — country-representative ETFs
AVAILABLE_TICKERS: list[str] = ["SPY", "EWZ", "INDA", "FXI", "EWU", "ERUS", "RSX"]

# Default tickers for finance block — country-representative ETFs / major stocks
DEFAULT_TICKERS: list[str] = ["SPY", "EWZ", "INDA"]

# Country → ticker mapping for sidebar country selector
COUNTRY_TICKER_MAP: dict[str, str] = {
    "USA":     "SPY",
    "Brazil":  "EWZ",
    "India":   "INDA",
    "China":   "FXI",
    "Britain": "EWU",
    "Russia":  "ERUS",
}

# Country display metadata for sidebar cards
COUNTRY_DISPLAY: dict[str, dict] = {
    "USA":     {"flag": "\U0001F1FA\U0001F1F8", "etf": "SPY",  "bloc": "NATO",  "color": "#3498db"},
    "Russia":  {"flag": "\U0001F1F7\U0001F1FA", "etf": "ERUS", "bloc": "CIS",   "color": "#e74c3c"},
    "China":   {"flag": "\U0001F1E8\U0001F1F3", "etf": "FXI",  "bloc": "SCO",   "color": "#e67e22"},
    "Britain": {"flag": "\U0001F1EC\U0001F1E7", "etf": "EWU",  "bloc": "NATO",  "color": "#2980b9"},
    "India":   {"flag": "\U0001F1EE\U0001F1F3", "etf": "INDA", "bloc": "NAM",   "color": "#2ecc71"},
    "Brazil":  {"flag": "\U0001F1E7\U0001F1F7", "etf": "EWZ",  "bloc": "BRICS", "color": "#9b59b6"},
}

AVAILABLE_COUNTRIES: list[str] = list(COUNTRY_TICKER_MAP.keys())
DEFAULT_COUNTRIES: list[str] = ["USA", "Brazil", "India"]


def countries_to_tickers(countries: list[str]) -> list[str]:
    """Convert selected country names to their representative ETF tickers."""
    return [COUNTRY_TICKER_MAP[c] for c in countries if c in COUNTRY_TICKER_MAP]


# ---------------------------------------------------------------------------
# Kernel & projection thresholds (centralized — not scattered as magic numbers)
# ---------------------------------------------------------------------------
# Minimum fraction of total region loading for a region to be listed as
# "contributing" to a kernel label.  With N=5 regions, 1/N = 0.20.
KERNEL_CONTRIBUTING_REGION_THRESHOLD: float = 0.15

# Minimum absolute block loading (U column) for a block to appear in
# kernel block-contribution lists.
KERNEL_BLOCK_CONTRIBUTION_MIN: float = 0.1

# Minimum kernel importance (fraction of total variance) to trigger
# semantic narrator invocation.
KERNEL_NARRATOR_IMPORTANCE_MIN: float = 0.15

# --------------------------------------------------------------------------- #
# Tab-level governance / display thresholds (centralized from tab modules)    #
# --------------------------------------------------------------------------- #
# Finance tab: forecast confidence bands
FORECAST_CONFIDENCE_HIGH: float = 0.10
FORECAST_CONFIDENCE_MODERATE: float = 0.30

# Clusters tab: topic coverage outlier ratios
OUTLIER_RATIO_CRITICAL: float = 0.30
OUTLIER_RATIO_MODERATE: float = 0.10

# Agents tab: resource Gini concentration
GINI_CONCENTRATION_HIGH: float = 0.50
GINI_CONCENTRATION_MODERATE: float = 0.25

# Interpreter tab: kernel expander auto-expand threshold
KERNEL_EXPANDER_THRESHOLD: float = 0.20

# Counterfactual tab: reality regression cosine stability
CF_STABILITY_HIGH: float = 0.95
CF_STABILITY_MODERATE: float = 0.80

# Drift monitoring thresholds (from core/drift_monitor.py)
DRIFT_REGRESSION_COSINE_THRESHOLD: float = 0.85
DRIFT_IMPORTANCE_COSINE_THRESHOLD: float = 0.80
DRIFT_STABILITY_DELTA_THRESHOLD: float = -0.10

# --------------------------------------------------------------------------- #
# Semantic Canvas coupling weights
# --------------------------------------------------------------------------- #
# Coupling weights are now computed via the SharedProjection energy-based
# mechanism. Explicit static constants removed under the emergence contract.

# --------------------------------------------------------------------------- #
# Disk-based API data cache (survives app restarts)                            #
# --------------------------------------------------------------------------- #
import pathlib as _pathlib
DATA_CACHE_DIR: _pathlib.Path = _pathlib.Path(__file__).resolve().parent.parent / ".data_cache"
DATA_CACHE_TTL_FINANCE: int = 3600       # 1 hour  (matches yfinance st.cache_data TTL)
DATA_CACHE_TTL_NEWS: int = 1800          # 30 min  (matches GDELT/RSS TTL)
DATA_CACHE_TTL_POLITICAL: int = 86400    # 24 hours (matches Dataverse TTL)
DATA_CACHE_TTL_SPATIAL: int = 86400      # 24 hours

# Feature flags
FEATURE_FLAGS: dict[str, bool] = {
    # Shared-latent prototype remains shadow-only until parity + governance
    # transparency criteria are met.
    "shared_latent_shadow": True,
}

# --------------------------------------------------------------------------- #
# SPEC-4: Cross-Modal Contrastive Alignment                                    #
# --------------------------------------------------------------------------- #
ENABLE_CONTRASTIVE_ALIGNMENT: bool = False
CONTRASTIVE_WEIGHT: float = 0.0           # 0.0 = pure SVD; 1.0 = pure contrastive
CONTRASTIVE_TEMPERATURE: float = 0.07     # InfoNCE temperature τ
CONTRASTIVE_LATENT_DIM: int = 32          # shared latent dimension per block encoder

# --------------------------------------------------------------------------- #
# SPEC-5: Temporal World-Model Memory                                          #
# --------------------------------------------------------------------------- #
ENABLE_TEMPORAL_MEMORY: bool = False
TEMPORAL_ENCODER_HIDDEN_DIM: int = 32     # GRU hidden dimension
TEMPORAL_MIN_RUNS: int = 3                # minimum runs before predictions
TEMPORAL_GOVERNANCE_CONFIDENCE: float = 0.8  # min confidence for governance surfacing
TEMPORAL_GOVERNANCE_MIN_RUNS: int = 5     # min runs at high confidence for governance

# --------------------------------------------------------------------------- #
# SPEC-6: Mechanistic Probes                                                   #
# --------------------------------------------------------------------------- #
PROBE_CANVAS_LOADING_THRESHOLD: float = 0.1   # min |loading| to count as canvas influence
PROBE_MIN_RUNS_FOR_LINEAR: int = 10           # min KernelMemory runs for linear probing

# --------------------------------------------------------------------------- #
# SPEC-7: Knowledge Persistence & Transfer                                     #
# --------------------------------------------------------------------------- #
KERNEL_TRANSFER_ENABLED: bool = False
KERNEL_LINEAGE_COSINE_THRESHOLD: float = 0.85   # min cosine to assign existing lineage
KERNEL_DISTILLATION_MIN_RUNS: int = 5            # consecutive runs before distillation
KERNEL_LIBRARY_PATH: str = "kernel_library.json"
KERNEL_STALE_THRESHOLD_RUNS: int = 20            # runs without sighting → archive

# Pipeline step names
PIPELINE_STEPS: list[str] = [
    "data_fetch",
    "finance_block",
    "cluster_block",
    "graph_block",
    "agent_sim",
    "final_interpretation",
]

# --------------------------------------------------------------------------- #
# Semantic feature names: human-readable labels for each UKT feature index    #
# --------------------------------------------------------------------------- #
# In an emergent architecture, feature labels are derived from the active
# feature metadata and registry. As a stable fallback we expose positional
# names.
FEATURE_NAMES: list[str] = [f"dim_{i}" for i in range(UKT_FEATURE_DIM)]

# Region-level semantic descriptions for human reports
REGION_DESCRIPTIONS: dict[str, str] = {
    "temporal-pattern": (
        "Temporal attention patterns from the TFT encoder — captures which historical "
        "time-steps the model attends to most when predicting future values. High "
        "activation on recent steps indicates momentum-driven markets; spread across "
        "distant steps indicates long-memory regimes."
    ),
    "semantic-embedding": (
        "Topic distribution and embedding centroids from BERTopic clustering — "
        "captures the informational landscape. A dominant topic share indicates "
        "a focused discourse (e.g., one geopolitical event dominating headlines); "
        "flatter distributions indicate a fragmented information environment."
    ),
    "structural-centrality": (
        "Node centrality features from the geopolitical graph — degree, betweenness, "
        "eigenvector, and PageRank for each actor. High betweenness for a node means "
        "it is a critical bridge in the alliance network; high eigenvector centrality "
        "means it is connected to other powerful nodes."
    ),
    "dynamic-agent": (
        "Agent simulation outcomes — resource distribution and alliance matrix "
        "eigenvalues after bounded-rational agents interact for N steps. The "
        "eigenvalue spectrum of the alliance matrix captures structural polarization: "
        "a single dominant eigenvalue means one cohesive bloc, multiple comparable "
        "eigenvalues indicate a multipolar world."
    ),
    "geospatial-kernel": (
        "Multimodal spatial raster features kernelized via SVD across 10 data layers "
        "(elevation, temperature, humidity, precipitation, GDP PPP, debt, military "
        "spending, school enrollment, conflict event density, conflict fatality density) "
        "sampled at 6 geopolitical nodes. The dominant kernel importances and node "
        "loadings reveal which physical and socioeconomic dimensions co-vary most "
        "strongly across the international system."
    ),
}

# --------------------------------------------------------------------------- #
# Governance: Data source jurisdiction labels                                  #
# --------------------------------------------------------------------------- #
DATA_SOURCE_JURISDICTIONS: dict[str, dict] = {
    "yfinance":            dict(tag="US-REGULATED",    color="#3060c0", note="SEC-regulated market data"),
    "GDELT":               dict(tag="OPEN-PUBLIC",     color="#208040", note="Open-access global event database"),
    "Harvard Dataverse":   dict(tag="ACADEMIC-LICENSED", color="#806020", note="Academic open-data license (CC0/CC-BY)"),
    "synthetic":           dict(tag="SYNTHETIC",       color="#802020", note="Machine-generated data — not real-world observations"),
    "fallback":            dict(tag="SYNTHETIC",       color="#802020", note="Machine-generated fallback — not real-world observations"),
}

# --------------------------------------------------------------------------- #
# Governance: interpretability score card thresholds                          #
# --------------------------------------------------------------------------- #
SCORECARD_THRESHOLDS: dict[str, dict] = {
    "feature_traceability": dict(
        label="Feature Traceability",
        unit=f"/ {UKT_FEATURE_DIM}",
        threshold=int(UKT_FEATURE_DIM * 0.9),  # ≥90% of features
        description="Features with semantic metadata labels attached.",
    ),
    "kernel_stability": dict(
        label="Kernel Stability (cosine)",
        unit="",
        threshold=0.75,
        description="Mean cosine similarity of reality regression across 8 noisy runs. ≥0.75 = stable.",
    ),
    "concept_activation_rate": dict(
        label="Concept Activation Rate",
        unit="%",
        threshold=30.0,
        description="Percentage of SAE concepts with above-mean activation. ≥30% = adequate concept coverage.",
    ),
    "data_source_diversity": dict(
        label="Live Data Sources",
        unit="/ 4",
        threshold=2,
        description="Number of pipeline blocks using live (non-synthetic) data. ≥2 = adequate real-world grounding.",
    ),
    "governance_flags": dict(
        label="Governance Flags Active",
        unit="",
        threshold=0,        # 0 = PASS; any flag = WARN
        description="Auto-detected data quality, bias, or coverage issues. 0 = no issues detected.",
    ),
    "legacy_retrieval_at_1": dict(
        label="Legacy Retrieval@1",
        unit="",
        threshold=0.15,
        description="Baseline paired-window retrieval accuracy from the legacy normalized 80-d pathway. ≥0.15 = above random chance.",
    ),
    "shared_latent_retrieval_at_1": dict(
        label="Shared-Latent Retrieval@1",
        unit="",
        threshold=0.20,
        description="Prototype paired-window retrieval accuracy in shared latent space (shadow mode). ≥0.20 = encoder learning signal.",
    ),
    "shared_latent_probe_cosine": dict(
        label="Shared-Latent Probe Cosine",
        unit="",
        threshold=0.10,
        description="Average positive-pair cosine alignment for shared-latent prototype (shadow mode). ≥0.10 = learned alignment.",
    ),
    "faithfulness_confidence": dict(
        label="Faithfulness Confidence",
        unit="",
        threshold=0.50,
        description="Overall confidence from mechanistic faithfulness checks. ≥0.50 = narratives are evidence-grounded.",
    ),
}

# --------------------------------------------------------------------------- #
# Governance: Policy language mode — kernel label replacements                #
# Maps dominant_region + dominant_block combinations to policy-friendly names #
# --------------------------------------------------------------------------- #
POLICY_KERNEL_NAMES: dict[str, str] = {
    "temporal-pattern":      "Market Momentum Indicator",
    "semantic-embedding":    "Information Landscape Signal",
    "structural-centrality": "Alliance Network Structure",
    "dynamic-agent":         "Geopolitical Power Distribution",
    "geospatial-kernel":     "Physical & Economic Terrain Signal",
}

POLICY_CONFIDENCE_BANDS: list[tuple[float, str, str]] = [
    # (threshold, label, explanation)
    (0.40, "Strong signal",   "This factor explains a dominant share of the observed variance across all modalities."),
    (0.25, "Moderate signal", "This factor accounts for a meaningful but not dominant share of variance."),
    (0.10, "Weak signal",     "This factor explains a minor portion of variance; treat with caution."),
    (0.00, "Negligible",      "This factor explains very little variance and may reflect noise."),
]

# --------------------------------------------------------------------------- #
# Governance: Glossary definitions for non-technical delegates                #
# --------------------------------------------------------------------------- #
GLOSSARY: dict[str, str] = {
    "Universal Knowledge Tensor (UKT)": (
        "A unified mathematical structure that combines data from multiple sources "
        "(financial markets, news, geopolitical networks, agent simulations) into a "
        "single comparable format. Think of it as a structured table where each row "
        "is a data domain and each column is a measurable dimension."
    ),
    "SVD Kernel": (
        "A statistically discovered 'theme' or pattern that cuts across multiple data "
        "domains simultaneously. Each kernel explains a portion of the total variation "
        "observed. Kernels are ranked by how much variation they explain (importance %)."
    ),
    "Reality Regression": (
        "A weighted summary vector that combines all kernels into a single direction "
        "representing the system's overall 'reading' of the current state. It is the "
        "system's best approximation of what is happening across all data domains at once."
    ),
    "Sparse Autoencoder (SAE)": (
        "A machine learning tool that compresses data into a small number of 'concepts' "
        "and then reconstructs it. 'Sparse' means most concepts are inactive for any "
        "given input — only the relevant ones activate. This makes the internal reasoning "
        "more human-interpretable than a standard neural network."
    ),
    "Feature Traceability": (
        "The degree to which every number fed into the system can be traced back to its "
        "original source, collection method, time period, and entity. Full traceability "
        "means any conclusion can be audited back to its raw data inputs."
    ),
    "Reconstruction Error": (
        "How much information is lost when the system compresses data into kernels and "
        "then reconstructs it. Lower is better. High reconstruction error means the "
        "kernel structure does not fully capture the data — conclusions are less reliable."
    ),
    "Kernel Stability": (
        "A test of how much the system's conclusions change when small random noise is "
        "added to the input data. High stability (cosine similarity ≥ 0.75) means the "
        "conclusions are robust. Low stability means conclusions are sensitive to minor "
        "data variations — a governance risk."
    ),
    "Counterfactual Analysis": (
        "A method of testing AI conclusions by asking 'What would the system have "
        "concluded if one data source were removed or altered?' This is a core technique "
        "for auditing AI systems and detecting over-reliance on any single data stream."
    ),
    "BERTopic": (
        "A multilingual topic modelling algorithm that groups documents (news articles, "
        "reports) into thematic clusters automatically. It uses neural sentence embeddings "
        "to ensure semantically similar texts are grouped together regardless of language."
    ),
    "Temporal Fusion Transformer (TFT)": (
        "A neural network architecture designed for time-series forecasting. It uses "
        "'attention' mechanisms to learn which historical time periods are most relevant "
        "for predicting the future. The attention weights are interpretable."
    ),
    "Governance Flag": (
        "An automatically generated warning raised when the system detects a potential "
        "data quality issue, analytical bias, or coverage gap. Flags are machine-readable "
        "(coded GOV-001 through GOV-006) and included in all exported reports."
    ),
    "Provenance Chain": (
        "The complete documented history of a single data point: what it measures, "
        "where it came from, when it was collected, and how it influenced the final "
        "conclusion. Provenance chains enable due process — any claim can be challenged "
        "by examining its evidence chain."
    ),
}

# --------------------------------------------------------------------------- #
# Governance: Flag code definitions                                             #
# --------------------------------------------------------------------------- #
GOVERNANCE_FLAG_CODES: dict[str, dict] = {
    "GOV-001": dict(
        label="Modality Imbalance",
        description=(
            "More than 50% of active features originate from a single data block. "
            "Conclusions may over-represent one modality (e.g., financial data) "
            "at the expense of others (geopolitical, informational)."
        ),
        severity="warning",
    ),
    "GOV-002": dict(
        label="Temporal Coverage Gap",
        description=(
            "Finance and news data cover different date ranges. Cross-modal conclusions "
            "drawn from these sources may reflect different time periods, reducing "
            "causal coherence."
        ),
        severity="warning",
    ),
    "GOV-003": dict(
        label="Geopolitical Centrality Skew",
        description=(
            "One geopolitical actor's centrality score exceeds 2× the network average. "
            "The structural analysis may disproportionately reflect that actor's "
            "position, potentially introducing systemic bias."
        ),
        severity="warning",
    ),
    "GOV-004": dict(
        label="Low Concept Coverage",
        description=(
            "More than 60% of SAE concepts are dormant (below-mean activation). "
            "The system found limited interpretable structure in the data. "
            "Conclusions should be treated as low-confidence."
        ),
        severity="warning",
    ),
    "GOV-005": dict(
        label="Synthetic Data Active",
        description=(
            "One or more pipeline blocks are using machine-generated synthetic data "
            "instead of live observations. All conclusions derived from synthetic "
            "inputs are illustrative only and must not be treated as empirical findings."
        ),
        severity="info",
    ),
    "GOV-006": dict(
        label="Spatial Data Unavailable",
        description=(
            "Spatial raster data (elevation, climate, World Bank indicators, "
            "environmental layers) could not be fetched. The pipeline proceeded "
            "without spatial enrichment. Geospatial kernel analysis and spatial "
            "block outputs are absent from this run."
        ),
        severity="warning",
    ),
}
