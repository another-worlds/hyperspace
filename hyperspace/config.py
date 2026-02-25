"""Constants, CSS, geopolitical structures, and news snippets."""
from __future__ import annotations

DARK_CSS = """
<style>
/* Global dark card styling */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
div[data-testid="stMetric"] label { color: #a8b2d1 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #64ffda !important; font-weight: 700;
}
.concept-badge {
    display: inline-block; padding: 4px 12px; margin: 2px;
    border-radius: 20px; font-size: 0.82em; font-weight: 600;
    background: linear-gradient(90deg, #0f3460, #533483);
    color: #e0e0ff;
}
.log-entry {
    font-family: 'Fira Code', monospace; font-size: 0.82em;
    padding: 4px 8px; margin: 2px 0; border-left: 3px solid #64ffda;
    background: #0d1117; color: #c9d1d9;
}
.source-badge {
    display: inline-block; padding: 2px 8px; margin: 2px;
    border-radius: 10px; font-size: 0.75em; font-weight: 600;
    background: #1a472a; color: #64ffda; border: 1px solid #2d6a4f;
}
.source-badge.fallback {
    background: #4a2020; color: #ff6b6b; border: 1px solid #6a2d2d;
}
</style>
"""

# Plotly dark layout defaults
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
)

GEOPOLITICAL_NODES: dict[str, dict] = {
    "USA":        dict(influence=0.95, lat=38.9, lon=-77.0, color="#3498db", bloc="NATO"),
    "NATO-EU":    dict(influence=0.80, lat=50.8, lon=4.4,   color="#2980b9", bloc="NATO"),
    "Russia":     dict(influence=0.70, lat=55.8, lon=37.6,  color="#e74c3c", bloc="CIS"),
    "China":      dict(influence=0.88, lat=39.9, lon=116.4, color="#e67e22", bloc="SCO"),
    "India":      dict(influence=0.72, lat=28.6, lon=77.2,  color="#2ecc71", bloc="NAM"),
    "Kazakhstan": dict(influence=0.35, lat=51.2, lon=71.4,  color="#f1c40f", bloc="CIS"),
    "CIS-bloc":   dict(influence=0.40, lat=53.9, lon=27.6,  color="#e74c3c", bloc="CIS"),
    "ASEAN":      dict(influence=0.45, lat=13.8, lon=100.5, color="#1abc9c", bloc="NAM"),
}

GEOPOLITICAL_EDGES: list[tuple] = [
    ("USA", "NATO-EU",     0.92,  "alliance",       "Financial + military integration"),
    ("USA", "India",       0.58,  "alignment",      "Post-1991 convergence trajectory"),
    ("USA", "China",      -0.45,  "competition",    "Trade tension + tech decoupling"),
    ("Russia", "China",    0.74,  "alignment",      "Strategic partnership deepening"),
    ("Russia", "CIS-bloc", 0.65,  "alliance",       "Post-Soviet integration"),
    ("Kazakhstan", "China", 0.84, "financial_flow", "BRI investment corridor"),
    ("Kazakhstan", "Russia", 0.55, "alliance",      "CSTO + EAEU membership"),
    ("Kazakhstan", "USA",   0.22, "alignment",      "Multi-vector diplomacy"),
    ("CIS-bloc", "NATO-EU", -0.68, "competition",   "Negative-sum structural tension"),
    ("India", "Russia",     0.40, "alignment",      "Legacy defence partnership"),
    ("ASEAN", "China",      0.35, "financial_flow", "Trade corridor + BRI"),
    ("ASEAN", "USA",        0.30, "alignment",      "Security partnerships"),
]

NEWS_SNIPPETS: list[str] = [
    "India and the US strengthen defense ties in the Indo-Pacific, signaling a post-1991 strategic shift away from non-alignment.",
    "Kazakhstan abstains on key UN General Assembly vote, balancing between Russian and Western pressure.",
    "China's Belt and Road Initiative expands into Central Asian energy infrastructure, deepening Kazakhstan dependence.",
    "NATO accelerates Eastern European deployments amid renewed tensions with CIS bloc members.",
    "Russia and China conduct joint naval exercises in the South China Sea for the third consecutive year.",
    "ASEAN nations express concern over South China Sea militarization but avoid direct confrontation with Beijing.",
    "Kazakhstan's multi-vector foreign policy tested as sanctions pressure mounts on Russian trade partners.",
    "India-US nuclear cooperation deal marks new era of Indo-American strategic partnership.",
    "CIS economic integration falters as member states seek alternative trade routes to avoid sanctions.",
    "UN voting records show Kazakhstan shifting from automatic Russia-alignment toward selective abstention.",
    "Die Europaeische Union verstaerkt Sanktionen gegen russische Energieimporte nach 2022.",
    "China strengthens investments in Central Asian infrastructure through the Silk Road corridor.",
    "L'ASEAN negocie de nouveaux accords commerciaux multilateraux face aux tensions sino-americaines.",
    "India expands military-technical cooperation with Israel and France in parallel with Russian ties.",
    "Turkey balances NATO membership with independent Middle East policy and S-400 procurement.",
    "Brazil-India-South Africa trilateral dialogue deepens on climate and trade reform.",
    "African Union members split on UN vote patterns, reflecting China-US influence competition.",
    "Post-2022 energy crisis reshapes European dependency maps, accelerating renewable transition.",
    "Central Asian water disputes intensify amid climate change and upstream dam projects.",
    "Cybersecurity alliances emerge as new axis of geopolitical alignment in 2025-2026.",
]

# UKT feature dimension (shared across all blocks)
UKT_FEATURE_DIM: int = 64

# Default tickers for finance block
DEFAULT_TICKERS: list[str] = ["AAPL", "TSLA", "NVDA"]

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
FEATURE_NAMES: list[str] = [
    # 0-15: temporal-pattern region (populated by Finance / TFT)
    "attention_recent_1d",      # 0  — attention weight on most-recent encoder step
    "attention_recent_2d",      # 1
    "attention_recent_3d",      # 2
    "attention_mid_week",       # 3  — attention on ~5-day horizon
    "attention_mid_2wk",        # 4
    "attention_mid_month",      # 5
    "attention_long_6wk",       # 6
    "attention_long_2mo",       # 7
    "attention_long_quarter",   # 8
    "attention_decay_fast",     # 9
    "attention_decay_slow",     # 10
    "attention_regime_shift",   # 11 — attention spike at regime boundary
    "attention_trend_strength", # 12
    "attention_volatility",     # 13
    "attention_tail_15",        # 14
    "attention_tail_16",        # 15

    # 16-31: semantic-embedding region (populated by Clusters / BERTopic)
    "topic_share_dominant",     # 16 — fraction of docs in largest topic
    "topic_share_2nd",          # 17
    "topic_share_3rd",          # 18
    "topic_share_4th",          # 19
    "topic_share_5th",          # 20
    "topic_share_6th",          # 21
    "topic_share_minor",        # 22
    "topic_share_outlier",      # 23
    "encoder_var_imp_1",        # 24 — TFT encoder variable importance
    "encoder_var_imp_2",        # 25
    "encoder_var_imp_3",        # 26
    "encoder_var_imp_4",        # 27
    "decoder_var_imp_1",        # 28 — TFT decoder variable importance
    "decoder_var_imp_2",        # 29
    "decoder_var_imp_3",        # 30
    "decoder_var_imp_4",        # 31

    # 32-47: structural-centrality region (populated by Graph engine)
    "centrality_node_0",        # 32 — flattened node centrality (degree, betw, eig, pr)
    "centrality_node_1",        # 33
    "centrality_node_2",        # 34
    "centrality_node_3",        # 35
    "centrality_node_4",        # 36
    "centrality_node_5",        # 37
    "centrality_node_6",        # 38
    "centrality_node_7",        # 39
    "centrality_node_8",        # 40
    "centrality_node_9",        # 41
    "centrality_node_10",       # 42
    "centrality_node_11",       # 43
    "centrality_node_12",       # 44
    "centrality_node_13",       # 45
    "centrality_node_14",       # 46
    "centrality_node_15",       # 47

    # 48-63: dynamic-agent region (populated by Agent simulation)
    "graph_density",            # 48
    "graph_avg_clustering",     # 49
    "graph_n_communities",      # 50
    "agent_res_share_0",        # 51 — normalized resource share per agent
    "agent_res_share_1",        # 52
    "agent_res_share_2",        # 53
    "agent_res_share_3",        # 54
    "agent_res_share_4",        # 55
    "alliance_eigen_1",         # 56 — top eigenvalue of alliance matrix
    "alliance_eigen_2",         # 57
    "alliance_eigen_3",         # 58
    "alliance_eigen_4",         # 59
    "alliance_eigen_5",         # 60
    "alliance_eigen_6",         # 61
    "alliance_eigen_7",         # 62
    "alliance_eigen_8",         # 63
]

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
}
