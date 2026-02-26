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

NEWS_SNIPPETS: list[str] = [
    "USA and Britain reaffirm Five Eyes intelligence sharing after new cybersecurity threats from Russia and China.",
    "Russia and China sign expanded energy cooperation deal, deepening strategic partnership in the face of Western sanctions.",
    "India navigates balancing act between US Quad partnership and legacy Russian defence ties over S-400 deliveries.",
    "Brazil positions itself as key BRICS mediator, hosting summit on global financial architecture reform.",
    "Britain imposes fresh sanctions on Russian oligarchs while expanding Indo-Pacific trade partnerships.",
    "China's Belt and Road Initiative reaches Latin America as Brazil signs infrastructure investment framework.",
    "US-China trade tensions escalate with new semiconductor export controls targeting advanced chip manufacturing.",
    "Russia conducts joint naval exercises with China in the Pacific, signaling deepening military coordination.",
    "India and Britain finalize free trade agreement, strengthening post-Brexit Commonwealth economic ties.",
    "Brazil-India-South Africa trilateral dialogue advances climate finance and UN Security Council reform.",
    "USA strengthens Quad alliance with India, Japan, and Australia to counter Chinese maritime expansion.",
    "Russia's energy pivot to China accelerates as European gas imports drop to historic lows.",
    "Britain expands AUKUS defence technology sharing amid growing Chinese military presence in the South China Sea.",
    "China and India hold border talks after renewed tensions along the Line of Actual Control.",
    "Brazil emerges as swing vote in UN General Assembly on Ukraine-related resolutions, reflecting BRICS dynamics.",
    "USA and India sign nuclear cooperation agreement extending civil nuclear energy collaboration.",
    "Russia and Brazil expand agricultural trade routes bypassing Western financial infrastructure.",
    "China invests heavily in British technology startups despite political friction over Hong Kong and Taiwan.",
    "India launches independent satellite navigation system, reducing reliance on US GPS and Russian GLONASS.",
    "BRICS New Development Bank approves infrastructure loans for Brazil and India, challenging World Bank dominance.",
]

# UKT feature dimension (shared across all blocks)
UKT_FEATURE_DIM: int = 64

# Default tickers for finance block — country-representative ETFs / major stocks
DEFAULT_TICKERS: list[str] = ["SPY", "EWZ", "INDA"]

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
