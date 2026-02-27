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
.gov-flag {
    display: inline-block; padding: 3px 10px; margin: 3px 2px;
    border-radius: 8px; font-size: 0.78em; font-weight: 700;
    background: #4a2000; color: #ffaa00; border: 1px solid #aa6600;
    font-family: 'Fira Code', monospace;
}
.gov-pass {
    display: inline-block; padding: 3px 10px; margin: 3px 2px;
    border-radius: 8px; font-size: 0.78em; font-weight: 700;
    background: #0a2a1a; color: #64ffda; border: 1px solid #2d6a4f;
    font-family: 'Fira Code', monospace;
}
.jurisdiction-badge {
    display: inline-block; padding: 2px 8px; margin: 2px;
    border-radius: 10px; font-size: 0.72em; font-weight: 600;
    background: #1a1a3e; color: #a0aaff; border: 1px solid #3040a0;
}
.synthetic-banner {
    background: linear-gradient(90deg, #2a1500, #3a2000);
    border: 1px solid #aa5500; border-radius: 8px;
    padding: 10px 16px; margin: 8px 0; color: #ffcc88;
    font-weight: 600; font-size: 0.92em;
}
.governance-header {
    background: linear-gradient(135deg, #0a1628 0%, #0f2040 100%);
    border: 1px solid #1a4080; border-radius: 12px;
    padding: 20px 24px; margin: 8px 0;
}
.run-id-watermark {
    font-family: 'Fira Code', monospace; font-size: 0.70em;
    color: #445566; padding: 2px 6px;
}
.contest-note {
    background: #0d1a2e; border-left: 3px solid #ffaa00;
    padding: 8px 12px; margin: 4px 0; font-size: 0.84em;
    color: #c9d1d9; border-radius: 0 6px 6px 0;
}
.annotation-tag {
    display: inline-block; padding: 2px 8px; margin: 2px;
    border-radius: 10px; font-size: 0.72em; font-weight: 600;
    background: #2a1a3e; color: #cc99ff; border: 1px solid #6040a0;
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
        unit="/ 64",
        threshold=57,       # ≥90% of 64 features
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
        "(coded GOV-001 through GOV-005) and included in all exported reports."
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
}
