"""Shared fixtures for Hyperspace integration tests.

All fixtures generate synthetic data so tests run offline without
external API calls or Streamlit session state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hyperspace.config import GEOPOLITICAL_NODES, UKT_FEATURE_DIM


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_finance_features(rng) -> tuple[np.ndarray, dict]:
    """80-dim feature vector mimicking TFT output (slots 0-31)."""
    features = np.zeros(UKT_FEATURE_DIM)
    # Attention weights (0-15)
    features[:16] = rng.uniform(0, 1, 16)
    # Encoder importance (16-23)
    features[16:24] = rng.uniform(0, 0.5, 8)
    # Decoder importance (24-26)
    features[24:27] = rng.uniform(0, 0.3, 3)
    # Macro features (27-31)
    features[27:32] = rng.uniform(0.2, 0.8, 5)

    meta = {
        i: {"label": f"tft_attention_lag_{i+1:02d}", "block": "finance",
            "metric": "attention_weight", "source": "synthetic"}
        for i in range(16)
    }
    return features, meta


@pytest.fixture()
def synthetic_cluster_features(rng) -> tuple[np.ndarray, dict]:
    """80-dim feature vector mimicking BERTopic output (slots 16-31)."""
    features = np.zeros(UKT_FEATURE_DIM)
    # Topic shares (16-23)
    shares = rng.dirichlet(np.ones(8))
    features[16:24] = shares
    # Embedding stats (24-31)
    features[24:32] = rng.normal(0, 1, 8)

    meta = {
        16 + i: {"label": f"topic_share_{i}", "block": "clusters",
                 "metric": "topic_distribution_share", "source": "synthetic"}
        for i in range(8)
    }
    return features, meta


@pytest.fixture()
def synthetic_agreement_matrix() -> pd.DataFrame:
    """6x6 pairwise agreement matrix for geopolitical nodes."""
    nodes = list(GEOPOLITICAL_NODES.keys())
    n = len(nodes)
    rng = np.random.default_rng(99)
    mat = rng.uniform(-0.5, 1.0, (n, n))
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1.0)
    return pd.DataFrame(mat, index=nodes, columns=nodes)


@pytest.fixture()
def synthetic_spatial_data(rng) -> dict:
    """Synthetic spatial raster + country scalars."""
    physical_raster = rng.uniform(0, 1, (4, 6, 9))
    country_scalars = rng.uniform(0, 1, (10, 6))
    return dict(
        physical_raster=physical_raster,
        country_scalars=country_scalars,
        scalar_names=[
            "elevation", "temperature", "humidity", "precipitation",
            "gdp_ppp", "debt_pct_gdp", "military_pct_gdp",
            "tertiary_enroll", "political_stability", "homicide_rate",
        ],
        node_order=["USA", "Russia", "China", "Britain", "India", "Brazil"],
    )


@pytest.fixture()
def timeframe_context() -> dict:
    return {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "min_year": 2025,
        "max_year": 2025,
    }
