"""Political/geopolitical data: UN votes from Harvard Dataverse with fallback."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.config import GEOPOLITICAL_EDGES, GEOPOLITICAL_NODES


# Harvard Dataverse: Erik Voeten UN General Assembly Voting Data
# This is the "descriptions" file which is smaller and more manageable
UN_VOTES_DESC_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/6358426"
)

# Key countries we care about (matching GEOPOLITICAL_NODES)
KEY_COUNTRIES = [
    "United States", "Russia", "China",
    "United Kingdom", "India", "Brazil",
]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_un_votes(min_year: int = 2000, max_year: int | None = None) -> pd.DataFrame | None:
    """Fetch UN General Assembly voting data. Returns None on failure.

    Falls back gracefully if Harvard Dataverse is unreachable.
    """
    try:
        import requests
        resp = requests.get(UN_VOTES_DESC_URL, timeout=15)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        # Filter to key countries and recent years
        if "Countryname" in df.columns and "year" in df.columns:
            df = df[df["Countryname"].isin(KEY_COUNTRIES)]
            df = df[df["year"] >= min_year]
            if max_year is not None:
                df = df[df["year"] <= max_year]
            return df if len(df) > 50 else None
        return None
    except Exception:
        return None


def compute_voting_agreement(un_df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute pairwise voting agreement between countries.

    Returns a correlation matrix (country x country).
    """
    try:
        if "vote" not in un_df.columns or "rcid" not in un_df.columns:
            return None
        pivot = un_df.pivot_table(
            index="rcid", columns="Countryname", values="vote",
        )
        if pivot.shape[1] < 3:
            return None
        agreement = pivot.corr()
        return agreement
    except Exception:
        return None


def build_synthetic_agreement() -> pd.DataFrame:
    """Build a synthetic agreement matrix from hardcoded geopolitical edges."""
    nodes = list(GEOPOLITICAL_NODES.keys())
    n = len(nodes)
    mat = np.eye(n)
    node_idx = {name: i for i, name in enumerate(nodes)}
    for src, dst, w, _, _ in GEOPOLITICAL_EDGES:
        if src in node_idx and dst in node_idx:
            mat[node_idx[src], node_idx[dst]] = w
            mat[node_idx[dst], node_idx[src]] = w
    return pd.DataFrame(mat, index=nodes, columns=nodes)


def get_political_data(min_year: int = 2000, max_year: int | None = None) -> tuple[pd.DataFrame | None, pd.DataFrame, str]:
    """Get political data with fallback.

    Returns:
        (un_votes_df_or_none, agreement_matrix, source_label)
    """
    un_df = fetch_un_votes(min_year=min_year, max_year=max_year)
    if un_df is not None:
        agreement = compute_voting_agreement(un_df)
        if agreement is not None:
            return un_df, agreement, f"Live: Harvard Dataverse UN Votes ({min_year}-{max_year or 'latest'})"

    # Fallback to synthetic
    return None, build_synthetic_agreement(), "Fallback: synthetic agreement"
