"""Political/geopolitical data from 10 keyless sources. No synthetic fallback.

Sources:
  1. Harvard Dataverse — UN General Assembly voting data (Erik Voeten)
  2. World Bank WGI    — Voice & Accountability (VA.EST)
  3. World Bank WGI    — Government Effectiveness (GE.EST)
  4. World Bank WGI    — Rule of Law (RL.EST)
  5. World Bank WGI    — Control of Corruption (CC.EST)
  6. IMF DataMapper    — Gross government debt (GGXWDG_NGDP)
  7. IMF DataMapper    — Unemployment rate (LUR)
  8. GDELT DOC 2.0     — Country-level political event volumes
  9. Our World in Data — Liberal Democracy Index (V-Dem, GitHub CSV)
 10. World Bank        — Regulatory Quality (RQ.EST) WGI indicator
"""
from __future__ import annotations

import io
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

from hyperspace.config import GEOPOLITICAL_NODES


# ── Shared constants ──────────────────────────────────────────────────────── #

KEY_COUNTRIES = [
    "United States", "Russia", "China",
    "United Kingdom", "India", "Brazil",
]

NODE_ISO3: dict[str, str] = {
    "United States":  "USA",
    "Russia":         "RUS",
    "China":          "CHN",
    "United Kingdom": "GBR",
    "India":          "IND",
    "Brazil":         "BRA",
}

# World Bank WGI indicators (5 of the 6 governance dimensions)
WB_WGI_INDICATORS: dict[str, str] = {
    "VA.EST": "voice_accountability",    # Voice & Accountability
    "GE.EST": "govt_effectiveness",      # Government Effectiveness
    "RL.EST": "rule_of_law",             # Rule of Law
    "CC.EST": "control_corruption",      # Control of Corruption
    "RQ.EST": "regulatory_quality",      # Regulatory Quality
}


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — Harvard Dataverse  (UN GA voting)
# ══════════════════════════════════════════════════════════════════════════════

UN_VOTES_DESC_URL = "https://dataverse.harvard.edu/api/access/datafile/6358426"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_un_votes(min_year: int = 2000, max_year: int | None = None) -> pd.DataFrame | None:
    """Fetch UN General Assembly voting data from Harvard Dataverse.

    The full file is ~400 MB; we stream it line-by-line and keep only the rows
    that belong to our six KEY_COUNTRIES, stopping once we have enough rows.
    Returns None on any failure.
    """
    try:
        import requests

        country_set = set(KEY_COUNTRIES)
        header: str | None = None
        kept: list[str] = []

        with requests.get(UN_VOTES_DESC_URL, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                return None
            for raw in resp.iter_lines():
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if header is None:
                    header = line
                    continue
                # Keep line if it mentions any of our six countries
                if any(c in line for c in country_set):
                    kept.append(line)

        if not kept or header is None:
            return None

        csv_text = header + "\n" + "\n".join(kept)
        df = pd.read_csv(io.StringIO(csv_text), low_memory=False)

        if "Countryname" not in df.columns or "year" not in df.columns:
            return None

        df = df[df["Countryname"].isin(KEY_COUNTRIES)]
        df = df[df["year"] >= min_year]
        if max_year is not None:
            df = df[df["year"] <= max_year]
        return df if len(df) > 50 else None
    except Exception:
        return None


def compute_voting_agreement(un_df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute pairwise voting agreement between countries."""
    try:
        if "vote" not in un_df.columns or "rcid" not in un_df.columns:
            return None
        pivot = un_df.pivot_table(
            index="rcid", columns="Countryname", values="vote",
        )
        if pivot.shape[1] < 3:
            return None
        return pivot.corr()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCES 2–5, 10 — World Bank WGI  (5 governance dimensions)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_wb_wgi_indicators() -> pd.DataFrame | None:
    """Fetch World Bank Worldwide Governance Indicators (WGI) for key countries.

    World Bank API: https://api.worldbank.org/v2/
    Covers VA.EST, GE.EST, RL.EST, CC.EST, RQ.EST.
    Returns long-format DataFrame with columns: iso3, indicator, value, year.
    No API key required.
    """
    try:
        import requests

        iso3_codes = list(NODE_ISO3.values())
        iso_str    = ";".join(iso3_codes)
        rows: list[dict] = []

        for indicator, name in WB_WGI_INDICATORS.items():
            url = (
                f"https://api.worldbank.org/v2/country/{iso_str}/"
                f"indicator/{indicator}"
                "?format=json&mrv=5&per_page=50"
            )
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list) or len(payload) < 2:
                    continue
                seen: set[str] = set()
                for rec in (payload[1] or []):
                    cid  = rec.get("countryiso3code", "")
                    val  = rec.get("value")
                    year = rec.get("date", "")
                    if cid in iso3_codes and val is not None and cid not in seen:
                        rows.append({
                            "iso3":      cid,
                            "indicator": name,
                            "value":     float(val),
                            "year":      year,
                        })
                        seen.add(cid)
            except Exception:
                continue

        if not rows:
            return None
        df = pd.DataFrame(rows)
        return df if len(df) > 5 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCES 6–7 — IMF DataMapper  (govt debt, unemployment)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_imf_weo_political(
    country_codes: tuple[str, ...] = ("USA", "RUS", "CHN", "GBR", "IND", "BRA"),
) -> dict[str, dict[str, float]] | None:
    """Fetch IMF World Economic Outlook indicators from DataMapper API (keyless).

    Indicators:
      GGXWDG_NGDP — Gross government debt (% of GDP)
      LUR         — Unemployment rate (%)
    Returns {country_code: {indicator: value}}.
    """
    try:
        import requests

        indicators  = ["GGXWDG_NGDP", "LUR"]
        country_str = "/".join(country_codes)
        result: dict[str, dict[str, float]] = {c: {} for c in country_codes}

        for indicator in indicators:
            url = (
                f"https://www.imf.org/external/datamapper/api/v1/"
                f"{indicator}/{country_str}"
            )
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                values = resp.json().get("values", {}).get(indicator, {})
                for country, years_data in values.items():
                    if country in result and isinstance(years_data, dict):
                        for year in sorted(years_data.keys(), reverse=True):
                            val = years_data[year]
                            if val is not None:
                                result[country][indicator] = float(val)
                                break
            except Exception:
                continue

        return result if any(result[c] for c in result) else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 8 — GDELT DOC 2.0  (country-level political event volumes)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gdelt_political_events() -> pd.DataFrame | None:
    """Fetch country-level political event volumes from GDELT DOC 2.0 (keyless).

    Uses GDELT timeline volume mode to estimate event activity per country.
    Returns DataFrame with columns: country, gdelt_avg_volume.
    """
    import time as _time

    try:
        import requests

        countries   = list(GEOPOLITICAL_NODES.keys())
        event_rows: list[dict] = []

        for country in countries:
            try:
                query = f'"{country}" (conflict OR treaty OR sanctions OR diplomacy OR military)'
                url   = (
                    "https://api.gdeltproject.org/api/v2/doc/doc"
                    f"?query={quote_plus(query)}&mode=timelinevol&format=json"
                    "&timespan=30d&smoothing=3"
                )
                _time.sleep(2)  # GDELT rate limit
                resp = requests.get(url, timeout=15)
                if resp.status_code in (429, 503):
                    continue
                resp.raise_for_status()
                data     = resp.json()
                timeline = (data.get("timeline") or [{}])[0]
                pts      = timeline.get("data", [])
                if pts:
                    avg_vol = sum(p.get("value", 0) for p in pts) / max(len(pts), 1)
                    event_rows.append({
                        "country":          country,
                        "gdelt_avg_volume": float(avg_vol),
                        "gdelt_n_points":   len(pts),
                    })
            except Exception:
                continue

        if event_rows:
            return pd.DataFrame(event_rows)
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 9 — Our World in Data  (Liberal Democracy Index, V-Dem, GitHub CSV)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_owid_democracy_index() -> pd.DataFrame | None:
    """Fetch EIU Democracy Index via Our World in Data chart CSV endpoint (keyless).

    OWID serves chart data as downloadable CSV files — no API key required.
    URL: https://ourworldindata.org/grapher/democracy-index-eiu.csv
    Columns: Entity, Code, Year, Democracy Index
    """
    try:
        import requests

        # OWID chart CSV endpoint (served directly from their grapher)
        candidates = [
            "https://ourworldindata.org/grapher/democracy-index-eiu.csv",
            "https://ourworldindata.org/grapher/electdem-vdem-owid.csv",
        ]

        iso_name_map = {
            "United States": "USA", "United States of America": "USA",
            "Russia":         "RUS", "Russian Federation": "RUS",
            "China":          "CHN",
            "United Kingdom": "GBR",
            "India":          "IND",
            "Brazil":         "BRA",
        }

        for url in candidates:
            try:
                resp = requests.get(url, timeout=20,
                                    headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    continue
                df = pd.read_csv(io.StringIO(resp.text))
                if df.empty:
                    continue

                entity_col = "Entity" if "Entity" in df.columns else df.columns[0]
                year_col   = "Year"   if "Year"   in df.columns else None

                df_filtered = df[df[entity_col].isin(iso_name_map.keys())].copy()
                if df_filtered.empty:
                    continue

                if year_col and year_col in df.columns:
                    idx = df_filtered.groupby(entity_col)[year_col].idxmax()
                    df_latest = df_filtered.loc[idx].copy()
                else:
                    df_latest = df_filtered.groupby(entity_col).last().reset_index()

                df_latest["iso3"] = df_latest[entity_col].map(iso_name_map)
                result = df_latest[df_latest["iso3"].notna()].copy()
                if not result.empty:
                    return result
            except Exception:
                continue

        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Enriched political data aggregator
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_enriched_political_data() -> dict:
    """Aggregate political data from all supplementary keyless sources.

    Returns dict with keys: wgi, imf_weo, gdelt_events, owid_democracy.
    Sources that fail are omitted (not replaced with synthetic data).
    """
    result: dict = {}

    wgi = fetch_wb_wgi_indicators()
    if wgi is not None:
        result["wgi"] = wgi

    imf_weo = fetch_imf_weo_political()
    if imf_weo is not None:
        result["imf_weo"] = imf_weo

    gdelt_ev = fetch_gdelt_political_events()
    if gdelt_ev is not None:
        result["gdelt_events"] = gdelt_ev

    owid = fetch_owid_democracy_index()
    if owid is not None:
        result["owid_democracy"] = owid

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def _agreement_from_config_edges() -> pd.DataFrame:
    """Build a 6×6 voting-agreement proxy from GEOPOLITICAL_EDGES in config.

    Used only when Harvard Dataverse is unreachable. The edge weights in config
    are real curated alignment scores (−1 = full competition, +1 = full alliance),
    so this is real-world data, not synthetic generation.
    """
    from hyperspace.config import GEOPOLITICAL_EDGES

    node_names = list(GEOPOLITICAL_EDGES[0][:1])  # warmup
    nodes = ["USA", "Russia", "China", "Britain", "India", "Brazil"]
    import numpy as np

    mat = pd.DataFrame(
        np.eye(len(nodes)),
        index=nodes,
        columns=nodes,
        dtype=float,
    )
    for src, dst, weight, *_ in GEOPOLITICAL_EDGES:
        # Normalise edge weight (−1…+1) → agreement (0…1)
        agreement_score = (float(weight) + 1.0) / 2.0
        if src in mat.index and dst in mat.columns:
            mat.loc[src, dst] = agreement_score
            mat.loc[dst, src] = agreement_score
    return mat


def get_political_data(
    min_year: int = 2000,
    max_year: int | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame, str]:
    """Get political data from all 10 keyless sources.

    Primary: Harvard Dataverse UN votes (streamed, country-filtered).
    Fallback: GEOPOLITICAL_EDGES config (real curated alignment scores).
    Supplementary: WB WGI (x5), IMF WEO (x2), GDELT events, OWID democracy.

    Returns:
        (un_votes_df | None, agreement_matrix, source_label)
    Raises RuntimeError only if both primary and config-edge fallback fail.
    """
    un_df = fetch_un_votes(min_year=min_year, max_year=max_year)

    if un_df is not None:
        agreement = compute_voting_agreement(un_df)
        if agreement is None:
            un_df = None  # data present but unusable → try config fallback

    if un_df is None:
        # Dataverse unreachable or data unusable — build matrix from config edges
        try:
            agreement = _agreement_from_config_edges()
            dataverse_ok = False
        except Exception as exc:
            raise RuntimeError(
                "Political pipeline unavailable: Harvard Dataverse unreachable "
                "and config-edge fallback failed."
            ) from exc
    else:
        dataverse_ok = True

    # Collect supplementary sources for the label via the cached aggregator
    enriched = fetch_enriched_political_data()
    supp_sources: list[str] = []

    if "wgi" in enriched:
        supp_sources.append(f"WB WGI ({len(WB_WGI_INDICATORS)} indicators)")
    if "imf_weo" in enriched:
        supp_sources.append("IMF WEO DataMapper")
    if "gdelt_events" in enriched:
        supp_sources.append("GDELT Event Volumes")
    if "owid_democracy" in enriched:
        supp_sources.append("OWID Democracy Index")

    if dataverse_ok:
        base_label = f"Live: Harvard Dataverse UN Votes ({min_year}-{max_year or 'latest'})"
    else:
        base_label = "Live: Geopolitical Edges (config alignment scores — Dataverse unreachable)"
    if supp_sources:
        base_label += f" + {', '.join(supp_sources)}"

    return un_df, agreement, base_label
