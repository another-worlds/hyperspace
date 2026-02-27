"""Geographic map data: REST Countries API (keyless, public)."""
from __future__ import annotations

import streamlit as st

# ISO-3166 alpha-3 codes for the 6 geopolitical nodes
NODE_ISO3: dict[str, str] = {
    "USA":     "USA",
    "Russia":  "RUS",
    "China":   "CHN",
    "Britain": "GBR",
    "India":   "IND",
    "Brazil":  "BRA",
}

REST_COUNTRIES_URL = "https://restcountries.com/v3.1/alpha"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_country_stats() -> dict[str, dict] | None:
    """Fetch live country metadata from REST Countries API.

    Returns dict keyed by node name (USA, Russia, etc.) with fields:
        population, area_km2, capital, region, subregion, un_member, flag_emoji
    Returns None on failure.
    """
    try:
        import requests

        codes = ",".join(NODE_ISO3.values())
        resp = requests.get(
            f"{REST_COUNTRIES_URL}?codes={codes}&fields=name,cca3,capital,region,"
            "subregion,population,area,unMember,flag",
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()

        iso3_to_node = {v: k for k, v in NODE_ISO3.items()}
        stats: dict[str, dict] = {}
        for entry in raw:
            cca3 = entry.get("cca3", "")
            node = iso3_to_node.get(cca3)
            if not node:
                continue
            capitals = entry.get("capital", [])
            stats[node] = {
                "population":  entry.get("population", 0),
                "area_km2":    entry.get("area", 0),
                "capital":     capitals[0] if capitals else "N/A",
                "region":      entry.get("region", ""),
                "subregion":   entry.get("subregion", ""),
                "un_member":   entry.get("unMember", False),
                "flag_emoji":  entry.get("flag", ""),
                "iso3":        cca3,
            }
        return stats if len(stats) == len(NODE_ISO3) else None
    except Exception:
        return None


def get_country_stats() -> tuple[dict[str, dict], str]:
    """Get country stats. Raises RuntimeError if API unavailable."""
    stats = fetch_country_stats()
    if stats:
        return stats, "Live: REST Countries API"
    raise RuntimeError("REST Countries API unavailable. Check network connectivity.")
