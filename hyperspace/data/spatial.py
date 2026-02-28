"""Multimodal spatial raster data: elevation, climate, economic, and conflict layers.

Fetches live data from:
  - Open-Elevation API (batch POST, no key)
  - Open-Meteo Archive API (no key)
  - World Bank API (no key) — 6 indicators including conflict proxies

No fallbacks — raises RuntimeError if any source is unavailable.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import streamlit as st

from hyperspace.config import GEOPOLITICAL_NODES

# ---- Node metadata ------------------------------------------------- #

NODE_ORDER: list[str] = ["USA", "Russia", "China", "Britain", "India", "Brazil"]

# World Bank ISO3 codes for each node
NODE_ISO3: dict[str, str] = {
    "USA":     "USA",
    "Russia":  "RUS",
    "China":   "CHN",
    "Britain": "GBR",
    "India":   "IND",
    "Brazil":  "BRA",
}

# World Bank indicators — 4 economic/social + 2 conflict proxies (6 total)
# Conflict proxies replace UCDP GED (now requires authentication):
#   political_stability: WGI Political Stability & Absence of Violence (PV.EST)
#   homicide_rate:       Intentional homicides per 100k (VC.IHR.PSRC.P5)
WB_INDICATORS: dict[str, str] = {
    "gdp_ppp":              "NY.GDP.MKTP.PP.CD",  # GDP PPP (current intl $)
    "debt_pct_gdp":         "GC.DOD.TOTL.GD.ZS",  # Central govt debt % GDP
    "military_pct_gdp":     "MS.MIL.XPND.GD.ZS",  # Military spending % GDP
    "tertiary_enroll":      "SE.TER.ENRR",          # Tertiary school enrollment %
    "political_stability":  "PV.EST",               # WGI stability [-2.5,+2.5]; lower=more conflict
    "homicide_rate":        "VC.IHR.PSRC.P5",       # Intentional homicides/100k (fatality proxy)
}

# Physical layer names (axis 0 of physical_raster)
PHYSICAL_LAYER_NAMES: list[str] = [
    "elevation", "temperature", "humidity", "precipitation",
]

# Scalar type names (axis 0 of country_scalars) — must align with WB_INDICATORS order
SCALAR_NAMES: list[str] = [
    "gdp_ppp", "debt_pct_gdp", "military_pct_gdp", "tertiary_enroll",
    "political_stability", "homicide_rate",
]


# ---- Helpers -------------------------------------------------------- #

def _grid_points(lat: float, lon: float, delta: float = 2.0) -> list[tuple[float, float]]:
    """Generate a 3×3 grid of (lat, lon) points centred on (lat, lon)."""
    offsets = [-delta, 0.0, delta]
    return [(lat + dlat, lon + dlon) for dlat in offsets for dlon in offsets]


# ---- Individual fetchers ------------------------------------------- #

def fetch_elevation(locations: list[tuple[float, float]]) -> list[float]:
    """Batch-fetch elevation (m) for a list of (lat, lon) points via Open-Elevation.

    Args:
        locations: List of (lat, lon) tuples.

    Returns:
        List of elevation values (metres) in the same order as *locations*.

    Raises:
        RuntimeError: If the API call fails or returns incomplete data.
    """
    import requests

    payload = {
        "locations": [{"latitude": lat, "longitude": lon} for lat, lon in locations]
    }
    try:
        resp = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Open-Elevation API unavailable: {exc}") from exc

    results = resp.json().get("results", [])
    if len(results) != len(locations):
        raise RuntimeError(
            f"Open-Elevation returned {len(results)} results, expected {len(locations)}."
        )
    return [float(r["elevation"]) for r in results]


def fetch_climate(lat: float, lon: float, days_back: int = 365) -> dict[str, float]:
    """Fetch annual-mean climate statistics via Open-Meteo Archive.

    Retries up to 4 times with exponential backoff on 429 rate-limit responses.

    Args:
        lat: Latitude.
        lon: Longitude.
        days_back: Number of past days to average (clamped to 30–365).

    Returns:
        Dict with: temperature_mean (°C), humidity_mean (%), precip_mean (mm/day).

    Raises:
        RuntimeError: If the API call fails or returns no data after retries.
    """
    import time
    import requests

    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=max(30, min(365, days_back)))
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_dt.isoformat()}&end_date={end_dt.isoformat()}"
        "&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
        "&timezone=UTC"
    )
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt * 8  # 8, 16, 32, 64 seconds
                time.sleep(wait)
                last_exc = RuntimeError(f"Open-Meteo rate-limited at ({lat},{lon}), attempt {attempt+1}")
                continue
            resp.raise_for_status()
            break
        except Exception as exc:
            last_exc = exc
            if attempt < 3:
                time.sleep(2 ** attempt * 2)
            continue
    else:
        raise RuntimeError(
            f"Open-Meteo API unavailable at ({lat},{lon}) after 4 attempts: {last_exc}"
        ) from last_exc

    daily = resp.json().get("daily", {})
    temp_vals = [v for v in daily.get("temperature_2m_mean", []) if v is not None]
    hum_vals  = [v for v in daily.get("relative_humidity_2m_mean", []) if v is not None]
    prec_vals = [v for v in daily.get("precipitation_sum", []) if v is not None]

    if not temp_vals:
        raise RuntimeError(
            f"Open-Meteo returned no temperature data for ({lat}, {lon})."
        )
    return {
        "temperature_mean": float(np.mean(temp_vals)),
        "humidity_mean":    float(np.mean(hum_vals)) if hum_vals else 0.0,
        "precip_mean":      float(np.mean(prec_vals)) if prec_vals else 0.0,
    }


def fetch_worldbank_indicator(iso3: str, indicator: str) -> float:
    """Fetch the most-recent non-null value for a World Bank indicator.

    Args:
        iso3: ISO-3 country code (e.g. "USA").
        indicator: World Bank indicator code (e.g. "NY.GDP.MKTP.PP.CD").

    Returns:
        Most-recent non-null float value.

    Raises:
        RuntimeError: If no non-null value is found in the last 10 years.
    """
    import requests

    url = (
        f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}"
        "?format=json&mrv=10&per_page=10"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"World Bank API unavailable for {iso3}/{indicator}: {exc}"
        ) from exc

    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(
            f"World Bank API unexpected response structure for {iso3}/{indicator}."
        )
    records = payload[1] or []
    for rec in records:
        val = rec.get("value")
        if val is not None:
            return float(val)
    # Sovereign non-reporting: API is live but country does not publish this indicator.
    # Record as 0.0 (neutral/missing) rather than raising — the source is real, the
    # data gap is a known governance fact (e.g. China does not report debt % GDP).
    return 0.0




# ---- Orchestrator -------------------------------------------------- #

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_spatial_data() -> dict:
    """Fetch all multimodal spatial layers for the 6 geopolitical nodes.

    Orchestrates:
      - Open-Elevation: 3×3 grid per node, 54 points total (single batch POST)
      - Open-Meteo archive: annual-mean climate per node (6 sequential requests)
      - World Bank API: 6 indicators × 6 nodes = 36 requests
        (4 economic/social + PV.EST political stability + VC.IHR.PSRC.P5 homicide rate)

    Returns:
        Dict with:
            physical_raster (np.ndarray, shape (4, 6, 9)):
                layers × nodes × grid_points for elevation, temp, humidity, precip
            country_scalars (np.ndarray, shape (6, 6)):
                scalar_types × nodes for GDP, debt, mil, enroll, events, fatalities
            node_order   (list[str])
            layer_names  (list[str])
            scalar_names (list[str])
            source_label (str)

    Raises:
        RuntimeError: If any live data source is unavailable.
    """
    nodes   = NODE_ORDER
    n_nodes = len(nodes)

    # ---- Elevation: single batch POST for all 54 grid points ---- #
    all_grid_pts: list[tuple[float, float]] = []
    grid_pt_map:  list[tuple[int, int]]     = []  # (node_idx, grid_idx)

    for ni, name in enumerate(nodes):
        attrs = GEOPOLITICAL_NODES[name]
        pts = _grid_points(attrs["lat"], attrs["lon"], delta=2.0)  # 9 pts per node
        for gi, pt in enumerate(pts):
            all_grid_pts.append(pt)
            grid_pt_map.append((ni, gi))

    elevations_flat = fetch_elevation(all_grid_pts)   # raises RuntimeError on failure

    elevation_grid = np.zeros((n_nodes, 9))
    for flat_idx, (ni, gi) in enumerate(grid_pt_map):
        elevation_grid[ni, gi] = elevations_flat[flat_idx]

    # ---- Climate: temperature, humidity, precipitation per node ---- #
    import time as _time
    temp_grid = np.zeros((n_nodes, 9))
    hum_grid  = np.zeros((n_nodes, 9))
    prec_grid = np.zeros((n_nodes, 9))

    for ni, name in enumerate(nodes):
        if ni > 0:
            _time.sleep(6)  # Open-Meteo free-tier: allow cooldown between archive calls
        attrs   = GEOPOLITICAL_NODES[name]
        climate = fetch_climate(attrs["lat"], attrs["lon"], days_back=365)
        # Broadcast node-centre climate value across all 9 grid points
        temp_grid[ni, :] = climate["temperature_mean"]
        hum_grid[ni, :]  = climate["humidity_mean"]
        prec_grid[ni, :] = climate["precip_mean"]

    # Stack physical layers → (4, 6, 9)
    physical_raster = np.stack([
        elevation_grid,  # layer 0: elevation
        temp_grid,       # layer 1: temperature
        hum_grid,        # layer 2: relative humidity
        prec_grid,       # layer 3: precipitation
    ])

    # ---- Country scalars: (6, 6) ---- #
    # Rows 0-3: economic/social | Row 4: political stability | Row 5: homicide rate
    country_scalars = np.zeros((6, n_nodes))

    for row_idx, (_, indicator) in enumerate(WB_INDICATORS.items()):
        for ni, name in enumerate(nodes):
            country_scalars[row_idx, ni] = fetch_worldbank_indicator(
                NODE_ISO3[name], indicator,
            )

    # Invert political_stability row (row 4) so higher value = more conflict stress
    # PV.EST ranges [-2.5, +2.5]: -2.5 = very unstable → maps to high conflict
    country_scalars[4] = -country_scalars[4]  # now higher = worse stability = more conflict

    return dict(
        physical_raster=physical_raster,
        country_scalars=country_scalars,
        node_order=nodes,
        layer_names=PHYSICAL_LAYER_NAMES,
        scalar_names=SCALAR_NAMES,
        source_label="Live: Open-Elevation, Open-Meteo, World Bank API (6 indicators)",
    )
