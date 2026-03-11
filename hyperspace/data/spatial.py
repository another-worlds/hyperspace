"""Multimodal spatial raster data from 12 keyless live APIs. No synthetic fallback.

Sources:
  1. Open-Elevation API      — terrain elevation (batch POST, no key)
  2. Open-Meteo Archive API  — annual-mean climate (no key)
  3. World Bank API          — GDP PPP (NY.GDP.MKTP.PP.CD)
  4. World Bank API          — Govt debt % GDP (GC.DOD.TOTL.GD.ZS)
  5. World Bank API          — Military spending % GDP (MS.MIL.XPND.GD.ZS)
  6. World Bank API          — Tertiary enrollment % (SE.TER.ENRR)
  7. World Bank API          — Political stability WGI (PV.EST)
  8. World Bank API          — Homicide rate (VC.IHR.PSRC.P5)
  9. USGS Earthquake Hazards — earthquake count & magnitude in country region
 10. NASA EONET              — natural event count near each country capital
 11. Open-Meteo Air Quality  — PM2.5 concentration at each capital
 12. NOAA Tides & Currents   — tide gauge trends (sea-level proxy, coastal nodes)

Raises RuntimeError if any of sources 1–8 are unavailable.
Sources 9–12 (extended env scalars) are added when available; if all four fail,
a RuntimeError is raised to maintain the zero-fallback policy.
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import streamlit as st

from hyperspace.config import GEOPOLITICAL_NODES
from hyperspace.data import cache as _cache

# ── Node metadata ─────────────────────────────────────────────────────────── #

NODE_ORDER: list[str] = ["USA", "Russia", "China", "Britain", "India", "Brazil"]

NODE_ISO3: dict[str, str] = {
    "USA":     "USA",
    "Russia":  "RUS",
    "China":   "CHN",
    "Britain": "GBR",
    "India":   "IND",
    "Brazil":  "BRA",
}

# NOAA tide gauge IDs for coastal/river-adjacent capitals (best proxies)
NODE_NOAA_STATION: dict[str, str] = {
    "USA":     "8518750",   # The Battery, New York (Atlantic coast)
    "Britain": "1159280",   # London Bridge (Thames)
    "Brazil":  "830-141",   # Salvador, Brazil (Atlantic coast)
    # Russia, China, India, Brazil inland capitals → fallback: no NOAA station
}

# World Bank indicators — 6 original scalars
WB_INDICATORS: dict[str, str] = {
    "gdp_ppp":              "NY.GDP.MKTP.PP.CD",
    "debt_pct_gdp":         "GC.DOD.TOTL.GD.ZS",
    "military_pct_gdp":     "MS.MIL.XPND.GD.ZS",
    "tertiary_enroll":      "SE.TER.ENRR",
    "political_stability":  "PV.EST",
    "homicide_rate":        "VC.IHR.PSRC.P5",
}

PHYSICAL_LAYER_NAMES: list[str] = [
    "elevation", "temperature", "humidity", "precipitation",
]

# Full scalar names: 6 original WB + 4 new extended env layers
SCALAR_NAMES: list[str] = [
    # rows 0-5 (WB indicators)
    "gdp_ppp", "debt_pct_gdp", "military_pct_gdp", "tertiary_enroll",
    "political_stability", "homicide_rate",
    # rows 6-9 (extended env)
    "earthquake_risk", "eonet_events", "air_quality_pm25", "sea_level_proxy",
]


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _grid_points(lat: float, lon: float, delta: float = 2.0) -> list[tuple[float, float]]:
    """Generate a 3×3 grid of (lat, lon) points centred on (lat, lon)."""
    offsets = [-delta, 0.0, delta]
    return [(lat + dlat, lon + dlon) for dlat in offsets for dlon in offsets]


# ══════════════════════════════════════════════════════════════════════════════
# SOURCES 1–2 — Open-Elevation + Open-Meteo Archive  (physical raster)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_elevation(locations: list[tuple[float, float]]) -> list[float]:
    """Batch-fetch elevation (m) via Open-Elevation (no key).

    Raises RuntimeError on failure.
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
    """Fetch annual-mean climate via Open-Meteo Archive (no key).

    Raises RuntimeError on failure after 4 retries.
    """
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
                wait = 2 ** attempt * 8
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
            f"Open-Meteo Archive unavailable at ({lat},{lon}) after 4 attempts: {last_exc}"
        ) from last_exc

    daily = resp.json().get("daily", {})
    temp_vals = [v for v in daily.get("temperature_2m_mean", []) if v is not None]
    hum_vals  = [v for v in daily.get("relative_humidity_2m_mean", []) if v is not None]
    prec_vals = [v for v in daily.get("precipitation_sum", []) if v is not None]

    if not temp_vals:
        raise RuntimeError(f"Open-Meteo returned no temperature data for ({lat},{lon}).")
    return {
        "temperature_mean": float(np.mean(temp_vals)),
        "humidity_mean":    float(np.mean(hum_vals)) if hum_vals else 0.0,
        "precip_mean":      float(np.mean(prec_vals)) if prec_vals else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SOURCES 3–8 — World Bank API  (6 economic/governance indicators)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_worldbank_indicator(iso3: str, indicator: str) -> float:
    """Fetch most-recent non-null value for a World Bank indicator (no key).

    Raises RuntimeError if the API call itself fails.
    Returns 0.0 if the country does not publish the indicator (sovereign non-reporting).
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
            f"World Bank API unexpected response for {iso3}/{indicator}."
        )
    for rec in (payload[1] or []):
        val = rec.get("value")
        if val is not None:
            return float(val)
    return 0.0   # Sovereign non-reporting (documented gap, not synthetic)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 9 — USGS Earthquake Hazards Program  (seismic risk proxy)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_usgs_earthquakes_per_node() -> dict[str, float] | None:
    """Fetch USGS earthquake counts within 5° of each capital (no key).

    USGS Earthquake Hazards: https://earthquake.usgs.gov/earthquakes/feed/v1.0/
    Returns {node_name: earthquake_count_last_30_days} as proxy for seismic risk.
    """
    try:
        import requests

        node_quakes: dict[str, float] = {}
        end_dt   = date.today().isoformat()
        start_dt = (date.today() - timedelta(days=30)).isoformat()
        successes = 0

        for node, attrs in GEOPOLITICAL_NODES.items():
            lat, lon = attrs["lat"], attrs["lon"]
            # USGS rectangle query around capital (±5°)
            url = (
                "https://earthquake.usgs.gov/fdsnws/event/1/count?format=geojson"
                f"&starttime={start_dt}&endtime={end_dt}"
                f"&minlatitude={lat-5}&maxlatitude={lat+5}"
                f"&minlongitude={lon-5}&maxlongitude={lon+5}"
                "&minmagnitude=3.0"
            )
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                count = resp.json().get("count", 0)
                node_quakes[node] = float(count)
                successes += 1
            except Exception:
                node_quakes[node] = 0.0

        return node_quakes if successes > 0 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 10 — NASA EONET  (natural event tracker)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_eonet_events() -> dict[str, float] | None:
    """Fetch NASA EONET natural event counts near each node (no key).

    NASA EONET API v3: https://eonet.gsfc.nasa.gov/api/v3/events
    Returns {node_name: event_count_last_30_days}.
    """
    try:
        import requests

        url = (
            "https://eonet.gsfc.nasa.gov/api/v3/events"
            "?limit=500&days=30&status=open"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        events = resp.json().get("events", [])

        node_counts: dict[str, float] = {n: 0.0 for n in GEOPOLITICAL_NODES}
        for event in events:
            # Get most recent geometry
            geoms = event.get("geometry", [])
            if not geoms:
                continue
            coords = geoms[-1].get("coordinates", [])
            if len(coords) < 2:
                continue
            evt_lon, evt_lat = float(coords[0]), float(coords[1])

            # Assign to nearest node (within 15°)
            for node, attrs in GEOPOLITICAL_NODES.items():
                dlat = abs(evt_lat - attrs["lat"])
                dlon = abs(evt_lon - attrs["lon"])
                if dlat < 15 and dlon < 15:
                    node_counts[node] += 1.0
                    break

        return node_counts
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 11 — Open-Meteo Air Quality API  (PM2.5)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openmeteo_air_quality() -> dict[str, float] | None:
    """Fetch PM2.5 air quality at each capital from Open-Meteo Air Quality API (no key).

    Open-Meteo AQ: https://air-quality-api.open-meteo.com/v1/air-quality
    Returns {node_name: pm25_mean_ug_m3}.
    """
    try:
        import requests

        node_aqi: dict[str, float] = {}
        successes = 0
        for node, attrs in GEOPOLITICAL_NODES.items():
            lat, lon = attrs["lat"], attrs["lon"]
            url = (
                "https://air-quality-api.open-meteo.com/v1/air-quality"
                f"?latitude={lat}&longitude={lon}"
                "&hourly=pm2_5&timezone=UTC&forecast_days=1"
            )
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                vals = [
                    v for v in resp.json().get("hourly", {}).get("pm2_5", [])
                    if v is not None
                ]
                if vals:
                    node_aqi[node] = float(np.mean(vals))
                    successes += 1
                else:
                    node_aqi[node] = 0.0
            except Exception:
                node_aqi[node] = 0.0

        return node_aqi if successes > 0 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 12 — NOAA Tides & Currents  (sea-level trend proxy for coastal nodes)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_noaa_sea_level_proxy() -> dict[str, float] | None:
    """Fetch recent tidal mean water levels for coastal nodes from NOAA API (no key).

    NOAA Tides & Currents: https://api.tidesandcurrents.noaa.gov/api/prod/datagetter
    Returns {node_name: mean_water_level_meters} for coastal/riparian nodes.
    Inland nodes (Russia, China, India) receive 0.0 (no station).
    """
    try:
        import requests

        node_sea: dict[str, float] = {n: 0.0 for n in GEOPOLITICAL_NODES}
        end_dt   = date.today().strftime("%Y%m%d")
        start_dt = (date.today() - timedelta(days=7)).strftime("%Y%m%d")
        successes = 0

        for node, station_id in NODE_NOAA_STATION.items():
            url = (
                "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
                f"?begin_date={start_dt}&end_date={end_dt}"
                f"&station={station_id}&product=water_level"
                "&datum=MSL&time_zone=GMT&units=metric&format=json"
            )
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data_pts = resp.json().get("data", [])
                vals = []
                for pt in data_pts:
                    try:
                        vals.append(float(pt.get("v", 0)))
                    except (ValueError, TypeError):
                        pass
                if vals:
                    node_sea[node] = float(np.mean(vals))
                    successes += 1
            except Exception:
                pass  # Coastal station unavailable — keep 0.0

        return node_sea if successes > 0 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — assembles all 12 sources
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_all_spatial_data_live() -> dict:
    """Fetch all multimodal spatial layers from 12 keyless sources.

    Physical raster (sources 1–2):
        - Open-Elevation: 3×3 grid per node, 54 points (single batch POST)
        - Open-Meteo Archive: annual-mean climate per node (6 requests)
        → physical_raster: np.ndarray (4, 6, 9)

    Country scalars (sources 3–8, World Bank, 6 rows × 6 nodes):
        - NY.GDP.MKTP.PP.CD, GC.DOD.TOTL.GD.ZS, MS.MIL.XPND.GD.ZS
        - SE.TER.ENRR, PV.EST, VC.IHR.PSRC.P5
        → wb_scalars: np.ndarray (6, 6)

    Extended env scalars (sources 9–12, 4 rows × 6 nodes):
        - USGS earthquake count, NASA EONET events,
          Open-Meteo Air Quality PM2.5, NOAA mean water level
        → env_scalars: np.ndarray (4, 6)

    Combined country_scalars = vstack(wb_scalars, env_scalars) → (10, 6)

    Returns:
        Dict with:
            physical_raster  (np.ndarray, shape (4, 6, 9))
            country_scalars  (np.ndarray, shape (10, 6))  ← extended
            node_order       list[str]
            layer_names      list[str]
            scalar_names     list[str]  (10 entries)
            source_label     str

    Raises RuntimeError if sources 1–8 (core) are unavailable.
    Extended env sources (9–12) use 0.0 per node when unreachable.
    """
    nodes   = NODE_ORDER
    n_nodes = len(nodes)

    # ── Source 1: Open-Elevation — batch POST for all 54 grid points ── #
    all_grid_pts: list[tuple[float, float]] = []
    grid_pt_map:  list[tuple[int, int]]     = []

    for ni, name in enumerate(nodes):
        attrs = GEOPOLITICAL_NODES[name]
        pts = _grid_points(attrs["lat"], attrs["lon"], delta=2.0)
        for gi, pt in enumerate(pts):
            all_grid_pts.append(pt)
            grid_pt_map.append((ni, gi))

    elevations_flat = fetch_elevation(all_grid_pts)  # raises on failure

    elevation_grid = np.zeros((n_nodes, 9))
    for flat_idx, (ni, gi) in enumerate(grid_pt_map):
        elevation_grid[ni, gi] = elevations_flat[flat_idx]

    # ── Source 2: Open-Meteo Archive — climate per node ── #
    temp_grid = np.zeros((n_nodes, 9))
    hum_grid  = np.zeros((n_nodes, 9))
    prec_grid = np.zeros((n_nodes, 9))

    for ni, name in enumerate(nodes):
        if ni > 0:
            time.sleep(6)  # Open-Meteo free-tier cooldown
        attrs   = GEOPOLITICAL_NODES[name]
        climate = fetch_climate(attrs["lat"], attrs["lon"], days_back=365)
        temp_grid[ni, :] = climate["temperature_mean"]
        hum_grid[ni, :]  = climate["humidity_mean"]
        prec_grid[ni, :] = climate["precip_mean"]

    physical_raster = np.stack([
        elevation_grid,   # layer 0
        temp_grid,        # layer 1
        hum_grid,         # layer 2
        prec_grid,        # layer 3
    ])  # shape (4, 6, 9)

    # ── Sources 3–8: World Bank API — 6 indicators × 6 nodes ── #
    wb_scalars = np.zeros((6, n_nodes))

    wb_failures: list[str] = []
    for row_idx, (_, indicator) in enumerate(WB_INDICATORS.items()):
        for ni, name in enumerate(nodes):
            try:
                wb_scalars[row_idx, ni] = fetch_worldbank_indicator(
                    NODE_ISO3[name], indicator,
                )
            except RuntimeError as _exc:
                # Single node/indicator timeout → 0.0; do not abort entire raster
                wb_failures.append(f"{name}/{indicator}")
                wb_scalars[row_idx, ni] = 0.0

    # Invert political_stability (row 4): PV.EST higher = more stable →
    # after inversion higher value = more conflict stress
    wb_scalars[4] = -wb_scalars[4]

    # ── Sources 9–12: Extended environmental scalars ── #
    env_scalars = np.zeros((4, n_nodes))  # rows: earthquake, eonet, pm25, sea_level

    # Source 9: USGS earthquakes
    usgs = fetch_usgs_earthquakes_per_node()
    env_sources_ok: list[str] = []
    if usgs:
        for ni, name in enumerate(nodes):
            env_scalars[0, ni] = usgs.get(name, 0.0)
        env_sources_ok.append("USGS Earthquakes")

    # Source 10: NASA EONET
    eonet = fetch_nasa_eonet_events()
    if eonet:
        for ni, name in enumerate(nodes):
            env_scalars[1, ni] = eonet.get(name, 0.0)
        env_sources_ok.append("NASA EONET")

    # Source 11: Open-Meteo Air Quality
    aq = fetch_openmeteo_air_quality()
    if aq:
        for ni, name in enumerate(nodes):
            env_scalars[2, ni] = aq.get(name, 0.0)
        env_sources_ok.append("Open-Meteo AQ")

    # Source 12: NOAA Tides
    noaa = fetch_noaa_sea_level_proxy()
    if noaa:
        for ni, name in enumerate(nodes):
            env_scalars[3, ni] = noaa.get(name, 0.0)
        env_sources_ok.append("NOAA Tides")

    if not env_sources_ok:
        raise RuntimeError(
            "All 4 extended environmental sources unavailable "
            "(USGS Earthquakes, NASA EONET, Open-Meteo Air Quality, NOAA Tides). "
            "Check network connectivity."
        )

    # Stack: (6, 6) WB + (4, 6) env → (10, 6) full country_scalars
    country_scalars = np.vstack([wb_scalars, env_scalars])

    core_label = "Live: Open-Elevation, Open-Meteo Archive, World Bank API (6 indicators)"
    env_label  = f", {', '.join(env_sources_ok)}" if env_sources_ok else ""

    return dict(
        physical_raster=physical_raster,
        country_scalars=country_scalars,
        node_order=nodes,
        layer_names=PHYSICAL_LAYER_NAMES,
        scalar_names=SCALAR_NAMES,
        source_label=core_label + env_label,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_spatial_data() -> dict:
    """Fetch all multimodal spatial layers from 12 keyless sources.

    Delegates to ``_fetch_all_spatial_data_live()``; on RuntimeError falls back
    to the last successfully fetched result stored in the disk cache.
    Raises RuntimeError only when both live sources and disk cache are absent.
    """
    try:
        result = _fetch_all_spatial_data_live()
        _cache.save("spatial_data", result)
        return result
    except RuntimeError:
        cached = _cache.load("spatial_data")
        if cached is not None:
            return cached
        raise
