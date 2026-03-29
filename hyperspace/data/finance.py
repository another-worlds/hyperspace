"""Real financial data via 10 keyless APIs with synthetic GBM fallback.

Sources:
  1. yfinance (Yahoo Finance) — primary OHLCV
  2. Stooq.com CSV API       — backup OHLCV
  3. ECB Statistical Data Warehouse (SDW) REST — EUR FX rates
  4. IMF DataMapper API      — GDP growth, inflation
  5. US Treasury FiscalData API — yield curve interest rates
  6. CoinGecko API           — global market cap / risk sentiment
  7. Open.er-api.com         — live USD FX rates (free, no key)
  8. World Bank API (financial indicators) — CPI, market cap, FDI, M2
  9. BIS Statistics Portal   — central bank policy rates
 10. BLS Public API v1       — US CPI (no registration required)
"""
from __future__ import annotations

from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ── Ticker → country / currency mappings ────────────────────────────────── #

TICKER_COUNTRY_MAP: dict[str, str] = {
    "SPY":  "USA",
    "EWZ":  "BRA",
    "INDA": "IND",
    "FXI":  "CHN",
    "EWU":  "GBR",
    "ERUS": "RUS",
    "RSX":  "RUS",
}

COUNTRY_CURRENCY_MAP: dict[str, str] = {
    "USA": "USD",
    "BRA": "BRL",
    "IND": "INR",
    "CHN": "CNY",
    "GBR": "GBP",
    "RUS": "RUB",
}

IMF_COUNTRY_CODES: tuple[str, ...] = ("USA", "BRA", "IND", "CHN", "GBR", "RUS")


# ── Helper ───────────────────────────────────────────────────────────────── #

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that yfinance sometimes returns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — yfinance  (Yahoo Finance OHLCV)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_real_ohlcv(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame | None:
    """Fetch real OHLCV data via yfinance. Returns None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False, timeout=15)
            if df is not None and len(df) > 20:
                df = _flatten_yf_columns(df.reset_index())
                df["Ticker"] = ticker
                frames.append(df)
        except Exception:
            pass

    if frames:
        return pd.concat(frames, ignore_index=True)

    try:
        ticker_str = " ".join(tickers)
        df = yf.download(ticker_str, period=period, progress=False,
                         timeout=20, group_by="ticker")
        if df is not None and len(df) > 20:
            df = df.reset_index()
            result_frames = []
            for ticker in tickers:
                try:
                    if isinstance(df.columns, pd.MultiIndex) and ticker in df.columns.get_level_values(0):
                        tdf = df[["Date", ticker]].copy() if "Date" in df.columns else df[[ticker]].copy()
                        tdf = tdf.droplevel(0, axis=1) if isinstance(tdf.columns, pd.MultiIndex) else tdf
                    else:
                        tdf = df.copy()
                    tdf = _flatten_yf_columns(tdf)
                    tdf["Ticker"] = ticker
                    if "Close" in tdf.columns and len(tdf) > 20:
                        result_frames.append(tdf)
                except Exception:
                    continue
            if result_frames:
                return pd.concat(result_frames, ignore_index=True)
    except Exception:
        pass

    if period != "6mo":
        try:
            for ticker in tickers:
                df = yf.download(ticker, period="6mo", progress=False, timeout=15)
                if df is not None and len(df) > 20:
                    df = _flatten_yf_columns(df.reset_index())
                    df["Ticker"] = ticker
                    frames.append(df)
            if frames:
                return pd.concat(frames, ignore_index=True)
        except Exception:
            pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — Stooq.com  (backup OHLCV, keyless CSV)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stooq_ohlcv(ticker: str) -> pd.DataFrame | None:
    """Fetch OHLCV from Stooq.com (keyless CSV download, backup to yfinance).

    URL pattern: https://stooq.com/q/d/l/?s={ticker}.us&i=d
    No API key required.
    """
    try:
        import requests

        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None

        text = resp.text.strip()
        if not text or "No data" in text[:200] or len(text) < 100:
            return None

        df = pd.read_csv(StringIO(text))
        if df.empty or "Date" not in df.columns or "Close" not in df.columns:
            return None

        df["Date"] = pd.to_datetime(df["Date"])
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
        df = df[df["Date"] >= cutoff].copy()
        df["Ticker"] = ticker
        return df if len(df) > 20 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — ECB Statistical Data Warehouse  (EUR FX rates)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ecb_fx_rates() -> dict[str, float] | None:
    """Fetch EUR-based exchange rates from ECB daily XML feed (keyless).

    ECB Reference Rates: https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml
    Simpler than the SDW REST endpoint; returns {currency_code: units_per_EUR}.
    """
    try:
        import requests
        import xml.etree.ElementTree as ET

        resp = requests.get(
            "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml",
            timeout=15,
        )
        resp.raise_for_status()

        ns = "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"
        root = ET.fromstring(resp.text)
        relevant = {"USD", "GBP", "CNY", "INR", "BRL", "RUB", "JPY"}
        rates: dict[str, float] = {}
        for cube in root.iter(f"{{{ns}}}Cube"):
            ccy  = cube.get("currency")
            rate = cube.get("rate")
            if ccy in relevant and rate:
                try:
                    rates[ccy] = float(rate)
                except ValueError:
                    pass

        return rates if rates else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — IMF DataMapper  (GDP growth, inflation)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_imf_macro(country_codes: tuple[str, ...]) -> dict[str, dict[str, float]] | None:
    """Fetch macroeconomic indicators from IMF DataMapper API (keyless).

    IMF DataMapper: https://www.imf.org/external/datamapper/api/v1/
    Indicators: NGDP_RPCH (real GDP growth %), PCPIPCH (CPI inflation %).
    Returns {country_code: {indicator: value}}.
    """
    try:
        import requests

        indicators = ["NGDP_RPCH", "PCPIPCH"]
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
# SOURCE 5 — US Treasury FiscalData API  (yield curve)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_treasury_yield_curve() -> dict[str, float] | None:
    """Fetch US Treasury yield curve via Treasury.gov OData XML feed (keyless).

    US Treasury: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/
    Parses the Atom/OData XML for the most-recent daily yield curve data.
    Returns {maturity_label: yield_pct}.
    """
    try:
        import requests
        import xml.etree.ElementTree as ET
        from datetime import date, timedelta

        atom_ns = "http://www.w3.org/2005/Atom"
        d_ns    = "http://schemas.microsoft.com/ado/2007/08/dataservices"
        m_ns    = "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"

        # Try current month, then previous month (in case current month has <1 day of data)
        today = date.today()
        months_to_try = [
            today.strftime("%Y%m"),
            (today.replace(day=1) - timedelta(days=1)).strftime("%Y%m"),
        ]
        entries: list = []
        for ym in months_to_try:
            url = (
                "https://home.treasury.gov/resource-center/data-chart-center/"
                "interest-rates/pages/xml?data=daily_treasury_yield_curve"
                f"&field_tdr_date_value_month={ym}"
            )
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root    = ET.fromstring(resp.text)
            entries = root.findall(f"{{{atom_ns}}}entry")
            if entries:
                break

        if not entries:
            return None

        props = entries[0].find(f".//{{{m_ns}}}properties")
        if props is None:
            return None

        maturity_map = {
            "BC_1MONTH": "1M",  "BC_3MONTH":  "3M",  "BC_6MONTH": "6M",
            "BC_1YEAR":  "1Y",  "BC_2YEAR":   "2Y",  "BC_5YEAR":  "5Y",
            "BC_10YEAR": "10Y", "BC_20YEAR": "20Y",  "BC_30YEAR": "30Y",
        }
        yields: dict[str, float] = {}
        for field, label in maturity_map.items():
            elem = props.find(f"{{{d_ns}}}{field}")
            if elem is not None and elem.text:
                try:
                    yields[label] = float(elem.text)
                except ValueError:
                    pass

        return yields if yields else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 6 — CoinGecko API  (global market cap / risk sentiment)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_coingecko_global() -> dict[str, float] | None:
    """Fetch global market-cap and risk-sentiment from CoinGecko API (keyless).

    CoinGecko: https://api.coingecko.com/api/v3/global
    Returns total_market_cap_usd, btc_dominance_pct, market_cap_change_24h_pct.
    """
    try:
        import requests

        resp = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=10,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})

        result: dict[str, float] = {}
        for key, path in [
            ("total_market_cap_usd",     ("total_market_cap", "usd")),
            ("btc_dominance_pct",        ("market_cap_percentage", "btc")),
            ("market_cap_change_24h_pct", ("market_cap_change_percentage_24h_usd",)),
            ("active_cryptos",           ("active_cryptocurrencies",)),
        ]:
            try:
                val: Any = data
                for k in path:
                    val = val[k]
                if val is not None:
                    result[key] = float(val)
            except (KeyError, TypeError):
                pass

        return result if result else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 7 — open.er-api.com  (live USD FX rates, free / no key)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_open_er_fx() -> dict[str, float] | None:
    """Fetch live USD-based FX rates from open.er-api.com (free, no API key).

    Open ER API: https://open.er-api.com/v6/latest/USD
    Returns {currency_code: units_per_USD}.
    """
    try:
        import requests

        resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("result") == "success":
            rates = data.get("rates", {})
            relevant = {"EUR", "GBP", "CNY", "INR", "BRL", "RUB", "JPY"}
            return {k: float(v) for k, v in rates.items() if k in relevant}
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 8 — World Bank API  (financial indicators)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_wb_financial_indicators(iso3_codes: tuple[str, ...]) -> dict[str, dict[str, float]] | None:
    """Fetch World Bank financial-sector indicators (keyless).

    World Bank API: https://api.worldbank.org/v2/
    Indicators: CPI inflation, market cap (% GDP), FDI inflows (% GDP), M2 (% GDP).
    Returns {iso3: {name: value}}.
    """
    try:
        import requests

        indicators = {
            "FP.CPI.TOTL.ZG":    "cpi_inflation_pct",
            "CM.MKT.LCAP.GD.ZS": "market_cap_gdp_pct",
            "BX.KLT.DINV.WD.GD.ZS": "fdi_inflows_gdp_pct",
            "FM.LBL.BMNY.GD.ZS": "m2_gdp_pct",
        }

        result: dict[str, dict[str, float]] = {iso3: {} for iso3 in iso3_codes}
        iso_str = ";".join(iso3_codes)

        for indicator, name in indicators.items():
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
                    cid = rec.get("countryiso3code", "")
                    val = rec.get("value")
                    if cid in result and val is not None and cid not in seen:
                        result[cid][name] = float(val)
                        seen.add(cid)
            except Exception:
                continue

        return result if any(result[c] for c in result) else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 9 — BIS Statistics Portal  (central bank policy rates)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_bis_policy_rates() -> dict[str, float] | None:
    """Fetch central bank policy rates from BIS Statistics Portal (keyless).

    BIS SDMX REST: https://stats.bis.org/api/v1/data/WS_CBPOL/
    Returns {country_code: policy_rate_pct}.
    """
    try:
        import requests

        url = (
            "https://stats.bis.org/api/v1/data/WS_CBPOL/"
            "M.US+GB+CN+IN+BR+RU"
            "?startPeriod=2024-01&lastNObservations=1&format=csv"
        )
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None

        df = pd.read_csv(StringIO(resp.text))
        if df.empty:
            return None

        rates: dict[str, float] = {}
        if "OBS_VALUE" in df.columns and "REF_AREA" in df.columns:
            for _, row in df.iterrows():
                try:
                    country = str(row["REF_AREA"])
                    val = row["OBS_VALUE"]
                    if country and val is not None:
                        rates[country] = float(val)
                except (ValueError, TypeError):
                    pass

        return rates if rates else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 10 — BLS Public API v1  (US CPI, no registration)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_bls_cpi_us() -> dict[str, float] | None:
    """Fetch US CPI from BLS Public Data API v1 (no registration required).

    BLS API: https://api.bls.gov/publicAPI/v1/timeseries/data/
    Series CUUR0000SA0: CPI-U all items, not seasonally adjusted.
    Returns {YYYY-MM: index_value}.
    """
    try:
        import requests

        resp = requests.get(
            "https://api.bls.gov/publicAPI/v1/timeseries/data/CUUR0000SA0",
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
            return None

        series = data.get("Results", {}).get("series", [])
        if not series:
            return None

        result: dict[str, float] = {}
        for item in series[0].get("data", [])[:12]:
            try:
                key = f"{item['year']}-{item['period'].replace('M', '')}"
                result[key] = float(item["value"])
            except (KeyError, ValueError, TypeError):
                pass

        return result if result else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Macro feature aggregator
# ══════════════════════════════════════════════════════════════════════════════

def get_macro_features(tickers: list[str]) -> tuple[dict, str]:
    """Aggregate macro features from all 8 supplementary keyless APIs.

    Tries: ECB SDW, IMF DataMapper, US Treasury FiscalData, CoinGecko,
    Open.er-api, World Bank Financial, BIS Policy Rates, BLS CPI.

    Returns:
        (features_dict, source_label)
    Raises RuntimeError if none of the 8 macro sources return data.
    """
    sources_used: list[str] = []
    features: dict = {}

    # ECB FX
    ecb = fetch_ecb_fx_rates()
    if ecb:
        features["ecb_fx"] = ecb
        sources_used.append("ECB SDW")

    # IMF macro
    imf_codes = tuple(sorted({TICKER_COUNTRY_MAP.get(t, "USA") for t in tickers}))
    imf = fetch_imf_macro(imf_codes)
    if imf:
        features["imf_macro"] = imf
        sources_used.append("IMF DataMapper")

    # Treasury yield curve
    treasury = fetch_treasury_yield_curve()
    if treasury:
        features["treasury_yields"] = treasury
        sources_used.append("US Treasury FiscalData")

    # CoinGecko
    cg = fetch_coingecko_global()
    if cg:
        features["crypto_market"] = cg
        sources_used.append("CoinGecko")

    # Open ER FX
    oper = fetch_open_er_fx()
    if oper:
        features["open_er_fx"] = oper
        sources_used.append("Open.er-api")

    # World Bank financial
    iso3_codes = tuple(sorted({TICKER_COUNTRY_MAP.get(t, "USA") for t in tickers}))
    wb_fin = fetch_wb_financial_indicators(iso3_codes)
    if wb_fin:
        features["wb_financial"] = wb_fin
        sources_used.append("World Bank Financial")

    # BIS policy rates
    bis = fetch_bis_policy_rates()
    if bis:
        features["bis_policy_rates"] = bis
        sources_used.append("BIS Policy Rates")

    # BLS CPI
    bls = fetch_bls_cpi_us()
    if bls:
        features["bls_cpi_us"] = bls
        sources_used.append("BLS CPI")

    if not sources_used:
        raise RuntimeError(
            "All 8 macro financial APIs unavailable "
            "(ECB SDW, IMF DataMapper, US Treasury FiscalData, CoinGecko, "
            "Open.er-api, World Bank Financial, BIS Policy Rates, BLS CPI). "
            "Check network connectivity."
        )

    return features, f"Live: {', '.join(sources_used)}"


# ══════════════════════════════════════════════════════════════════════════════
# TFT dataset preparation (enriched with macro features)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_tft_dataset_from_real(
    ohlcv_df: pd.DataFrame,
    encoder_len: int = 48,
    prediction_len: int = 12,
    macro_features: dict | None = None,
) -> pd.DataFrame | None:
    """Convert real OHLCV + macro features into TFT-compatible format.

    Returns DataFrame with columns:
        time_idx, group, target, month, regime,
        gdp_growth, inflation, fx_rate, cpi_inflation, market_cap_gdp
    (the last five are static reals per ticker/country).
    Returns None if data is insufficient.
    """
    try:
        rows = []
        for ticker in ohlcv_df.Ticker.unique():
            tdf = (
                ohlcv_df[ohlcv_df.Ticker == ticker]
                .sort_values("Date")
                .reset_index(drop=True)
            )
            if len(tdf) < encoder_len + prediction_len + 10:
                continue

            tdf["ma50"] = tdf["Close"].rolling(50, min_periods=1).mean()
            tdf["regime"] = np.where(tdf["Close"] > tdf["ma50"], "bull", "bear")

            country = TICKER_COUNTRY_MAP.get(ticker, "USA")
            ccy     = COUNTRY_CURRENCY_MAP.get(country, "USD")

            # ── macro values: NaN when API returned no data (no synthetic fallback) ──
            gdp_growth     = float("nan")
            inflation      = float("nan")
            fx_rate        = float("nan")
            cpi_inflation  = float("nan")
            market_cap_gdp = float("nan")

            if macro_features:
                imf = macro_features.get("imf_macro", {})
                if country in imf:
                    raw_gdp_growth = imf[country].get("NGDP_RPCH")
                    if raw_gdp_growth is not None:
                        gdp_growth = float(np.clip(float(raw_gdp_growth), -20.0, 20.0))
                    raw_inflation = imf[country].get("PCPIPCH")
                    if raw_inflation is not None:
                        inflation = float(np.clip(float(raw_inflation), 0.0, 100.0))

                # FX rate: prefer Open ER (USD base), fall back to ECB (EUR base)
                oper = macro_features.get("open_er_fx", {})
                ecb  = macro_features.get("ecb_fx", {})
                if ccy in oper:
                    fx_rate = float(max(0.001, float(oper[ccy])))
                elif ccy in ecb:
                    fx_rate = float(max(0.001, float(ecb[ccy])))

                wb = macro_features.get("wb_financial", {})
                if country in wb:
                    raw_cpi_inflation = wb[country].get("cpi_inflation_pct")
                    if raw_cpi_inflation is not None:
                        cpi_inflation = float(np.clip(float(raw_cpi_inflation), 0.0, 100.0))
                    raw_market_cap_gdp = wb[country].get("market_cap_gdp_pct")
                    if raw_market_cap_gdp is not None:
                        market_cap_gdp = float(max(0.0, float(raw_market_cap_gdp)))

            for t in range(len(tdf)):
                row = tdf.iloc[t]
                rows.append(dict(
                    time_idx=t,
                    group=ticker,
                    target=float(row["Close"]),
                    month=int(row["Date"].month) if hasattr(row["Date"], "month") else (t % 12),
                    regime=row["regime"],
                    # Static macro reals (constant per group)
                    gdp_growth=gdp_growth,
                    inflation=inflation,
                    fx_rate=fx_rate,
                    cpi_inflation=cpi_inflation,
                    market_cap_gdp=market_cap_gdp,
                ))

        if len(rows) > encoder_len + prediction_len:
            result = pd.DataFrame(rows)
            # Drop macro columns that contain any NaN — indicates the API returned
            # no real data for at least one ticker, so including them in
            # static_reals would introduce synthetic placeholders.
            _macro_cols = [
                "gdp_growth", "inflation", "fx_rate",
                "cpi_inflation", "market_cap_gdp",
            ]
            present = [c for c in _macro_cols if c in result.columns]
            has_nan = result[present].isna().any()
            cols_to_drop = has_nan[has_nan].index.tolist()
            if cols_to_drop:
                result = result.drop(columns=cols_to_drop)
            return result
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_ohlcv(
    tickers: list[str],
    n_days: int = 252,
) -> pd.DataFrame | None:
    """Generate synthetic OHLCV via geometric Brownian motion.

    Used as last-resort fallback when all live sources are unavailable.
    Produces plausible price series for governance demo purposes.
    Labeled transparently as synthetic for audit trail compliance.
    """
    import numpy as np

    frames: list[pd.DataFrame] = []
    end_date = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(end=end_date, periods=n_days)

    for ticker in tickers:
        seed = sum(ord(c) for c in ticker) * 137
        rng = np.random.RandomState(seed)

        mu, sigma = 0.0005, 0.015
        returns = rng.normal(mu, sigma, n_days)
        prices = 100.0 * np.exp(np.cumsum(returns))

        # Derive OHLCV from close prices
        noise = rng.uniform(0.005, 0.015, n_days)
        opens = prices * (1 + rng.uniform(-0.005, 0.005, n_days))
        highs = prices * (1 + noise)
        lows = prices * (1 - noise)
        volumes = rng.uniform(5e6, 50e6, n_days).astype(int)

        df = pd.DataFrame({
            "Date": dates,
            "Ticker": ticker,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        })
        frames.append(df)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


def get_ohlcv(tickers: list[str], period: str = "1y") -> tuple[pd.DataFrame, str]:
    """Get OHLCV from yfinance (primary), Stooq.com (backup), or synthetic GBM (fallback).

    The synthetic fallback uses geometric Brownian motion to generate
    plausible price series when all live sources are unavailable.
    Transparently labeled for governance audit compliance.
    """
    real = fetch_real_ohlcv(tuple(tickers), period)
    if real is not None and len(real) > 50:
        return real, "Live: yfinance"

    # Stooq.com backup
    frames = []
    for t in tickers:
        df = fetch_stooq_ohlcv(t)
        if df is not None:
            frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if len(combined) > 50:
            return combined, "Live: Stooq.com"

    # Synthetic fallback: geometric Brownian motion
    synthetic = _generate_synthetic_ohlcv(tickers)
    if synthetic is not None and len(synthetic) > 50:
        return synthetic, "Synthetic: GBM simulation"

    raise RuntimeError(
        f"OHLCV data unavailable from yfinance, Stooq.com, and synthetic fallback "
        f"for tickers {tickers}. Check network connectivity."
    )


def get_tft_data(
    tickers: list[str],
    encoder_len: int = 48,
    prediction_len: int = 12,
) -> tuple[pd.DataFrame, str]:
    """Get TFT training data: real OHLCV enriched with live macro features.

    Returns (dataframe, source_label). Raises RuntimeError if unavailable.
    """
    ohlcv, ohlcv_src = get_ohlcv(tickers)

    macro_features: dict | None = None
    macro_src = ""
    try:
        macro_features, macro_src = get_macro_features(tickers)
    except RuntimeError:
        pass  # Macro enrichment is supplementary; OHLCV is critical

    tft_df = prepare_tft_dataset_from_real(ohlcv, encoder_len, prediction_len, macro_features)
    if tft_df is not None:
        source = ohlcv_src + (f" + {macro_src}" if macro_src else "")
        return tft_df, source

    raise RuntimeError(
        "Unable to build TFT dataset from real OHLCV data. "
        "Insufficient rows after filtering."
    )
