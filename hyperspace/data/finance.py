"""Real financial data via yfinance with synthetic fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.data.synthetic import generate_ohlcv, generate_tft_dataset


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that yfinance sometimes returns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]
    return df


@st.cache_resource(ttl=3600, show_spinner=False)
def fetch_real_ohlcv(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame | None:
    """Fetch real OHLCV data via yfinance. Returns None on failure.

    Tries individual ticker downloads, then a batch download as fallback.
    """
    try:
        import yfinance as yf
    except ImportError:
        st.warning("yfinance not installed — using synthetic data.")
        return None

    frames: list[pd.DataFrame] = []

    # Strategy 1: individual downloads (more reliable per-ticker)
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False, timeout=15)
            if df is not None and len(df) > 20:
                df = _flatten_yf_columns(df.reset_index())
                df["Ticker"] = ticker
                frames.append(df)
        except Exception as e:
            st.caption(f"yfinance: {ticker} individual download failed ({e})")

    if frames:
        return pd.concat(frames, ignore_index=True)

    # Strategy 2: batch download (sometimes works when individual fails)
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
    except Exception as e:
        st.caption(f"yfinance: batch download failed ({e})")

    # Strategy 3: try shorter period
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


def get_ohlcv(tickers: list[str], period: str = "1y") -> tuple[pd.DataFrame, str]:
    """Get OHLCV data with fallback chain.

    Returns:
        (dataframe, source_label)
    """
    # Try real data first
    real = fetch_real_ohlcv(tuple(tickers), period)
    if real is not None and len(real) > 50:
        return real, "Live: yfinance"

    # Fallback to synthetic
    frames = [generate_ohlcv(t, s=hash(t) % 10000) for t in tickers]
    return pd.concat(frames, ignore_index=True), "Fallback: synthetic"


def prepare_tft_dataset_from_real(
    ohlcv_df: pd.DataFrame,
    encoder_len: int = 48,
    prediction_len: int = 12,
) -> pd.DataFrame | None:
    """Convert real OHLCV into TFT-compatible format.

    Returns DataFrame with columns: time_idx, group, target, month, regime.
    Returns None if data is insufficient.
    """
    try:
        rows = []
        for ticker in ohlcv_df.Ticker.unique():
            tdf = ohlcv_df[ohlcv_df.Ticker == ticker].sort_values("Date").reset_index(drop=True)
            if len(tdf) < encoder_len + prediction_len + 10:
                continue
            # Compute regime from MA50
            tdf["ma50"] = tdf["Close"].rolling(50, min_periods=1).mean()
            tdf["regime"] = np.where(tdf["Close"] > tdf["ma50"], "bull", "bear")
            for t in range(len(tdf)):
                rows.append(dict(
                    time_idx=t,
                    group=ticker,
                    target=float(tdf.iloc[t]["Close"]),
                    month=int(tdf.iloc[t]["Date"].month) if hasattr(tdf.iloc[t]["Date"], "month") else t % 12,
                    regime=tdf.iloc[t]["regime"],
                ))
        if len(rows) > encoder_len + prediction_len:
            return pd.DataFrame(rows)
        return None
    except Exception:
        return None


def get_tft_data(
    tickers: list[str],
    encoder_len: int = 48,
    prediction_len: int = 12,
) -> tuple[pd.DataFrame, str]:
    """Get TFT training data with fallback.

    Returns:
        (dataframe, source_label)
    """
    # Try real OHLCV first
    ohlcv, src = get_ohlcv(tickers)
    if "yfinance" in src:
        tft_df = prepare_tft_dataset_from_real(ohlcv, encoder_len, prediction_len)
        if tft_df is not None:
            return tft_df, "Live: yfinance"

    # Fallback to synthetic
    return generate_tft_dataset(
        n_groups=max(3, len(tickers)),
        length=encoder_len + prediction_len + 20,
    ), "Fallback: synthetic"
