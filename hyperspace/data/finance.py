"""Real financial data via yfinance with synthetic fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.data.synthetic import generate_ohlcv, generate_tft_dataset


@st.cache_resource(ttl=3600, show_spinner=False)
def fetch_real_ohlcv(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame | None:
    """Fetch real OHLCV data via yfinance. Returns None on failure.

    Args:
        tickers: Tuple of ticker symbols (tuple for hashability).
        period: yfinance period string.
    """
    try:
        import yfinance as yf
        frames = []
        for ticker in tickers:
            df = yf.download(ticker, period=period, progress=False, timeout=10)
            if df is not None and len(df) > 20:
                df = df.reset_index()
                # Handle MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if c[1] == '' else c[0] for c in df.columns]
                df["Ticker"] = ticker
                frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return None
    except Exception:
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
    frames = [generate_ohlcv(t, seed=hash(t) % 10000) for t in tickers]
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
