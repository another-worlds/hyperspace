"""Real financial data via yfinance. No synthetic fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that yfinance sometimes returns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_real_ohlcv(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame | None:
    """Fetch real OHLCV data via yfinance. Returns None on failure.

    Tries individual ticker downloads, then a batch download as fallback.
    """
    try:
        import yfinance as yf
    except ImportError:
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
        except Exception:
            pass

    if frames:
        return pd.concat(frames, ignore_index=True)

    # Strategy 2: batch download
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

    # Strategy 3: shorter period
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
    """Get OHLCV data from yfinance. Raises RuntimeError if unavailable.

    Returns:
        (dataframe, source_label)
    """
    real = fetch_real_ohlcv(tuple(tickers), period)
    if real is not None and len(real) > 50:
        return real, "Live: yfinance"

    raise RuntimeError(
        f"yfinance data unavailable for tickers {tickers}. "
        "Check network connectivity."
    )


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
    """Get TFT training data from real OHLCV. Raises RuntimeError if unavailable.

    Returns:
        (dataframe, source_label)
    """
    ohlcv, src = get_ohlcv(tickers)
    tft_df = prepare_tft_dataset_from_real(ohlcv, encoder_len, prediction_len)
    if tft_df is not None:
        return tft_df, "Live: yfinance"

    raise RuntimeError(
        "Unable to build TFT dataset from real OHLCV data. "
        "Insufficient rows after filtering."
    )
