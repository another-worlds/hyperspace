"""Synthetic data generators (fallback when real sources unavailable)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def seed(s: int = 42) -> np.random.Generator:
    return np.random.default_rng(s)


def generate_ohlcv(ticker: str, days: int = 252, s: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with regime shifts."""
    rng = seed(s)
    base = 100 + rng.normal(0, 1) * 30
    returns = np.concatenate([
        rng.normal(0.001, 0.012, days // 3),
        rng.normal(-0.002, 0.025, days // 3),
        rng.normal(0.0005, 0.015, days - 2 * (days // 3)),
    ])
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.02, days))
    low = close * (1 - rng.uniform(0, 0.02, days))
    opn = close * (1 + rng.normal(0, 0.005, days))
    vol = rng.lognormal(15, 0.5, days).astype(int)
    dates = pd.bdate_range(end=pd.Timestamp("2026-02-20"), periods=days)
    return pd.DataFrame(dict(
        Date=dates, Open=opn, High=high, Low=low, Close=close, Volume=vol,
    )).assign(Ticker=ticker)


def generate_tft_dataset(
    n_groups: int = 12, length: int = 120, s: int = 42,
) -> pd.DataFrame:
    """Stallion-style synthetic time-series for pytorch-forecasting."""
    rng = seed(s)
    rows = []
    for g in range(n_groups):
        trend = np.linspace(0, rng.uniform(0.5, 2.0), length)
        seasonal = 0.3 * np.sin(np.linspace(0, 4 * np.pi, length) + rng.uniform(0, np.pi))
        noise = rng.normal(0, 0.15, length)
        target = np.exp(trend + seasonal + noise)
        for t in range(length):
            rows.append(dict(
                time_idx=t, group=f"sector_{g:02d}",
                target=float(target[t]),
                month=t % 12,
                regime="bull" if t < length // 3 else (
                    "bear" if t < 2 * length // 3 else "recovery"),
            ))
    return pd.DataFrame(rows)
