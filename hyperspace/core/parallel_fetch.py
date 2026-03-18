"""Parallel data fetching orchestration for the Hyperspace pipeline.

Parallelizes independent data fetch operations to maximize throughput:
- Financial OHLCV data (yfinance)
- News/text documents (GDELT/RSS fallback)
- Political agreement matrices (GDELT)
- Multimodal spatial rasters

Expected speedup: 4-6× (from ~100-150s to ~50-70s) for typical runs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable
import pandas as pd


class ParallelFetchResult:
    """Container for parallel fetch results with timing data."""

    def __init__(self):
        self.ohlcv_df: pd.DataFrame | None = None
        self.ohlcv_source: str | None = None
        self.docs: list[str] | None = None
        self.docs_source: str | None = None
        self.agreement: pd.DataFrame | None = None
        self.agreement_source: str | None = None
        self.spatial_data: dict | None = None
        self.spatial_source: str | None = None
        self.errors: dict[str, Exception] = {}
        self.timings: dict[str, float] = {}

    def is_complete(self) -> bool:
        """Check if all required data was fetched successfully."""
        return (
            self.ohlcv_df is not None
            and self.docs is not None
            and self.agreement is not None
            and self.spatial_data is not None
        )

    def get_missing(self) -> list[str]:
        """Return list of missing data sources."""
        missing = []
        if self.ohlcv_df is None:
            missing.append("finance")
        if self.docs is None:
            missing.append("documents")
        if self.agreement is None:
            missing.append("political agreement")
        if self.spatial_data is None:
            missing.append("spatial rasters")
        return missing


def fetch_all_data_parallel(
    tickers: tuple[str, ...] = ("SPY", "EWZ", "INDA"),
    fetch_ohlcv_fn: Callable[..., tuple[pd.DataFrame, str]] | None = None,
    fetch_docs_fn: Callable[..., tuple[list[str], str]] | None = None,
    fetch_political_fn: Callable[..., tuple[Any, pd.DataFrame, str]] | None = None,
    fetch_spatial_fn: Callable[[], dict] | None = None,
    finance_start: datetime | None = None,
    finance_end: datetime | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    max_workers: int = 4,
) -> ParallelFetchResult:
    """Fetch all required data sources in parallel.

    Args:
        tickers: Financial tickers to fetch
        fetch_ohlcv_fn: Function to fetch OHLCV data: () -> (DataFrame, source_name)
        fetch_docs_fn: Function to fetch documents: (start_date, end_date) -> (list[str], source_name)
        fetch_political_fn: Function to fetch political data: (min_year, max_year) -> (any, DataFrame, source_name)
        fetch_spatial_fn: Function to fetch spatial rasters: () -> dict
        finance_start: Start date for document/political fetch alignment
        finance_end: End date for document/political fetch alignment
        min_year: Minimum year for political data
        max_year: Maximum year for political data
        max_workers: Max threads for parallel execution

    Returns:
        ParallelFetchResult with all fetched data and error information
    """
    result = ParallelFetchResult()

    if fetch_ohlcv_fn is None:
        from hyperspace.data.finance import get_financial_data
        fetch_ohlcv_fn = lambda: get_financial_data(tickers=tickers)

    if fetch_docs_fn is None:
        from hyperspace.data.news import get_text_data
        fetch_docs_fn = lambda: get_text_data(start_date=finance_start, end_date=finance_end)

    if fetch_political_fn is None:
        from hyperspace.data.political import get_political_data
        fetch_political_fn = lambda: get_political_data(min_year=min_year, max_year=max_year)

    if fetch_spatial_fn is None:
        from hyperspace.data.spatial import fetch_all_spatial_data
        fetch_spatial_fn = lambda: (fetch_all_spatial_data(), "spatial_rasters")

    # Define wrapper tasks for parallel execution
    def fetch_ohlcv_task():
        """Fetch financial data."""
        import time
        start = time.time()
        try:
            ohlcv_df, source = fetch_ohlcv_fn()
            result.ohlcv_df = ohlcv_df
            result.ohlcv_source = source
        except Exception as e:
            result.errors["ohlcv"] = e
        finally:
            result.timings["ohlcv"] = time.time() - start

    def fetch_docs_task():
        """Fetch document data."""
        import time
        start = time.time()
        try:
            docs, source = fetch_docs_fn()
            result.docs = docs
            result.docs_source = source
        except Exception as e:
            result.errors["documents"] = e
        finally:
            result.timings["documents"] = time.time() - start

    def fetch_political_task():
        """Fetch political data."""
        import time
        start = time.time()
        try:
            _, agreement, source = fetch_political_fn()
            result.agreement = agreement
            result.agreement_source = source
        except Exception as e:
            result.errors["political"] = e
        finally:
            result.timings["political"] = time.time() - start

    def fetch_spatial_task():
        """Fetch spatial data."""
        import time
        start = time.time()
        try:
            spatial_data, source = fetch_spatial_fn()
            result.spatial_data = spatial_data
            result.spatial_source = source
        except Exception as e:
            result.errors["spatial"] = e
        finally:
            result.timings["spatial"] = time.time() - start

    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_ohlcv_task): "ohlcv",
            executor.submit(fetch_docs_task): "documents",
            executor.submit(fetch_political_task): "political",
            executor.submit(fetch_spatial_task): "spatial",
        }

        for future in as_completed(futures):
            task_name = futures[future]
            try:
                future.result()
            except Exception as e:
                result.errors[task_name] = e

    return result


def fetch_models_parallel(
    tickers: tuple[str, ...] | None = None,
    docs: list[str] | None = None,
    docs_source: str | None = None,
    fit_tft_fn: Callable | None = None,
    fit_bertopic_fn: Callable | None = None,
    max_workers: int = 3,
) -> dict[str, Any]:
    """Train/fit models in parallel (TFT, BERTopic).

    Args:
        tickers: Financial tickers for TFT
        docs: Documents for BERTopic
        docs_source: Document source name for caching
        fit_tft_fn: Function to train TFT: (tickers) -> tft_result
        fit_bertopic_fn: Function to fit BERTopic: (docs_hash, docs, source) -> result
        max_workers: Max threads for parallel execution

    Returns:
        Dict with "tft_result" and "bertopic_result" keys
    """
    results = {}

    if fit_tft_fn is None:
        from hyperspace.models.tft_forecast import fit_tft
        fit_tft_fn = lambda tk: fit_tft(
            tickers=tk,
            hidden=32,
            encoder_len=48,
            prediction_len=12,
        )

    if fit_bertopic_fn is None:
        from hyperspace.models.topic_model import fit_topic_model
        import hashlib
        def default_bertopic(docs_list, docs_src):
            docs_hash = hashlib.md5("".join(docs_list[:5]).encode()).hexdigest()[:8]
            return fit_topic_model(docs_hash, docs=docs_list, data_source=docs_src)
        fit_bertopic_fn = lambda d, ds: default_bertopic(d, ds)

    # Define wrapper tasks
    def train_tft_task():
        """Train TFT model."""
        import time
        start = time.time()
        try:
            result = fit_tft_fn(tickers)
            results["tft_result"] = result
        except Exception as e:
            results["tft_error"] = e
        finally:
            results["tft_duration"] = time.time() - start

    def fit_bertopic_task():
        """Fit BERTopic model."""
        import time
        start = time.time()
        try:
            result = fit_bertopic_fn(docs, docs_source or "news")
            results["bertopic_result"] = result
        except Exception as e:
            results["bertopic_error"] = e
        finally:
            results["bertopic_duration"] = time.time() - start

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(train_tft_task),
            executor.submit(fit_bertopic_task),
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                pass  # Errors already captured in results dict

    return results
