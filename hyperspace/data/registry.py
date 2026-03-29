"""Unified data fetcher registry for extensible, keyless data sources.

Architecture:
  - Each data source (finance, news, politics, spatial) registers a fetcher function
  - Fetchers receive parameters from session state and return (data, source_label)
  - Pipeline orchestrates all fetchers, handles errors gracefully, tracks provenance
  - Individual modules retain backward-compatible helpers for direct use
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import streamlit as st


@dataclass
class DataFetcher:
    """Metadata for a registered data source fetcher.
    
    Attributes:
        name: Display name (e.g., "Finance", "News", "Politics")
        fetch: Callable that takes params dict and returns (data, source_label)
        requires: List of parameter keys needed from session state
        optional: Whether this fetcher is allowed to fail without blocking pipeline
    """
    name: str
    fetch: Callable[[dict], tuple[Any, str]]
    requires: list[str] = field(default_factory=list)
    optional: bool = False


# Global registry of data fetchers
_FETCHERS: dict[str, DataFetcher] = {}


def register_fetcher(
    name: str,
    requires: list[str] | None = None,
    optional: bool = False,
) -> Callable:
    """Decorator to register a data fetcher function.
    
    Args:
        name: Unique identifier for this fetcher (e.g., "Finance")
        requires: List of parameter keys the fetcher needs (e.g., ["tickers"])
        optional: If True, pipeline continues even if this fetcher fails
        
    Example:
        @register_fetcher("Finance", requires=["tickers"])
        def fetch_finance_data(params: dict) -> tuple[pd.DataFrame, str]:
            tickers = params["tickers"]
            df, src = get_ohlcv(tickers)
            return df, src
    """
    def decorator(func: Callable[[dict], tuple[Any, str]]) -> Callable:
        _FETCHERS[name] = DataFetcher(
            name=name,
            fetch=func,
            requires=requires or [],
            optional=optional,
        )
        return func
    return decorator


def get_fetcher(name: str) -> DataFetcher | None:
    """Retrieve a registered fetcher by name."""
    return _FETCHERS.get(name)


def list_fetchers() -> list[str]:
    """Return names of all registered fetchers."""
    return list(_FETCHERS.keys())


def run_all_fetchers(
    session_state: dict,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Execute all registered fetchers and return data source labels.
    
    Args:
        session_state: Dictionary (typically st.session_state) with parameters
        status_callback: Optional callable to report progress (e.g., st.write)
        
    Returns:
        Dict mapping fetcher name to source label (e.g., {"Finance": "Live: yfinance"})
        
    Side effects:
        - Stores fetched data in session_state with key pattern: raw_{name}
        - Raises RuntimeError if a required fetcher fails
        - Logs warnings if optional fetchers fail
        
    Example:
        data_sources = run_all_fetchers(st.session_state, st.write)
        st.session_state.data_sources = data_sources
    """
    data_sources: dict[str, str] = {}
    errors: list[tuple[str, str]] = []
    
    for name, fetcher in _FETCHERS.items():
        if status_callback:
            status_callback(f"Fetching {name}...")
        
        # Build parameter dict from session state
        params = {}
        missing_params = []
        for key in fetcher.requires:
            if key in session_state:
                params[key] = session_state[key]
            else:
                missing_params.append(key)
        
        if missing_params:
            msg = f"{name} fetcher missing required parameters: {missing_params}"
            if fetcher.optional:
                if status_callback:
                    status_callback(f"⚠ {msg} (skipping optional fetcher)")
                continue
            else:
                errors.append((name, msg))
                continue
        
        # Execute the fetcher
        try:
            data, source_label = fetcher.fetch(params)
            
            # Store in session state with predictable key
            session_state[f"raw_{name}"] = data
            data_sources[name] = source_label
            
            if status_callback:
                status_callback(f"✓ {name}: {source_label}")
                
        except Exception as exc:
            error_msg = f"{name} fetcher failed: {str(exc)}"
            
            if fetcher.optional:
                if status_callback:
                    status_callback(f"⚠ {error_msg} (optional, continuing)")
                # Store error indicator
                data_sources[name] = f"Unavailable: {str(exc)[:50]}"
            else:
                errors.append((name, error_msg))
                if status_callback:
                    status_callback(f"✗ {error_msg}")
    
    # Report any critical errors
    if errors:
        error_details = "\n".join(f"  • {name}: {msg}" for name, msg in errors)
        raise RuntimeError(
            f"Pipeline blocked by {len(errors)} required data source(s):\n{error_details}\n\n"
            "Check network connectivity and input parameters."
        )
    
    return data_sources


def run_fetcher(
    name: str,
    params: dict,
    store_in_state: dict | None = None,
) -> tuple[Any, str]:
    """Run a single fetcher by name.
    
    Args:
        name: Fetcher name
        params: Parameter dict to pass to fetcher
        store_in_state: Optional dict (e.g., st.session_state) to store result
        
    Returns:
        (data, source_label) tuple
        
    Raises:
        ValueError: If fetcher not found
        RuntimeError: If fetcher execution fails
    """
    fetcher = get_fetcher(name)
    if fetcher is None:
        raise ValueError(f"No fetcher registered with name '{name}'")
    
    # Check required parameters
    missing = [k for k in fetcher.requires if k not in params]
    if missing:
        raise ValueError(
            f"{name} fetcher requires parameters: {missing}"
        )
    
    try:
        data, source_label = fetcher.fetch(params)
        
        if store_in_state is not None:
            store_in_state[f"raw_{name}"] = data
            
        return data, source_label
        
    except Exception as exc:
        raise RuntimeError(
            f"{name} fetcher failed: {str(exc)}"
        ) from exc
