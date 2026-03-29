"""Shared utilities for keyless API requests and fallback management."""
from __future__ import annotations

import logging
import time
from typing import Any

import requests


logger = logging.getLogger(__name__)


def safe_request(
    url: str,
    timeout: int = 20,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> requests.Response | None:
    """Make a safe HTTP GET request with retries and error handling.
    
    Args:
        url: Target URL.
        timeout: Request timeout in seconds.
        headers: Optional HTTP headers.
        params: Optional query parameters.
        max_retries: Number of retry attempts on failure.
        retry_delay: Seconds to wait between retries.
    
    Returns:
        Response object on success, None on failure.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers=headers,
                params=params,
            )
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries + 1}: {url}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error {e.response.status_code} on {url}: {e}")
            return None  # Don't retry on HTTP errors (4xx, 5xx)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed on attempt {attempt + 1}/{max_retries + 1}: {e}")
        
        if attempt < max_retries:
            time.sleep(retry_delay)
    
    return None


def parse_json_safely(response: requests.Response) -> dict | list | None:
    """Safely parse JSON response.
    
    Args:
        response: Requests response object.
    
    Returns:
        Parsed JSON (dict or list) on success, None on failure.
    """
    try:
        return response.json()
    except (ValueError, TypeError) as e:
        logger.warning(f"JSON parse error: {e}")
        return None


def log_api_fallback(source_name: str, tier: int, total_tiers: int) -> None:
    """Log API fallback tier for debugging.
    
    Args:
        source_name: Name of data source (e.g., "Finance", "News").
        tier: Current tier being attempted.
        total_tiers: Total number of fallback tiers.
    """
    logger.info(f"{source_name}: Attempting Tier {tier}/{total_tiers}")


def create_source_label(
    source_name: str,
    tier: int,
    api_name: str,
    details: str = "",
    is_demo: bool = False,
) -> str:
    """Create standardized source label for governance tracking.
    
    Args:
        source_name: Data domain (e.g., "Finance", "News").
        tier: Tier number in fallback chain.
        api_name: Specific API/service name.
        details: Optional detail string (record count, date range, etc.).
        is_demo: Whether this is synthetic demo data.
    
    Returns:
        Formatted source label string.
    
    Examples:
        >>> create_source_label("Finance", 1, "Yahoo CSV", "252 days")
        'Live: Yahoo CSV (252 days) [Tier 1]'
        >>> create_source_label("News", 9, "Synthetic", "200 headlines", is_demo=True)
        'Demo: Synthetic (200 headlines) [Tier 9]'
    """
    prefix = "Demo" if is_demo else "Live"
    detail_str = f" ({details})" if details else ""
    tier_str = f" [Tier {tier}]" if tier > 1 else ""
    return f"{prefix}: {api_name}{detail_str}{tier_str}"
