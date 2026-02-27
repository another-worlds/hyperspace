"""Country-focused news ingestion from open APIs with static snippet fallback."""
from __future__ import annotations

from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st

from hyperspace.config import GEOPOLITICAL_NODES, NEWS_SNIPPETS


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gdelt_country_news(days_back: int = 30, max_records: int = 120) -> list[str] | None:
    """Fetch country-focused geopolitical news from GDELT DOC 2.0 API.

    GDELT is open and keyless. We query for configured countries and return
    lightweight text snippets that include title/source/date.
    """
    try:
        import requests

        countries = list(GEOPOLITICAL_NODES.keys())
        query = " OR ".join([f'"{c}"' for c in countries])
        timespan = f"{max(1, days_back)}d"
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc?"
            f"query={quote_plus(query)}&mode=ArtList&format=json"
            f"&maxrecords={max_records}&sort=DateDesc&timespan={timespan}"
        )

        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        articles = payload.get("articles", [])

        docs: list[str] = []
        for a in articles:
            title = (a.get("title") or "").strip()
            seen = (a.get("seendate") or "").strip()
            source = (a.get("sourceCommonName") or "").strip()
            text = " ".join(p for p in [title, source, seen] if p)
            if len(text) > 20:
                docs.append(text)

        return docs if len(docs) >= 20 else None
    except Exception:
        return None


def get_text_data(start_date: datetime | None = None,
                  end_date: datetime | None = None) -> tuple[list[str], str]:
    """Get country-focused text documents from open news APIs only.

    Time window is derived from finance dates when provided to keep modal alignment.
    Raises RuntimeError if the live API is unavailable.
    """
    days_back = 30
    if start_date and end_date:
        days_back = max(7, min(90, (end_date.date() - start_date.date()).days))

    gdelt_docs = fetch_gdelt_country_news(days_back=days_back)
    if gdelt_docs:
        return gdelt_docs, f"Live: GDELT DOC 2.0 ({days_back}d window)"

    # Emergency fallback: curated geopolitical snippets from config
    # Used when GDELT is unavailable (offline, rate-limited, or CI environments).
    if NEWS_SNIPPETS:
        return list(NEWS_SNIPPETS), "Fallback: static geopolitical snippets"

    raise RuntimeError(
        "No news data source available (GDELT unavailable and no static snippets configured)."
    )
