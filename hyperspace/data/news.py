"""Country-focused news ingestion: GDELT (primary) + public RSS feeds (secondary)."""
from __future__ import annotations

import time
from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st

from hyperspace.config import GEOPOLITICAL_NODES

# Public geopolitics RSS feeds — no API keys required
RSS_FEEDS: list[tuple[str, str]] = [
    ("BBC World",       "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Reuters World",   "https://feeds.reuters.com/Reuters/worldNews"),
    ("Al Jazeera",      "https://www.aljazeera.com/xml/rss/all.xml"),
    ("Guardian World",  "https://www.theguardian.com/world/rss"),
    ("France24",        "https://www.france24.com/en/rss"),
]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gdelt_country_news(days_back: int = 30, max_records: int = 120) -> list[str] | None:
    """Fetch country-focused geopolitical news from GDELT DOC 2.0 API.

    Respects the 5-second rate limit with up to 3 retries on 429.
    Returns None on failure.
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

        for attempt in range(3):
            resp = requests.get(url, timeout=20)
            if resp.status_code == 429:
                time.sleep(6 * (attempt + 1))
                continue
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

        return None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_rss_news() -> list[str] | None:
    """Fetch geopolitics headlines from public RSS feeds.

    Aggregates across multiple outlets; returns None if all fail.
    """
    try:
        import feedparser

        docs: list[str] = []
        for outlet, feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:30]:
                    title = (entry.get("title") or "").strip()
                    summary = (entry.get("summary") or "").strip()[:200]
                    text = f"{title} {summary}".strip()
                    if len(text) > 20:
                        docs.append(text)
            except Exception:
                continue

        return docs if len(docs) >= 20 else None
    except Exception:
        return None


def get_text_data(start_date: datetime | None = None,
                  end_date: datetime | None = None) -> tuple[list[str], str]:
    """Get country-focused text documents from live sources.

    Tries GDELT first, then public RSS feeds.
    Raises RuntimeError if all live sources are unavailable.
    """
    days_back = 30
    if start_date and end_date:
        days_back = max(7, min(90, (end_date.date() - start_date.date()).days))

    gdelt_docs = fetch_gdelt_country_news(days_back=days_back)
    if gdelt_docs:
        return gdelt_docs, f"Live: GDELT DOC 2.0 ({days_back}d window)"

    rss_docs = fetch_rss_news()
    if rss_docs:
        return rss_docs, f"Live: RSS ({len(RSS_FEEDS)} outlets, {len(rss_docs)} headlines)"

    raise RuntimeError(
        "All news sources unavailable (GDELT rate-limited or unreachable, RSS feeds failed). "
        "Check network connectivity."
    )
