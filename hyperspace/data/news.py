"""Country-focused news ingestion from 11 keyless sources.

Sources:
  1. GDELT DOC 2.0 API           — primary geopolitical news events
  2. BBC World RSS               — international headlines
  3. Reuters World RSS           — international headlines
  4. Al Jazeera RSS              — international headlines
  5. Guardian World RSS          — international headlines
  6. France24 RSS                — international headlines
  7. Wikipedia API               — country-specific article searches
  8. Reddit public JSON          — r/worldnews, r/geopolitics
  9. Hacker News Algolia API     — geopolitics-tagged stories
 10. UN News RSS                 — Security Council / peace & security
 11. GDELT GKG 2.0              — global knowledge graph (broader query)
"""
from __future__ import annotations

import time
from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st

from hyperspace.config import GEOPOLITICAL_NODES

# ── Public RSS feeds ────────────────────────────────────────────────────── #

RSS_FEEDS: list[tuple[str, str]] = [
    ("BBC World",      "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Reuters World",  "https://feeds.reuters.com/Reuters/worldNews"),
    ("Al Jazeera",     "https://www.aljazeera.com/xml/rss/all.xml"),
    ("Guardian World", "https://www.theguardian.com/world/rss"),
    ("France24",       "https://www.france24.com/en/rss"),
]

UN_RSS_FEEDS: list[tuple[str, str]] = [
    ("UN Security Council",  "https://news.un.org/feed/subscribe/en/news/topic/security-council/feed/rss.xml"),
    ("UN Peace & Security",  "https://news.un.org/feed/subscribe/en/news/topic/peace-and-security/feed/rss.xml"),
    ("UN Human Rights",      "https://news.un.org/feed/subscribe/en/news/topic/human-rights/feed/rss.xml"),
]


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — GDELT DOC 2.0 API
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gdelt_country_news(days_back: int = 30, max_records: int = 120) -> list[str] | None:
    """Fetch country-focused geopolitical news from GDELT DOC 2.0 API.

    Respects the 5-second rate limit with up to 3 retries on 429.
    Returns None on failure.
    """
    try:
        import requests

        countries = list(GEOPOLITICAL_NODES.keys())
        query   = " OR ".join([f'"{c}"' for c in countries])
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
            articles = resp.json().get("articles", [])
            docs: list[str] = []
            for a in articles:
                title  = (a.get("title") or "").strip()
                seen   = (a.get("seendate") or "").strip()
                source = (a.get("sourceCommonName") or "").strip()
                text   = " ".join(p for p in [title, source, seen] if p)
                if len(text) > 20:
                    docs.append(text)
            return docs if len(docs) >= 20 else None

        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCES 2–6 — Public RSS feeds (BBC, Reuters, Al Jazeera, Guardian, France24)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_rss_news() -> list[str] | None:
    """Fetch geopolitics headlines from 5 public RSS feeds.

    Aggregates across BBC, Reuters, Al Jazeera, Guardian, France24.
    Returns None if all fail.
    """
    try:
        import feedparser

        docs: list[str] = []
        for outlet, feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:30]:
                    title   = (entry.get("title") or "").strip()
                    summary = (entry.get("summary") or "").strip()[:200]
                    text    = f"{title} {summary}".strip()
                    if len(text) > 20:
                        docs.append(text)
            except Exception:
                continue

        return docs if len(docs) >= 20 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 7 — Wikipedia API  (country-specific searches)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_wikipedia_events() -> list[str] | None:
    """Fetch geopolitics articles from Wikipedia Search API (keyless).

    Wikipedia API: https://en.wikipedia.org/w/api.php
    Searches geopolitics + each key country. No API key required.
    """
    try:
        import requests

        countries = ["United States", "Russia", "China",
                     "United Kingdom", "India", "Brazil"]
        queries   = [f"{c} international relations" for c in countries] + [
            "geopolitics 2025", "NATO 2025", "UN Security Council",
            "sanctions diplomacy", "trade war 2025",
        ]

        docs: list[str] = []
        for q in queries:
            try:
                url = (
                    "https://en.wikipedia.org/w/api.php"
                    f"?action=query&list=search&srsearch={quote_plus(q)}"
                    "&format=json&srlimit=5&srprop=snippet"
                )
                resp = requests.get(url, timeout=10,
                                    headers={"User-Agent": "HyperspaceBot/1.0"})
                resp.raise_for_status()
                for result in resp.json().get("query", {}).get("search", []):
                    title   = result.get("title", "")
                    snippet = (result.get("snippet") or "")
                    # Strip HTML tags
                    import re
                    snippet = re.sub(r"<[^>]+>", "", snippet)
                    text = f"{title}: {snippet}".strip()
                    if len(text) > 30:
                        docs.append(text)
            except Exception:
                continue

        return docs if len(docs) >= 10 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 8 — Reddit public JSON  (r/worldnews, r/geopolitics)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_reddit_worldnews() -> list[str] | None:
    """Fetch top posts from geopolitics-focused subreddits (public JSON, no key).

    Reddit provides unauthenticated JSON for all public subreddits.
    No API key or OAuth required.
    """
    try:
        import requests

        headers = {"User-Agent": "HyperspaceBot/1.0 (geopolitics research)"}
        subreddits = ["worldnews", "geopolitics", "InternationalNews"]
        docs: list[str] = []

        for sub in subreddits:
            try:
                url  = f"https://www.reddit.com/r/{sub}/top.json?limit=25&t=week"
                resp = requests.get(url, timeout=10, headers=headers)
                resp.raise_for_status()
                for post in resp.json().get("data", {}).get("children", []):
                    title = (post.get("data", {}).get("title") or "").strip()
                    if len(title) > 20:
                        docs.append(title)
            except Exception:
                continue

        return docs if len(docs) >= 10 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 9 — Hacker News Algolia API
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_hn_geopolitics() -> list[str] | None:
    """Fetch geopolitics-related stories from Hacker News Algolia API (keyless).

    HN Algolia: https://hn.algolia.com/api/v1/
    No API key required.
    """
    try:
        import requests

        cutoff  = int(time.time()) - 7 * 86400  # last 7 days
        queries = ["geopolitics", "sanctions 2025",
                   "Ukraine Russia war", "China Taiwan",
                   "NATO alliance", "IMF World Bank"]
        docs: list[str] = []

        for q in queries:
            try:
                url = (
                    "https://hn.algolia.com/api/v1/search"
                    f"?query={quote_plus(q)}&tags=story"
                    f"&numericFilters=created_at_i>{cutoff}"
                    "&hitsPerPage=10"
                )
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                for hit in resp.json().get("hits", []):
                    title = (hit.get("title") or "").strip()
                    if len(title) > 20:
                        docs.append(title)
            except Exception:
                continue

        return docs if len(docs) >= 5 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 10 — UN News RSS  (Security Council, Peace & Security, Human Rights)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_un_news_rss() -> list[str] | None:
    """Fetch UN News headlines from official RSS feeds (keyless).

    UN News provides public RSS for Security Council, peace & security,
    and human rights topics.
    """
    try:
        import feedparser

        docs: list[str] = []
        for name, url in UN_RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    title   = (entry.get("title") or "").strip()
                    summary = (entry.get("summary") or "").strip()[:200]
                    text    = f"{title} {summary}".strip()
                    if len(text) > 20:
                        docs.append(text)
            except Exception:
                continue

        return docs if len(docs) >= 5 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 11 — GDELT GKG 2.0  (global knowledge graph, broader query)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gdelt_gkg_themes() -> list[str] | None:
    """Fetch broader geopolitical news via GDELT GKG 2.0 themes (keyless).

    Uses a broader query targeting conflict, diplomacy, and sanctions themes.
    Distinct from SOURCE 1 which targets country names.
    """
    try:
        import requests

        query = (
            "conflict OR diplomacy OR sanctions OR alliance OR ceasefire "
            "OR treaty OR nuclear OR military"
        )
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={quote_plus(query)}&mode=artlist&format=json"
            "&maxrecords=60&sort=DateDesc&timespan=7d"
        )

        for attempt in range(3):
            resp = requests.get(url, timeout=20)
            if resp.status_code == 429:
                time.sleep(6 * (attempt + 1))
                continue
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            docs: list[str] = []
            for a in articles:
                title = (a.get("title") or "").strip()
                if len(title) > 20:
                    docs.append(title)
            return docs if len(docs) >= 5 else None

        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def get_text_data(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> tuple[list[str], str]:
    """Get country-focused text documents from all 11 live keyless sources.

    Cascade order:
      GDELT DOC 2.0  → RSS feeds (x5) → Wikipedia → Reddit →
      HN Algolia → UN News RSS → GDELT GKG

    At least one source must succeed; raises RuntimeError if all fail.
    Returns (docs, source_label).
    """
    days_back = 30
    if start_date and end_date:
        days_back = max(7, min(90, (end_date.date() - start_date.date()).days))

    all_docs: list[str] = []
    sources_used: list[str] = []

    # 1. GDELT DOC 2.0
    gdelt = fetch_gdelt_country_news(days_back=days_back)
    if gdelt:
        all_docs.extend(gdelt)
        sources_used.append(f"GDELT DOC 2.0 ({days_back}d)")

    # 2–6. Public RSS feeds
    rss = fetch_rss_news()
    if rss:
        all_docs.extend(rss)
        sources_used.append(f"RSS ({len(RSS_FEEDS)} outlets)")

    # 7. Wikipedia
    wiki = fetch_wikipedia_events()
    if wiki:
        all_docs.extend(wiki)
        sources_used.append("Wikipedia API")

    # 8. Reddit
    reddit = fetch_reddit_worldnews()
    if reddit:
        all_docs.extend(reddit)
        sources_used.append("Reddit (r/worldnews, r/geopolitics)")

    # 9. HN Algolia
    hn = fetch_hn_geopolitics()
    if hn:
        all_docs.extend(hn)
        sources_used.append("HN Algolia")

    # 10. UN News RSS
    un_rss = fetch_un_news_rss()
    if un_rss:
        all_docs.extend(un_rss)
        sources_used.append(f"UN News RSS ({len(UN_RSS_FEEDS)} feeds)")

    # 11. GDELT GKG
    gdelt_gkg = fetch_gdelt_gkg_themes()
    if gdelt_gkg:
        all_docs.extend(gdelt_gkg)
        sources_used.append("GDELT GKG 2.0")

    if not all_docs or len(all_docs) < 10:
        raise RuntimeError(
            "All 11 news sources unavailable "
            "(GDELT DOC 2.0, 5x RSS, Wikipedia API, Reddit, HN Algolia, "
            "UN News RSS, GDELT GKG 2.0). Check network connectivity."
        )

    # De-duplicate (exact duplicates)
    seen: set[str] = set()
    deduped: list[str] = []
    for doc in all_docs:
        if doc not in seen:
            seen.add(doc)
            deduped.append(doc)

    source_label = f"Live: {' + '.join(sources_used)} ({len(deduped)} docs)"
    return deduped, source_label
