"""Real news/text data via RSS feeds and sklearn, with fallback."""
from __future__ import annotations

import streamlit as st

from hyperspace.config import NEWS_SNIPPETS


@st.cache_resource(ttl=3600, show_spinner=False)
def fetch_rss_headlines(max_per_feed: int = 25) -> list[str] | None:
    """Fetch real headlines from RSS feeds. Returns None on failure."""
    try:
        import feedparser
        feeds = {
            "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
            "BBC Tech": "http://feeds.bbci.co.uk/news/technology/rss.xml",
            "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
        }
        headlines: list[str] = []
        for name, url in feeds.items():
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = f"{title}. {summary}".strip() if summary else title.strip()
                if len(text) > 20:
                    headlines.append(text)
        return headlines if len(headlines) >= 10 else None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def fetch_newsgroups(n_docs: int = 200) -> list[str] | None:
    """Fetch from sklearn 20newsgroups -- always available offline."""
    try:
        from sklearn.datasets import fetch_20newsgroups
        categories = [
            "talk.politics.misc", "talk.politics.mideast",
            "talk.politics.guns", "sci.space", "sci.crypt",
            "soc.religion.christian",
        ]
        data = fetch_20newsgroups(
            subset="train", categories=categories,
            remove=("headers", "footers", "quotes"),
        )
        docs = [d[:500].strip() for d in data.data if len(d.strip()) > 50]
        return docs[:n_docs] if len(docs) >= 20 else None
    except Exception:
        return None


def get_text_data() -> tuple[list[str], str]:
    """Get text documents with fallback chain.

    Returns:
        (documents, source_label)
    """
    # Try RSS feeds first
    rss = fetch_rss_headlines()
    if rss and len(rss) >= 15:
        return rss, "Live: RSS feeds"

    # Try 20newsgroups
    ng = fetch_newsgroups(200)
    if ng and len(ng) >= 20:
        return ng, "Offline: 20newsgroups"

    # Ultimate fallback
    return NEWS_SNIPPETS, "Fallback: built-in snippets"
