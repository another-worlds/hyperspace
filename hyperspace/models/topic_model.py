"""BERTopic clustering on real documents with embedding extraction for UKT."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.config import UKT_FEATURE_DIM
from hyperspace.data.news import get_text_data
from hyperspace.data.synthetic import seed


@st.cache_resource(show_spinner=False)
def fit_topic_model(docs_key: str) -> dict | None:
    """Fit BERTopic on real documents.

    Args:
        docs_key: Cache key (hash of docs content) for st.cache_resource.

    Returns dict with: model, topics, probs, doc_embeddings, topic_embeddings,
    features_for_ukt, data_source.
    """
    try:
        from bertopic import BERTopic

        docs, data_source = get_text_data()

        model = BERTopic(
            language="multilingual",
            min_topic_size=3,
            nr_topics="auto",
        )
        topics, probs = model.fit_transform(docs)

        # Extract embeddings if available
        doc_embeddings = None
        topic_embeddings = None
        try:
            topic_embeddings = np.array(model.topic_embeddings_)
        except Exception:
            pass

        # Build UKT feature vector from topic distribution statistics
        features_for_ukt = np.zeros(UKT_FEATURE_DIM)
        topic_counts = pd.Series(topics).value_counts().values.astype(float)
        # Normalize topic distribution
        if len(topic_counts) > 0:
            topic_counts = topic_counts / (topic_counts.sum() + 1e-8)
        # Place in semantic-embedding region (indices 16-31)
        features_for_ukt[16:16 + min(16, len(topic_counts))] = topic_counts[:16]
        # Add topic embedding means if available
        if topic_embeddings is not None and len(topic_embeddings) > 0:
            te_mean = topic_embeddings.mean(axis=0)
            n_fill = min(16, len(te_mean))
            features_for_ukt[32:32 + n_fill] = te_mean[:n_fill]

        return dict(
            model=model,
            topics=topics,
            probs=probs,
            docs=docs,
            doc_embeddings=doc_embeddings,
            topic_embeddings=topic_embeddings,
            features_for_ukt=features_for_ukt,
            data_source=data_source,
        )
    except Exception as e:
        st.warning(f"BERTopic unavailable ({e}); using mock clusters.")
        return None


def mock_clusters(docs: list[str], s: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Simple keyword-based mock clustering fallback.

    Returns (dataframe, ukt_features).
    """
    from hyperspace.config import NEWS_SNIPPETS

    rng = seed(s)
    keywords = {
        0: ["USA", "US", "America", "NATO", "sanction", "Quad"],
        1: ["China", "BRI", "Belt", "Beijing", "semiconductor"],
        2: ["Russia", "Moscow", "Kremlin", "energy", "sanctions"],
        3: ["India", "Indo", "nuclear", "BRICS", "Delhi"],
        4: ["Britain", "UK", "British", "AUKUS", "Commonwealth"],
        5: ["Brazil", "BRICS", "Latin", "Mercosur", "climate"],
    }
    topic_names = {
        0: "US-Led Western Alliance & Sanctions",
        1: "China Economic Expansion & Tech Competition",
        2: "Russia Strategic Posture & Energy Diplomacy",
        3: "India Multi-Alignment & Defence Partnerships",
        4: "Britain Post-Brexit Global Pivot",
        5: "Brazil BRICS Mediation & South-South Ties",
    }
    labels = []
    for doc in docs:
        scores = {t: sum(1 for kw in kws if kw.lower() in doc.lower())
                  for t, kws in keywords.items()}
        best = max(scores, key=scores.get) if max(scores.values()) > 0 else rng.integers(0, 6)
        labels.append(best)

    # Build UKT feature vector matching real BERTopic's layout:
    #   indices 16-31 = normalized topic distribution (semantic-embedding region)
    #   indices 32-47 = simulated topic embedding means
    features = np.zeros(UKT_FEATURE_DIM)
    topic_counts = pd.Series(labels).value_counts().values.astype(float)
    if len(topic_counts) > 0:
        topic_counts = topic_counts / (topic_counts.sum() + 1e-8)
    features[16:16 + min(16, len(topic_counts))] = topic_counts[:16]
    # Simulated topic embedding centroids
    features[32:37] = rng.uniform(0.1, 0.5, 5)

    df = pd.DataFrame(dict(
        Document=docs,
        Topic=labels,
        Topic_Name=[topic_names.get(l, "Misc") for l in labels],
        Confidence=rng.uniform(0.6, 0.98, len(docs)),
    ))
    return df, features
