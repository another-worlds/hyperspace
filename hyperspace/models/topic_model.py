"""BERTopic clustering on live country-focused documents for UKT."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.config import UKT_FEATURE_DIM
from hyperspace.data.news import get_text_data


def _topic_feature_meta(model, topics: list[int]) -> dict[int, dict]:
    """Build semantic feature metadata from discovered topics and keywords."""
    meta: dict[int, dict] = {}
    topic_ids = [t for t in pd.Series(topics).value_counts().index.tolist() if t != -1]
    for rank, topic_id in enumerate(topic_ids[:16]):
        label = f"topic_share_{topic_id}"
        try:
            terms = model.get_topic(topic_id) or []
            top_terms = [t[0] for t in terms[:3] if t and t[0]]
            if top_terms:
                label = f"topic_{topic_id}_{'_'.join(top_terms)}"
        except Exception:
            pass
        idx = 16 + rank
        meta[idx] = {
            "label": label,
            "block": "clusters",
            "metric": "topic_distribution_share",
            "entity": f"topic_{topic_id}",
            "source": "BERTopic",
        }
    return meta


@st.cache_resource(show_spinner=False)
def fit_topic_model(docs_key: str, docs: list[str] | None = None, data_source: str | None = None) -> dict | None:
    """Fit BERTopic on real documents.

    Args:
        docs_key: Cache key (hash of docs content) for st.cache_resource.

    Returns dict with: model, topics, probs, doc_embeddings, topic_embeddings,
    features_for_ukt, data_source.
    """
    try:
        from bertopic import BERTopic

        if docs is None:
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

        # Build UKT feature vector from topic distribution statistics.
        # Only the semantic-embedding region (indices 16-31) is populated here.
        # The structural-centrality region (32-47) is reserved exclusively for
        # the graph engine block to preserve UKT semantic coherence.
        features_for_ukt = np.zeros(UKT_FEATURE_DIM)
        topic_counts = pd.Series(topics).value_counts().values.astype(float)
        # Normalize topic distribution
        if len(topic_counts) > 0:
            topic_counts = topic_counts / (topic_counts.sum() + 1e-8)
        # Place topic shares in semantic-embedding region (indices 16-31)
        features_for_ukt[16:16 + min(16, len(topic_counts))] = topic_counts[:16]
        # If topic embeddings are available, summarize them into the remaining
        # semantic slots (24-31) using the L2 norm of the mean embedding — a
        # scalar summary that stays within the semantic region and does not
        # collide with the structural-centrality region (32-47).
        if topic_embeddings is not None and len(topic_embeddings) > 0:
            te_mean = topic_embeddings.mean(axis=0)
            # Scalar statistics: norm, std, min, max of mean embedding vector
            te_stats = np.array([
                float(np.linalg.norm(te_mean)),
                float(np.std(te_mean)),
                float(np.min(te_mean)),
                float(np.max(te_mean)),
                float(np.mean(np.abs(te_mean))),
                float(np.percentile(te_mean, 25)),
                float(np.percentile(te_mean, 75)),
                float(np.linalg.norm(te_mean) / (np.std(te_mean) + 1e-8)),
            ])
            features_for_ukt[24:32] = te_stats

        feature_meta = _topic_feature_meta(model, topics)

        return dict(
            model=model,
            topics=topics,
            probs=probs,
            docs=docs,
            doc_embeddings=doc_embeddings,
            topic_embeddings=topic_embeddings,
            features_for_ukt=features_for_ukt,
            feature_meta=feature_meta,
            data_source=data_source,
        )
    except Exception as e:
        st.warning(f"BERTopic unavailable ({e}); no clustering fallback is used.")
        return None

