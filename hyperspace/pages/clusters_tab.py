"""Info Clusters tab: BERTopic on real text data."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from hyperspace.config import PLOTLY_LAYOUT
from hyperspace.data.news import get_text_data
from hyperspace.models.topic_model import fit_topic_model
from hyperspace.pages._report_section import render_interpretability_report
from hyperspace.viz.charts import source_badge


def render() -> None:
    """Render the Informational Cluster Mapping tab."""
    st.markdown("## Informational Cluster Mapping")
    st.markdown(
        "BERTopic multilingual clustering on live country-focused news. "
        "*Backbone: BERTopic + sentence-transformers.*"
    )

    cluster_btn = st.button("Fit Topic Model", type="primary", key="cluster_fit")

    # Use pipeline result if available
    cluster_result = st.session_state.get("cluster_result")

    if cluster_btn or cluster_result:
        if cluster_btn:
            with st.spinner("Fitting BERTopic on real documents..."):
                try:
                    docs, src = get_text_data()
                except RuntimeError as exc:
                    st.error(str(exc))
                    return
                docs_hash = hashlib.md5("".join(docs[:5]).encode()).hexdigest()[:8]
                result = fit_topic_model(docs_hash, docs=docs, data_source=src)
                if result is None:
                    st.error("BERTopic unavailable; cluster fallback was intentionally removed.")
                    return
                st.session_state.cluster_result = result
                cluster_result = result

        if cluster_result:
            src = cluster_result.get("data_source", "unknown")
            st.markdown(f"**Data source**: {source_badge(src)}", unsafe_allow_html=True)

            # Display topics
            if "model" in cluster_result and cluster_result["model"] is not None:
                model = cluster_result["model"]
                topics = cluster_result.get("topics", [])
                if not topics:
                    st.warning("No topic assignments available.")
                    return
                docs = cluster_result.get("docs", [])

                st.markdown("### Discovered Topics")
                try:
                    topic_info = model.get_topic_info()
                    st.dataframe(topic_info.head(15), use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display topic info: {e}")

                # Topic distribution
                st.markdown("### Topic Distribution")
                tc = pd.Series(topics).value_counts().head(10)
                fig = px.bar(
                    x=[str(x) for x in tc.index], y=tc.values,
                    color=tc.values, color_continuous_scale="Viridis",
                    title="Document Count per Topic",
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=350)
                st.plotly_chart(fig, use_container_width=True, key="clusters_topic_distribution")
                st.caption(
                    "v3.0 — Topic distribution populates UKT indices 16–23 "
                    "(semantic-embedding region). A dominant single topic signals "
                    "focused discourse; flat distributions indicate fragmented "
                    "information — both are traceable governance signals."
                )

                # Show sample docs per topic
                with st.expander("Sample Documents by Topic"):
                    for topic_id in sorted(set(topics))[:5]:
                        if topic_id == -1:
                            continue
                        st.markdown(f"**Topic {topic_id}**")
                        indices = [i for i, t in enumerate(topics) if t == topic_id][:3]
                        for idx in indices:
                            if idx < len(docs):
                                st.caption(docs[idx][:200] + "...")

            # UKT contribution
            if "features_for_ukt" in cluster_result:
                with st.expander("UKT Contribution (Cluster Feature Vector)"):
                    fv = cluster_result["features_for_ukt"]
                    st.bar_chart(pd.DataFrame({"Value": fv}))

            # ----------------------------------------------------------- #
            # Fitting Metrics                                              #
            # ----------------------------------------------------------- #
            st.markdown("---")
            st.markdown("### Fitting Metrics")
            if "model" in cluster_result and cluster_result["model"] is not None:
                model = cluster_result["model"]
                topics = cluster_result.get("topics", [])
                n_topics = len(set(t for t in topics if t != -1))
                n_outliers = sum(1 for t in topics if t == -1)
                n_docs = len(topics)

                fm1, fm2, fm3, fm4 = st.columns(4)
                fm1.metric("Topics Discovered", str(n_topics))
                fm2.metric("Documents Processed", str(n_docs))
                fm3.metric("Outlier Documents", str(n_outliers))
                fm4.metric("Outlier Ratio", f"{n_outliers / max(n_docs, 1):.1%}")

                te = cluster_result.get("topic_embeddings")
                if te is not None and len(te) > 1:
                    # Inter-topic distance (mean pairwise cosine distance)
                    from numpy.linalg import norm
                    norms = norm(te, axis=1, keepdims=True) + 1e-8
                    cosine_sim = (te @ te.T) / (norms @ norms.T)
                    np.fill_diagonal(cosine_sim, 0)
                    n_te = cosine_sim.shape[0]
                    mean_sim = float(cosine_sim.sum() / (n_te * (n_te - 1) + 1e-8))
                    st.metric("Mean Inter-Topic Similarity", f"{mean_sim:.3f}",
                              help="Lower = more distinct topics; higher = overlapping topics")
            else:
                st.info("Run topic model to generate fitting metrics.")

            # ----------------------------------------------------------- #
            # Test Metrics                                                 #
            # ----------------------------------------------------------- #
            st.markdown("### Test Metrics")
            if "topics" in cluster_result:
                topics = cluster_result["topics"]
                tc = pd.Series(topics)
                valid = tc[tc != -1]

                if len(valid) > 0:
                    # Topic entropy (uniformity of assignment)
                    probs = valid.value_counts(normalize=True).values
                    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
                    max_entropy = float(np.log(len(probs) + 1e-8))
                    normalized_entropy = entropy / (max_entropy + 1e-8)

                    # Topic concentration (Gini of topic sizes)
                    sizes = valid.value_counts().values.astype(float)
                    gini = float(
                        np.sum(np.abs(np.subtract.outer(sizes, sizes)))
                        / (2 * len(sizes) * (np.sum(sizes) + 1e-8))
                    )

                    # Largest topic share
                    largest_share = float(sizes.max() / sizes.sum())

                    tm1, tm2, tm3 = st.columns(3)
                    tm1.metric("Topic Entropy", f"{entropy:.3f}",
                               help="Higher = more uniform topic distribution")
                    tm2.metric("Normalized Entropy", f"{normalized_entropy:.2%}",
                               help="100% = perfectly uniform, 0% = single topic dominates")
                    tm3.metric("Topic Size Gini", f"{gini:.3f}",
                               help="0 = equal sizes, 1 = one topic holds all docs")
                    st.metric("Largest Topic Share", f"{largest_share:.1%}")

                # Topic probability confidence (if probs available)
                probs_arr = cluster_result.get("probs")
                if probs_arr is not None:
                    try:
                        prob_np = np.array(probs_arr)
                        if prob_np.ndim == 1:
                            mean_conf = float(np.mean(prob_np))
                        else:
                            mean_conf = float(np.mean(np.max(prob_np, axis=1)))
                        st.metric("Mean Assignment Confidence", f"{mean_conf:.3f}",
                                  help="BERTopic probability of most likely topic per document")
                    except Exception:
                        pass
            else:
                st.info("Run topic model to generate test metrics.")

            # ----------------------------------------------------------- #
            # Interpretability Report                                      #
            # ----------------------------------------------------------- #
            st.markdown("---")
            render_interpretability_report("Clusters")
