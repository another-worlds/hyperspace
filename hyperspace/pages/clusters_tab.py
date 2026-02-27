"""Info Clusters tab: BERTopic on real text data."""
from __future__ import annotations

import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from hyperspace.config import PLOTLY_LAYOUT
from hyperspace.data.news import get_text_data
from hyperspace.models.topic_model import fit_topic_model
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
                st.plotly_chart(fig, use_container_width=True)

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
