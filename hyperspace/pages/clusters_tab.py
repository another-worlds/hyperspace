"""Info Clusters tab: BERTopic on real text data."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from hyperspace.config import PLOTLY_LAYOUT
from hyperspace.data.news import get_text_data
from hyperspace.models.topic_model import fit_topic_model, mock_clusters
from hyperspace.viz.charts import source_badge


def render() -> None:
    """Render the Informational Cluster Mapping tab."""
    st.markdown("## Informational Cluster Mapping")
    st.markdown(
        "BERTopic multilingual clustering on real news/text data. "
        "*Backbone: BERTopic + sentence-transformers.*"
    )

    cluster_btn = st.button("Fit Topic Model", type="primary", key="cluster_fit")

    # Use pipeline result if available
    cluster_result = st.session_state.get("cluster_result")

    if cluster_btn or cluster_result:
        if cluster_btn:
            with st.spinner("Fitting BERTopic on real documents..."):
                docs, src = get_text_data()
                docs_hash = hashlib.md5("".join(docs[:5]).encode()).hexdigest()[:8]
                result = fit_topic_model(docs_hash)
                if result is None:
                    mock_df, mock_features = mock_clusters(docs)
                    result = dict(
                        mock_df=mock_df, docs=docs,
                        features_for_ukt=mock_features,
                        data_source="Fallback: keyword clusters",
                    )
                st.session_state.cluster_result = result
                cluster_result = result

        if cluster_result:
            src = cluster_result.get("data_source", "unknown")
            st.markdown(f"**Data source**: {source_badge(src)}", unsafe_allow_html=True)

            # Display topics
            if "model" in cluster_result and cluster_result["model"] is not None:
                model = cluster_result["model"]
                topics = cluster_result["topics"]
                docs = cluster_result.get("docs", [])

                st.markdown("### Discovered Topics")
                topic_info = model.get_topic_info()
                st.dataframe(topic_info.head(15), use_container_width=True)

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

            elif "mock_df" in cluster_result:
                mock_df = cluster_result["mock_df"]
                st.markdown("### Cluster Results (Keyword Fallback)")
                st.dataframe(mock_df, use_container_width=True, height=350)

                tc = mock_df.Topic_Name.value_counts()
                fig = px.bar(
                    x=tc.index, y=tc.values,
                    color=tc.values, color_continuous_scale="Viridis",
                    title="Cluster Distribution",
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=350)
                st.plotly_chart(fig, use_container_width=True)

            # UKT contribution
            if "features_for_ukt" in cluster_result:
                with st.expander("UKT Contribution (Cluster Feature Vector)"):
                    fv = cluster_result["features_for_ukt"]
                    st.bar_chart(pd.DataFrame(fv, columns=["Value"]))
