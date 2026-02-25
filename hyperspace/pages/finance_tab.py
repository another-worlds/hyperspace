"""Finance-Neural Block tab: TFT forecasting with real data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hyperspace.config import DEFAULT_TICKERS, PLOTLY_LAYOUT
from hyperspace.data.finance import get_ohlcv
from hyperspace.models.tft_forecast import fit_tft, mock_forecast
from hyperspace.viz.charts import candlestick_chart, forecast_chart, source_badge


def render() -> None:
    """Render the Finance-Neural Block tab."""
    st.markdown("## Finance-Neural Block")
    st.markdown(
        "Temporal Fusion Transformer + cross-ticker correlations + multi-horizon "
        "probabilistic forecasting. *Backbone: TFT (Lim et al., 2021).*"
    )

    fc1, fc2 = st.columns(2)
    tickers = fc1.multiselect(
        "Tickers", ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN"],
        default=DEFAULT_TICKERS,
    )
    fc2.markdown("")

    sc1, sc2, sc3 = st.columns(3)
    encoder_length = sc1.slider("Encoder Length", 24, 90, 48)
    prediction_length = sc2.slider("Prediction Length", 6, 30, 12)
    hidden_size = sc3.slider("Hidden Size", 16, 64, 32)

    compute_tft = st.button("Compute TFT Forecast", type="primary", key="finance_compute")

    # Show results from pipeline if available
    finance_result = st.session_state.get("finance_result")

    if compute_tft or finance_result:
        # Price history
        ohlcv_df, ohlcv_src = get_ohlcv(tickers)
        st.markdown(f"### Price History {source_badge(ohlcv_src)}", unsafe_allow_html=True)

        for ticker in tickers:
            tdf = ohlcv_df[ohlcv_df.Ticker == ticker] if "Ticker" in ohlcv_df.columns else ohlcv_df
            if len(tdf) > 0:
                fig = candlestick_chart(ohlcv_df, ticker)
                st.plotly_chart(fig, use_container_width=True)

        # TFT forecast
        st.markdown("### Multi-Horizon Probabilistic Forecast")

        if compute_tft:
            with st.spinner("Fitting TFT (3 epochs, CPU)..."):
                result = fit_tft(tuple(tickers), hidden_size, encoder_length, prediction_length)
            if result is None:
                result = mock_forecast(prediction_length)
            st.session_state.finance_result = result
            finance_result = result

        if finance_result:
            if "quantiles" in finance_result:
                q = finance_result["quantiles"]
                if len(q.shape) == 3:
                    q_mean = q.mean(axis=0)
                    x_axis = list(range(q_mean.shape[0]))
                    fig = forecast_chart(
                        x_axis,
                        q_mean[:, 0] if q_mean.shape[1] > 0 else q_mean[:, 0],
                        q_mean[:, q_mean.shape[1] // 2],
                        q_mean[:, -1],
                        title="TFT Multi-Quantile Forecast",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif len(q.shape) == 2:
                    x_axis = list(range(q.shape[0]))
                    fig = forecast_chart(
                        x_axis, q[:, 0], q[:, q.shape[1] // 2], q[:, -1],
                        title="TFT Multi-Quantile Forecast",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            elif "q50" in finance_result:
                x_axis = list(range(len(finance_result["q50"])))
                fig = forecast_chart(
                    x_axis, finance_result["q10"], finance_result["q50"],
                    finance_result["q90"], title="Mock Multi-Quantile Forecast",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Show source badge
            src = finance_result.get("data_source", "unknown")
            st.markdown(f"**Model data**: {source_badge(src)}", unsafe_allow_html=True)

            # Interpretation (if real TFT)
            if "attention" in finance_result:
                with st.expander("TFT Interpretation (Attention + Variable Importance)"):
                    st.markdown("**Temporal Attention Weights**")
                    import plotly.express as px
                    att = finance_result["attention"]
                    fig = px.imshow(
                        att.reshape(-1, att.shape[-1]) if att.ndim > 2 else att.reshape(1, -1),
                        color_continuous_scale="Viridis",
                        **PLOTLY_LAYOUT,
                        title="Attention over Encoder Time Steps",
                    )
                    fig.update_layout(height=250, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
                    st.plotly_chart(fig, use_container_width=True)

                    if "encoder_importance" in finance_result:
                        enc = finance_result["encoder_importance"]
                        st.markdown("**Encoder Variable Importance**")
                        st.bar_chart(pd.DataFrame(
                            enc.flatten()[:10], columns=["Importance"],
                        ))

            # UKT contribution
            if "features_for_ukt" in finance_result:
                with st.expander("UKT Contribution (Finance Feature Vector)"):
                    fv = finance_result["features_for_ukt"]
                    st.bar_chart(pd.DataFrame(fv, columns=["Value"]))

            # Correlation heatmap
            st.markdown("### Cross-Ticker Correlation")
            if len(tickers) > 1 and "Ticker" in ohlcv_df.columns:
                pivot = ohlcv_df.pivot_table(
                    index="Date", columns="Ticker", values="Close",
                )
                corr = pivot.corr()
                import plotly.express as px
                fig = px.imshow(
                    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    **PLOTLY_LAYOUT,
                    title="Cross-Ticker Correlation Matrix",
                )
                fig.update_layout(height=350, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
                st.plotly_chart(fig, use_container_width=True)
