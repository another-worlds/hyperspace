"""Finance-Neural Block tab: TFT forecasting with real data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hyperspace.config import DEFAULT_TICKERS, PLOTLY_LAYOUT
from hyperspace.data.finance import get_ohlcv
from hyperspace.models.tft_forecast import fit_tft
from hyperspace.pages._report_section import render_interpretability_report
from hyperspace.viz.charts import candlestick_chart, forecast_chart, source_badge


def render() -> None:
    """Render the Finance-Neural Block tab."""
    st.markdown("## Finance-Neural Block")
    st.markdown(
        "Temporal Fusion Transformer + cross-ticker correlations + multi-horizon "
        "probabilistic forecasting. *Backbone: TFT (Lim et al., 2021).*"
    )

    # Use shared session state tickers (set by sidebar)
    tickers = st.session_state.get("tickers", DEFAULT_TICKERS) or DEFAULT_TICKERS
    st.caption(f"Using tickers from sidebar: {', '.join(tickers)}")

    sc1, sc2, sc3 = st.columns(3)
    encoder_length = sc1.slider("Encoder Length", 24, 90, 48, key="finance_encoder_len")
    prediction_length = sc2.slider("Prediction Length", 6, 30, 12, key="finance_pred_len")
    hidden_size = sc3.slider("Hidden Size", 16, 64, 32, key="finance_hidden_size")

    compute_tft = st.button("Compute TFT Forecast", type="primary", key="finance_compute")

    # Show results from pipeline if available
    finance_result = st.session_state.get("finance_result")

    if compute_tft or finance_result:
        # Price history
        try:
            ohlcv_df, ohlcv_src = get_ohlcv(tickers)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        st.markdown(f"### Price History {source_badge(ohlcv_src)}", unsafe_allow_html=True)

        for ticker in tickers:
            tdf = ohlcv_df[ohlcv_df.Ticker == ticker] if "Ticker" in ohlcv_df.columns else ohlcv_df
            if len(tdf) > 0:
                fig = candlestick_chart(ohlcv_df, ticker)
                st.plotly_chart(fig, use_container_width=True, key=f"finance_candlestick_{ticker}")
        st.caption(
            "v3.0 — Raw price history feeds the TFT encoder. Attention weights "
            "over these time steps populate UKT indices 0–15 (temporal-pattern region), "
            "enabling kernel-level traceability back to specific market regimes."
        )

        # TFT forecast
        st.markdown("### Multi-Horizon Probabilistic Forecast")

        if compute_tft:
            with st.spinner("Fitting TFT (3 epochs, CPU)..."):
                result = fit_tft(tuple(tickers), hidden_size, encoder_length, prediction_length)
            if result is None:
                st.error("TFT fitting failed. Ensure live market data is reachable.")
                return
            st.session_state.finance_result = result
            finance_result = result

        if finance_result:
            if "quantiles" in finance_result:
                q = finance_result["quantiles"]
                if len(q.shape) == 3:
                    q_mean = q.mean(axis=0)
                    x_axis = list(range(q_mean.shape[0]))
                    if q_mean.shape[1] < 2:
                        st.warning("Insufficient quantile columns in forecast output.")
                    else:
                        fig = forecast_chart(
                            x_axis,
                            q_mean[:, 0],
                            q_mean[:, q_mean.shape[1] // 2],
                            q_mean[:, -1],
                            title="TFT Multi-Quantile Forecast",
                        )
                        st.plotly_chart(fig, use_container_width=True, key="finance_forecast_3d")
                        st.caption(
                            "v3.0 — Quantile forecast uncertainty bounds support the "
                            "contestability guarantee: wide bands signal low confidence."
                        )
                elif len(q.shape) == 2:
                    x_axis = list(range(q.shape[0]))
                    fig = forecast_chart(
                        x_axis, q[:, 0], q[:, q.shape[1] // 2], q[:, -1],
                        title="TFT Multi-Quantile Forecast",
                    )
                    st.plotly_chart(fig, use_container_width=True, key="finance_forecast_2d")
                    st.caption(
                        "v3.0 — Quantile forecast uncertainty bounds support the "
                        "contestability guarantee: wide bands signal low confidence."
                    )

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
                        title="Attention over Encoder Time Steps",
                    )
                    fig.update_layout(**PLOTLY_LAYOUT, height=250)
                    st.plotly_chart(fig, use_container_width=True, key="finance_attention")
                    st.caption(
                        "v3.0 — Attention heatmap shows which encoder time steps "
                        "the TFT attends to, directly populating UKT temporal-pattern "
                        "features and enabling provenance tracing."
                    )

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
                    st.bar_chart(pd.DataFrame({"Value": fv}))

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
                    title="Cross-Ticker Correlation Matrix",
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=350)
                st.plotly_chart(fig, use_container_width=True, key="finance_correlation_heatmap")
                st.caption(
                    "v3.0 — Cross-ticker correlations reveal co-movement patterns "
                    "that inform the TFT's multi-entity encoder and contribute to "
                    "UKT kernel structure."
                )

            # ----------------------------------------------------------- #
            # Fitting Metrics                                              #
            # ----------------------------------------------------------- #
            st.markdown("---")
            st.markdown("### Fitting Metrics")
            fm1, fm2, fm3, fm4 = st.columns(4)
            fm1.metric("Model Parameters", f"{finance_result.get('model_params', 0):,}")
            fm2.metric("Training Epochs", "3")
            fm3.metric("Hidden Size", str(hidden_size))
            fm4.metric("Encoder Length", str(encoder_length))

            if "attention" in finance_result:
                att = finance_result["attention"]
                att_entropy = float(-np.sum(
                    att.flatten() * np.log(att.flatten() + 1e-8)
                ))
                st.metric("Attention Entropy", f"{att_entropy:.3f}",
                          help="Higher entropy = more distributed attention across time steps")

            # ----------------------------------------------------------- #
            # Test Metrics                                                 #
            # ----------------------------------------------------------- #
            st.markdown("### Test Metrics")
            if "quantiles" in finance_result:
                q = finance_result["quantiles"]
                if len(q.shape) == 3:
                    q_mean = q.mean(axis=0)
                elif len(q.shape) == 2:
                    q_mean = q
                else:
                    q_mean = None

                if q_mean is not None and q_mean.shape[1] >= 2:
                    q_low = q_mean[:, 0]
                    q_mid = q_mean[:, q_mean.shape[1] // 2]
                    q_high = q_mean[:, -1]

                    spread = float(np.mean(q_high - q_low))
                    mid_range = float(q_mid.max() - q_mid.min())
                    coverage = float(np.mean(q_high - q_low) / (np.mean(np.abs(q_mid)) + 1e-8))

                    tm1, tm2, tm3 = st.columns(3)
                    tm1.metric("Mean Quantile Spread (q90 − q10)", f"{spread:.4f}",
                               help="Average width of prediction interval")
                    tm2.metric("Median Forecast Range", f"{mid_range:.4f}",
                               help="Range of the median (q50) forecast")
                    tm3.metric("Relative Uncertainty", f"{coverage:.2%}",
                               help="Spread / |median| — lower = more confident")
            else:
                st.info("Run TFT forecast to compute test metrics.")

            if "encoder_importance" in finance_result:
                enc = finance_result["encoder_importance"].flatten()
                top_var = enc[:min(10, len(enc))]
                gini = float(
                    np.sum(np.abs(np.subtract.outer(top_var, top_var)))
                    / (2 * len(top_var) * (np.sum(top_var) + 1e-8))
                )
                st.metric("Encoder Variable Concentration (Gini)", f"{gini:.3f}",
                          help="0 = uniform importance, 1 = single variable dominates")

            # ----------------------------------------------------------- #
            # Interpretability Report                                      #
            # ----------------------------------------------------------- #
            st.markdown("---")
            render_interpretability_report("Finance")
