"""Finance-Neural Block tab: TFT forecasting with real data."""
from __future__ import annotations

import streamlit as st

from hyperspace.config import (
    DEFAULT_TICKERS,
    FORECAST_CONFIDENCE_HIGH,
    FORECAST_CONFIDENCE_MODERATE,
    PLOTLY_LAYOUT,
)
from hyperspace.core.caching import (
    get_or_compute_dataframe,
    get_or_compute_figure,
    hash_dataframe,
    hash_list,
)
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

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        compute_tft = st.button("Compute TFT Forecast", type="primary", key="finance_compute")
    with col_btn2:
        force_retrain = st.button("🔄 Retrain", key="finance_retrain",
                                  help="Clear cached forecast and refit TFT model")
    if force_retrain:
        st.session_state.pop("finance_result", None)
        compute_tft = True

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

        _ohlcv_hash = hash_dataframe(ohlcv_df, "ohlcv")
        for ticker in tickers:
            tdf = ohlcv_df[ohlcv_df.Ticker == ticker] if "Ticker" in ohlcv_df.columns else ohlcv_df
            if len(tdf) > 0:
                fig = get_or_compute_figure(
                    f"candle_{_ohlcv_hash}_{ticker}",
                    lambda t=ticker: candlestick_chart(ohlcv_df, t),
                    force_recompute=force_retrain,
                )
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

            # Correlation heatmap
            st.markdown("### Cross-Ticker Correlation")
            if len(tickers) > 1 and "Ticker" in ohlcv_df.columns:
                _tickers_hash = hash_list(sorted(tickers), "tickers")
                corr = get_or_compute_dataframe(
                    f"corr_{_ohlcv_hash}_{_tickers_hash}",
                    lambda: ohlcv_df.pivot_table(
                        index="Date", columns="Ticker", values="Close",
                    ).corr(),
                    force_recompute=force_retrain,
                )

                import plotly.express as px

                def _build_corr_fig():
                    _fig = px.imshow(
                        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Cross-Ticker Correlation Matrix",
                    )
                    _fig.update_traces(
                        hovertemplate="Ticker X: %{x}<br>Ticker Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
                    )
                    _fig.update_layout(**PLOTLY_LAYOUT, height=350)
                    return _fig

                fig = get_or_compute_figure(
                    f"corrfig_{_ohlcv_hash}_{_tickers_hash}",
                    _build_corr_fig,
                    force_recompute=force_retrain,
                )
                st.plotly_chart(fig, use_container_width=True, key="finance_correlation_heatmap")
                st.caption(
                    "v3.0 — Cross-ticker correlations reveal co-movement patterns "
                    "that inform the TFT's multi-entity encoder and contribute to "
                    "UKT kernel structure."
                )

            # ----------------------------------------------------------- #
            # Forecast Confidence Summary                                 #
            # ----------------------------------------------------------- #
            if "quantiles" in finance_result:
                import numpy as np
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
                    coverage = float(np.mean(q_high - q_low) / (np.mean(np.abs(q_mid)) + 1e-8))

                    if coverage < FORECAST_CONFIDENCE_HIGH:
                        st.success(
                            f"Forecast confidence: **HIGH** — uncertainty band is {coverage:.1%} "
                            "of forecast magnitude. Conclusions from this block are robust."
                        )
                    elif coverage < FORECAST_CONFIDENCE_MODERATE:
                        st.warning(
                            f"Forecast confidence: **MODERATE** — uncertainty band is {coverage:.1%} "
                            "of forecast magnitude. Exercise caution when citing specific values."
                        )
                    else:
                        st.error(
                            f"Forecast confidence: **LOW** — uncertainty band is {coverage:.1%} "
                            "of forecast magnitude. Wide bands indicate high model uncertainty; "
                            "directional trends may still be informative."
                        )

            # ----------------------------------------------------------- #
            # Interpretability Report                                      #
            # ----------------------------------------------------------- #
            st.markdown("---")
            render_interpretability_report("Finance")
