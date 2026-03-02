"""TFT forecasting: trains on real OHLCV data, extracts interpretable features for UKT."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.config import UKT_FEATURE_DIM
from hyperspace.data.finance import get_ohlcv, get_tft_data


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a torch.Tensor or np.ndarray to numpy array safely."""
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _finance_feature_meta_from_attention(attention: np.ndarray, encoder_importance: np.ndarray,
                                         decoder_importance: np.ndarray) -> dict[int, dict]:
    """Build metadata labels for finance UKT slots from model interpretation output."""
    meta: dict[int, dict] = {}
    att_flat = attention.flatten()
    for i in range(min(16, len(att_flat))):
        lag = i + 1
        meta[i] = {
            "label": f"tft_attention_lag_{lag:02d}",
            "block": "finance",
            "metric": "attention_weight",
            "time_scope": f"lag_{lag}",
            "source": "TemporalFusionTransformer",
        }

    enc_flat = encoder_importance.flatten()
    for i in range(min(8, len(enc_flat))):
        idx = 16 + i
        meta[idx] = {
            "label": f"tft_encoder_importance_{i+1}",
            "block": "finance",
            "metric": "encoder_variable_importance",
            "source": "TemporalFusionTransformer",
        }

    dec_flat = decoder_importance.flatten()
    for i in range(min(3, len(dec_flat))):  # slots 24-26 only; 27-31 reserved for macro
        idx = 24 + i
        meta[idx] = {
            "label": f"tft_decoder_importance_{i+1}",
            "block": "finance",
            "metric": "decoder_variable_importance",
            "source": "TemporalFusionTransformer",
        }
    return meta


@st.cache_data(show_spinner=False)
def fit_tft(
    tickers: tuple[str, ...],
    hidden: int = 32,
    encoder_len: int = 48,
    prediction_len: int = 12,
    max_epochs: int = 3,
) -> dict | None:
    """Fit TFT on real OHLCV data. Return predictions + interpretation features.

    Returns dict with: quantiles, attention, encoder_importance, decoder_importance,
    model_params, features_for_ukt, feature_meta, data_source.
    Returns None on failure.
    """
    try:
        import lightning.pytorch as pl
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss

        pl.seed_everything(42)

        df, data_source = get_tft_data(list(tickers), encoder_len, prediction_len)

        max_time = df.time_idx.max()
        train_cutoff = max_time - prediction_len

        # Determine which static macro columns are present in the dataframe
        macro_cols = ["gdp_growth", "inflation", "fx_rate",
                      "cpi_inflation", "market_cap_gdp"]
        available_static_reals = [c for c in macro_cols if c in df.columns]

        training = TimeSeriesDataSet(
            df[df.time_idx <= train_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["group"],
            max_encoder_length=encoder_len,
            max_prediction_length=prediction_len,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["target"],
            time_varying_known_categoricals=["regime"],
            static_categoricals=["group"],
            static_reals=available_static_reals,   # real macro features per ticker
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training, df, predict=True, stop_randomization=True,
        )

        train_dl = training.to_dataloader(train=True, batch_size=32, num_workers=0)
        val_dl = validation.to_dataloader(train=False, batch_size=32, num_workers=0)

        model = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=hidden,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=max(8, hidden // 2),
            loss=QuantileLoss(),
            learning_rate=0.03,
            reduce_on_plateau_patience=2,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            gradient_clip_val=0.1,
            logger=False,
        )
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        preds = model.predict(val_dl, mode="quantiles", return_x=True)
        preds_out = preds.output if hasattr(preds, "output") else preds
        quantiles = _to_numpy(preds_out)

        raw_preds = model.predict(val_dl, mode="raw", return_x=True)
        raw_out = raw_preds.output if hasattr(raw_preds, "output") else raw_preds
        interpretation = model.interpret_output(raw_out, reduction="mean")

        attention = _to_numpy(interpretation["attention"])
        encoder_importance = _to_numpy(interpretation["encoder_variables"])
        decoder_importance = _to_numpy(interpretation["decoder_variables"])
        static_importance = _to_numpy(interpretation.get("static_variables", np.zeros(1)))

        features_for_ukt = np.zeros(UKT_FEATURE_DIM)
        att_flat = attention.flatten()
        features_for_ukt[:min(16, len(att_flat))] = att_flat[:16]
        enc_flat = encoder_importance.flatten()
        features_for_ukt[16:16 + min(8, len(enc_flat))] = enc_flat[:8]
        dec_flat = decoder_importance.flatten()
        features_for_ukt[24:24 + min(3, len(dec_flat))] = dec_flat[:3]  # 24-26; 27-31 reserved for macro

        feature_meta = _finance_feature_meta_from_attention(
            attention, encoder_importance, decoder_importance,
        )

        # Finance-macro features: packed into slots 27–31 (finance-owned range 0–31),
        # which is the tail of the decoder-importance sub-block and does NOT overlap
        # with the graph block's structural-centrality region (indices 32–47).
        # Both feature values and metadata are written for every populated slot.
        macro_slot_labels = {
            "gdp_growth":     ("IMF DataMapper",   "real_gdp_growth_pct"),
            "inflation":      ("IMF DataMapper",   "cpi_inflation_pct"),
            "fx_rate":        ("Open.er-api/ECB",  "fx_rate_vs_usd"),
            "cpi_inflation":  ("World Bank",       "cpi_inflation_pct"),
            "market_cap_gdp": ("World Bank",       "market_cap_gdp_pct"),
        }
        for i, col in enumerate(available_static_reals[:5]):  # max 5 slots (27-31)
            slot = 27 + i
            if col in macro_slot_labels:
                src, metric = macro_slot_labels[col]
                col_series = df[col].dropna()
                if col_series.empty:
                    continue
                col_min, col_max = float(col_series.min()), float(col_series.max())
                norm_val = float((col_series.iloc[-1] - col_min) / (col_max - col_min + 1e-8))
                features_for_ukt[slot] = norm_val
                feature_meta[slot] = {
                    "label":  f"macro_{col}",
                    "block":  "finance_macro",
                    "metric": metric,
                    "source": src,
                }

        return dict(
            quantiles=quantiles,
            x=getattr(preds, "x", None),
            attention=attention,
            encoder_importance=encoder_importance,
            decoder_importance=decoder_importance,
            static_importance=static_importance,
            model_params=sum(p.numel() for p in model.parameters()),
            features_for_ukt=features_for_ukt,
            feature_meta=feature_meta,
            data_source=data_source,
        )
    except Exception as e:
        st.error(f"TFT fitting failed: {e}")
        return None
