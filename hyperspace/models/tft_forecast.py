"""TFT forecasting: trains on real OHLCV data, extracts interpretable features for UKT."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hyperspace.config import UKT_FEATURE_DIM
from hyperspace.data.finance import get_ohlcv, get_tft_data
from hyperspace.data.synthetic import generate_tft_dataset, seed


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
    for i in range(min(8, len(dec_flat))):
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
    """Fit TFT on real or synthetic data. Return predictions + interpretation features.

    Returns dict with: quantiles, attention, encoder_importance, decoder_importance,
    model_params, features_for_ukt, feature_meta, data_source.
    Returns None on failure.
    """
    try:
        import lightning.pytorch as pl
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss

        pl.seed_everything(42)

        # Get data (real or fallback)
        df, data_source = get_tft_data(list(tickers), encoder_len, prediction_len)

        max_time = df.time_idx.max()
        train_cutoff = max_time - prediction_len

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

        # Quantile predictions
        preds = model.predict(val_dl, mode="quantiles", return_x=True)
        preds_out = preds.output if hasattr(preds, "output") else preds
        quantiles = _to_numpy(preds_out)

        # Interpretation: extract attention + variable importance
        raw_preds = model.predict(val_dl, mode="raw", return_x=True)
        raw_out = raw_preds.output if hasattr(raw_preds, "output") else raw_preds
        interpretation = model.interpret_output(raw_out, reduction="mean")

        attention = _to_numpy(interpretation["attention"])
        encoder_importance = _to_numpy(interpretation["encoder_variables"])
        decoder_importance = _to_numpy(interpretation["decoder_variables"])
        static_importance = _to_numpy(interpretation.get("static_variables", np.zeros(1)))

        # Build UKT feature vector from attention and variable importance
        features_for_ukt = np.zeros(UKT_FEATURE_DIM)
        att_flat = attention.flatten()
        features_for_ukt[:min(16, len(att_flat))] = att_flat[:16]
        enc_flat = encoder_importance.flatten()
        features_for_ukt[16:16 + min(8, len(enc_flat))] = enc_flat[:8]
        dec_flat = decoder_importance.flatten()
        features_for_ukt[24:24 + min(8, len(dec_flat))] = dec_flat[:8]

        feature_meta = _finance_feature_meta_from_attention(
            attention, encoder_importance, decoder_importance,
        )

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
        st.warning(f"TFT fitting unavailable ({e}); using synthetic fallback.")
        return None


def mock_forecast(prediction_len: int, s: int = 42) -> dict:
    """Generate a mock multi-quantile forecast when TFT unavailable."""
    rng = seed(s)
    base = np.cumsum(rng.normal(0.02, 0.1, prediction_len)) + 5
    # Produce a UKT feature vector matching real TFT's layout:
    #   indices 0-15  = temporal attention pattern
    #   indices 16-23 = encoder variable importance
    #   indices 24-31 = decoder variable importance
    features = np.zeros(UKT_FEATURE_DIM)
    # Simulated attention weights (temporal region)
    att = rng.dirichlet(np.ones(16))
    features[:16] = att
    # Simulated encoder importance
    enc = rng.uniform(0.05, 0.4, 8)
    features[16:24] = enc
    # Simulated decoder importance
    dec = rng.uniform(0.05, 0.3, 8)
    features[24:32] = dec
    feature_meta = {
        i: {"label": f"mock_attention_lag_{i+1:02d}", "block": "finance", "metric": "attention_weight"}
        for i in range(16)
    }
    return dict(
        q10=base - rng.uniform(0.3, 0.6, prediction_len),
        q50=base,
        q90=base + rng.uniform(0.3, 0.6, prediction_len),
        features_for_ukt=features,
        feature_meta=feature_meta,
        data_source="Fallback: mock forecast",
    )
