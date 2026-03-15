"""Integration-style tests for dashboard pipeline session hydration."""
from __future__ import annotations

import sys
import types

import pandas as pd

from tests.dashboard_fixtures import (
    EXPECTED_RUNNER_PAYLOAD_KEYS,
    SESSION_TO_PAYLOAD_KEY_MAP,
    build_runner_payload,
)


class _MockSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class _NoopStatus:
    def __init__(self):
        self.updates: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, **kwargs):
        self.updates.append(kwargs)


def _stub_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_dashboard_run_pipeline_hydrates_session_from_runner_payload(monkeypatch):
    """Dashboard run_pipeline must map PipelineRunner payload to session state exactly."""
    from hyperspace.pages import dashboard

    payload = build_runner_payload()

    # Strict drift guard: contract keys must exactly match fixture keys.
    assert set(payload.keys()) == set(EXPECTED_RUNNER_PAYLOAD_KEYS)

    # Stubs for runtime imports performed by run_pipeline().
    ohlcv_df = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-02"],
            "Close": [100.0, 101.0],
        },
    )
    docs = ["policy update", "market brief"]
    agreement = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["A", "B"], columns=["A", "B"])
    raw_spatial = {"physical_raster": [[[1.0]]], "country_scalars": [[0.5]], "node_order": ["A"]}

    _stub_module(
        monkeypatch,
        "hyperspace.data.finance",
        get_ohlcv=lambda tickers: (ohlcv_df, "finance_stub"),
    )
    _stub_module(
        monkeypatch,
        "hyperspace.data.news",
        get_text_data=lambda start_date, end_date: (docs, "news_stub"),
    )
    _stub_module(
        monkeypatch,
        "hyperspace.data.political",
        get_political_data=lambda min_year, max_year: (None, agreement, "politics_stub"),
    )
    _stub_module(
        monkeypatch,
        "hyperspace.models.tft_forecast",
        fit_tft=lambda **kwargs: {"tft": "ok", "kwargs": kwargs},
    )
    _stub_module(
        monkeypatch,
        "hyperspace.models.topic_model",
        fit_topic_model=lambda docs_hash, docs, data_source: {
            "topic_model": "ok",
            "docs_hash": docs_hash,
            "docs": docs,
            "data_source": data_source,
        },
    )
    _stub_module(
        monkeypatch,
        "hyperspace.data.spatial",
        fetch_all_spatial_data=lambda: raw_spatial,
    )

    status = _NoopStatus()
    mock_st = types.SimpleNamespace(
        session_state=_MockSessionState(tickers=["AAPL"]),
        status=lambda *args, **kwargs: status,
        write=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(dashboard, "st", mock_st)

    monkeypatch.setattr(
        dashboard.PipelineRunner,
        "run",
        lambda self, **kwargs: payload,
    )

    dashboard.run_pipeline()

    # Contract-to-session mapping assertions (presence + exact value mapping).
    for session_key, payload_key in SESSION_TO_PAYLOAD_KEY_MAP.items():
        assert session_key in mock_st.session_state
        assert mock_st.session_state[session_key] == payload[payload_key]

    # Non-runner fields that run_pipeline still hydrates directly.
    assert mock_st.session_state.raw_ohlcv.equals(ohlcv_df)
    assert mock_st.session_state.raw_docs == docs
    assert mock_st.session_state.timeframe_context == {
        "start_date": "2026-01-01",
        "end_date": "2026-01-02",
        "min_year": 2026,
        "max_year": 2026,
    }

    # Contract data flows from PipelineRunner payload (no longer rebuilt in dashboard).
    assert mock_st.session_state.interpretability_contract == payload["interpretability_contract"]
    assert mock_st.session_state.interpretability_contract_summary == payload["interpretability_contract_summary"]
    assert mock_st.session_state.pipeline_complete is True

    assert status.updates[-1] == {"label": "Pipeline complete!", "state": "complete"}
