import pytest

from hyperspace.models import semantic_narrator
from hyperspace.models.sparse_ae import enrich_concepts_with_narratives
from semantic_interpreter.narrator import TemplateNarrator, LLMNarrator


def test_template_narrator_methods_return_strings():
    canvas = type('C', (), {'get_accumulated_state': lambda self: {}})()
    template = TemplateNarrator()

    assert template.narrate_canvas(canvas, kernel_labels=[]) is not None
    assert template.narrate_layer(type('E', (), {'block_name': 'B', 'coordinates': [0.0], 'active_concepts': 0})(), canvas) is not None
    assert template.narrate_kernel({'kernel_id': 'K0', 'importance': 0.5, 'dominant_region': 'temporal', 'dominant_block': 'x', 'top_features': []}, canvas) is not None
    assert template.narrate_concept({'concept_id': 'C0', 'dominant_region': 'features', 'mean_activation': 0.2, 'top_features': []}, canvas) is not None
    assert template.narrate_reality_regression({'reality_regression': [0.1, 0.2], 'n_kernels': 1, 'kernel_labels': []}, canvas) is not None


def test_enrich_concepts_with_narratives_uses_concept_id_mean_activation_cache_keys(monkeypatch):
    seen_keys = []

    def fake_get_or_compute_narrative(cache_key, compute_fn, force_recompute=False):
        seen_keys.append(cache_key)
        return compute_fn()

    def fake_narrate_concept(cl, canvas):
        return f"narrative_{cl['concept_id']}"

    monkeypatch.setattr(semantic_narrator, 'narrate_concept', fake_narrate_concept)
    monkeypatch.setattr('hyperspace.core.caching.get_or_compute_narrative', fake_get_or_compute_narrative)

    import streamlit as st
    st.session_state.clear()

    sae_result = {
        'concept_labels': [
            {
                'concept_id': 'C00',
                'mean_activation': 0.9,
                'active': True,
                'top_features': [{'index': 1, 'name': 'f1', 'loading': 0.3}],
            },
            {
                'concept_id': 'C01',
                'mean_activation': 0.8,
                'active': True,
                'top_features': [{'index': 2, 'name': 'f2', 'loading': 0.4}],
            },
        ]
    }

    result = enrich_concepts_with_narratives(sae_result, canvas=object())

    assert result['concept_labels'][0]['semantic_narrative'] == 'narrative_C00'
    assert result['concept_labels'][1]['semantic_narrative'] == 'narrative_C01'
    assert len(seen_keys) == 2
    assert seen_keys[0] != seen_keys[1]
    assert 'concept_C00' in seen_keys[0] or 'concept_C00' in seen_keys[1]
    assert 'concept_C01' in seen_keys[0] or 'concept_C01' in seen_keys[1]


def test_get_llm_health_status_includes_expected_fields(monkeypatch):
    class DummyNarrator:
        model_name = 'dummy'
        _model = None
        _model_load_failed = False
        _timeout_count = 1
        _timeout_threshold = 3
        _generation_timeout = 30.0

    monkeypatch.setattr('hyperspace.models.semantic_narrator._get_narrator', lambda: DummyNarrator())

    status = semantic_narrator.get_llm_health_status()

    assert set(status.keys()) == {
        'model_name', 'model_loaded', 'model_load_failed',
        'timeout_count', 'timeout_threshold', 'generation_timeout',
        'permanently_disabled'
    }
    assert status['model_name'] == 'dummy'
    assert status['timeout_count'] == 1
    assert status['permanently_disabled'] is False
