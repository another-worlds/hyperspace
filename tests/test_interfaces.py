"""Integration tests for UKT model interfaces.

Tests LLMInterface, CNNInterface, DeepLinearInterface end-to-end:
training → extraction → UKT integration → cross-model compatibility.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from ukt import (
    UniversalKnowledgeTensor,
    FeatureRegionRegistry,
    LLMInterface,
    CNNInterface,
    DeepLinearInterface,
    LayerSpec,
    unified_registry,
    extract_all,
    ModelInterface,
)
from ukt.utils import _pad_or_truncate


# ------------------------------------------------------------------ #
# Minimal model fixtures                                              #
# ------------------------------------------------------------------ #

class _DeepLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(16, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 8)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


class _TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.attn = nn.MultiheadAttention(16, 2, batch_first=True)
        self.mlp = nn.Linear(16, 16)
        self.head = nn.Linear(16, 64)

    def forward(self, x):
        e = self.embed(x)
        a, _ = self.attn(e, e, e)
        h = torch.relu(self.mlp(a))
        return self.head(h)


def _train(model, make_batch, steps=30):
    """Quick training loop for any model."""
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(steps):
        x, y, loss_fn = make_batch()
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


@pytest.fixture
def trained_dln():
    m = _DeepLinearNet()
    _train(m, lambda: (
        torch.randn(4, 16),
        torch.randn(4, 8),
        nn.MSELoss(),
    ))
    return m


@pytest.fixture
def trained_cnn():
    m = _TinyCNN()
    _train(m, lambda: (
        torch.randn(4, 1, 8, 8),
        torch.randint(0, 4, (4,)),
        nn.CrossEntropyLoss(),
    ))
    return m


@pytest.fixture
def trained_llm():
    m = _TinyTransformer()
    _train(m, lambda: (
        torch.randint(0, 64, (4, 8)),
        torch.randint(0, 64, (4, 8)).view(-1),
        lambda out, y: nn.CrossEntropyLoss()(out.view(-1, 64), y),
    ))
    return m


# ------------------------------------------------------------------ #
# ModelInterface ABC tests                                            #
# ------------------------------------------------------------------ #

class TestModelInterfaceABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ModelInterface(nn.Linear(4, 4))

    def test_model_type_is_abstract(self):
        assert hasattr(ModelInterface, "_model_type")


# ------------------------------------------------------------------ #
# DeepLinearInterface tests                                           #
# ------------------------------------------------------------------ #

class TestDeepLinearInterface:
    def test_registry_structure(self, trained_dln):
        iface = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        reg = iface.registry
        assert "layer_activations" in reg
        assert "weight_spectrum" in reg
        assert reg.total_dim == 40

    def test_model_type(self, trained_dln):
        iface = DeepLinearInterface(trained_dln)
        assert iface.model_type == "deep_linear"

    def test_extract_shape(self, trained_dln):
        iface = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        feats = iface.extract(torch.randn(1, 16))
        assert feats.shape == (40,)
        assert feats.dtype == np.float64

    def test_extract_nonzero(self, trained_dln):
        iface = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        feats = iface.extract(torch.randn(1, 16))
        assert np.count_nonzero(feats) > 0

    def test_weight_spectrum_normalized(self, trained_dln):
        iface = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        feats = iface.extract(torch.randn(1, 16))
        spectrum = feats[20:40]
        nonzero = spectrum[spectrum > 0]
        if len(nonzero) > 0:
            assert nonzero.max() <= 1.0 + 1e-6

    def test_extract_and_add(self, trained_dln):
        iface = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        ukt = UniversalKnowledgeTensor(iface.registry)
        snap = iface.extract_and_add(ukt, torch.randn(1, 16), block_name="dln_test")
        assert snap["block_name"] == "dln_test"
        assert snap["matrix"].shape == (1, 40)
        assert snap["n_kernels"] == 1

    def test_feature_meta(self, trained_dln):
        iface = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        meta = iface.feature_meta()
        assert len(meta) == 40
        assert all(m["source"] == "deep_linear" for m in meta.values())

    def test_detach(self, trained_dln):
        iface = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        iface.extract(torch.randn(1, 16))
        iface.detach()
        # Should be able to re-extract after detach
        feats = iface.extract(torch.randn(1, 16))
        assert feats.shape == (40,)

    def test_custom_layer_specs(self, trained_dln):
        iface = DeepLinearInterface(
            trained_dln,
            layer_specs=[
                LayerSpec("layer1", "layer_activations", reducer="mean"),
            ],
            activation_dim=20,
            spectrum_dim=20,
        )
        feats = iface.extract(torch.randn(1, 16))
        assert feats.shape == (40,)

    def test_deterministic_extraction(self, trained_dln):
        iface = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        x = torch.randn(1, 16)
        f1 = iface.extract(x)
        iface.detach()
        f2 = iface.extract(x)
        np.testing.assert_array_almost_equal(f1, f2)


# ------------------------------------------------------------------ #
# CNNInterface tests                                                  #
# ------------------------------------------------------------------ #

class TestCNNInterface:
    def test_registry_structure(self, trained_cnn):
        iface = CNNInterface(trained_cnn, conv_dim=12, deep_conv_dim=16, classifier_dim=12)
        reg = iface.registry
        assert "conv_features" in reg
        assert "deep_conv_features" in reg
        assert "classifier_features" in reg
        assert reg.total_dim == 40

    def test_model_type(self, trained_cnn):
        assert CNNInterface(trained_cnn).model_type == "cnn"

    def test_extract_shape(self, trained_cnn):
        iface = CNNInterface(trained_cnn, conv_dim=12, deep_conv_dim=16, classifier_dim=12)
        feats = iface.extract(torch.randn(1, 1, 8, 8))
        assert feats.shape == (40,)

    def test_extract_nonzero(self, trained_cnn):
        iface = CNNInterface(trained_cnn, conv_dim=12, deep_conv_dim=16, classifier_dim=12)
        feats = iface.extract(torch.randn(1, 1, 8, 8))
        assert np.count_nonzero(feats) > 0

    def test_auto_discovers_conv_layers(self, trained_cnn):
        iface = CNNInterface(trained_cnn)
        specs = iface._default_layer_specs()
        regions = {s.region for s in specs}
        assert "conv_features" in regions
        assert "classifier_features" in regions

    def test_extract_and_add(self, trained_cnn):
        iface = CNNInterface(trained_cnn, conv_dim=12, deep_conv_dim=16, classifier_dim=12)
        ukt = UniversalKnowledgeTensor(iface.registry)
        snap = iface.extract_and_add(ukt, torch.randn(1, 1, 8, 8))
        assert snap["block_name"] == "cnn"
        assert snap["n_kernels"] == 1

    def test_deterministic(self, trained_cnn):
        iface = CNNInterface(trained_cnn, conv_dim=12, deep_conv_dim=16, classifier_dim=12)
        x = torch.randn(1, 1, 8, 8)
        f1 = iface.extract(x)
        iface.detach()
        f2 = iface.extract(x)
        np.testing.assert_array_almost_equal(f1, f2)


# ------------------------------------------------------------------ #
# LLMInterface tests                                                  #
# ------------------------------------------------------------------ #

class TestLLMInterface:
    def test_registry_structure(self, trained_llm):
        iface = LLMInterface(trained_llm, attention_dim=10, hidden_dim=20, embedding_dim=10)
        reg = iface.registry
        assert "attention_patterns" in reg
        assert "hidden_repr" in reg
        assert "embedding_space" in reg
        assert reg.total_dim == 40

    def test_model_type(self, trained_llm):
        assert LLMInterface(trained_llm).model_type == "llm"

    def test_extract_with_explicit_specs(self, trained_llm):
        iface = LLMInterface(
            trained_llm,
            layer_specs=[
                LayerSpec("attn", "attention_patterns", reducer="mean"),
                LayerSpec("mlp", "hidden_repr", reducer="mean"),
                LayerSpec("embed", "embedding_space", reducer="mean"),
            ],
            attention_dim=10, hidden_dim=20, embedding_dim=10,
        )
        feats = iface.extract(torch.randint(0, 64, (1, 8)))
        assert feats.shape == (40,)
        assert np.count_nonzero(feats) > 0

    def test_auto_discovers_transformer_layers(self, trained_llm):
        iface = LLMInterface(trained_llm)
        specs = iface._default_layer_specs()
        regions = {s.region for s in specs}
        assert "attention_patterns" in regions
        assert "embedding_space" in regions

    def test_extract_and_add(self, trained_llm):
        iface = LLMInterface(
            trained_llm,
            layer_specs=[
                LayerSpec("attn", "attention_patterns", reducer="mean"),
                LayerSpec("mlp", "hidden_repr", reducer="mean"),
                LayerSpec("embed", "embedding_space", reducer="mean"),
            ],
            attention_dim=10, hidden_dim=20, embedding_dim=10,
        )
        ukt = UniversalKnowledgeTensor(iface.registry)
        snap = iface.extract_and_add(ukt, torch.randint(0, 64, (1, 8)), block_name="gpt")
        assert snap["block_name"] == "gpt"
        assert snap["n_kernels"] == 1

    def test_deterministic(self, trained_llm):
        iface = LLMInterface(
            trained_llm,
            layer_specs=[
                LayerSpec("attn", "attention_patterns", reducer="mean"),
                LayerSpec("mlp", "hidden_repr", reducer="mean"),
                LayerSpec("embed", "embedding_space", reducer="mean"),
            ],
            attention_dim=10, hidden_dim=20, embedding_dim=10,
        )
        x = torch.randint(0, 64, (1, 8))
        f1 = iface.extract(x)
        iface.detach()
        f2 = iface.extract(x)
        np.testing.assert_array_almost_equal(f1, f2)


# ------------------------------------------------------------------ #
# Cross-model interoperability tests                                  #
# ------------------------------------------------------------------ #

class TestIntermodelCompatibility:
    def _make_interfaces(self, trained_dln, trained_cnn, trained_llm):
        dln_i = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        cnn_i = CNNInterface(trained_cnn, conv_dim=12, deep_conv_dim=16, classifier_dim=12)
        llm_i = LLMInterface(
            trained_llm,
            layer_specs=[
                LayerSpec("attn", "attention_patterns", reducer="mean"),
                LayerSpec("mlp", "hidden_repr", reducer="mean"),
                LayerSpec("embed", "embedding_space", reducer="mean"),
            ],
            attention_dim=10, hidden_dim=20, embedding_dim=10,
        )
        return dln_i, cnn_i, llm_i

    def test_unified_registry_no_overlap(self, trained_dln, trained_cnn, trained_llm):
        dln_i, cnn_i, llm_i = self._make_interfaces(trained_dln, trained_cnn, trained_llm)
        reg = unified_registry(dln_i, cnn_i, llm_i)
        bounds = list(reg.region_bounds().values())
        # No region should overlap another
        for i, (s1, e1) in enumerate(bounds):
            for j, (s2, e2) in enumerate(bounds):
                if i != j:
                    assert e1 <= s2 or e2 <= s1, f"Overlap: [{s1},{e1}) vs [{s2},{e2})"

    def test_unified_registry_prefixed(self, trained_dln, trained_cnn, trained_llm):
        dln_i, cnn_i, llm_i = self._make_interfaces(trained_dln, trained_cnn, trained_llm)
        reg = unified_registry(dln_i, cnn_i, llm_i, prefix_regions=True)
        names = list(reg.region_bounds().keys())
        assert any(n.startswith("deep_linear/") for n in names)
        assert any(n.startswith("cnn/") for n in names)
        assert any(n.startswith("llm/") for n in names)

    def test_unified_registry_total_dim(self, trained_dln, trained_cnn, trained_llm):
        dln_i, cnn_i, llm_i = self._make_interfaces(trained_dln, trained_cnn, trained_llm)
        reg = unified_registry(dln_i, cnn_i, llm_i)
        expected = dln_i.registry.total_dim + cnn_i.registry.total_dim + llm_i.registry.total_dim
        assert reg.total_dim == expected

    def test_extract_all_into_shared_ukt(self, trained_dln, trained_cnn, trained_llm):
        dln_i, cnn_i, llm_i = self._make_interfaces(trained_dln, trained_cnn, trained_llm)
        reg = unified_registry(dln_i, cnn_i, llm_i)
        ukt = UniversalKnowledgeTensor(reg)

        snapshots = extract_all(
            [dln_i, cnn_i, llm_i],
            [torch.randn(1, 16), torch.randn(1, 1, 8, 8), torch.randint(0, 64, (1, 8))],
            ukt,
        )
        assert len(snapshots) == 3
        assert ukt.block_names == ["deep_linear", "cnn", "llm"]

        final = ukt.get_latest_snapshot()
        assert final["matrix"].shape == (3, reg.total_dim)
        assert final["n_kernels"] == 3

    def test_kernels_span_models(self, trained_dln, trained_cnn, trained_llm):
        """Kernels should have cross-model activation (not all zero for any block)."""
        dln_i, cnn_i, llm_i = self._make_interfaces(trained_dln, trained_cnn, trained_llm)
        reg = unified_registry(dln_i, cnn_i, llm_i)
        ukt = UniversalKnowledgeTensor(reg)

        extract_all(
            [dln_i, cnn_i, llm_i],
            [torch.randn(1, 16), torch.randn(1, 1, 8, 8), torch.randint(0, 64, (1, 8))],
            ukt,
        )
        final = ukt.get_latest_snapshot()
        activations = final["kernel_activation"]  # (3, n_kernels)
        # Each kernel should activate at least one block
        for k in range(final["n_kernels"]):
            assert np.any(np.abs(activations[:, k]) > 1e-6)

    def test_extract_all_with_prefix(self, trained_dln, trained_cnn, trained_llm):
        dln_i, cnn_i, llm_i = self._make_interfaces(trained_dln, trained_cnn, trained_llm)
        reg = unified_registry(dln_i, cnn_i, llm_i)
        ukt = UniversalKnowledgeTensor(reg)

        extract_all(
            [dln_i, cnn_i, llm_i],
            [torch.randn(1, 16), torch.randn(1, 1, 8, 8), torch.randint(0, 64, (1, 8))],
            ukt,
            block_prefix="exp1_",
        )
        assert ukt.block_names == ["exp1_deep_linear", "exp1_cnn", "exp1_llm"]

    def test_reality_regression_covers_all_regions(self, trained_dln, trained_cnn, trained_llm):
        """Reality regression should have nonzero values in regions from multiple models."""
        dln_i, cnn_i, llm_i = self._make_interfaces(trained_dln, trained_cnn, trained_llm)
        reg = unified_registry(dln_i, cnn_i, llm_i)
        ukt = UniversalKnowledgeTensor(reg)

        extract_all(
            [dln_i, cnn_i, llm_i],
            [torch.randn(1, 16), torch.randn(1, 1, 8, 8), torch.randint(0, 64, (1, 8))],
            ukt,
        )
        rr = ukt.get_latest_snapshot()["reality_regression"]
        assert rr.shape == (reg.total_dim,)
        assert np.count_nonzero(rr) > 0

    def test_multiple_runs_produce_different_snapshots(self, trained_dln, trained_cnn, trained_llm):
        """Two UKT runs with different inputs should produce different kernels."""
        dln_i, cnn_i, llm_i = self._make_interfaces(trained_dln, trained_cnn, trained_llm)
        reg = unified_registry(dln_i, cnn_i, llm_i)

        results = []
        for _ in range(2):
            ukt = UniversalKnowledgeTensor(reg)
            extract_all(
                [dln_i, cnn_i, llm_i],
                [torch.randn(1, 16), torch.randn(1, 1, 8, 8), torch.randint(0, 64, (1, 8))],
                ukt,
            )
            results.append(ukt.get_latest_snapshot()["S"].copy())
            dln_i.detach()
            cnn_i.detach()
            llm_i.detach()

        # Singular values should differ between runs
        assert not np.allclose(results[0], results[1])


# ------------------------------------------------------------------ #
# Edge case tests                                                     #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_single_model_in_unified_registry(self, trained_dln):
        iface = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        reg = unified_registry(iface)
        assert reg.total_dim == 40

    def test_extract_all_single_model(self, trained_dln):
        iface = DeepLinearInterface(trained_dln, activation_dim=20, spectrum_dim=20)
        ukt = UniversalKnowledgeTensor(iface.registry)
        snaps = extract_all([iface], [torch.randn(1, 16)], ukt)
        assert len(snaps) == 1

    def test_large_batch_input(self, trained_cnn):
        iface = CNNInterface(trained_cnn, conv_dim=12, deep_conv_dim=16, classifier_dim=12)
        feats = iface.extract(torch.randn(32, 1, 8, 8))
        assert feats.shape == (40,)

    def test_pad_or_truncate_utility(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert _pad_or_truncate(arr, 5).shape == (5,)
        assert _pad_or_truncate(arr, 2).shape == (2,)
        assert _pad_or_truncate(arr, 3).shape == (3,)
        np.testing.assert_array_equal(_pad_or_truncate(arr, 3), arr)
