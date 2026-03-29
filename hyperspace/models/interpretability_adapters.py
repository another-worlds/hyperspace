"""Interpretability adapters for pipeline blocks (SPEC-3).

Each adapter wraps the functional output of a pipeline block and implements the
``InterpretableModule`` protocol defined in ``hyperspace.core.types``.  Adapters
are read-only wrappers — they never modify the underlying model or its outputs.
All export methods complete in <100ms and return JSON-serializable dicts.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _safe_float(v: Any) -> float:
    """Coerce a value to a plain Python float for JSON serialization."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _top_n_indices(arr: np.ndarray, n: int = 5) -> list[int]:
    """Return indices of top-n absolute values in a 1-D array."""
    if arr is None or len(arr) == 0:
        return []
    return list(np.argsort(np.abs(arr))[::-1][:n])


# --------------------------------------------------------------------------- #
# TFT Adapter                                                                  #
# --------------------------------------------------------------------------- #

class TFTAdapter:
    """Wraps ``fit_tft()`` dict output."""

    def __init__(self, result: dict) -> None:
        self._r = result

    def export_latent_units(self) -> dict[str, Any]:
        attention = self._r.get("attention")
        att_list = attention.flatten().tolist() if attention is not None else []
        return {
            "unit_type": "temporal_attention_weights",
            "n_units": len(att_list),
            "units": att_list,
            "model_params": self._r.get("model_params", 0),
        }

    def export_feature_attributions(self, input_batch: Any | None = None) -> dict[str, Any]:
        enc = self._r.get("encoder_importance", np.array([]))
        dec = self._r.get("decoder_importance", np.array([]))
        enc_flat = enc.flatten() if hasattr(enc, "flatten") else np.array(enc).flatten()
        dec_flat = dec.flatten() if hasattr(dec, "flatten") else np.array(dec).flatten()
        attributions = []
        for i, v in enumerate(enc_flat):
            attributions.append({"index": i, "name": f"encoder_var_{i}", "value": _safe_float(v), "type": "encoder"})
        for i, v in enumerate(dec_flat):
            attributions.append({"index": len(enc_flat) + i, "name": f"decoder_var_{i}", "value": _safe_float(v), "type": "decoder"})
        return {"attributions": attributions}

    def export_alignment_report(self, reference_modalities: list[str] | None = None) -> dict[str, Any]:
        att = self._r.get("attention")
        if att is not None:
            att_flat = att.flatten()
            cross_corr = float(np.corrcoef(att_flat[:len(att_flat)//2], att_flat[len(att_flat)//2:len(att_flat)])[0, 1]) if len(att_flat) >= 2 else 0.0
        else:
            cross_corr = 0.0
        return {
            "modality": "finance",
            "alignment_proxy": "cross_ticker_attention_correlation",
            "score": cross_corr,
            "data_source": self._r.get("data_source", "unknown"),
        }

    def explain_prediction(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        quantiles = self._r.get("quantiles")
        if quantiles is not None:
            q_arr = np.array(quantiles)
            median_pred = float(q_arr[..., q_arr.shape[-1] // 2].mean()) if q_arr.ndim > 1 else float(q_arr.mean())
        else:
            median_pred = 0.0
        return {
            "summary": f"TFT forecast: median prediction {median_pred:.4f}. "
                       f"Model has {self._r.get('model_params', 0)} parameters.",
            "median_prediction": median_pred,
            "context": context,
        }


# --------------------------------------------------------------------------- #
# BERTopic Adapter                                                             #
# --------------------------------------------------------------------------- #

class BERTopicAdapter:
    """Wraps ``fit_topic_model()`` dict output."""

    def __init__(self, result: dict) -> None:
        self._r = result

    def export_latent_units(self) -> dict[str, Any]:
        te = self._r.get("topic_embeddings")
        topics = self._r.get("topics", [])
        unique_topics = sorted(set(t for t in topics if t != -1))
        return {
            "unit_type": "topic_embeddings",
            "n_topics": len(unique_topics),
            "topic_ids": unique_topics[:20],
            "embedding_dim": te.shape[1] if te is not None and len(te.shape) > 1 else 0,
        }

    def export_feature_attributions(self, input_batch: Any | None = None) -> dict[str, Any]:
        import pandas as pd
        topics = self._r.get("topics", [])
        counts = pd.Series(topics).value_counts()
        total = counts.sum()
        attributions = []
        for topic_id, count in counts.items():
            attributions.append({
                "index": int(topic_id),
                "name": f"topic_{topic_id}",
                "value": _safe_float(count / (total + 1e-8)),
                "count": int(count),
            })
        return {"attributions": attributions}

    def export_alignment_report(self, reference_modalities: list[str] | None = None) -> dict[str, Any]:
        topics = self._r.get("topics", [])
        n_outliers = sum(1 for t in topics if t == -1)
        n_total = len(topics)
        outlier_ratio = n_outliers / (n_total + 1e-8)
        return {
            "modality": "information",
            "alignment_proxy": "topic_coverage",
            "outlier_ratio": _safe_float(outlier_ratio),
            "n_docs": n_total,
            "data_source": self._r.get("data_source", "unknown"),
        }

    def explain_prediction(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        topics = self._r.get("topics", [])
        unique = set(t for t in topics if t != -1)
        import pandas as pd
        counts = pd.Series(topics).value_counts()
        dominant = int(counts.index[0]) if len(counts) > 0 else -1
        outlier_pct = sum(1 for t in topics if t == -1) / (len(topics) + 1e-8) * 100
        return {
            "summary": f"BERTopic found {len(unique)} topics across {len(topics)} documents. "
                       f"Dominant topic: {dominant}. Outlier rate: {outlier_pct:.1f}%.",
            "n_topics": len(unique),
            "dominant_topic": dominant,
            "outlier_pct": _safe_float(outlier_pct),
            "context": context,
        }


# --------------------------------------------------------------------------- #
# Graph Engine Adapter                                                         #
# --------------------------------------------------------------------------- #

class GraphEngineAdapter:
    """Wraps ``analyze_graph()`` dict + NetworkX graph."""

    def __init__(self, graph: Any, analysis: dict) -> None:
        self._G = graph
        self._a = analysis

    def export_latent_units(self) -> dict[str, Any]:
        communities = self._a.get("communities", [])
        return {
            "unit_type": "graph_communities",
            "n_communities": len(communities),
            "communities": communities,
            "n_nodes": len(self._a.get("node_names", [])),
            "n_edges": self._G.number_of_edges() if self._G is not None else 0,
        }

    def export_feature_attributions(self, input_batch: Any | None = None) -> dict[str, Any]:
        attributions = []
        for node in self._a.get("node_names", []):
            attributions.append({
                "index": len(attributions),
                "name": node,
                "degree": _safe_float(self._a.get("degree_centrality", {}).get(node, 0)),
                "betweenness": _safe_float(self._a.get("betweenness", {}).get(node, 0)),
                "eigenvector": _safe_float(self._a.get("eigenvector", {}).get(node, 0)),
                "pagerank": _safe_float(self._a.get("pagerank", {}).get(node, 0)),
            })
        return {"attributions": attributions}

    def export_alignment_report(self, reference_modalities: list[str] | None = None) -> dict[str, Any]:
        communities = self._a.get("communities", [[]])
        # Inter-community coupling: fraction of edges between communities
        if self._G is not None and len(communities) > 1:
            inter = 0
            total = self._G.number_of_edges()
            comm_map = {}
            for ci, comm in enumerate(communities):
                for node in comm:
                    comm_map[node] = ci
            for u, v in self._G.edges():
                if comm_map.get(u) != comm_map.get(v):
                    inter += 1
            coupling = inter / (total + 1e-8)
        else:
            coupling = 0.0
        return {
            "modality": "geopolitics",
            "alignment_proxy": "inter_community_coupling",
            "coupling_score": _safe_float(coupling),
            "density": _safe_float(self._a.get("density", 0)),
        }

    def explain_prediction(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        # Find dominant actor
        pr = self._a.get("pagerank", {})
        dominant = max(pr, key=pr.get) if pr else "unknown"
        return {
            "summary": f"Graph analysis: {len(self._a.get('node_names', []))} actors, "
                       f"{len(self._a.get('communities', []))} communities. "
                       f"Dominant actor: {dominant} (PageRank). "
                       f"Density: {self._a.get('density', 0):.3f}.",
            "dominant_actor": dominant,
            "density": _safe_float(self._a.get("density", 0)),
            "avg_clustering": _safe_float(self._a.get("avg_clustering", 0)),
            "context": context,
        }


# --------------------------------------------------------------------------- #
# Agent Simulation Adapter                                                     #
# --------------------------------------------------------------------------- #

class AgentSimAdapter:
    """Wraps ``run_simulation()`` output."""

    def __init__(
        self,
        agents: dict,
        log_entries: list[str],
        features: np.ndarray,
    ) -> None:
        self._agents = agents
        self._logs = log_entries
        self._features = features

    def export_latent_units(self) -> dict[str, Any]:
        agent_states = []
        for name, agent in self._agents.items():
            agent_states.append({
                "name": name,
                "resources": _safe_float(agent.resources),
                "n_alliances": len(agent.alliances),
                "top_alliance": max(agent.alliances, key=agent.alliances.get) if agent.alliances else None,
            })
        return {
            "unit_type": "agent_state_vectors",
            "n_agents": len(agent_states),
            "agents": agent_states,
        }

    def export_feature_attributions(self, input_batch: Any | None = None) -> dict[str, Any]:
        attributions = []
        resources = np.array([a.resources for a in self._agents.values()])
        total = resources.sum() + 1e-8
        for i, (name, agent) in enumerate(self._agents.items()):
            attributions.append({
                "index": i,
                "name": name,
                "value": _safe_float(agent.resources / total),
                "resources": _safe_float(agent.resources),
            })
        return {"attributions": attributions}

    def export_alignment_report(self, reference_modalities: list[str] | None = None) -> dict[str, Any]:
        # Alliance matrix eigenvalue spectrum
        names = list(self._agents.keys())
        n = len(names)
        mat = np.zeros((n, n))
        for i, name in enumerate(names):
            for j, other in enumerate(names):
                mat[i, j] = self._agents[name].alliances.get(other, 0.0)
        eigenvalues = np.sort(np.linalg.eigvalsh(mat))[::-1]
        return {
            "modality": "agent_simulation",
            "alignment_proxy": "alliance_eigenvalue_spectrum",
            "eigenvalues": eigenvalues.tolist(),
            "dominant_eigenvalue": _safe_float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
        }

    def explain_prediction(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        resources = np.array([a.resources for a in self._agents.values()])
        total = resources.sum() + 1e-8
        shares = resources / total
        gini = float(np.sum(np.abs(shares[:, None] - shares[None, :])) / (2 * len(shares) * shares.mean() + 1e-8))
        # Determine polarity
        if gini > 0.5:
            regime = "unipolar (one dominant actor)"
        elif gini > 0.25:
            regime = "bipolar (two major blocs)"
        else:
            regime = "multipolar (distributed power)"
        return {
            "summary": f"Agent simulation: {len(self._agents)} agents, "
                       f"Gini={gini:.3f} → {regime}. "
                       f"{len(self._logs)} events logged.",
            "gini": _safe_float(gini),
            "regime": regime,
            "n_events": len(self._logs),
            "context": context,
        }


# --------------------------------------------------------------------------- #
# Spatial Kernels Adapter                                                      #
# --------------------------------------------------------------------------- #

class SpatialKernelsAdapter:
    """Wraps ``kernelize_spatial()`` / ``get_spatial_features()`` dict output."""

    def __init__(self, result: dict) -> None:
        self._r = result

    def export_latent_units(self) -> dict[str, Any]:
        S = self._r.get("S")
        Vt = self._r.get("Vt")
        importance = (S / (S.sum() + 1e-8)).tolist() if S is not None else []
        return {
            "unit_type": "spatial_svd_kernels",
            "n_kernels": len(importance),
            "kernel_importances": importance,
            "node_order": self._r.get("node_order", []),
        }

    def export_feature_attributions(self, input_batch: Any | None = None) -> dict[str, Any]:
        Vt = self._r.get("Vt")
        node_order = self._r.get("node_order", [])
        attributions = []
        if Vt is not None and len(node_order) > 0:
            for i, node in enumerate(node_order):
                if i < Vt.shape[1]:
                    attributions.append({
                        "index": i,
                        "name": node,
                        "value": _safe_float(Vt[0, i]),
                        "type": "dominant_kernel_loading",
                    })
        return {"attributions": attributions}

    def export_alignment_report(self, reference_modalities: list[str] | None = None) -> dict[str, Any]:
        full_matrix = self._r.get("full_matrix")
        if full_matrix is not None:
            corr = np.corrcoef(full_matrix.T)  # (6, 6) cross-region correlation
            mean_corr = float(np.mean(np.abs(corr[np.triu_indices_from(corr, k=1)])))
        else:
            mean_corr = 0.0
        return {
            "modality": "geospatial",
            "alignment_proxy": "cross_region_correlation",
            "mean_abs_correlation": _safe_float(mean_corr),
        }

    def explain_prediction(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        S = self._r.get("S")
        Vt = self._r.get("Vt")
        node_order = self._r.get("node_order", [])
        if S is not None:
            importance = S / (S.sum() + 1e-8)
            top_kernel_imp = _safe_float(importance[0])
        else:
            top_kernel_imp = 0.0
        dominant_node = "unknown"
        if Vt is not None and len(node_order) > 0:
            dominant_idx = int(np.argmax(np.abs(Vt[0])))
            dominant_node = node_order[dominant_idx] if dominant_idx < len(node_order) else "unknown"
        return {
            "summary": f"Spatial SVD: dominant kernel explains {top_kernel_imp:.1%} of variance. "
                       f"Dominant geographic loading: {dominant_node}.",
            "dominant_kernel_importance": top_kernel_imp,
            "dominant_node": dominant_node,
            "context": context,
        }
