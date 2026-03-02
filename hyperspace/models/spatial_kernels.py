"""Spatial raster kernelization: SVD on the (R, 6) full feature matrix.

R = 4 physical layers + N_scalar rows (6 WB + 4 extended env = 10 scalars total).
Full matrix shape with all sources active: (14, 6).

Produces UKT feature block at indices [64:80] (geospatial-kernel region):
    [64:70]  S / S.sum()    — 6 normalised cross-layer kernel importances
    [70:76]  Vt[0, :]       — 6 node loadings on the dominant spatial kernel
    [76:80]  4 summary scalars — mean_elev_norm, mean_temp_norm,
                                 mean_conflict_norm, mean_econ_norm

Row layout in full_matrix (agent_sim.py safe indices are 0-9; 10+ are new env):
    0: elevation   1: temperature   2: humidity   3: precipitation
    4: gdp_ppp     5: debt_pct_gdp  6: military_pct_gdp  7: tertiary_enroll
    8: political_stability (inverted)   9: homicide_rate
   10: earthquake_risk  11: eonet_events  12: air_quality_pm25  13: sea_level_proxy
"""
from __future__ import annotations

import numpy as np

from hyperspace.config import UKT_FEATURE_DIM

# Canonical node order (matches hyperspace/data/spatial.py)
NODE_ORDER: list[str] = ["USA", "Russia", "China", "Britain", "India", "Brazil"]


def build_full_feature_matrix(
    physical_raster: np.ndarray,   # (4, 6, 9)
    country_scalars: np.ndarray,   # (N_scalar, 6) — 6 WB + 4 env = 10 rows
) -> np.ndarray:                   # (4 + N_scalar, 6) — normalized rows
    """Build and row-normalize the spatial feature matrix.

    Combines:
      - Physical raster means: physical_raster.mean(axis=2) → (4, 6)
      - Country scalars: (N_scalar, 6) — originally 6 WB rows; extended to
        10 rows when USGS/EONET/AQ/NOAA sources are available.
    Stacks and normalises each row to [0, 1].

    Args:
        physical_raster: Array of shape (4, 6, 9).
        country_scalars: Array of shape (N_scalar, 6).

    Returns:
        Row-normalised feature matrix of shape (4 + N_scalar, 6).
    """
    phys_mean = physical_raster.mean(axis=2)             # (4, 6)
    full      = np.vstack([phys_mean, country_scalars])  # (10, 6)

    out = np.zeros_like(full, dtype=float)
    for i in range(full.shape[0]):
        row = full[i]
        rng = float(row.max() - row.min())
        if rng > 1e-10:
            out[i] = (row - row.min()) / rng
        else:
            out[i] = 0.5  # constant row → neutral midpoint
    return out


def kernelize_spatial(full_matrix: np.ndarray) -> dict:
    """SVD on the (10, 6) spatial feature matrix → UKT features [64:80].

    Args:
        full_matrix: Row-normalised (10, 6) matrix from build_full_feature_matrix.

    Returns:
        Dict containing:
            U                (10, 6)   — layer contributions per kernel
            S                (6,)      — singular values
            Vt               (6, 6)    — node loadings per kernel
            features_for_ukt (80,)     — zeros except [64:80]
            feature_meta     dict[int, dict]
            per_node_vectors dict[str, np.ndarray]  node_name → (10,) spatial vec
    """
    U, S, Vt = np.linalg.svd(full_matrix, full_matrices=False)
    # U: (10, 6), S: (6,), Vt: (6, 6)
    importance = S / (S.sum() + 1e-8)

    features_for_ukt = np.zeros(UKT_FEATURE_DIM)
    feature_meta: dict[int, dict] = {}

    # [64:70] — normalized singular values (cross-layer kernel importances)
    features_for_ukt[64:70] = importance
    for i in range(6):
        feature_meta[64 + i] = {
            "label":  f"spatial_kernel_importance_{i}",
            "metric": "singular_value_importance",
            "block":  "geospatial",
            "source": "svd_spatial_raster",
        }

    # [70:76] — dominant kernel's per-node loadings (Vt row 0)
    features_for_ukt[70:76] = Vt[0]
    for i, node in enumerate(NODE_ORDER):
        feature_meta[70 + i] = {
            "label":  f"node_spatial_loading_{node}",
            "entity": node,
            "metric": "svd_dominant_kernel_loading",
            "block":  "geospatial",
            "source": "svd_spatial_raster",
        }

    # [76:80] — domain-level summary scalars (mean of normalised rows per domain)
    # Rows 0=elevation, 1=temperature, 2=humidity, 3=precip,
    # 4=GDP, 5=debt, 6=military, 7=enrollment,
    # 8=political_stability (inverted), 9=homicide_rate
    mean_elev     = float(full_matrix[0].mean())
    mean_temp     = float(full_matrix[1].mean())
    mean_conflict = float(full_matrix[8:10].mean())  # WB conflict proxies
    mean_econ     = float(full_matrix[4:8].mean())

    features_for_ukt[76] = mean_elev
    features_for_ukt[77] = mean_temp
    features_for_ukt[78] = mean_conflict
    features_for_ukt[79] = mean_econ

    feature_meta[76] = {
        "label": "spatial_summary_elevation",
        "block": "geospatial", "metric": "mean_elevation_norm",
    }
    feature_meta[77] = {
        "label": "spatial_summary_temperature",
        "block": "geospatial", "metric": "mean_temperature_norm",
    }
    feature_meta[78] = {
        "label": "spatial_summary_conflict",
        "block": "geospatial", "metric": "mean_conflict_norm",
    }
    feature_meta[79] = {
        "label": "spatial_summary_economic",
        "block": "geospatial", "metric": "mean_econ_norm",
    }

    # Per-node spatial vectors: column i of full_matrix → shape (10,)
    per_node_vectors: dict[str, np.ndarray] = {
        node: full_matrix[:, i].copy() for i, node in enumerate(NODE_ORDER)
    }

    return dict(
        U=U,
        S=S,
        Vt=Vt,
        features_for_ukt=features_for_ukt,
        feature_meta=feature_meta,
        per_node_vectors=per_node_vectors,
    )


def get_spatial_features(
    physical_raster: np.ndarray,
    country_scalars: np.ndarray,
    scalar_names: list[str],
    node_order: list[str],
    timeframe_context: dict,
) -> dict:
    """Orchestrate spatial feature extraction for UKT and agent initialisation.

    Args:
        physical_raster:  (4, 6, 9) raster tensor.
        country_scalars:  (6, 6) scalar matrix.
        scalar_names:     List of 6 scalar-type name strings.
        node_order:       List of 6 node-name strings.
        timeframe_context: Pipeline timeframe dict.

    Returns:
        Dict with all SVD outputs plus:
            full_matrix     (10, 6) normalised matrix
            scalar_names    list[str]
            node_order      list[str]
            timeframe_context dict
    """
    full_matrix = build_full_feature_matrix(physical_raster, country_scalars)
    result      = kernelize_spatial(full_matrix)
    result.update(dict(
        full_matrix=full_matrix,
        scalar_names=scalar_names,
        node_order=node_order,
        timeframe_context=timeframe_context,
    ))
    return result
