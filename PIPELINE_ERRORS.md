# Hyperspace Pipeline — Critical Error Report

**Date:** 2026-03-09
**Branch:** `claude/run-pipeline-errors-JRuk1`
**Test Suite:** 217/217 tests PASS

---

## Summary

The full pipeline was executed end-to-end (data fetch → TFT → BERTopic → Graph → Spatial → Agents → SAE → Narratives). **No critical runtime errors were found** — the pipeline completes successfully with all live data sources.

However, the following **operational issues and warnings** were identified:

---

## 1. CRITICAL: Missing Dependencies Cause Silent Failures

### 1a. `pytorch-forecasting` and `lightning` not installed → TFT returns `None`

**Location:** `hyperspace/models/tft_forecast.py:73-202`
**Symptom:** `fit_tft()` catches all exceptions in a bare `except Exception` block (line 200), calls `st.error()`, and returns `None`. If `lightning` or `pytorch-forecasting` is not installed, TFT silently fails.
**Impact:** The dashboard pipeline (`dashboard.py:430-433`) treats `None` as a hard block — it shows "Pipeline blocked: TFT fitting failed" and returns early. However, the **error message is misleading** ("Ensure live market data is reachable") when the actual cause is a missing Python package.
**Fix:** Add specific `ImportError` handling before the general `Exception` catch to provide an accurate error message.

### 1b. `bertopic` and `sentence-transformers` not installed → BERTopic returns `None`

**Location:** `hyperspace/models/topic_model.py`
**Symptom:** Similar to TFT — if BERTopic or sentence-transformers is not installed, `fit_topic_model()` returns `None`.
**Impact:** Pipeline blocks with "Pipeline blocked: BERTopic unavailable" (dashboard.py:454-456).
**Fix:** Same as 1a — add explicit `ImportError` handling.

### 1c. `requirements.txt` build failures

**Location:** `requirements.txt:13-14`
**Symptom:** `yfinance` depends on `multitasking` and `sgmllib3k`, which fail to build wheels on Python 3.11+ (`AttributeError: install_layout`).
**Impact:** `pip install -r requirements.txt` fails completely. The app uses `stooq.com` via pandas-datareader as the actual data source (not yfinance), so `yfinance` may be a vestigial dependency.
**Fix:** Either remove `yfinance` from requirements.txt if unused, or pin to a version that builds cleanly on Python 3.11+.

---

## 2. WARNING: Governance Flags Detected (Expected Behavior)

The pipeline correctly identifies and flags these governance issues:

| Code | Flag | Detail |
|------|------|--------|
| GOV-001 | Modality Imbalance | Graph block dominates ~62% of total variance when only Graph+Agents blocks are active (no Finance/Clusters) |
| GOV-003 | Geopolitical Centrality Skew | USA eigenvector centrality (0.556) exceeds 2x network average (0.091) |

These are **not bugs** — they are the governance system working as designed.

---

## 3. WARNING: Scorecard Failures (Expected with Partial Data)

| Dimension | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| Feature Traceability | 28/80 | >= 72/80 | WARN |
| Governance Flags Active | 2 | 0 | WARN |

- Feature traceability is low when Finance and Cluster blocks are skipped (only 28 of 80 features get metadata).
- With all 5 blocks active (Finance + Clusters + Graph + Spatial + Agents), traceability reaches the threshold.

---

## 4. INFO: Non-Critical Warnings

### 4a. Streamlit cache warnings outside runtime
```
WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
```
**Impact:** None. Expected when importing modules decorated with `@st.cache_data` outside a running Streamlit server.

### 4b. HuggingFace unauthenticated rate limiting
```
Warning: You are sending unauthenticated requests to the HF Hub.
```
**Impact:** Low. Sentence-transformers downloads models from HuggingFace Hub. Without a token, downloads are rate-limited but functional.

### 4c. Attention mask warning from semantic narrator
```
The attention mask is not set and cannot be inferred from input because pad token is same as eos token.
```
**Location:** `hyperspace/models/semantic_narrator.py` (uses a tiny LLM for narratives)
**Impact:** None. The narrator produces valid output despite this warning.

---

## 5. Architecture Note: Dual Governance Flag Computation

**Observation:** Governance flags are computed in two places:
1. `hyperspace/core/pipeline.py:301-388` — `PipelineRunner._compute_governance_flags()` (used by tests)
2. `hyperspace/pages/dashboard.py:29-127` — `_compute_governance_flags()` (used by Streamlit UI)

These two implementations use **different detection logic**:
- `pipeline.py` checks kernel importance via SVD (`importance.max() > 0.50`)
- `dashboard.py` checks reality regression region sums (`max_region_share > 0.50`)
- `pipeline.py` checks eigenvector centrality for GOV-003
- `dashboard.py` checks degree centrality for GOV-003

**Impact:** The same pipeline data may produce different governance flags depending on whether it runs through the headless `PipelineRunner` or the Streamlit `dashboard.run_pipeline()`.
**Recommendation:** Consolidate to a single governance flag computation to ensure consistency.
