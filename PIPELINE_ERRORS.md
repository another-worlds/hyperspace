# Hyperspace Pipeline — Critical Error Report

**Date:** 2026-03-09
**Last reviewed:** 2026-03-17
**Branch:** `claude/run-pipeline-errors-JRuk1`
**Test Suite:** 290/290 tests PASS

---

## Summary

The full pipeline was executed end-to-end (data fetch → TFT → BERTopic → Graph → Spatial → Agents → SAE → Narratives). **No critical runtime errors were found** — the pipeline completes successfully with all live data sources.

However, the following **operational issues and warnings** were identified:

---

## 1. CRITICAL: Missing Dependencies Cause Silent Failures

### 1a. `pytorch-forecasting` and `lightning` not installed → TFT returns `None`

**Status:** **RESOLVED** (verified 2026-03-17)
**Location:** `hyperspace/models/tft_forecast.py:73-82`
**Original symptom:** `fit_tft()` caught all exceptions in a bare `except Exception` block with a misleading error message.
**Resolution:** Specific `ImportError` handling has been added at lines 73-82 of `fit_tft()`. The catch provides an accurate error message: "TFT fitting failed: missing dependency — {e}. Install with: pip install pytorch-forecasting lightning". The general `except Exception` block remains at line 207 as a fallback for non-import errors.

### 1b. `bertopic` and `sentence-transformers` not installed → BERTopic returns `None`

**Status:** **RESOLVED** (verified 2026-03-17)
**Location:** `hyperspace/models/topic_model.py:46-53`
**Original symptom:** BERTopic import failure was caught by a generic exception handler with no installation guidance.
**Resolution:** Specific `ImportError` handling added at lines 46-53: "BERTopic unavailable: missing dependency — {e}. Install with: pip install bertopic sentence-transformers". General exception handler remains at lines 119-121.

### 1c. `requirements.txt` build failures

**Status:** **OPEN** (verified 2026-03-17)
**Location:** `requirements.txt:13`
**Symptom:** `yfinance>=0.2.41,<1.0` is still listed and depends on `multitasking` and `sgmllib3k`, which fail to build wheels on Python 3.11+ (`AttributeError: install_layout`).
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

**Status:** **RESOLVED** (verified 2026-03-17)

**Original observation:** Governance flags were computed in two places with divergent logic:
1. `hyperspace/core/pipeline.py` — `PipelineRunner._compute_governance_flags()`
2. `hyperspace/pages/dashboard.py` — `_compute_governance_flags()`

**Resolution:** The duplicate implementation in `dashboard.py` has been removed. Governance flag computation is now consolidated in `PipelineRunner._compute_governance_flags()` (pipeline.py lines 531-635). Dashboard receives pre-computed flags from the `PipelineRunner` result and only renders them. All governance flags (GOV-001 through GOV-005) use consistent detection logic regardless of execution path.

---

## 6. UI-Layer Issues (identified 2026-03-18)

A comprehensive UI audit identified additional issues at the rendering layer.
These are not pipeline runtime errors but affect how results are presented:

- **ERR-009** (HIGH): Reality Regression charts show feature indices 0–79 instead of names
- **ERR-010** (HIGH): Dark theme CSS fails WCAG AA contrast for captions, labels, tabs
- **H-006**: Mission Control has 15+ flat sections with no content hierarchy
- **H-007**: SAE, counterfactual SVD, UVT, USE uncached between button clicks

See `docs/critical-errors-and-future-proposals.md` and `STRATEGY_UI.md` for
full details and remediation plans.
