# Hyperspace v3.0 — Critical Errors & Future Feature Proposals

**Date:** 2026-03-09
**Last reviewed:** 2026-03-17
**Scope:** Full codebase audit of `/home/user/hyperspace`
**Status:** Living document — update as items are resolved

---

## Part 1: Critical Errors

### ERR-001: Dashboard Bypasses PipelineRunner — Dual Pipeline Paths

**Severity:** CRITICAL
**Status:** **RESOLVED** (verified 2026-03-17)
**Files:** `hyperspace/pages/dashboard.py`, `hyperspace/core/pipeline.py`

**Description:**
`dashboard.py:run_pipeline()` previously implemented its own complete pipeline orchestration that **duplicated** `core/pipeline.py:PipelineRunner.run()`. The test suite validated `PipelineRunner`, but the production Streamlit app never called it.

**Resolution:**
Dashboard now imports `PipelineRunner` (line 19) and delegates all core orchestration to `runner.run()` (lines 224-238). Data fetching and model pretraining remain in dashboard (for `st.status` progress UI), but all UKT/SAE/governance logic runs through the canonical runner. The `_compute_governance_flags()` and `_compute_scorecard()` functions have been removed from dashboard entirely.

---

### ERR-002: Governance Flag Logic Diverges Between Dashboard and PipelineRunner

**Severity:** HIGH
**Status:** **RESOLVED** (verified 2026-03-17)
**Files:** `hyperspace/core/pipeline.py:531-635`

**Description:**
Both files previously implemented `_compute_governance_flags()` with semantically different detection logic for GOV-001 through GOV-005.

**Resolution:**
Governance flag computation is now consolidated in a single canonical location: `PipelineRunner._compute_governance_flags()` in `hyperspace/core/pipeline.py` (lines 531-635). The duplicate implementation was removed from `dashboard.py`. Dashboard receives pre-computed flags from the `PipelineRunner` result and only renders them via `_build_governance_report_markdown()`. All flags (GOV-001 through GOV-005) now use consistent logic regardless of execution path.

---

### ERR-003: Landing Page States "64 Input Dimensions" — Actual Value Is 80

**Severity:** MEDIUM
**Status:** **RESOLVED** (verified 2026-03-17)
**File:** `hyperspace/pages/dashboard.py`

**Description:**
The System Accountability Statement previously read "64 input dimensions" instead of 80.

**Resolution:**
The hardcoded "64" reference has been removed. No such text exists in the current codebase. `UKT_FEATURE_DIM` is correctly defined as 80 in `config.py`.

---

### ERR-004: Scorecard "Feature Traceability" Uses Different Counting Methods

**Severity:** MEDIUM
**Status:** **RESOLVED** (verified 2026-03-17)
**Files:** `hyperspace/core/pipeline.py:654-667`

**Description:**
Dashboard and pipeline previously used different counting methods for feature traceability.

**Resolution:**
Scorecard computation is now consolidated in `PipelineRunner._compute_scorecard()` (pipeline.py lines 641-782). Dashboard does not have its own scorecard implementation — it receives the scorecard directly from the `PipelineRunner` result. Single counting method ensures consistency.

---

### ERR-005: Counterfactual Tab Comment Says "64" But UKT Is 80-Dimensional

**Severity:** LOW
**Status:** **RESOLVED** (verified 2026-03-17)
**File:** `hyperspace/pages/counterfactual_tab.py`

**Description:**
A stale comment referenced `# shape (n_blocks, 64)`.

**Resolution:**
The stale comment has been removed. The file now correctly references 80 dimensions (e.g., line 372: "all 80 feature dimensions").

---

### ERR-006: `_report_section.py` Is Dead Code in Production

**Severity:** LOW
**Status:** **RESOLVED** (verified 2026-03-17)
**File:** `hyperspace/pages/_report_section.py`

**Description:**
`_report_section.py` was previously reported as dead code with no production imports.

**Resolution:**
The module now exports `render_interpretability_report()` and is actively imported by 4 tab modules: `finance_tab.py`, `clusters_tab.py`, `agents_tab.py`, and `politics_tab.py`. It is no longer dead code.

---

### ERR-007: Agent Simulation Seed Parameter Named `s` Instead of `seed`

**Severity:** LOW
**Status:** **OPEN** (verified 2026-03-17)
**File:** `hyperspace/models/agent_sim.py:146`

**Description:**
```python
def run_simulation(agents, steps=50, resource_flow=5.0,
                   alliance_fluidity=0.5, shock_prob=0.1, s=42):
```

The seed parameter is still named `s` — a single-letter name that violates PEP 8 readability guidelines and the project's style guide. Every other function in the codebase uses `seed`. Used at line 153: `rng = np.random.default_rng(s)`. The pipeline call site (pipeline.py line 237) does not pass `s`, so renaming is safe.

**Fix:**
Rename `s` to `seed`.

---

### ERR-008: Scorecard `governance_flags` Pass Condition Diverges

**Severity:** MEDIUM
**Status:** **RESOLVED** (verified 2026-03-17)
**Files:** `hyperspace/core/pipeline.py:770-780`

**Description:**
Dashboard previously hardcoded `passed=flag_count == 0` while pipeline used threshold-based `n_flags <= thresh["threshold"]`.

**Resolution:**
Scorecard computation is now consolidated in `PipelineRunner._compute_scorecard()` (pipeline.py lines 770-780). The single implementation uses the config-driven threshold: `passed=n_flags <= thresh["threshold"]`. Dashboard no longer has its own scorecard logic. Threshold is defined in `config.py:548-553` as `threshold=0`.

---

## Part 2: Future Feature Proposals

### FEAT-001: Temporal Windowed UKT — Rolling Kernel Analysis

**Priority:** HIGH
**Status:** **PARTIALLY IMPLEMENTED** (verified 2026-03-17)

**Rationale:** Currently the UKT accumulates blocks as a static (n_blocks x 80) matrix with one row per modality. There is no temporal dimension — a pipeline run captures a single point in time.

**What exists:**
- `DriftMonitor` (`hyperspace/core/drift_monitor.py:99-247`): Rolling window history of run snapshots, windowed cosine drift detection (DRIFT-001/002/003), configurable alert thresholds, disk persistence via JSON.
- `KernelMemory` (`hyperspace/core/temporal_memory.py:60-283`): Per-block kernel evolution tracking, importance trends, reconstruction error trends, regression cosines between consecutive runs, disk persistence.
- Pipeline integration: Both are wired into `PipelineRunner.run()` (pipeline.py lines 423-427, 453-494) and results exported to `PipelineResult`.

**What remains unimplemented:**
- `TemporalUKT` class with SVD on time-stacked (n_runs, 80) matrix to produce temporal modes (U) and feature modes (Vt).
- Trend decomposition (trend + cyclical + noise) of kernel evolution.
- Per-kernel drift detail (which specific features drove the drift).

**Note:** The codebase chose pairwise cosine similarity-based drift detection instead of the time-stacked SVD approach. This addresses the governance use case ("Has the system's understanding changed?") through a different mechanism.

**Governance Value:** Enables the question "Has the system's understanding changed recently?" — critical for detecting data distribution shifts.

---

### FEAT-002: Causal Graph Overlay on Kernel Structure

**Priority:** HIGH
**Status:** **NOT IMPLEMENTED** (verified 2026-03-17)

**Rationale:** The current system discovers **correlational** structure (SVD kernels capture co-variation). Users and policymakers frequently misinterpret correlation as causation. The governance framing explicitly warns against this but offers no tools to explore causal hypotheses.

**Proposal:**
Integrate a lightweight **causal discovery** layer:
1. Use the PC algorithm (from `causal-learn` or `pgmpy`) on the UKT feature matrix to infer a DAG skeleton
2. Overlay the causal DAG edges on the existing kernel visualization
3. Distinguish "kernel loads on features X and Y" (correlation) from "X → Y is a plausible causal path" (directed edge)
4. Add a "Causal Audit" governance flag (GOV-006) when a kernel's top features have no discoverable causal links — indicating the pattern may be spurious

**Current state:** No causal discovery libraries are imported. No DAG inference code exists. GOV-006 is not defined (config.py only has GOV-001 through GOV-005). Kernel visualizations in `hyperspace/viz/kernel_viz.py` show correlational structure only.

**Governance Value:** Directly addresses the "correlation ≠ causation" limitation. Enables policymakers to ask "Is this pattern causal or coincidental?"

---

### FEAT-003: Multi-Run Governance Audit Trail with Persistent Storage

**Priority:** HIGH
**Status:** **PARTIALLY IMPLEMENTED** (verified 2026-03-17)

**Rationale:** Currently, `run_id` and `run_timestamp` exist only in Streamlit session state — they vanish when the browser tab closes. For a governance tool, non-repudiation requires persistent audit trails.

**What exists:**
- `DriftMonitor` persists run history to disk via JSON (`hyperspace/core/drift_monitor.py:249-284`): `save_to_disk()` / `load_from_disk()` to `.hyperspace/drift_history.json`.
- `KernelMemory` persists kernel evolution to disk (`hyperspace/core/temporal_memory.py`): `.save()` / `.load()` to `.hyperspace/kernel_memory.json`.
- `LatentVersionTrail` persists latent space versions (`hyperspace/core/latent_versioning.py:180-324`): structural change detection, vocabulary drift detection, concept audit records to `.hyperspace/latent_versions.json`.
- Dashboard loads/saves all three on startup/completion (`dashboard.py:28-100`).

**What remains unimplemented:**
- SQLite backend (`hyperspace/storage/audit.py`) — storage is JSON-only across 3 separate files.
- "Run History" tab for browsing and comparing historical runs.
- Cross-run governance queries ("Has GOV-001 fired in the last 7 runs?").
- Unified run table with `run_id`, `data_sources`, `governance_flags`, `scorecard`, `reality_regression`, and `kernel_labels` in a single queryable store.

**Governance Value:** Enables regulatory compliance — auditors can inspect historical runs without re-executing the pipeline.

---

### FEAT-004: Adversarial Robustness Testing for UKT

**Priority:** MEDIUM
**Status:** **PARTIALLY IMPLEMENTED** (verified 2026-03-17) — random perturbation only, no adversarial testing

**Rationale:** The stability test (8 noisy runs with Gaussian perturbation at std=0.01) is a good start but only tests random noise. It does not test **adversarial** perturbations — targeted modifications designed to flip a kernel's conclusion.

**What exists:**
- Gaussian stability test (`ukt/stability.py:12-60`): `estimate_regression_stability()` adds random noise and measures cosine similarity across perturbed regression vectors.
- Stability visualization in governance panel (`hyperspace/pages/governance.py:494-514`) with traffic-light verdicts.

**What remains unimplemented:**
- `find_adversarial_budget()` — targeted perturbation finder.
- GOV-007 "Adversarially Fragile Kernel" flag — not defined in config.py.
- Per-kernel adversarial sensitivity heatmap (kernels x features).

**Governance Value:** Answers "How much would someone need to manipulate the data to change this conclusion?" — directly relevant to AI safety and manipulation resistance.

---

### FEAT-005: Interactive Feature Attribution Drilldown

**Priority:** MEDIUM
**Status:** **NOT IMPLEMENTED** (verified 2026-03-17) — static provenance exists, no interactive drilldown

**Rationale:** The provenance trace panel (`governance.py:71-170`) provides per-feature audit chains, but users must navigate a sidebar selectbox for one feature at a time. There is no visual way to "click on a kernel and see what drives it."

**What exists:**
- Static provenance panel (`hyperspace/pages/governance.py:71-170`): per-feature audit chains via sidebar selectbox, kernel loadings metadata, reality regression weights, SAE concept connections, jurisdiction badges, stakeholder annotations.

**What remains unimplemented:**
- Plotly `customdata` + `clickData` for clickable kernel labels.
- Inline attribution panel on kernel click.
- Cross-kernel coupling view ("which other kernels share these features").
- Counterfactual impact estimates ("zeroing this feature changes importance by X%").
- Feature Dependency Graph visualization (features as nodes, kernels as hyperedges).

**Governance Value:** Enables non-technical policymakers to interactively explore "Why did the system conclude X?" without understanding SVD mathematics.

---

### FEAT-006: Expand Geopolitical Graph Beyond 6 Actors

**Priority:** MEDIUM
**Status:** **NOT IMPLEMENTED** (verified 2026-03-17)

**Rationale:** The system hardcodes 6 geopolitical actors (USA, Russia, China, Britain, India, Brazil). The `GEOPOLITICAL_NODES` and `GEOPOLITICAL_EDGES` in `config.py:306-313` are static. The System Accountability Statement already acknowledges "systemic omissions exist."

**Current state:** `GEOPOLITICAL_NODES` is a static dict with exactly 6 entries. Agent count is derived from this dict (`agent_sim.py:61`). No sidebar multiselect for actor selection exists. Pipeline passes `list(GEOPOLITICAL_NODES.keys())` directly to spatial module.

**Proposal:**
1. Make the node set configurable via sidebar multiselect (from a larger pool of ~20 actors)
2. Add nodes for: EU (aggregate), Japan, South Korea, Saudi Arabia, Iran, Turkey, South Africa, Nigeria, Indonesia, Australia, Pakistan, Mexico, Egypt, Germany, France
3. Dynamically fetch UN voting data and World Bank indicators for selected nodes
4. Scale the spatial raster grid to accommodate variable node counts
5. Update the agent simulation to handle N agents (currently assumes exactly 6)

**Constraints:** The UKT feature dimension is fixed at 80. The graph region (32-47) has 16 slots. With more than 6 nodes, the flattened centrality vector (6 nodes x 4 measures = 24, truncated to 16) would need a different encoding — e.g., top-K centralities or graph-level summary statistics only.

---

### FEAT-007: Unified Pipeline Runner in Dashboard (Architecture Consolidation)

**Priority:** HIGH (prerequisite for all other features)
**Status:** **FULLY IMPLEMENTED** (verified 2026-03-17)

**Rationale:** ERR-001 and ERR-002 describe the dual-pipeline problem. This proposal formalizes the fix.

**Resolution:**
Dashboard now follows the exact target architecture:
- **Phase A** (`dashboard.py:137-220`): Data fetching with `st.status` progress UI.
- **Phase B** (`dashboard.py:226-238`): `runner = PipelineRunner(...)` then `result = runner.run(...)`.
- **Phase C** (`dashboard.py:240-267`): Hydrates `st.session_state` from `PipelineResult` dict.
- `_compute_governance_flags()` and `_compute_scorecard()` have been deleted from dashboard.
- All governance logic is canonical in `PipelineRunner` (pipeline.py).

**Governance Value:** Single source of truth for all governance computations. Tests validate exactly the code that runs in production.

---

### FEAT-008: Export Pipeline Results as Machine-Readable Governance Package

**Priority:** MEDIUM
**Status:** **PARTIALLY IMPLEMENTED** (verified 2026-03-17) — individual exports exist, no unified package

**Rationale:** Current exports are Markdown reports and CSV metrics. For regulatory compliance and inter-system interoperability, a structured machine-readable format is needed.

**What exists** (`dashboard.py:968-1059`):
- Markdown governance report download (stamped with `run_id`)
- Metrics CSV (kernel importances per run)
- Scorecard CSV (pass/fail dimensions)
- Contract compliance CSV (per-module compliance)
- Diagnostics CSV (alignment metrics, faithfulness, drift)

**What remains unimplemented:**
- Unified zip archive packaging (`hyperspace_governance_pkg_{run_id}.zip`)
- `manifest.json`, `signatures.json` (SHA-256 tamper detection)
- `reality_regression.npy`, `kernel_labels.json`, `provenance_chains.json`
- Counterfactual diffs in package

**Governance Value:** Enables downstream systems, regulatory bodies, or audit frameworks to ingest Hyperspace outputs programmatically. The `signatures.json` provides tamper detection.

---

### FEAT-009: Semantic Canvas as Shared Embedding Space Across Runs

**Priority:** LOW
**Status:** **NOT IMPLEMENTED** (verified 2026-03-17) — canvas exists within-run only

**Rationale:** The Semantic Canvas (12 named dimensions) currently resets every run. Its value as a "shared semantic coordinate system" is limited to within a single pipeline execution.

**Current state:** `SemanticCanvas` (`semantic_interpreter/canvas.py:36-232`) accumulates coordinates within a single run via `project_block()` and `get_accumulated_state()`. It has no `save()`/`load()` methods for cross-run persistence. Canvas is destroyed on Streamlit rerun.

**What remains unimplemented:**
1. Persist canvas coordinates across runs
2. Compute canvas **velocity** (how fast coordinates shift between runs)
3. Visualize a "semantic trajectory" across time
4. Alert when velocity exceeds a threshold

**Governance Value:** Provides an intuitive, non-technical view of "Is the world changing according to this system?" that policymakers can monitor without understanding SVD.

---

### FEAT-010: Real-Time Data Streaming with Incremental UKT Updates

**Priority:** LOW
**Status:** **NOT IMPLEMENTED** (verified 2026-03-17)

**Rationale:** The current architecture is batch-mode: fetch all data, run full pipeline, display results. For operational use, incremental updates as new data arrives would reduce latency.

**Current state:** Architecture is entirely synchronous and batch-based. No WebSocket listeners, no `asyncio` usage, no incremental SVD, no stale data timers. Pipeline executes as a blocking `result = runner.run(...)` call.

**Proposal:**
1. Add a websocket listener for streaming data sources (yfinance real-time, GDELT GKG streaming)
2. Implement incremental SVD update (rank-1 update to existing decomposition) instead of full re-decomposition
3. Push kernel drift alerts when live updates shift kernels beyond the stability threshold
4. Maintain a "stale data" timer per block — flag when a block hasn't been updated within its expected cadence

**Constraints:** Incremental SVD (e.g., via `scipy.linalg.svd_lowrank` or Brand's algorithm) trades accuracy for speed. Must validate that incremental results match full re-computation within acceptable tolerance.

---

## Appendix: Error Priority Matrix

| ID | Severity | Effort | Impact if Unfixed | Status |
|----|----------|--------|-------------------|--------|
| ERR-001 | CRITICAL | Large | Tests don't validate production code | **RESOLVED** |
| ERR-002 | HIGH | Medium | Governance flags inconsistent across paths | **RESOLVED** |
| ERR-003 | MEDIUM | Trivial | Incorrect governance statement | **RESOLVED** |
| ERR-004 | MEDIUM | Small | Scorecard values diverge | **RESOLVED** |
| ERR-005 | LOW | Trivial | Misleading comment | **RESOLVED** |
| ERR-006 | LOW | Small | Dead code | **RESOLVED** |
| ERR-007 | LOW | Trivial | Style violation | **OPEN** |
| ERR-008 | MEDIUM | Trivial | Config not honored in production | **RESOLVED** |

## Appendix: Feature Priority Matrix

| ID | Priority | Effort | Governance Value | Status |
|----|----------|--------|-----------------|--------|
| FEAT-007 | HIGH | Medium | Prerequisite for correctness | **DONE** |
| FEAT-001 | HIGH | Medium | Temporal drift detection | **PARTIAL** — DriftMonitor + KernelMemory done; TemporalUKT SVD not done |
| FEAT-002 | HIGH | Large | Causal vs correlational clarity | **NOT STARTED** |
| FEAT-003 | HIGH | Medium | Persistent audit trail | **PARTIAL** — JSON persistence done; SQLite/queries not done |
| FEAT-004 | MEDIUM | Medium | Adversarial robustness | **PARTIAL** — Gaussian stability done; adversarial testing not done |
| FEAT-005 | MEDIUM | Medium | Interactive attribution | **NOT STARTED** — static provenance only |
| FEAT-006 | MEDIUM | Large | Broader geopolitical coverage | **NOT STARTED** |
| FEAT-008 | MEDIUM | Small | Machine-readable exports | **PARTIAL** — 5 exports done; zip package not done |
| FEAT-009 | LOW | Medium | Cross-run semantic tracking | **NOT STARTED** |
| FEAT-010 | LOW | Large | Real-time operation | **NOT STARTED** |
