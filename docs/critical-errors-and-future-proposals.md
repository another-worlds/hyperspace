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
**Rationale:** Currently the UKT accumulates blocks as a static (n_blocks x 80) matrix with one row per modality. There is no temporal dimension — a pipeline run captures a single point in time.

**Proposal:**
Add a **rolling window mode** where the UKT maintains the last N pipeline runs (e.g., daily runs over 30 days). Each run becomes a row, enabling:
- Kernel **drift detection**: alert when kernel structure shifts significantly between runs
- **Temporal kernel evolution**: track how the same kernel's feature loadings change over time
- **Trend decomposition**: decompose kernels into trend + cyclical + noise components

**Implementation Sketch:**
```python
class TemporalUKT:
    def __init__(self, window_size=30, feature_dim=80):
        self.window: deque[np.ndarray] = deque(maxlen=window_size)

    def add_run(self, run_regression: np.ndarray) -> dict:
        self.window.append(run_regression)
        matrix = np.stack(list(self.window))
        # SVD on time-windowed matrix gives temporal kernels
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        # U columns = temporal modes (which runs activate which kernel)
        # Vt rows = feature modes (same as current kernels)
        return {"temporal_kernels": Vt, "run_activations": U, ...}
```

**Governance Value:** Enables the question "Has the system's understanding changed recently?" — critical for detecting data distribution shifts.

---

### FEAT-002: Causal Graph Overlay on Kernel Structure

**Priority:** HIGH
**Rationale:** The current system discovers **correlational** structure (SVD kernels capture co-variation). Users and policymakers frequently misinterpret correlation as causation. The governance framing explicitly warns against this (dashboard.py line 302) but offers no tools to explore causal hypotheses.

**Proposal:**
Integrate a lightweight **causal discovery** layer:
1. Use the PC algorithm (from `causal-learn` or `pgmpy`) on the UKT feature matrix to infer a DAG skeleton
2. Overlay the causal DAG edges on the existing kernel visualization
3. Distinguish "kernel loads on features X and Y" (correlation) from "X → Y is a plausible causal path" (directed edge)
4. Add a "Causal Audit" governance flag (GOV-006) when a kernel's top features have no discoverable causal links — indicating the pattern may be spurious

**Governance Value:** Directly addresses the "correlation ≠ causation" limitation. Enables policymakers to ask "Is this pattern causal or coincidental?"

---

### FEAT-003: Multi-Run Governance Audit Trail with Persistent Storage

**Priority:** HIGH
**Rationale:** Currently, `run_id` and `run_timestamp` exist only in Streamlit session state — they vanish when the browser tab closes. For a governance tool, non-repudiation requires persistent audit trails.

**Proposal:**
1. Add a lightweight SQLite backend (`hyperspace/storage/audit.py`)
2. After each pipeline run, persist: `run_id`, `run_timestamp`, `data_sources`, `governance_flags`, `scorecard`, `reality_regression` vector, `kernel_labels`
3. Add a "Run History" tab showing prior runs with diff capabilities
4. Enable cross-run governance queries: "Has GOV-001 fired in the last 7 runs?"

**Schema:**
```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT,
    data_sources JSON,
    governance_flags JSON,
    scorecard JSON,
    reality_regression BLOB,  -- numpy array serialized
    n_kernels INTEGER
);
```

**Governance Value:** Enables regulatory compliance — auditors can inspect historical runs without re-executing the pipeline.

---

### FEAT-004: Adversarial Robustness Testing for UKT

**Priority:** MEDIUM
**Rationale:** The stability test (8 noisy runs with Gaussian perturbation at std=0.01) is a good start but only tests random noise. It does not test **adversarial** perturbations — targeted modifications designed to flip a kernel's conclusion.

**Proposal:**
1. Implement a **minimal adversarial perturbation** finder: for each kernel, find the smallest feature change that flips the kernel's dominant region
2. Report the "adversarial budget" per kernel — how much targeted noise is needed to change the conclusion
3. Add GOV-007: "Adversarially Fragile Kernel" when the budget is below a threshold
4. Visualize adversarial sensitivity as a heatmap (kernels x features)

**Implementation Sketch:**
```python
def find_adversarial_budget(matrix, target_kernel, max_iters=100):
    """Binary search for minimal perturbation that flips kernel dominant region."""
    eps_lo, eps_hi = 0.0, 1.0
    for _ in range(max_iters):
        eps = (eps_lo + eps_hi) / 2
        perturbed = matrix.copy()
        # Perturb only the target kernel's dominant region
        perturbed[:, lo:hi] += eps * gradient_direction
        # Re-SVD and check if dominant region changed
        ...
    return eps  # adversarial budget
```

**Governance Value:** Answers "How much would someone need to manipulate the data to change this conclusion?" — directly relevant to AI safety and manipulation resistance.

---

### FEAT-005: Interactive Feature Attribution Drilldown

**Priority:** MEDIUM
**Rationale:** The provenance trace panel (`governance.py:71-100`) provides per-feature audit chains, but users must navigate a sidebar selectbox for one feature at a time. There is no visual way to "click on a kernel and see what drives it."

**Proposal:**
1. Make kernel labels in the Plotly charts **clickable** (using Plotly `customdata` + Streamlit click events)
2. On click, expand an inline attribution panel showing:
   - Top-5 features with full provenance metadata
   - Data source, entity, metric, time scope
   - Which other kernels share these features (cross-kernel coupling)
   - Counterfactual impact: "If this feature were zeroed, kernel importance would change by X%"
3. Add a "Feature Dependency Graph" visualization: features as nodes, kernels as hyperedges

**Governance Value:** Enables non-technical policymakers to interactively explore "Why did the system conclude X?" without understanding SVD mathematics.

---

### FEAT-006: Expand Geopolitical Graph Beyond 6 Actors

**Priority:** MEDIUM
**Rationale:** The system hardcodes 6 geopolitical actors (USA, Russia, China, Britain, India, Brazil). The `GEOPOLITICAL_NODES` and `GEOPOLITICAL_EDGES` in `config.py` are static. The System Accountability Statement already acknowledges "systemic omissions exist."

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
**Rationale:** ERR-001 and ERR-002 describe the dual-pipeline problem. This proposal formalizes the fix.

**Proposal:**
1. Refactor `dashboard.py:run_pipeline()` into two phases:
   - **Phase A: Data Fetch** (keeps `st.status` UI, handles errors with `st.error`)
   - **Phase B: Pipeline Execution** (calls `PipelineRunner.run()` with fetched data)
2. Map `PipelineResult` fields to `st.session_state` in a single function
3. Delete `_compute_governance_flags()` and `_compute_scorecard()` from `dashboard.py`
4. Update tests to verify that `PipelineRunner` output matches what the UI expects

**Target State:**
```python
# dashboard.py — Phase B becomes:
def run_pipeline():
    # Phase A: fetch data (with st.status UI)
    finance_result, cluster_result, agreement, spatial = _fetch_all_data()

    # Phase B: delegate to tested runner
    runner = PipelineRunner(on_step=lambda s, m: st.write(m))
    result = runner.run(
        finance_result=finance_result,
        cluster_result=cluster_result,
        agreement_matrix=agreement,
        spatial_data=spatial,
    )

    # Phase C: map result to session state
    _store_pipeline_result(result)
```

**Governance Value:** Single source of truth for all governance computations. Tests validate exactly the code that runs in production.

---

### FEAT-008: Export Pipeline Results as Machine-Readable Governance Package

**Priority:** MEDIUM
**Rationale:** Current exports are Markdown reports and CSV metrics. For regulatory compliance and inter-system interoperability, a structured machine-readable format is needed.

**Proposal:**
Add a "Governance Package" export (JSON + numpy arrays in a zip archive):
```
hyperspace_governance_pkg_{run_id}.zip
├── manifest.json           # run_id, timestamp, version, data_sources
├── governance_flags.json   # all detected flags with detail
├── scorecard.json          # pass/fail for each dimension
├── reality_regression.npy  # 80-dim vector
├── kernel_labels.json      # all kernel metadata
├── provenance_chains.json  # per-feature audit trail
├── counterfactual/         # if run
│   ├── {block}_removed.json
│   └── {block}_rr_diff.npy
└── signatures.json         # SHA-256 hashes of all included files
```

**Governance Value:** Enables downstream systems, regulatory bodies, or audit frameworks to ingest Hyperspace outputs programmatically. The `signatures.json` provides tamper detection.

---

### FEAT-009: Semantic Canvas as Shared Embedding Space Across Runs

**Priority:** LOW
**Rationale:** The Semantic Canvas (12 named dimensions) currently resets every run. Its value as a "shared semantic coordinate system" is limited to within a single pipeline execution.

**Proposal:**
1. Persist canvas coordinates across runs (using FEAT-003's SQLite backend)
2. Compute canvas **velocity** (how fast coordinates shift between runs)
3. Visualize a "semantic trajectory" — the system's understanding moving through the 12-dim space over time
4. Alert when velocity exceeds a threshold (rapid semantic shift → possible data regime change)

**Governance Value:** Provides an intuitive, non-technical view of "Is the world changing according to this system?" that policymakers can monitor without understanding SVD.

---

### FEAT-010: Real-Time Data Streaming with Incremental UKT Updates

**Priority:** LOW
**Rationale:** The current architecture is batch-mode: fetch all data, run full pipeline, display results. For operational use, incremental updates as new data arrives would reduce latency.

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

| ID | Priority | Effort | Governance Value |
|----|----------|--------|-----------------|
| FEAT-007 | HIGH | Medium | Prerequisite for correctness |
| FEAT-001 | HIGH | Medium | Temporal drift detection |
| FEAT-002 | HIGH | Large | Causal vs correlational clarity |
| FEAT-003 | HIGH | Medium | Persistent audit trail |
| FEAT-004 | MEDIUM | Medium | Adversarial robustness |
| FEAT-005 | MEDIUM | Medium | Interactive attribution |
| FEAT-006 | MEDIUM | Large | Broader geopolitical coverage |
| FEAT-008 | MEDIUM | Small | Machine-readable exports |
| FEAT-009 | LOW | Medium | Cross-run semantic tracking |
| FEAT-010 | LOW | Large | Real-time operation |
