# Alpha 1.0 Issue Tracker (Severity-Ordered)

This tracker consolidates active issues and progress toward an **Alpha 1.0**
release. Items are ordered by severity first, then by architectural dependency.

Status legend:
- `OPEN` — not started
- `IN_PROGRESS` — active engineering work
- `BLOCKED` — waiting on prerequisite decisions/work
- `DONE` — implemented and validated

---

## Alpha 1.0 Progress Snapshot

- **Overall Alpha 1.0 readiness:** **92%** (was 99% — UI audit identified critical gaps)
- **Backend governance + interpretability:** **100%**
- **UI-layer governance compliance:** **60%** (**NEW** — feature names, contrast, hierarchy, caching)
- **Headless/UI parity:** **99%**
- **UTK learned shared-latent goals:** **95%**
- **Mechanistic/faithfulness validation:** **99%**
- **Vision compliance (dynamic architecture):** **95%**

These percentages reflect current implementation plus existing roadmap and gap
analysis documented in:
- `docs/critical-errors-and-future-proposals.md`
- `docs/vision-assessment-and-redesign.md`

### Independent verification (2026-03-17)

Error audit against current codebase confirms:
- **7/8 ERR items resolved** (ERR-001 through ERR-006, ERR-008). Only ERR-007 remains (trivial `s`→`seed` rename).
- **PIPELINE_ERRORS.md**: 4/5 issues resolved (1a, 1b, 5 fixed; 1c `yfinance` still open).
- **FEAT-007** (unified pipeline runner): fully implemented — dashboard delegates to `PipelineRunner`.
- **FEAT-001** (temporal UKT): partially done via `DriftMonitor` + `KernelMemory` (different approach than proposed).
- **FEAT-003** (audit trail): partially done via JSON persistence (no SQLite/queries yet).
- **FEAT-002, 005, 006, 009, 010**: not started.
- See `docs/critical-errors-and-future-proposals.md` for full status on all 10 feature proposals.

---


## Latest Progress Update (Current Cycle)

- **Overall readiness:** **99%**
- **Delta vs previous checkpoint:** **+1 percentage point**

### Area deltas
- **UTK universality / learned multimodal substrate:** 95% (**Δ +5pp**)
- **Interpretability + governance architecture:** 100% (**Δ 0pp**)
- **Headless/UI parity + reliability:** 99% (**Δ 0pp**)
- **Mechanistic/faithfulness validation:** 99% (**Δ +1pp**)
- **Vision compliance (dynamic architecture):** 95% (**NEW**)

### Completed in this cycle
1. **Dynamic UKT feature registry:** `BLOCK_REGION_MAP` auto-derived from `HYPERSPACE_REGISTRY`. `_normalize_features()` and `_label_kernel()` iterate registry-discovered regions. `UniversalKnowledgeTensor.feature_dim` defaults to `HYPERSPACE_REGISTRY.total_dim`. Adding new blocks only requires `registry.register()`.
2. **Dynamic Semantic Canvas:** New `REGION_SEMANTIC_SPEC` defines per-region semantic dimensions and projection weights. `_build_canvas_from_spec()` auto-assembles `CANVAS_DIMENSIONS`, `REGION_TO_CANVAS`, and `CANVAS_DIM` from the spec. Canvas `__init__` rebuilds from registry at instantiation time. Adding a region + spec entry auto-extends the canvas.
3. **Registry-driven downstream consumers:** `faithfulness.py` region masking, `counterfactual_tab.py` domain impact, `cross_block_net.py` variance modes, `semantic_narrator.py` region bounds — all now derive from `HYPERSPACE_REGISTRY` instead of hardcoded tuples.
4. **Embedded Tiny-LLM semantic translator:** Installed `transformers` library. `arnir0/Tiny-LLM` (13M param Llama) verified loading and generating on CPU. LLM is the primary narrator path — translates machine neuron clusters (UKT kernels, SAE concepts, canvas coordinates) into human-readable semantics. Falls back to `TemplateNarrator` when generation times out (CPU-speed adaptive).
5. **LLM generation timeout protection:** `LLMNarrator._generate()` uses `ThreadPoolExecutor` with configurable timeout. After first timeout, `_llm_too_slow` flag disables further LLM calls for the run (graceful degradation, no pipeline blocking).
6. **Scorecard threshold dynamic:** `feature_traceability` threshold now computed as `int(UKT_FEATURE_DIM * 0.9)` instead of hardcoded 72.
7. **290 tests passing** across 5 test files (76 system + 45 shipping + 73 drift/faithfulness + 1 dashboard + 95 full pipeline).

### Previous cycle completions
1. Scorecard threshold tuning (calibrated baselines).
2. Kernel evolution dashboard panel (cross-run trend charts).
3. Contract coverage expansion (12 modules enrolled).
4. Parity audit + `.hyperspace/` gitignore.
5. Zero TODO/FIXME markers.
6. Faithfulness numpy bug fix.
7. Full integration test suite unblocked.
8. Latent space versioning (Phase 3).
9. Concept vocabulary audit trails (Phase 3).

### Next critical actions
1. **H-004**: Add feature names to Reality Regression charts (critical vision failure).
2. **H-005**: Fix WCAG AA contrast failures in dark theme CSS.
3. **H-006**: Restructure Mission Control with executive summary and content grouping.
4. **H-007**: Cache SAE/UVT/USE/counterfactual SVD by input hash in session state.
5. Scorecard threshold validation against real pipeline runs.
6. Shared-latent promotion from shadow-only to optional production path (post-alpha).

---

## Severity: CRITICAL

## C-001 — Single canonical pipeline path in production
- **Status:** `DONE`
- **Why critical:** If dashboard and headless runner diverge, tests cannot
  guarantee production correctness.
- **Root cause:** Historical duplication between UI orchestration and
  `PipelineRunner` logic.
- **Current state:** Dashboard delegates fully to `PipelineRunner.run()`.
  Dead governance/scorecard wrapper functions removed from dashboard.
  Parity test fixture covers all payload keys including new faithfulness
  and drift fields.
- **Alpha exit criteria:**
  1. ~~Dashboard orchestration delegates fully to `PipelineRunner.run()`.~~ DONE
  2. ~~No duplicated governance/scorecard implementations remain in dashboard.~~ DONE
  3. ~~Parity tests cover headless vs UI outputs for all governance artifacts.~~ DONE
- **Progress:** **98%**

## C-002 — Enforced interpretability contract across core modules
- **Status:** `IN_PROGRESS`
- **Why critical:** Alpha governance requires machine-checkable evidence that
  explanation interfaces exist and emit structured payloads.
- **Current state:**
  - `InterpretableModule` protocol exists.
  - Structured module reports + summary are produced.
  - UKT + SemanticCanvas are compliant and surfaced in headless and UI paths.
  - All remaining alpha-scope modules have explicit N/A policies with
    owner and rationale.
- **Remaining risk:** If future modules are added without policy enrollment,
  `enforce_alpha_scope_contract_coverage()` will raise.
- **Alpha exit criteria:**
  1. ~~All major model blocks implement contract (or explicit N/A policy).~~ DONE
  2. ~~Pipeline summary reaches 100% compliant or explicit N/A for alpha scope.~~ DONE (11 modules enrolled)
- **Progress:** **98%**

---

## Severity: HIGH

## H-001 — Learned cross-modal shared latent alignment
- **Status:** `DONE`
- **Why high:** Core UTK vision requires learned intermodality, not only fixed
  concatenated regions.
- **Current state:** Two-layer nonlinear encoders (ReLU activation) with proper
  InfoNCE contrastive gradients, Xavier initialization, and L2 weight decay.
  Training shows measurable loss decrease and retrieval improvement over legacy.
  Leave-one-out transfer learning evaluation measures cross-modal generalization.
  Feature-flagged and shadow-only. UKT feature space is now fully dynamic —
  feature dimensions, region bounds, and block mappings are derived from the
  `HYPERSPACE_REGISTRY` at runtime. Adding new modalities only requires
  `registry.register()`. Semantic Canvas dimensions auto-extend via
  `REGION_SEMANTIC_SPEC`.
- **Alpha exit criteria:**
  1. ~~Shared latent projector prototype trained on paired windows.~~ DONE
  2. ~~Alignment metrics reported (retrieval/probe-based).~~ DONE
  3. ~~Shadow comparison against legacy UKT in governance panel.~~ DONE
  4. ~~Richer encoder architecture (nonlinear projectors, deeper training).~~ DONE
  5. ~~Extended evaluation: cross-block transfer learning metrics.~~ DONE
  6. ~~Dynamic feature registry — no hardcoded dimensions/regions.~~ DONE
  7. ~~Embedded LLM translator (Tiny-LLM) for latent-to-semantics.~~ DONE
- **Progress:** **95%**

## H-002 — Temporal memory and drift monitoring
- **Status:** `DONE`
- **Why high:** Alpha reliability requires observing representation drift and
  transition behavior, not single-run snapshots only.
- **Current state:** `DriftMonitor` class implemented in
  `hyperspace/core/drift_monitor.py`. Tracks reality regression vectors,
  kernel importances, and stability across runs. Computes windowed cosine
  drift, L2 distance, and stability deltas. Three configurable alert
  thresholds (DRIFT-001/002/003). Exportable history rows for CSV.
  Integrated into `PipelineRunner.run()` and dashboard rendering.
  DriftMonitor persisted in Streamlit session state and on disk via JSON
  serialization (`save_to_disk` / `load_from_disk`). Cross-session
  persistence survives Streamlit restarts.
- **Alpha exit criteria:**
  1. ~~Windowed run history for kernel/regression drift.~~ DONE
  2. ~~Alerting thresholds for significant drift.~~ DONE
  3. ~~Session persistence across Streamlit re-runs.~~ DONE
  4. ~~Cross-session disk serialization.~~ DONE
- **Progress:** **100%**

## H-003 — Narrative faithfulness / mechanistic checks
- **Status:** `DONE`
- **Why high:** Human-readable narratives must be constrained by measurable
  attributions/interventions to avoid unsupported claims.
- **Current state:** Six intervention-style checks implemented in
  `hyperspace/core/faithfulness.py`:
  - Kernel removal attribution (UKT) — verifies importance via SVD ablation
  - Feature region masking (UKT) — zeros regions and validates energy-delta monotonicity
  - Importance-delta rank consistency (UKT) — top-3 rank agreement
  - Canvas dimension grounding (SemanticCanvas) — validates dimension ordering
  - SAE concept-kernel consistency — checks concept activation grounding
  - Concept ablation region alignment — validates concept-kernel region correspondence
  Fail-safe downgrade replaces narratives with disclaimer when confidence
  drops below threshold. Integrated into pipeline and dashboard exports.
  35 dedicated tests passing.
- **Alpha exit criteria:**
  1. ~~At least one intervention-style check per major module in alpha scope.~~ DONE
  2. ~~Fail-safe downgrade when explanation confidence is low.~~ DONE
- **Progress:** **90%**

---

## Severity: HIGH (UI Layer)

## H-004 — Feature names missing from Reality Regression charts
- **Status:** `OPEN`
- **Why high:** The Reality Regression bar chart is the system's primary
  feature-importance visualization. It shows feature *indices* (0–79) instead
  of `FEATURE_NAMES` / `registry.feature_name(idx)`. This breaks
  interpretability for non-technical stakeholders — the core vision promise.
- **Root cause:** `kernel_viz.plot_reality_regression()` does not pass feature
  names to the Plotly figure's x-axis or `hovertemplate`. The `FEATURE_NAMES`
  list exists in `config.py` and is registered in `HYPERSPACE_REGISTRY`.
- **Affected locations:** `interpreter_tab.py:264`, `counterfactual_tab.py:121-166`
  (`_plot_rr_diff`), `kernel_viz.py:plot_reality_regression()`.
- **Alpha exit criteria:**
  1. Reality Regression chart x-axis shows feature names.
  2. Hover template shows: feature name, region, loading value.
  3. Counterfactual RR diff chart shows feature names.
- **Progress:** **0%**

## H-005 — WCAG AA contrast failures in dark theme
- **Status:** `OPEN`
- **Why high:** Multiple text elements fail WCAG AA minimum contrast (4.5:1):
  caption text (`#2d4a66` on `#070d1a` ≈ 1.8:1), metric labels (`#3d5673` ≈
  2.5:1), inactive tab text (`#3d5673` ≈ 2.5:1), expander summaries (`#567090`
  ≈ 3:1). Governance badges (`#fbbf24` on `#1c0e00` ≈ 4.2:1) barely pass.
- **Root cause:** CSS colors in `config.py` lines 33–194 were chosen for
  aesthetic preference without contrast ratio verification.
- **Alpha exit criteria:**
  1. All body text ≥ 4.5:1 contrast ratio.
  2. All large text (≥18px) ≥ 3:1 contrast ratio.
- **Progress:** **0%**

## H-006 — Mission Control lacks content hierarchy and executive summary
- **Status:** `OPEN`
- **Why high:** Tab 0 renders 15+ sections in flat scroll. No executive summary,
  no content grouping. Governance outputs (flags, scorecard) are buried below
  technical metrics (TFT params, recon error). Policy officers cannot find
  actionable information without scrolling through ML diagnostics.
- **Root cause:** `dashboard.py` renders all results sequentially with
  `st.markdown("---")` dividers. No grouping logic, no tab/accordion structure
  within Mission Control.
- **Proposed structure:**
  1. Executive Summary (narrative + top 3 findings)
  2. Governance Status (flags + scorecard + faithfulness)
  3. Data Provenance (sources + jurisdictions + freshness)
  4. System Diagnostics (collapsed: drift + kernel evolution + alignment)
  5. Export (all formats)
- **Progress:** **0%**

## H-007 — Uncached expensive UI-layer computations
- **Status:** `OPEN`
- **Why high:** SAE training (100 epochs, 5–10s), counterfactual baseline SVD
  (3–8s), UVT (120 epochs, 10–20s), and USE (150 epochs, 15–25s) all re-run
  on every button click. This wastes user time and produces slightly different
  results (torch stochastic init), conflicting with the reproducibility
  invariant.
- **Root cause:** Tab modules call training functions directly without checking
  session state for cached results. No hash-based cache key mechanism.
- **Alpha exit criteria:**
  1. SAE, UVT, USE cached in `session_state` by input matrix hash.
  2. Counterfactual reuses pipeline's baseline SVD from `ukt_snapshots[-1]`.
  3. Re-click on unchanged data returns cached result instantly.
- **Progress:** **0%**

---

## Severity: MEDIUM

## M-001 — Governance artifact consistency in UI exports
- **Status:** `DONE`
- **Why medium:** Downloaded reports/CSVs should include contract compliance
  artifacts consistently for auditability.
- **Current state:** All five export buttons present:
  1. Report (Markdown) — includes scorecard, contract, alignment, faithfulness, drift
  2. Metrics (CSV) — kernel importances per run
  3. Score Card (CSV) — pass/fail dimensions
  4. Contract Compliance (CSV) — per-module compliance status
  5. Diagnostics (CSV) — alignment metrics, faithfulness checks, drift stats
- **Alpha exit criteria:**
  1. ~~Contract report + summary included in all governance exports.~~ DONE
  2. ~~Report template references compliance status per module.~~ DONE
- **Progress:** **90%**

## M-002 — Counterfactual/regional mapping maintainability
- **Status:** `DONE`
- **Why medium:** Hardcoded region boundaries can drift from canonical feature
  registry.
- **Current state:** Counterfactual region coloring now uses canonical
  `FEATURE_REGION_LABELS`; stale 64-d comments corrected to 80-d.
- **Progress:** **100%**

## M-004 — Hardcoded governance/display thresholds in tab modules
- **Status:** `OPEN`
- **Why medium:** 10 threshold values used in governance interpretation and
  display logic are hardcoded in tab modules instead of centralized in
  `config.py`. This conflicts with the "Centralized configuration" compliance
  principle and makes auditing difficult.
- **Thresholds to centralize:**
  - `finance_tab.py`: forecast confidence 0.10/0.30
  - `clusters_tab.py`: outlier ratio 0.30/0.10
  - `agents_tab.py`: Gini concentration 0.50/0.25
  - `interpreter_tab.py`: kernel expander threshold 0.20
  - `counterfactual_tab.py`: RR cosine stability 0.95/0.80, shock range 32–47
- **Progress:** **0%**

## M-005 — Pipeline progress lacks per-block timing and API status
- **Status:** `OPEN`
- **Why medium:** The 15–30s pipeline run shows only flat text lines like
  "Running Finance block..." in a single `st.status` container. No progress bar,
  no per-block timing, no per-source API status during the 11-source news cascade.
- **Progress:** **0%**

## M-006 — Tab buttons silently overwrite pipeline results
- **Status:** `OPEN`
- **Why medium:** Each tab has a "Compute X" button that runs independently.
  Clicking "Compute TFT Forecast" after the pipeline overwrites the pipeline's
  finance result in session state with no warning or diff. Users cannot tell if
  they're viewing pipeline results or manual overrides.
- **Progress:** **0%**

## M-007 — Counterfactual has no scenario memory
- **Status:** `OPEN`
- **Why medium:** Each counterfactual run overwrites the previous result. A
  governance auditor cannot compare "remove Finance" vs. "remove Clusters"
  side-by-side. This limits the practical utility of the contestability mechanism.
- **Progress:** **0%**

## M-003 — Session-state lifecycle coverage for governance keys
- **Status:** `DONE`
- **Why medium:** Missing defaults/reset logic causes stale governance data in
  UI runs.
- **Current state:** Dedicated tests validate init/reset behavior for
  interpretability contract keys. New `faithfulness_report` and `drift_result`
  keys added to both init and reset paths.
- **Progress:** **100%**

---

## Severity: LOW

## L-001 — Residual stale comments and naming cleanup
- **Status:** `DONE`
- **Why low:** Developer clarity/maintenance, minimal runtime risk.
- **Current state:** All stale 64-d references cleared. `_report_section.py`
  region map updated to include geospatial-kernel (64-79). No remaining
  TODO/FIXME markers in Python source.
- **Progress:** **90%**

## L-002 — Dead/underused UI helper consolidation
- **Status:** `DONE`
- **Why low:** Reduces maintenance overhead and unused code surface.
- **Current state:** Dashboard `_compute_governance_flags` and
  `_compute_scorecard` wrappers removed (were dead delegates).
  `bar_chart()` and `heatmap_chart()` removed from `charts.py` (never
  imported). Unused `plotly.express` and `numpy` imports cleaned.
  Note: `_report_section.py` is actively imported by 4 tab modules
  (finance, clusters, politics, agents) — not dead code.
  `_flatten_yf_columns()` is used internally in `finance.py` — not dead.
- **Progress:** **90%**

---

## Alpha 1.0 Milestones

## Milestone A — Governance foundation lock (target: 80% complete)
- [x] Structured interpretability contract reports
- [x] Contract summary aggregation
- [x] UI visibility in Mission Control
- [x] Export/report parity for all governance artifacts

**Milestone A progress:** **100%**

## Milestone B — Pipeline parity lock (target: 100% complete)
- [x] Scorecard parity (dashboard delegates to pipeline logic)
- [x] Full orchestration parity (single path via runner)
- [x] End-to-end parity tests for key outputs

**Milestone B progress:** **98%**

## Milestone C — UTK vision alpha prototype (target: >=50% complete)
- [x] Shared latent projector prototype
- [x] Alignment metrics dashboard
- [x] Temporal drift panel
- [x] Richer contrastive encoder architecture (nonlinear 2-layer ReLU)
- [x] Cross-session drift persistence (Streamlit session state + disk)
- [x] Cross-block transfer learning evaluation (leave-one-out protocol)
- [x] Temporal memory integration (cross-run kernel persistence)
- [x] Latent space versioning (config fingerprints + structural change detection)
- [x] Concept vocabulary audit trails (per-run SAE snapshot with drift detection)

**Milestone C progress:** **100%**

---

## Weekly Update Template

Use this section for iterative updates without changing the tracker structure.

- **Week of:** YYYY-MM-DD
- **Overall readiness:** XX%
- **Delta vs previous week:** +X / -X percentage points
- **Top completed item:**
- **Top blocker:**
- **Next critical action:**


### Example Filled Update

- **Week of:** 2026-03-13
- **Overall readiness:** 80%
- **Delta vs previous week:** +22 percentage points
- **Top completed item:** 6 faithfulness checks + transfer learning evaluation + nonlinear shared-latent encoders.
- **Top blocker:** Cross-session drift serialization to disk.
- **Next critical action:** Add temporal memory integration for cross-run kernel persistence.
