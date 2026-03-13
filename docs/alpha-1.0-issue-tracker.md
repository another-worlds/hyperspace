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

- **Overall Alpha 1.0 readiness:** **87%**
- **Governance + interpretability compliance layer:** **92%**
- **Headless/UI parity:** **96%**
- **UTK learned shared-latent goals:** **85%**
- **Mechanistic/faithfulness validation:** **90%**

These percentages reflect current implementation plus existing roadmap and gap
analysis documented in:
- `docs/critical-errors-and-future-proposals.md`
- `docs/vision-assessment-and-redesign.md`

---


## Latest Progress Update (Current Cycle)

- **Overall readiness:** **87%**
- **Delta vs previous checkpoint:** **+7 percentage points**

### Area deltas
- **UTK universality / learned multimodal substrate:** 85% (**Δ +15pp**)
- **Interpretability + governance architecture:** 92% (**Δ +2pp**)
- **Headless/UI parity + reliability:** 96% (**Δ +0pp**)
- **Mechanistic/faithfulness validation:** 90% (**Δ +5pp**)

### Completed in this cycle
1. **H-002 (cross-session persistence):** DriftMonitor now serializes to disk via JSON (`save_to_disk` / `load_from_disk`). Dashboard loads persisted drift history on startup and saves after each pipeline run. History survives across separate Streamlit sessions. 3 new tests (round-trip, missing file, max_history cap).
2. **Temporal memory (Milestone C):** Built `KernelMemory` class in `hyperspace/core/temporal_memory.py` — stores per-block kernel snapshots (importance, reality regression, reconstruction error, activation rows) across runs. Provides `get_block_history()`, `compute_kernel_evolution()` (consecutive cosines, importance stability, reconstruction trends), and `get_evolution_summary()` for governance export. Full disk serialization via JSON. Integrated into `PipelineRunner` and dashboard with session-state + disk persistence. 9 new tests.
3. **L-002 (dead code cleanup):** Removed unused `bar_chart()` and `heatmap_chart()` from `charts.py`. Removed unused `plotly.express` and `numpy` imports from charts module.

### Previous cycle completed
1. **H-002 (temporal drift):** Built `DriftMonitor` with windowed run history, cosine-based regression/importance drift, configurable alert thresholds (DRIFT-001/002/003), and CSV-exportable history. Integrated into `PipelineRunner.run()` via optional `drift_monitor` parameter. DriftMonitor is now persisted across Streamlit re-runs via session state.
2. **H-003 (faithfulness):** Implemented six intervention-style checks:
   - Kernel removal attribution (UKT)
   - Feature region masking with monotonicity validation (UKT)
   - Importance-delta rank consistency (UKT)
   - Canvas dimension grounding (SemanticCanvas)
   - SAE concept-kernel consistency (SparseAutoencoder)
   - Concept ablation region alignment (SparseAutoencoder)
   Added fail-safe narrative downgrade when confidence drops below threshold. 35 dedicated tests all passing.
3. **C-001 (pipeline parity):** Removed dead `_compute_governance_flags` and `_compute_scorecard` wrapper functions from `dashboard.py`. Dashboard now has zero duplicated governance logic — all flows through `PipelineRunner.run()`.
4. **M-001 (export parity):** Added 5th export column ("Diagnostics CSV") containing alignment metrics, faithfulness check results, and drift statistics. Faithfulness + drift sections added to markdown governance report.
5. **H-001 (shared-latent):** Upgraded from linear projectors to nonlinear two-layer encoders with ReLU activation, proper InfoNCE gradients, Xavier initialization, and L2 weight decay. Added leave-one-out cross-modal transfer learning evaluation metrics. Training shows measurable loss decrease and improved retrieval metrics vs legacy.
6. **L-001/L-002:** Fixed `_report_section.py` missing Spatial/geospatial-kernel region (indices 64-79). Stale 64-d references fully cleared. Corrected tracker entry — `_report_section.py` is actively used by 4 tab modules.

### Next critical actions
1. Scorecard threshold tuning based on real pipeline run baselines.
2. Kernel evolution dashboard panel (visualize importance trends across runs).
3. Final alpha contract coverage enforcement pass.

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
- **Progress:** **95%**

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
  2. Pipeline summary reaches 100% compliant modules for alpha scope.
- **Progress:** **85%**

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
  Feature-flagged and shadow-only. 7 dedicated tests.
- **Alpha exit criteria:**
  1. ~~Shared latent projector prototype trained on paired windows.~~ DONE
  2. ~~Alignment metrics reported (retrieval/probe-based).~~ DONE
  3. ~~Shadow comparison against legacy UKT in governance panel.~~ DONE
  4. ~~Richer encoder architecture (nonlinear projectors, deeper training).~~ DONE
  5. ~~Extended evaluation: cross-block transfer learning metrics.~~ DONE
- **Progress:** **85%**

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

**Milestone A progress:** **95%**

## Milestone B — Pipeline parity lock (target: 100% complete)
- [x] Scorecard parity (dashboard delegates to pipeline logic)
- [x] Full orchestration parity (single path via runner)
- [x] End-to-end parity tests for key outputs

**Milestone B progress:** **95%**

## Milestone C — UTK vision alpha prototype (target: >=50% complete)
- [x] Shared latent projector prototype
- [x] Alignment metrics dashboard
- [x] Temporal drift panel
- [x] Richer contrastive encoder architecture (nonlinear 2-layer ReLU)
- [x] Cross-session drift persistence (Streamlit session state + disk)
- [x] Cross-block transfer learning evaluation (leave-one-out protocol)
- [x] Temporal memory integration (cross-run kernel persistence)

**Milestone C progress:** **95%**

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
