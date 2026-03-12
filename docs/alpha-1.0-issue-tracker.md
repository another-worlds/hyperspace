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

- **Overall Alpha 1.0 readiness:** **56%**
- **Governance + interpretability compliance layer:** **78%**
- **Headless/UI parity:** **82%**
- **UTK learned shared-latent goals:** **30%**
- **Mechanistic/faithfulness validation:** **25%**

These percentages reflect current implementation plus existing roadmap and gap
analysis documented in:
- `docs/critical-errors-and-future-proposals.md`
- `docs/vision-assessment-and-redesign.md`

---


## Latest Progress Update (Current Cycle)

- **Overall readiness:** **58%**
- **Delta vs previous checkpoint:** **+4 percentage points**

### Area deltas
- **UTK universality / learned multimodal substrate:** 30% (**Δ +0pp**)
- **Interpretability + governance architecture:** 78% (**Δ +3pp**)
- **Headless/UI parity + reliability:** 93% (**Δ +4pp**)

### Completed in this cycle
1. Dashboard orchestration now delegates graph/spatial/agents/interpretability/governance phases to canonical `PipelineRunner.run()` after data fetching and preprocessing.
2. Session-state outputs are hydrated from runner output payload to preserve headless/UI parity for governance + interpretability artifacts.
3. Governance exports now include interpretability contract markdown/CSV artifacts with dedicated regression tests.

### Next critical actions
1. Add an integration regression test that stubs `PipelineRunner.run()` and verifies dashboard `run_pipeline()` maps the complete payload contract into session state.
2. Extend contract enrollment to additional upstream modules (or explicit N/A policy) for alpha-scope compliance completeness.
3. Begin shared latent projector prototype for UTK alpha milestone C.

---

## Severity: CRITICAL

## C-001 — Single canonical pipeline path in production
- **Status:** `IN_PROGRESS`
- **Why critical:** If dashboard and headless runner diverge, tests cannot
  guarantee production correctness.
- **Root cause:** Historical duplication between UI orchestration and
  `PipelineRunner` logic.
- **Current state:** Governance scorecard logic has been unified via dashboard
  delegation to runner scorecard helpers; full orchestration parity still needs
  completion and simplification.
- **Alpha exit criteria:**
  1. Dashboard orchestration delegates fully to `PipelineRunner.run()`.
  2. No duplicated governance/scorecard implementations remain in dashboard.
  3. Parity tests cover headless vs UI outputs for all governance artifacts.
- **Progress:** **65%**

## C-002 — Enforced interpretability contract across core modules
- **Status:** `IN_PROGRESS`
- **Why critical:** Alpha governance requires machine-checkable evidence that
  explanation interfaces exist and emit structured payloads.
- **Current state:**
  - `InterpretableModule` protocol exists.
  - Structured module reports + summary are produced.
  - UKT + SemanticCanvas are compliant and surfaced in headless and UI paths.
- **Remaining risk:** Upstream model blocks beyond UKT/canvas are not yet
  universally enrolled in the same contract.
- **Alpha exit criteria:**
  1. All major model blocks implement contract (or explicit N/A policy).
  2. Pipeline summary reaches 100% compliant modules for alpha scope.
- **Progress:** **72%**

---

## Severity: HIGH

## H-001 — Learned cross-modal shared latent alignment
- **Status:** `OPEN`
- **Why high:** Core UTK vision requires learned intermodality, not only fixed
  concatenated regions.
- **Current state:** Existing UKT remains SVD over fixed 80-d regioned vectors.
- **Alpha exit criteria:**
  1. Shared latent projector prototype trained on paired windows.
  2. Alignment metrics reported (retrieval/probe-based).
  3. Shadow comparison against legacy UKT in governance panel.
- **Progress:** **20%**

## H-002 — Temporal memory and drift monitoring
- **Status:** `OPEN`
- **Why high:** Alpha reliability requires observing representation drift and
  transition behavior, not single-run snapshots only.
- **Current state:** Stability estimate exists; no temporal UTK memory backend.
- **Alpha exit criteria:**
  1. Windowed run history for kernel/regression drift.
  2. Alerting thresholds for significant drift.
- **Progress:** **25%**

## H-003 — Narrative faithfulness / mechanistic checks
- **Status:** `IN_PROGRESS`
- **Why high:** Human-readable narratives must be constrained by measurable
  attributions/interventions to avoid unsupported claims.
- **Current state:** Payload-shape checks are in place; mechanistic
  intervention/faithfulness tests are limited.
- **Alpha exit criteria:**
  1. At least one intervention-style check per major module in alpha scope.
  2. Fail-safe downgrade when explanation confidence is low.
- **Progress:** **30%**

---

## Severity: MEDIUM

## M-001 — Governance artifact consistency in UI exports
- **Status:** `IN_PROGRESS`
- **Why medium:** Downloaded reports/CSVs should include contract compliance
  artifacts consistently for auditability.
- **Current state:** Mission Control surfaces compliance; export parity is
  partial.
- **Alpha exit criteria:**
  1. Contract report + summary included in all governance exports.
  2. Report template references compliance status per module.
- **Progress:** **55%**

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
  interpretability contract keys.
- **Progress:** **100%**

---

## Severity: LOW

## L-001 — Residual stale comments and naming cleanup
- **Status:** `IN_PROGRESS`
- **Why low:** Developer clarity/maintenance, minimal runtime risk.
- **Current state:** Multiple stale references already fixed; continue sweep in
  tabs/docs where dimensionality or legacy naming may remain.
- **Progress:** **70%**

## L-002 — Dead/underused UI helper consolidation
- **Status:** `OPEN`
- **Why low:** Reduces maintenance overhead and unused code surface.
- **Progress:** **15%**

---

## Alpha 1.0 Milestones

## Milestone A — Governance foundation lock (target: 80% complete)
- [x] Structured interpretability contract reports
- [x] Contract summary aggregation
- [x] UI visibility in Mission Control
- [ ] Export/report parity for all governance artifacts

**Milestone A progress:** **82%**

## Milestone B — Pipeline parity lock (target: 100% complete)
- [x] Scorecard parity (dashboard delegates to pipeline logic)
- [ ] Full orchestration parity (single path via runner)
- [ ] End-to-end parity tests for key outputs

**Milestone B progress:** **82%**

## Milestone C — UTK vision alpha prototype (target: >=50% complete)
- [ ] Shared latent projector prototype
- [ ] Alignment metrics dashboard
- [ ] Temporal drift panel

**Milestone C progress:** **25%**

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

- **Week of:** 2026-03-12
- **Overall readiness:** 54%
- **Delta vs previous week:** +1 percentage point
- **Top completed item:** Counterfactual region mapping made config-driven.
- **Top blocker:** Full UI orchestration still not fully delegated to runner.
- **Next critical action:** Complete runner delegation and export parity checks.
