# Core System Vision Assessment and Redesign Plan

## Scope

This document evaluates the current Hyperspace architecture against the target
vision for:

1. Universal Knowledge Tensor (UKT) kernels as a modality-agnostic, shared
   representation substrate
2. Embedded semantic interpretability as an intrinsic interface for each model
   component

It also proposes a redesign path that prioritizes universality,
cross-modality integration, and intrinsic interpretability over raw benchmark
optimization.

---

## Executive Assessment

## Vision Scorecard

| Area | Target | Current status | Assessment |
|---|---|---|---|
| Shared latent tensor | One joint space across modalities | UKT is a single 80-d vector space with five fixed 16-d regions | **Partial** |
| Cross-modal alignment | Learned intermodality alignment | Regions are concatenated and independently normalized; no explicit alignment objective | **Gap** |
| Reusable knowledge across tasks | Task/domain transfer from shared kernels | SVD kernels are descriptive per run, but no pretraining/fine-tuning loop for reuse | **Gap** |
| Monolithic knowledge substrate | Unified substrate across model families | One matrix exists, but each upstream block remains siloed and hand-engineered | **Partial** |
| Intrinsic interpretability | Every component exposes semantics | UKT kernel labels + stage SAE + canvas + narratives are integrated into pipeline | **Strong partial** |
| Mechanistic introspection | Causal/feature pathway explanations | Mostly feature loading narratives; limited mechanistic probing/circuit analysis | **Gap** |

Bottom line: the system has a strong interpretability-first shell and a usable
shared matrix abstraction, but it does **not yet** satisfy the stricter UKT
vision of learned, modality-agnostic, cross-domain representation alignment.

---

## Evidence From Current Implementation

### What is already aligned with the vision

1. **Single shared tensor object exists**
   - `UniversalKnowledgeTensor` appends each block as a row into one matrix and
     computes global SVD kernels and a reality regression vector.
2. **Interpretability is integrated, not bolt-on**
   - Every `add_block()` call runs a per-stage SAE, projects to a shared
     semantic canvas, and optionally generates narratives.
3. **Pipeline-level semantic outputs are first-class**
   - The interpreter tab and pipeline reports expose kernel labels, semantic
     dimensions, and layer-level narratives.

**Note (2026-03-18):** A UI-layer audit (`STRATEGY_UI.md`) found that while
the backend is well-aligned, the **rendering layer** undermines several of
these strengths — e.g., the Reality Regression chart drops feature names at
display time (showing indices instead), Mission Control buries governance
outputs under technical diagnostics, and expensive computations (SAE, UVT)
re-run on every click. See `docs/vision-compliance.md` §6 for compliance
matrix updates and `docs/alpha-1.0-issue-tracker.md` H-004 through H-007
for tracked issues.

### Structural gaps vs target UKT design

1. **Latent space is hand-partitioned, not learned jointly**
   - The 80-d space is fixed into modality-owned slices (0-15, 16-31, etc.).
   - This enforces modality boundaries instead of discovering shared structure.
2. **No explicit cross-modal contrastive/alignment objective**
   - SVD is applied after concatenation; there is no paired objective that pulls
     semantically related cross-modal representations together.
3. **No world-model style temporal memory**
   - Current UKT snapshotting is per pipeline run; it lacks sequence modeling
     over transitions and interventions.
4. **Knowledge reuse is weakly defined**
   - Kernels summarize variance in the current matrix but are not versioned,
     distilled, or transferred as reusable modules.

### Structural gaps vs embedded interpretability vision

1. **Interpretability coverage is uneven upstream**
   - UKT and semantic layers are interpretable, but several upstream model
     blocks do not expose standardized concept interfaces.
2. **Semantic narratives are mostly descriptive**
   - Narratives are generated from loadings/canvas coordinates; they do not yet
     provide faithful path-level attribution across the full pipeline graph.
3. **No common interpretability contract**
   - There is no required interface such as
     `export_latent_subspaces() / export_concepts() / explain_decision()` that
     every block must implement.

---

## Redesign Strategy (Research-Grounded)

## A. Evolve UKT into a learned multimodal latent substrate

### A1) Introduce modality encoders + shared projector heads

- Keep existing block feature extraction for backward compatibility.
- Add per-modality encoders `E_m` that emit embeddings into a shared latent
  dimensionality `d_shared` (e.g., 256/512).
- Add projection heads `P_m` and train with multi-positive contrastive losses
  (InfoNCE variants) across aligned events/time windows.

**Open-source anchors:** OpenCLIP-style contrastive alignment, ImageBind-like
cross-modal embedding alignment patterns.

### A2) Replace rigid region concatenation with tensorized factorization

- Represent UKT state as a factorized tensor `Z[t, m, k]` (time, modality,
  latent channel) instead of only `(n_blocks, 80)`.
- Add low-rank tensor factorization (CP/Tucker) or linear attention pooling to
  derive global kernels while preserving modality interactions.

### A3) Add temporal world-model memory

- Introduce recurrent/transformer state over UKT updates so kernels encode
  transitions, not just static co-variance.
- Train auxiliary next-state prediction and masked-modality reconstruction.

---

## B. Make interpretability an enforced interface contract

### B1) Add a mandatory `InterpretableModule` protocol

Each core model block should implement:

- `export_latent_units()` (neurons/subspaces/concepts)
- `export_feature_attributions(input_batch)`
- `export_alignment_report(reference_modalities)`
- `explain_prediction(context)` returning structured JSON + narrative

This preserves flexibility while forcing intrinsic interpretability coverage.

### B2) Standardize concept bottlenecks for cross-block comparability

- Keep stage SAEs, but standardize concept dictionaries and naming schema.
- Attach concept confidence/calibration and provenance metadata.
- Add sparse probing heads for important downstream decisions.

### B3) Add mechanistic probes where feasible

- Attention-pattern probes for transformer-like components.
- Linear probes / activation patching tests on shared latent channels.
- Counterfactual interventions: zero/boost latent channels and report outcome
  deltas.

---

## C. Preserve existing strengths while migrating

1. Keep current UKT SVD report as a governance-facing baseline.
2. Run old and new UKT paths in shadow mode with parity dashboards.
3. Promote canvas + narrative outputs to consume both legacy and learned latent
   kernels.
4. Add strict faithfulness checks so narratives are constrained by measurable
   attributions.

---

## Implementation Roadmap

### Phase 0 (Hardening, 1–2 weeks)

- Define `InterpretableModule` protocol and retrofit existing blocks.
- Add unified schema for concept export and attribution export.
- Add tests that fail if any block lacks interpretability endpoints.

### Phase 1 (Shared latent MVP, 2–4 weeks)

- Add modality encoders/projectors and contrastive alignment training.
- Keep legacy 80-d features as auxiliary inputs for compatibility.
- Extend UKT class with a dual-state backend (`legacy_matrix`,
  `shared_latent_tensor`).

### Phase 2 (Temporal and mechanistic depth, 4–8 weeks)

- Add temporal memory module and transition objectives.
- Add intervention-based interpretability reports per kernel/channel.
- Introduce cross-run kernel persistence and drift tracking.

### Phase 3 (Governance-grade deployment)

- Version latent spaces and concept vocabularies.
- Add audit trails for explanations and alignment metrics.
- Define fail-safe policy when interpretability confidence is low.

---

## Evaluation Framework (Pass/Fail Criteria)

A system revision should be considered compliant with the vision only when all
conditions below pass.

### UKT kernel criteria

1. **Shared latent tensor**
   - Pass if all modalities map to a common latent space with measurable overlap
     (e.g., retrieval and clustering metrics across modalities).
2. **Intermodality alignment**
   - Pass if aligned pairs score significantly above mismatched pairs on
     contrastive retrieval and probing tasks.
3. **Knowledge reuse**
   - Pass if kernels/concepts learned on one task improve downstream performance
     or sample efficiency on other tasks without retraining from scratch.

### Interpretability criteria

1. **Coverage**
   - Pass if every model block implements `InterpretableModule` endpoints.
2. **Semantic mapping quality**
   - Pass if latent units map to stable concepts with calibration and
     inter-annotator agreement checks where human labels are used.
3. **Narrative faithfulness**
   - Pass if generated explanations remain consistent under attribution and
     intervention tests (no unsupported claims).

---

## Phase 0-2 Completion Status (March 2026)

### Completed Hardening Work

**Phase 0** focused on governance-first UI redesign and performance optimization (not on learned modality alignment):

✅ **Config Centralization**: All 10 governance thresholds consolidated to `hyperspace/config.py`.
✅ **WCAG Accessibility**: Run-ID watermark contrast fixed to 4.5:1 ratio.
✅ **Mission Control Restructure**: 5-section hierarchy with governance outputs first.
✅ **Intelligent Caching**: Hash-based caching for SAE (5-10s saved), SVD reuse, stability estimation.
✅ **Parallel Processing**: 4-6× speedup via ThreadPoolExecutor for data fetching, kernel labeling, stability estimation.
✅ **Structured Logging**: Event-based governance audit trails with text-based execution traces.
✅ **Counterfactual Scenario Memory**: Last 5 runs stored for side-by-side comparison.

See `docs/vision-compliance.md` §6 and `docs/alpha-1.0-issue-tracker.md` for detailed completion status.

### Recommended Immediate Actions (Updated)

The Phase 0-2 work above addressed **urgent governance and usability issues** (UI rendering layer compliance).
The research roadmap below remains valid for **Phase 3+ (post-Alpha)** direction:

1. **Feature names in Reality Regression** (Phase 2, READY): Add FEATURE_NAMES to chart x-axis and hovertemplate.
   FEATURE_NAMES are available in HYPERSPACE_REGISTRY; implementation is straightforward Plotly template update.

2. **Learned modality alignment** (Phase 3, RESEARCH): Formalize the interpretability interface contract and prototype
   a shared latent projector with contrastive alignment. Keep current UKT as governance-facing baseline; run new path
   in shadow mode with parity dashboards.

3. **Cross-modal evaluation** (Phase 3): Add shadow evaluation dashboard comparing:
   - legacy UKT kernel quality
   - shared-latent alignment quality (InfoNCE contrastive losses on paired windows)
   - explanation faithfulness metrics (attribution/intervention tests)

4. **Promote shared-latent path** (Phase 3): Only promote the learned UKT path to default once it matches current
   governance transparency and exceeds current cross-modal utility (measured via kernel coherence, cross-block transfer
   learning, and explanation stability).
