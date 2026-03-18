# Vision Compliance Document — Hyperspace Alpha 1.0

## Purpose

This document defines the architectural roles of the three core subsystems —
the **Pipeline**, the **Universal Knowledge Tensor (UKT) framework**, and the
**Semantic Interpretability framework** — and specifies how each addresses
fundamental challenges in machine learning, big data, and AI governance.

It serves as the binding contract between the codebase and the vision: any
future change must preserve the invariants described here, or explicitly
document why they were relaxed.

---

## 1. Subsystem Roles

### 1.1 The Pipeline (`hyperspace/core/pipeline.py`, `hyperspace/pages/`)

**Role:** Data flow orchestration and multi-modal feature generation.

The pipeline runs five heterogeneous model blocks in sequence — Finance (TFT),
Clusters (BERTopic), Graph Engine (NetworkX), Agent Simulation, and Spatial
Raster — and feeds each block's outputs into a shared feature space. The
pipeline's job is to:

- Execute each block with its own domain-appropriate model architecture.
- Extract a fixed-length feature vector (16 dimensions) from each block.
- Pass raw features to the UKT for standardization and kernel discovery.
- Orchestrate the Semantic Interpretability layer after each block addition.
- Store raw features alongside projected features so that any subsystem
  (including counterfactual analysis) can rebuild state from ground truth.

**What the pipeline does NOT do:** The pipeline does not decide what cross-domain
patterns exist. It generates the raw material; emergence is the UKT's job.

### 1.2 The UKT Framework (`ukt/`)

**Role:** Standardization, cross-domain mixing, and emergent kernel discovery.

The UKT is the central analytical engine. It takes raw, heterogeneous feature
vectors from the pipeline and transforms them into a space where SVD can
discover genuine cross-domain structure. Its components are:

| Component | File | Function |
|-----------|------|----------|
| Feature Region Registry | `ukt/registry.py` | Defines named, non-overlapping regions in the shared feature vector |
| Normalization | `hyperspace/models/knowledge_matrix.py` | Per-region [0,1] scaling BEFORE projection (preserves cross-region energy ratios) |
| Shared Projection | `ukt/projection.py` | Adaptive mixing matrix P that creates cross-region coupling from data |
| SVD Decomposition | `ukt/kernels.py` | Extracts kernels (principal directions of variance) from the projected matrix |
| Kernel Labeling | `ukt/kernels.py` | Generates data-grounded semantic labels for each kernel |
| Stability Estimation | `ukt/stability.py` | Bootstrap confidence intervals on the reality regression |

**The emergence contract:** Kernels are NOT pre-defined. They emerge from the
structure in the data. The pipeline does not name them, the regions do not
determine them, and the projection topology does not restrict which regions can
couple. The only structural prior is the region layout (which features belong
to which domain); everything else — coupling strength, coupling direction,
kernel count, kernel importance — comes from the data.

The critical invariants that guarantee emergence:

1. **Normalize before project.** Per-region normalization scales each block's
   features to [0,1] so they enter the mixing space on comparable scales.
   Normalizing AFTER projection would re-isolate regions and erase the
   cross-region energy ratios the projection encoded.

2. **Open topology.** All region pairs can couple. The adaptive strength
   computation (energy ratio: `min(E_src, E_tgt) / max(E_src, E_tgt)`) gates
   weak couplings to near-zero without hardcoded topology restrictions.

3. **Data-driven blend.** The ratio between data-driven rank-1 direction and
   random orthogonal fallback tracks coupling strength: `alpha = strength`.
   Strong signals get data-driven coupling; weak signals get safe fallback.

4. **Full replay after each block.** When block N arrives and the projection
   matrix P is rebuilt, ALL blocks 1..N are re-projected through the updated P.
   No block retains stale coordinates from an earlier projection epoch.

### 1.3 The Semantic Interpretability Framework (`semantic_interpreter/`, `hyperspace/models/semantic_canvas.py`)

**Role:** Human-readable translation of emergent structure into named
coordinates and natural-language narratives.

The interpretability framework sits downstream of the UKT and provides three
layers of explanation:

| Layer | Component | Purpose |
|-------|-----------|---------|
| Coordinate System | Semantic Canvas (`semantic_interpreter/canvas.py`) | Projects features onto named semantic dimensions (e.g., "Feature Strength", "Cross-Layer Coupling") |
| Pattern Labeling | Kernel Narratives (`ukt/kernels.py`) | Generates structured text explaining what each kernel captures |
| Contestability | Counterfactual Analysis (`hyperspace/pages/counterfactual_tab.py`) | Rebuilds the entire pipeline from raw features with blocks removed |

**The interpretive lens contract:** The Semantic Canvas is a fixed interpretive
frame, not a learning system. Its dimensions are defined by domain knowledge
and remain stable across runs. What changes is the DATA-DRIVEN coordinates:
these depend on the actual feature distribution (entropy, concentration, energy),
not just topology or template weights. The canvas answers "where does this data
land in our interpretive space?" — not "what space should we use?"

This separation is deliberate: the UKT discovers emergent structure; the canvas
provides a stable vocabulary for communicating that structure to humans. Mixing
these roles (e.g., having the canvas discover its own dimensions) would make
explanations unstable and harder to audit.

---

## 2. Addressing Machine Learning Challenges

### 2.1 The Black Box Problem

**Challenge:** Deep learning models are powerful but opaque. Hidden-layer
activations are high-dimensional vectors with no inherent human meaning. Users
cannot inspect why a model produced a particular output.

**How Hyperspace addresses it:**

- **UKT kernels decompose the black box.** Instead of a single opaque prediction,
  the UKT produces N interpretable kernels, each explaining a fraction of total
  variance. Each kernel has a data-grounded label showing which features drive
  it, which blocks contribute, and which regions it spans.

- **The Semantic Canvas translates activations into coordinates.** Raw 80-dim
  vectors become positions in a named semantic space. A stakeholder can see
  "strong Feature Strength, moderate Cross-Layer Coupling" instead of a
  floating-point vector.

- **Narratives adapt to complexity.** Single-region kernels get brief, factual
  descriptions. Two-region couplings highlight the cross-domain link and its
  evidence. Three+ region patterns emphasize the emergent multi-domain
  structure. The narrative shape reflects the kernel's nature.

### 2.2 Cross-Modal Integration

**Challenge:** Real-world systems span multiple modalities — financial time
series, text corpora, network graphs, geospatial data, agent simulations.
Standard ML treats these as separate pipelines with late fusion. Emergent
cross-modal patterns are invisible.

**How Hyperspace addresses it:**

- **The Shared Projection creates genuine feature mixing.** After normalization,
  the projection matrix P maps features across regions so that Finance features
  appear in the Geospatial slice and vice versa. SVD then discovers patterns
  that span modalities — not because we told it to, but because the projected
  data contains cross-region correlations.

- **Adaptive coupling strength is purely data-driven.** If Finance and Graph
  features are both strongly active, they couple. If Agent features are weak,
  they decouple automatically. No hardcoded cross-modal weights.

- **The reality regression is a single multi-modal summary.** The weighted
  combination `importance @ Vt` produces an 80-dim vector that represents the
  system's best single-direction summary of all discovered structure, spanning
  all modalities simultaneously.

### 2.3 Reproducibility and Stability

**Challenge:** ML results must be reproducible. Small perturbations in input
data should not produce wildly different interpretations.

**How Hyperspace addresses it:**

- **Bootstrap stability estimation** (`ukt/stability.py`) provides confidence
  intervals on the reality regression. If the direction is stable under
  resampling, the interpretation is trustworthy.

- **Deterministic projection fallback.** The orthogonal fallback basis in
  `SharedProjection` uses a fixed seed, so when data signals are weak, the
  projection produces reproducible results.

- **Centralized thresholds** (`hyperspace/config.py`) ensure that magic numbers
  like the contributing-region threshold (15%), block contribution minimum (0.1),
  and narrator importance minimum (0.15) are documented, auditable, and
  consistent across the codebase.

---

## 3. Addressing Big Data Challenges

### 3.1 Multi-Source Heterogeneity

**Challenge:** Big data systems ingest streams from fundamentally different
sources — structured (financial), unstructured (text), relational (graphs),
spatial (rasters), and behavioral (agent simulations). These sources have
different dimensionalities, distributions, and update frequencies.

**How Hyperspace addresses it:**

- **The Feature Region Registry** standardizes heterogeneous outputs into a
  uniform 80-dim vector with named, non-overlapping 16-dim slices. Each source
  extracts its own features using domain-appropriate methods, then maps them to
  its assigned region.

- **Per-region normalization** ensures that different scales (financial prices
  vs. graph centrality scores vs. topic probabilities) do not dominate the
  shared space. Each region is independently scaled to [0,1] before projection.

- **The UKT matrix grows incrementally.** Each block's features are appended as
  a new row. The SVD is recomputed after each addition, so the system handles
  streaming multi-source data without requiring all sources to be available
  simultaneously.

### 3.2 Dimensionality and Scalability

**Challenge:** Combining five 16-dim feature vectors into an 80-dim shared space
is manageable. Scaling to hundreds of sources or thousands of features requires
the same architecture to hold.

**How Hyperspace addresses it:**

- **The UKT framework is source-count agnostic.** The `FeatureRegionRegistry`
  supports arbitrary numbers of regions. Adding a sixth data source means
  registering a new region and extending the feature vector. The projection,
  SVD, and kernel labeling scale naturally with matrix dimensions.

- **SVD truncation is automatic.** The number of kernels equals
  `min(n_blocks, feature_dim)`. As more sources are added, more kernels become
  available, but the importance weighting ensures only meaningful ones surface
  in labels and narratives.

- **Projection sparsity scales linearly.** In the full-topology projection, the
  number of coupling blocks is O(n_regions²). For modest region counts (5–20),
  this is negligible. For larger systems, topology can be restricted to
  semantically meaningful pairs while still letting data drive strength.

### 3.3 Data Quality and Missing Sources

**Challenge:** In real-world big data systems, sources fail. A financial data
feed goes down; a geospatial raster is unavailable. The system must degrade
gracefully.

**How Hyperspace addresses it:**

- **Missing blocks produce zero features.** If a pipeline block fails, its
  region stays at zero. The projection's energy-ratio gating means zero-energy
  regions automatically decouple — no special handling needed.

- **Counterfactual analysis quantifies source impact.** The counterfactual tab
  rebuilds the ENTIRE pipeline — including the projection matrix — from
  remaining blocks' raw features. This answers "what would we know without
  Finance?" with mathematical precision, not approximation.

- **Graceful fallback in all components.** SVD handles degenerate matrices
  (fallback to identity). The canvas handles empty entries. Narratives handle
  single-block systems. The system never crashes on missing data.

---

## 4. Addressing AI Governance Challenges

### 4.1 Contestability

**Challenge:** EU AI Act and emerging governance frameworks require that
AI-generated decisions be contestable — stakeholders must be able to challenge
the basis of a decision and receive a meaningful response.

**How Hyperspace addresses it:**

- **Kernel provenance is fully traceable.** Each kernel label includes:
  the specific features that drive it (with loading values), the blocks that
  contribute (with activation strengths), the regions it spans (with
  percentage contributions), and a narrative explaining the pattern.

- **Counterfactual contestability.** The counterfactual tab provides a
  mathematically rigorous answer to "what if we removed this data source?"
  by rebuilding the projection from remaining raw features. The resulting
  kernel differences show exactly how each source shapes the overall analysis.

- **Feature evidence links.** Every canvas entry and kernel label carries
  `feature_evidence` — a structured trace from the high-level interpretation
  back to specific feature indices, their values, and their source regions.

### 4.2 Transparency and Auditability

**Challenge:** Governance requires that the system's reasoning process be
transparent and auditable. An external auditor must be able to trace any
output back to its inputs and understand the transformation chain.

**How Hyperspace addresses it:**

- **The transformation chain is explicit and auditable:**
  ```
  Raw Features → Per-Region Normalization → Shared Projection (P)
  → Projected Matrix → SVD → Kernels → Labels + Narratives
  ```
  Every step is a documented mathematical operation with inspectable
  intermediate results stored in snapshots.

- **Snapshots preserve full state.** Each `add_block()` call produces a
  snapshot containing: raw features, the projected matrix, SVD decomposition,
  kernel labels, canvas entries, stability estimates, and the projection
  matrix itself. An auditor can reconstruct any interpretation from the
  snapshot.

- **Centralized, documented thresholds.** All magic numbers are defined in
  `hyperspace/config.py` with clear names and descriptions. An auditor can
  review `KERNEL_CONTRIBUTING_REGION_THRESHOLD = 0.15` and understand its
  role without reading the implementation.

### 4.3 Faithfulness of Explanations

**Challenge:** Generated explanations must faithfully reflect the model's
actual reasoning, not post-hoc rationalizations. An explanation that sounds
plausible but doesn't match the underlying computation is worse than no
explanation.

**How Hyperspace addresses it:**

- **Narratives are generated from data, not templates.** Kernel narratives
  branch on actual kernel complexity (single-region, two-region coupling,
  multi-domain pattern) and cite specific feature loadings, block
  contributions, and region scores. Every claim in a narrative is backed by
  a number from the SVD decomposition.

- **The canvas is an honest interpretive lens.** The Semantic Canvas does not
  pretend to discover new dimensions — it projects onto a fixed set of named
  axes defined by domain knowledge. Coordinates are data-driven (entropy,
  concentration, energy), but the axes are stable. This prevents the common
  failure mode of post-hoc explanation systems that generate different
  "important features" on each run.

- **Reconstruction error is reported.** The UKT reports how well the kernel
  decomposition reconstructs the original matrix. High reconstruction error
  means the kernels don't capture the data well — a built-in honesty signal.

### 4.4 Bias Detection and Multi-Stakeholder Accountability

**Challenge:** AI systems serving multiple stakeholders (financial analysts,
policymakers, military planners) must make their assumptions visible so that
each stakeholder can assess whether the system's framing aligns with their
values.

**How Hyperspace addresses it:**

- **Region scores expose domain weighting.** Each kernel's `region_scores`
  dict shows exactly how much each domain (Finance, Text, Graph, Agent,
  Spatial) contributes. If Finance dominates every kernel, that imbalance is
  immediately visible.

- **Contributing regions expose coupling assumptions.** The
  `contributing_regions` field (regions contributing >15% of total loading)
  makes cross-domain assumptions explicit. A stakeholder can challenge
  whether Finance-Graph coupling should drive a particular interpretation.

- **The projection matrix is inspectable.** The `SharedProjection.active_couplings`
  property lists all active cross-region couplings with their data-derived
  weights. An auditor can verify that coupling strength reflects data, not
  hardcoded preferences.

---

## 5. Compliance Matrix

| Principle | Pipeline | UKT | Interpretability | Status |
|-----------|----------|-----|-----------------|--------|
| Emergence (kernels not predetermined) | Generates raw features | Shared projection + SVD discovers patterns | Reports what was discovered | **Compliant** |
| Data-driven coupling | N/A | Energy ratio gates strength; rank-1 direction from features | N/A | **Compliant** |
| Normalize-before-project | N/A | Per-region [0,1] then P@ | N/A | **Compliant** |
| Full replay on projection rebuild | Stores raw features | Re-projects all blocks through updated P | Resets and replays canvas for all blocks | **Compliant** |
| Faithful narratives | N/A | Kernel labels cite specific loadings | Canvas coordinates from data distribution | **Compliant** |
| Contestability | Stores raw features in snapshots | Counterfactual rebuilds projection from remaining raw features | Narrative differences shown per block removal | **Compliant** |
| Auditability | Snapshot preservation | Inspectable projection matrix, coupling list, thresholds | Feature evidence traces, reconstruction error | **Compliant** |
| Graceful degradation | try/except with fallback data | Zero-energy regions decouple; SVD handles degenerate matrices | Canvas/narrative handle empty inputs | **Compliant** |
| Centralized configuration | N/A | Thresholds in `config.py` | Canvas dimensions in `canvas.py` | **Compliant** |
| UI interpretability | Feature names in charts | N/A | Reality Regression uses FEATURE_NAMES in hover/labels | **Non-Compliant** — shows indices 0–79, not names |
| UI accessibility | WCAG AA contrast | N/A | All text meets 4.5:1 contrast on dark background | **Non-Compliant** — captions, labels, tabs fail |
| UI progressive disclosure | Technical vs governance content separated | N/A | Mission Control groups governance outputs before diagnostics | **Non-Compliant** — 15+ flat sections, no hierarchy |
| Threshold centralization | N/A | All thresholds in `config.py` | Display thresholds in `config.py` | **Partial** — 10 governance thresholds scattered in tab modules |
| Computation caching | N/A | N/A | Expensive operations cached between clicks | **Non-Compliant** — SAE, counterfactual SVD, UVT, USE uncached |

---

## 6. UI-Layer Compliance Gaps

The Alpha 1.0 backend satisfies the emergence and governance contracts above.
However, the **UI rendering layer** introduces compliance gaps that undermine
the governance promise at the point of stakeholder interaction:

### 6.1 Interpretability Broken at Display Time

The Reality Regression chart (`interpreter_tab.py:264`, `counterfactual_tab.py:368`)
is the system's primary feature-importance visualization. It displays 80 bars
labeled by **feature index** (0, 1, 2...) instead of `FEATURE_NAMES` ("attention_recent_1d",
"China: Eigenvector Centrality", etc.). The full provenance chain (raw data →
feature → kernel → narrative) is intact in the backend, but the UI drops human-readable
names at the final rendering step. This is a **critical vision failure**: the
system is interpretable internally but opaque at the user interface.

**Affected components**: `kernel_viz.plot_reality_regression()`, `_plot_rr_diff()`,
counterfactual domain-level impact.

### 6.2 Governance Content Buried Under Diagnostics

Mission Control (Tab 0) renders 15+ sections in flat scroll order: 5 summary
metrics → governance flags → alignment comparison → scorecard → narratives →
faithfulness → drift → kernel evolution → contract → export. A policy officer
seeking the governance scorecard must scroll past technical metrics they don't
understand. The tab lacks an executive summary, content grouping, or table of
contents.

### 6.3 Contestability UX Incomplete

The counterfactual tab correctly rebuilds the pipeline from raw features (§4.1
above), but the UI does not support **scenario memory** — each counterfactual
run overwrites the previous result. A governance auditor cannot compare
"remove Finance" vs. "remove Clusters" side-by-side, which limits the practical
utility of the contestability mechanism.

### 6.4 Uncached Computation Degrades Repeatability

SAE training (100 epochs), stability estimation (6+ SVD runs), and counterfactual
baseline SVD all recompute on every button click with no session-state caching.
This means identical inputs produce slightly different outputs (due to torch
stochastic initialization), which conflicts with the reproducibility invariant
(§2.3 above).

See `STRATEGY_UI.md` for the full strategic UI audit and issue tracker.

---

## 7. Open Directions (Post-Alpha)

The Alpha 1.0 backend architecture satisfies the emergence and governance
contracts above. The UI-layer gaps (§6) must be resolved before the system
can be considered compliant at the stakeholder-facing level. The following
directions are identified for future work:

1. **Learned cross-modal alignment.** Replace SVD with contrastive alignment
   objectives (InfoNCE) for stronger cross-modal binding.
2. **Temporal world-model memory.** Add recurrent state over UKT snapshots so
   kernels encode transitions, not just static covariance.
3. **InterpretableModule protocol.** Require every pipeline block to implement
   `export_latent_units()`, `export_feature_attributions()`, and
   `explain_prediction()` for full mechanistic transparency.
4. **Kernel persistence and drift tracking.** Version kernels across runs to
   detect concept drift and enable knowledge transfer.
5. **Calibrated concept confidence.** Attach uncertainty estimates to SAE
   concepts and canvas coordinates.

These are documented in `docs/vision-assessment-and-redesign.md` with a
phased implementation roadmap.

---

*Document version: Alpha 1.0 — March 2026*
*Binding for: `hyperspace/` and `ukt/` packages on branch `claude/alpha-1.0-vision-redesign-2mvlf`*
