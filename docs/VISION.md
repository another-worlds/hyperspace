# Vision — Hyperspace

## Mission

Hyperspace is a multimodal geopolitical intelligence system that fuses heterogeneous data streams — financial, textual, relational, spatial, and behavioral — into a single emergent analytical substrate, then translates that substrate into human-readable, contestable, and auditable interpretations.

The system exists to prove that **AI accountability and analytical power are not trade-offs**. A system can be simultaneously powerful (discovering cross-domain patterns invisible to siloed models), interpretable (explaining what it found in human language), and governable (allowing any stakeholder to challenge any conclusion and trace it back to evidence).

---

## Core Philosophy

1. **Emergence over prescription.** The system does not define what patterns exist. It creates the conditions for patterns to emerge from data, then reports what it discovers. Kernels, couplings, and cross-domain links are never hardcoded — they are computed from evidence.

2. **Interpretability as first-class architecture.** Interpretability is not a post-hoc overlay. It is woven into every layer: the feature space is monolithic with block-level provenance labels, projections are inspectable, kernels carry structured labels, canvas dimensions emerge from sparse concept discovery, and narratives are generated from measured quantities — not templates. No semantic label — kernel name, block label, canvas axis, concept ID — is hardcoded. All are derived from data.

3. **Governance by design.** Every output is traceable from high-level narrative back to specific feature values, through documented mathematical transformations, with inspectable intermediate states. An auditor can reconstruct any conclusion from stored snapshots without re-running the system.

4. **Faithful representation.** Explanations must reflect actual computation, not plausible-sounding approximations. Narratives cite specific loadings, block scores, and feature evidence. Reconstruction error is reported as an honesty signal. The system admits when its model is a poor fit.

5. **Graceful degradation.** When data sources fail, the system continues with reduced scope rather than crashing or producing silent errors. Missing modalities decouple automatically through energy-based gating; no special-case handling is required.

6. **Open topology.** Any domain can couple with any other domain. The system imposes no structural prior on which modalities should interact — only data-driven evidence determines coupling strength and direction.

---

## Three Pillars

### The Pipeline — Data Flow Orchestration

The Pipeline executes heterogeneous model blocks in sequence, each using domain-appropriate architectures (temporal forecasting, topic modeling, graph analysis, spatial decomposition, agent simulation). Its sole responsibility is generating raw features from diverse sources and feeding them into the shared analytical substrate.

**What the Pipeline does NOT do:** It does not decide what cross-domain patterns exist. It generates raw material; emergence is the substrate's job.

### The Universal Knowledge Tensor — Emergent Structure Discovery

The UKT is the central analytical engine. It standardizes heterogeneous features into a monolithic 80-dim space with block-level provenance labels, applies adaptive cross-block projection to create genuine feature mixing, then decomposes the result via SVD to discover emergent kernels — principal directions of variance that span multiple blocks.

**The Emergence Contract:** Kernels are NOT pre-defined. They emerge from the structure in the data. The pipeline does not name them, the blocks do not determine them, and the projection topology does not restrict which blocks can couple. The only structural prior is the block feature allocation (which features belong to which block for provenance); everything else — coupling strength, coupling direction, kernel count, kernel importance — comes from the data.

**Kernels are features of information, not sources of it.** A kernel is a learned direction in the shared projected feature space — a pattern along which multiple blocks co-vary. Block names that appear attached to a kernel (e.g. "K0 (Finance 0.60, Graph 0.27)") are a *provenance decomposition* of where the kernel's mass falls, not a statement of what the kernel is "about." Any sentence that puts a kernel in the grammatical position of a subject with a topic — "K0 encodes Finance," "K0 is a finance kernel" — mis-frames a cross-block regularity as a single-block source and silently erases the emergent structure the UKT exists to discover. The correct grammar puts the kernel in the predicate's object position: "K0 is the axis along which Finance and Graph features co-vary." Sources are the data blocks; kernels are regularities the system found in the joint signal those sources produced.

### The Semantic Interpretability Framework — Human Translation

The Interpretability Framework sits downstream of the UKT and translates emergent structure into named coordinates and natural-language narratives. It provides three layers of explanation: an emergent coordinate system (the Semantic Canvas, whose axes are discovered SAE concepts), pattern labeling (kernel narratives grounded in SVD evidence), and contestability (counterfactual analysis).

**The Interpretive Lens Contract:** The Semantic Canvas is a fully emergent interpretive system. Its dimensions are discovered — not prescribed — from SAE concept decomposition of the projected feature space. Each run produces its own set of canvas axes corresponding to active sparse autoencoder concepts. What varies is both the dimensions themselves (which concepts emerge) and the coordinates (how strongly each concept activates). The Canvas translates emergent structure into human-readable coordinates; it does not impose structure. The UKT discovers kernels; the SAE discovers concepts; the Canvas names them.

The "features of information, not sources of it" grammar rule from the UKT subsection above applies verbatim to SAE concepts. A concept whose top loadings span multiple blocks is not a concept "about" the dominant block; it is a concept *of the joint pattern* across those blocks. Labels that collapse this to a single block are a violation of both the Emergence Contract and the Interpretive Lens Contract.

---

## Critical Invariants

These properties must hold in every version of the system. Violating any of them undermines the core promise.

1. **Normalize before project.** Global min-max normalization scales the full 80-dim feature vector to comparable ranges before projection. This ensures coupling reflects genuine cross-block interactions, not just per-block energy differences. Normalizing after projection would destroy the holistic structure.

2. **Open topology.** All block pairs can couple. Coupling strength is gated by data-driven energy ratios, never by hardcoded topology restrictions.

3. **Data-driven blend.** The ratio between data-driven coupling direction and random orthogonal fallback tracks coupling strength. Strong signals get data-driven coupling (rank-1 outer product); weak signals get safe fallback.

4. **Full replay after each block.** When a new block arrives and the projection matrix is rebuilt, ALL blocks are re-projected through the updated matrix. No block retains stale coordinates from an earlier projection epoch.

5. **Faithful narratives.** Every claim in a kernel narrative is backed by a measured quantity from the SVD decomposition. Narrative shape reflects kernel complexity (single-block, two-block coupling, multi-block pattern). A governance-grade narrative must contain all of the following; a narrative missing any of them is incomplete and must not be presented as a governance output:

   (a) **Anchor** — the decision, forecast, or event this narrative explains (not a free-floating "state of the system" summary).
   (b) **Direction of influence** — drawn from the reality-regression vector or counterfactual deltas, not from SVD co-variance alone. Co-variation is not causation; the narrative must cite the actual gradient/counterfactual signal when making directional claims.
   (c) **Temporal context** — when the pattern activated, and whether it is a change from the prior run (drift deltas, stability across snapshots).
   (d) **Cross-item synthesis** — how the active concepts relate to each other and to the dominant kernels. Parallel independent bullets ("C05 is X. C01 is Y. C00 is Z.") are not synthesis; the narrative must describe relationships between items.
   (e) **World-grounded feature names** — resolved through the feature registry to human-meaningful terms, not bare indices like `Finance_19` or `Agents_74`.

8. **Regions are provenance metadata only.** Block-to-feature-range mappings exist for provenance tracking and feature naming, never for structural enforcement. No block "owns" reserved indices; the feature space is monolithic.

6. **Counterfactual contestability.** Removing a data source rebuilds the ENTIRE pipeline — including the projection matrix — from remaining blocks' raw features. Impact is measured with mathematical precision, not approximation.

7. **Snapshot preservation.** Each analytical step produces a snapshot containing raw features, projected matrix, SVD decomposition, kernel labels, canvas entries, stability estimates, and the projection matrix. Any interpretation can be reconstructed from its snapshot.

8. **Centralized thresholds.** All magic numbers used in governance and display logic are defined in a single configuration source with descriptive names, making them auditable and consistent.

9. **LLM narration is a vital, non-optional part of the interpretability framework.** The language model (currently `arnir0/Tiny-LLM`, but the role, not the identity, is what matters) is the only component in the pipeline capable of translating emergent cross-block structure into language that a non-engineer can audit. Templates cannot perform this role, for three structural reasons:

   1. **Emergent concepts are cross-block by construction.** A concept that loads (+0.137) on a Finance feature, (+0.103) on an Agents feature, and (-0.093) on a Graph feature is, by definition, not a "Finance concept" — it is a relationship between Finance, Agents, and Graph. A template can only slot-fill a single label per concept and therefore must collapse the concept to its single most-dominant block. That collapse silently destroys the exact structure the UKT was built to discover.
   2. **Synthesis across concepts requires a language model.** A faithful narrative must relate multiple kernels and concepts to one another ("when K0 shifts in this direction, concept C05 rises together with C01, implying …"). This is synthesis, not slot-filling. Templates are per-item; they have no mechanism for cross-item reasoning.
   3. **Governance audiences are non-engineers.** Policy officers, auditors, and UN stakeholders do not read loading matrices. The narrative layer is the only interface through which they can interrogate the system's conclusions. If that layer is a statistics stitch, the interpretability chain is complete on paper (narrative → kernel → SVD → features → data) but broken in practice, because the "narrative" link carries no semantic content beyond what a spreadsheet would.

   Consequences for implementation:
   - LLM unavailability **must be treated as a hard failure in the governance pipeline**, not a silent fallback. The system should refuse to emit a "System Summary" if the LLM is down, and must never present template output as if it were LLM-generated interpretation.
   - Templates are acceptable **only for auxiliary provenance text** (e.g., listing top feature loadings verbatim), never for claims about what a concept, kernel, or reality-regression direction *means*.
   - Concept labels produced by the SAE must preserve their cross-block signature (signed contributions per block) as structured data. The narrator, given that structured signature and an LLM, produces the human-readable claim. The SAE does not get to decide the concept's semantic label.

10. **Label–evidence consistency.** No narrative sentence may assign a single-block label to a concept or kernel whose top feature loadings span multiple blocks. If the evidence printed alongside a label would, on its own, falsify the label, the label is not allowed to be emitted. This rule applies to templates and LLM output alike. A concrete example of the failure mode this invariant exists to prevent: a sentence that reads *"C05 primarily encodes Finance information"* printed directly above the evidence *"Top feature loadings: Finance_19 (+0.137), Agents_3 (+0.103), Graph_36 (−0.093)"* — three top features spanning three blocks with mixed sign, labeled as a single-block "Finance" concept. The evidence on the next line falsifies the label on the previous line. That sentence must never be emitted.

---

## Ideal End-State

The fully realized Hyperspace is a **learned, modality-agnostic, cross-domain representation substrate** with intrinsic interpretability at every layer:

- **Modality encoders** project diverse data types into a shared latent space through learned alignment, not hand-partitioned concatenation.
- **Cross-modal contrastive objectives** pull semantically related representations together across modalities, discovering genuine shared structure.
- **Temporal world-model memory** encodes transitions and interventions over time, not just static co-variance snapshots.
- **Mechanistic probes** provide causal/feature pathway explanations through attention-pattern analysis, linear probing, and intervention-based attribution.
- **Every model block** implements a standardized interpretability contract exposing latent units, feature attributions, alignment reports, and structured explanations.
- **Knowledge persistence** allows kernels to be versioned, distilled, and transferred as reusable analytical modules across runs.

The current system is a strong partial realization: it has a usable shared matrix abstraction, integrated interpretability, and governance compliance. The path from here to the ideal end-state is evolutionary, not revolutionary — extend the existing architecture rather than replace it.

---

## What Hyperspace Is NOT

- **Not a dashboard.** It is an analytical engine that happens to have a UI. The core computation is UI-framework-agnostic.
- **Not late fusion.** It does not run siloed models and merge results at the end. Cross-domain mixing happens in the shared projection space before decomposition.
- **Not pre-scripted analysis.** It does not output predetermined conclusions. What it discovers depends entirely on what the data contains.
- **Not a black box with a wrapper.** Interpretability is structural, not cosmetic. The system's explanations reflect its actual computations, verified by reconstruction error and feature evidence traces.
- **Not a static report generator.** It supports interactive exploration, counterfactual reasoning, and contestability — stakeholders interrogate the analysis, not just consume it.
