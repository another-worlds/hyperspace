# One-Shot: Phase 5 — Medium-Term Implementation

**Date compiled**: 2026-03-26
**Source specs**: `ARCHITECTURE.md` §§ SPEC-4, SPEC-6, SPEC-7
**Branch**: `claude/compile-medium-todos-SzpzK`

This is a self-contained implementation prompt for a stateless LLM session. It covers all three medium-term specs exactly as designed in `ARCHITECTURE.md`. Read this file in full before writing any code.

---

## Mandatory Pre-Read (before writing any code)

Read these files to orient yourself:

```
hyperspace/config.py                          # All constants — add new ones here
hyperspace/core/types.py                      # InterpretableModule protocol, BlockSnapshot
hyperspace/core/temporal_memory.py            # KernelMemory — SPEC-7 extends this
hyperspace/models/knowledge_matrix.py         # UKT core — SPEC-4 and SPEC-7 modify this
ukt/projection.py                             # SharedProjection — SPEC-7 adds warm-start
hyperspace/core/faithfulness.py               # Faithfulness checks — SPEC-6 refactors shared logic
hyperspace/pages/interpreter_tab.py           # Tab 5 — SPEC-6 adds Mechanistic Probes expander
hyperspace/pages/mission_control_tab.py       # Tab 0 — SPEC-7 adds Knowledge Base expander
hyperspace/pages/pipeline_tab.py              # Tab 6 — SPEC-7 adds lineage annotations
hyperspace/core/pipeline.py                   # Main pipeline — all 3 specs integrate here
```

---

## Global Constraints (apply to all three specs)

- **CPU-only** — all training and inference must run on CPU. No CUDA.
- **No external APIs** — all new components are self-contained; no network calls.
- **Feature flags off by default** — every spec introduces a feature flag in `config.py` defaulting to `False`. The system must be fully functional at the default settings (= current behaviour).
- **Backward compatibility** — existing SVD path, governance flags, and all tab UIs must continue to work exactly as before when feature flags are off.
- **JSON-serializable outputs** — all new data structures stored in session state or exported must be JSON-serializable (numpy arrays → nested lists, numpy scalars → Python float/int).
- **Graceful degradation** — wrap new computations in `try/except`; on failure, log via `st.warning()` and skip the new feature (never crash the pipeline).
- **< 2 seconds per spec on CPU** — each new computation path must complete within 2 s during a normal pipeline run.
- **PEP 8, type hints where natural, docstrings on every new public function.**

---

## SPEC-4: Cross-Modal Contrastive Alignment

### What to build

A learned alignment path that runs **in parallel** with the existing SVD path. At default config it has zero weight and is invisible. When `CONTRASTIVE_WEIGHT` > 0, it blends contrastive-derived kernels into the UKT output.

### New file: `hyperspace/models/contrastive_encoder.py`

```python
"""
Cross-modal contrastive encoders for SPEC-4.

Per-block MLP encoders project each block's native feature vector into a
shared 32-dim latent space. InfoNCE loss pulls semantically related
cross-block pairs together.
"""
```

Implement:

1. **`BlockEncoder(nn.Module)`** — small MLP: `Linear(in_dim, 64) → ReLU → Dropout(0.1) → Linear(64, 32)`. One per block. `in_dim` = length of that block's raw feature vector.

2. **`ContrastiveEncoderBank`** — holds one `BlockEncoder` per block name. Manages training and inference.
   - `encode(block_name: str, features: np.ndarray) -> np.ndarray` — returns 32-dim latent (numpy, detached).
   - `train_step(block_features: dict[str, np.ndarray]) -> float` — one Adam step with InfoNCE loss; returns loss value.
   - `alignment_score(block_features: dict[str, np.ndarray]) -> float` — mean pairwise cosine similarity across all encoded block pairs. This is the governance metric.

3. **InfoNCE loss** — temperature `τ = config.CONTRASTIVE_TEMPERATURE` (default 0.07). Positive pairs: `(encoder_i(f_i), encoder_j(f_j))` for all `i ≠ j` in the same run. Negatives: feature vectors with shuffled block assignments + Gaussian noise (σ=0.05).

4. **Encoder weights** — cached in `st.session_state["contrastive_encoder_bank"]`. Persisted to `kernel_library.json` alongside SPEC-7 data (optional, best-effort).

### Modify `hyperspace/models/knowledge_matrix.py`

Add a **dual-path backend**:

- After the existing SVD decomposition, if `config.ENABLE_CONTRASTIVE_ALIGNMENT` and `CONTRASTIVE_WEIGHT > 0.0`:
  1. Run `ContrastiveEncoderBank.train_step(block_features)` (one step per pipeline run — online learning).
  2. Encode each block, stack into a matrix, run SVD on the 32-dim latent space to get contrastive kernels.
  3. Blend: `final_kernels = (1 - CONTRASTIVE_WEIGHT) × svd_kernels + CONTRASTIVE_WEIGHT × contrastive_kernels`
- At `CONTRASTIVE_WEIGHT = 0.0` (default) the code path is unreachable after the flag check — pure SVD, no overhead.
- Store `alignment_score` in the snapshot under key `"contrastive_alignment_score"`.

### Modify `hyperspace/config.py`

```python
# SPEC-4: Cross-modal contrastive alignment
ENABLE_CONTRASTIVE_ALIGNMENT: bool = False
CONTRASTIVE_WEIGHT: float = 0.0          # 0.0 = pure SVD; 1.0 = pure contrastive
CONTRASTIVE_TEMPERATURE: float = 0.07   # InfoNCE temperature τ
CONTRASTIVE_LATENT_DIM: int = 32        # shared latent dimension
```

### Modify `hyperspace/core/pipeline.py`

After `ukt.add_block()` for the final block, call `contrastive_encoder_bank.train_step()` if flag is on. Store `alignment_score` in `st.session_state["contrastive_alignment_score"]` for display.

### Files affected

| File | Change |
|------|--------|
| `hyperspace/models/contrastive_encoder.py` | **NEW** |
| `hyperspace/models/knowledge_matrix.py` | Add dual-path blend |
| `hyperspace/config.py` | Add 4 constants |
| `hyperspace/core/pipeline.py` | Call train_step after final block |

---

## SPEC-6: Mechanistic Probes

### What to build

Three interactive probing tools in Tab 5 (Semantic Interpreter) that let users trace causal pathways through the UKT. All probes are **read-only** — they operate on copies of cached state and never modify the pipeline's outputs.

### New file: `hyperspace/core/mechanistic_probes.py`

```python
"""
Mechanistic probes for SPEC-6.

Three probe types: activation patching, linear probing, feature pathway tracing.
All operate on copies of snapshot data — never modify cached pipeline state.
"""
```

Implement these three functions:

#### 1. `activation_patch(snapshot, kernel_idx: int) -> dict`

Extend the logic from `faithfulness.check_kernel_attribution_faithfulness()` into a user-facing tool:

- Takes the pipeline snapshot and a user-selected `kernel_idx` (0-based).
- Zeros out `snapshot["singular_values"][kernel_idx]`.
- Recomputes `reality_regression` (dot product of zeroed-out reconstruction).
- Returns a `dict` with:
  - `"original_reality_regression"`: original values (list of float)
  - `"patched_reality_regression"`: patched values (list of float)
  - `"affected_canvas_dims"`: list of canvas dimension names whose values shifted > 0.05
  - `"reconstruction_error_delta"`: float (MSE difference)
  - `"kernel_label"`: str (from `snapshot["kernel_labels"][kernel_idx]`)

#### 2. `linear_probe(kernel_memory, governance_flags_history: list[dict]) -> dict`

Train one `sklearn.linear_model.LogisticRegression` per governance flag on `KernelMemory` cross-run data:

- Requires minimum 10 runs in `kernel_memory` — return `{"insufficient_history": True}` if fewer.
- Feature matrix X: `kernel_activation` matrix rows stacked across runs.
- Target y: binary presence of each flag (GOV-001 through GOV-006) across runs.
- For each flag, fit and report `coef_` per kernel as predictive weights.
- Returns `dict[flag_name, dict]` with keys: `"per_kernel_weights"`, `"most_predictive_kernel"`, `"predictive_weight"` (0–1 scale).
- Falls back to `{"fallback": "per_feature_correlation", "message": "..."}` when `sklearn` unavailable.

#### 3. `trace_feature_pathway(snapshot, feature_idx: int, feature_registry) -> dict`

Trace a single feature through the full computation chain:

```
feature_idx
  → feature_name       (registry.feature_name(feature_idx))
  → raw_value          (snapshot["raw_features"][feature_idx])
  → normalized_value   (snapshot["normalized_features"][feature_idx])
  → projected_value    (snapshot["projected_features"][feature_idx])
  → kernel_loadings    ({k_label: float for k in kernels} using Vt[:, feature_idx])
  → canvas_dimensions  (list of canvas dim names this feature influences, |loading| > 0.1)
  → narrative_mentions (list of kernel narrative strings that mention the feature name)
```

Returns a structured `FeaturePathway` dict with all 7 keys above.

### Modify `hyperspace/core/faithfulness.py`

Extract the kernel-zeroing logic from `check_kernel_attribution_faithfulness()` into a private helper `_zero_kernel_and_recompute(snapshot, kernel_idx)` shared by both the faithfulness check and the new `activation_patch()` probe.

### Modify `hyperspace/pages/interpreter_tab.py`

Add a new **"Mechanistic Probes"** `st.expander` below the existing "Advanced Diagnostics" section:

```
▶ Mechanistic Probes  [exploratory — not authoritative]
  Probe type:  [Activation Patching ▼]  [Linear Probing]  [Feature Pathway Tracing]

  [Activation Patching]
    Kernel: [K0 — temporal-volatility ▼]
    [Run Probe]
    → Bar chart: original vs patched reality regression (go.Bar, dark theme)
    → Affected canvas dimensions listed below chart

  [Linear Probing]
    [Run Probe]
    → Table: flag name | most predictive kernel | weight

  [Feature Pathway Tracing]
    Feature index: [0 ▼]  (dropdown of all 80 feature names)
    [Run Probe]
    → Expandable trace with values at each step
```

- All probe results stored in `st.session_state["probe_results"]` as a list (append each run).
- Each result dict includes `"probe_type"`, `"timestamp"`, and result payload.
- Add an "Include in governance export" checkbox (default off) — when on, probe results are appended to the exported report JSON.

### Modify `hyperspace/config.py`

```python
# SPEC-6: Mechanistic probes
PROBE_CANVAS_LOADING_THRESHOLD: float = 0.1   # min |loading| to count as canvas influence
PROBE_MIN_RUNS_FOR_LINEAR: int = 10           # min KernelMemory runs for linear probing
```

### Files affected

| File | Change |
|------|--------|
| `hyperspace/core/mechanistic_probes.py` | **NEW** |
| `hyperspace/core/faithfulness.py` | Extract shared helper |
| `hyperspace/pages/interpreter_tab.py` | Add Mechanistic Probes expander |
| `hyperspace/config.py` | Add 2 constants |

---

## SPEC-7: Knowledge Persistence & Transfer

### What to build

Three-layer system: **kernel versioning** (stable identity via cosine lineage), **distillation** (extract `KernelTemplate` from persistent kernels), **transfer** (optionally warm-start new runs). All layers are additive — the existing SVD path is unchanged.

### New file: `hyperspace/core/kernel_library.py`

```python
"""
Kernel library for SPEC-7.

Three-layer persistence: versioning (lineage tracking), distillation
(KernelTemplate extraction), and transfer (warm-start initialization).
"""
```

Implement:

#### `KernelTemplate` dataclass

```python
@dataclass
class KernelTemplate:
    lineage_id: str
    frozen_vt: list[float]              # (80,) feature loadings — list, not ndarray
    importance_range: tuple[float, float]
    stability_score: float
    dominant_region: str               # e.g. "temporal-pattern"
    canonical_narrative: str
    first_seen_run: str                # ISO timestamp
    run_count: int
    metadata: dict
```

#### `KernelLibrary` class

Manages versioning, distillation, and transfer. Persists to/from `kernel_library.json` in the working directory.

**Versioning methods:**
- `match_or_mint(vt_row: np.ndarray, label: str, importance: float, narrative: str, run_id: str) -> str` — compare against known lineage Vt rows via cosine similarity. If `cosine > KERNEL_LINEAGE_COSINE_THRESHOLD` (config, default 0.85), assign existing `lineage_id`. Otherwise mint a new UUID. Update lineage metadata.
- `record_run(lineage_id: str, importance: float, narrative: str, run_id: str)` — append to importance trend and narrative history for a lineage.

**Distillation methods:**
- `distill_eligible()` — scan all lineages; for any with `run_count >= KERNEL_DISTILLATION_MIN_RUNS` (config, default 5) that don't yet have a template, create a `KernelTemplate` and add to `self.templates`.
- Template's `frozen_vt` = mean of last `KERNEL_DISTILLATION_MIN_RUNS` observed Vt rows. `stability_score` = mean cosine across consecutive runs. `dominant_region` = region name of highest-loading feature.

**Transfer methods:**
- `get_transfer_bases() -> list[np.ndarray]` — returns list of `frozen_vt` arrays (as np.ndarray) for all templates, weighted by `stability_score`. Called by `SharedProjection` when `KERNEL_TRANSFER_ENABLED` is on.
- `prune_stale(current_run_count: int)` — archive (move to `self.archived`) any template not seen for 20+ runs (do not delete).

**Persistence:**
- `save(path: str = "kernel_library.json")` — serialize full state to JSON.
- `load(path: str = "kernel_library.json") -> KernelLibrary` — classmethod, returns new instance. Silently returns empty library if file not found.

### Modify `hyperspace/core/temporal_memory.py`

After each `KernelMemory.record()` call, pass each kernel's `vt_row`, `label`, `importance`, `narrative`, and `run_id` to `KernelLibrary.match_or_mint()` and `record_run()`. Then call `distill_eligible()`. Store the updated library in `st.session_state["kernel_library"]`.

### Modify `ukt/projection.py`

In `SharedProjection.__init__` or `rebuild()`, if `config.KERNEL_TRANSFER_ENABLED` is `True` and the library has templates, seed the fallback orthogonal basis with `kernel_library.get_transfer_bases()` instead of pure random orthogonal vectors. Weighting: blend toward template basis proportional to each template's `stability_score`. The SVD still runs fresh on actual data — templates only influence the `(1-α) × fallback` coupling term when coupling strength `α` is low.

### Modify `hyperspace/models/knowledge_matrix.py`

No functional change needed — the transfer happens at the `SharedProjection` level. Only add: store `kernel_library.get_cache_stats()` (list of lineage IDs + run counts) in the snapshot under key `"kernel_lineage_ids"` for traceability.

### Modify `hyperspace/pages/mission_control_tab.py`

Add a **"Knowledge Base"** `st.expander` (collapsed by default) in the Technical Diagnostics section:

- Table: `lineage_id (short)` | `label` | `run_count` | `stability_score` | `dominant_region` | `status (active/template/archived)`
- If no templates yet: show message "No templates distilled yet — requires {KERNEL_DISTILLATION_MIN_RUNS} consecutive stable runs."
- Show count of active lineages, distilled templates, archived templates.

### Modify `hyperspace/pages/pipeline_tab.py`

On the kernel evolution chart, annotate each kernel series with its `lineage_id` (short 6-char prefix) in the hover template. Color-code by lineage — kernels sharing a lineage ID across runs get the same color.

### Modify `hyperspace/config.py`

```python
# SPEC-7: Knowledge persistence & transfer
KERNEL_TRANSFER_ENABLED: bool = False
KERNEL_LINEAGE_COSINE_THRESHOLD: float = 0.85    # min cosine to assign existing lineage
KERNEL_DISTILLATION_MIN_RUNS: int = 5            # consecutive runs before distillation
KERNEL_LIBRARY_PATH: str = "kernel_library.json"
KERNEL_STALE_THRESHOLD_RUNS: int = 20            # runs without sighting → archive
```

### Files affected

| File | Change |
|------|--------|
| `hyperspace/core/kernel_library.py` | **NEW** |
| `hyperspace/core/temporal_memory.py` | Call library after each record() |
| `ukt/projection.py` | Accept template bases for warm-start fallback |
| `hyperspace/models/knowledge_matrix.py` | Store lineage IDs in snapshot |
| `hyperspace/pages/mission_control_tab.py` | Add Knowledge Base expander |
| `hyperspace/pages/pipeline_tab.py` | Lineage annotations on kernel evolution chart |
| `hyperspace/config.py` | Add 5 constants |

---

## Implementation Order

Implement in this order to manage dependencies:

1. **`hyperspace/config.py`** — add all new constants (SPEC-4, SPEC-6, SPEC-7) first
2. **SPEC-7** — `kernel_library.py` + `temporal_memory.py` extension + `ukt/projection.py` warm-start + Mission Control + Pipeline tab
3. **SPEC-6** — `mechanistic_probes.py` + `faithfulness.py` refactor + `interpreter_tab.py` UI
4. **SPEC-4** — `contrastive_encoder.py` + `knowledge_matrix.py` dual-path + `pipeline.py` integration

SPEC-4 is last because it optionally reads from SPEC-7's library for encoder weight persistence.

---

## Verification Checklist

After implementation, verify each of the following manually by running `streamlit run app.py`:

- [ ] **Feature flags off (default)**: App starts and pipeline runs without errors; all 8 tabs render; no new computations run; no regressions in existing governance flags or charts.
- [ ] **SPEC-7 versioning**: After 2+ pipeline runs, `st.session_state["kernel_library"]` contains lineage entries. Check Mission Control "Knowledge Base" expander shows lineage table.
- [ ] **SPEC-7 distillation**: After 5+ runs with stable kernels (run pipeline multiple times), at least one `KernelTemplate` appears in the Knowledge Base expander.
- [ ] **SPEC-7 transfer** (with `KERNEL_TRANSFER_ENABLED=True`): Pipeline runs without error; warm-start does not override SVD results when coupling strength is high.
- [ ] **SPEC-6 activation patching**: Select kernel K0 in Mechanistic Probes expander → Run Probe → bar chart renders with original vs patched values; no pipeline state modified.
- [ ] **SPEC-6 linear probing**: With < 10 runs, shows "insufficient history" message. With 10+ runs (mock or real), shows per-flag kernel weight table.
- [ ] **SPEC-6 pathway tracing**: Select feature 0 → Run Probe → expandable trace shows all 7 steps with values.
- [ ] **SPEC-4 disabled (default)**: `ENABLE_CONTRASTIVE_ALIGNMENT=False` → no overhead, no new session state keys.
- [ ] **SPEC-4 enabled** (with `ENABLE_CONTRASTIVE_ALIGNMENT=True`, `CONTRASTIVE_WEIGHT=0.3`): After pipeline run, `st.session_state["contrastive_alignment_score"]` is a float between 0 and 1; existing governance flags unchanged.
- [ ] **Faithfulness checks pass** (SPEC-6 refactor must not break existing checks): `faithfulness.py` still returns 6 results; overall confidence unchanged from pre-refactor for same snapshot.
- [ ] **JSON export**: Governance report export still works; probe results included when "Include in governance export" is checked.
- [ ] **WCAG compliance**: Any new text in Mission Control / Interpreter tab uses `config.CAPTION_COLOR` (`#8ab4cc`) for secondary text; no inline hex colors below 4.5:1 contrast against `#070d1a`.
