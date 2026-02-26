# Pipeline Iteration Proposal: News Source, Auto-Semantics, and GDP(PPP)-Debt Agent Resources

## 1) Open dataset/API options for geopolitical news (no fallback datasets)

### Candidate A — **GDELT DOC 2.0 API** (recommended as default)
- URL: `https://api.gdeltproject.org/api/v2/doc/doc`
- Access model: open, no API key required.
- Strengths:
  - High article volume and broad geographic coverage.
  - Queryable by keyword/country names, date windows, and sort order.
  - Good fit for country-focused BERTopic ingestion.
- Risks:
  - Upstream availability can fluctuate.
  - Metadata-centric fields may require text cleaning.

### Candidate B — **Mediastack** (free tier, API key)
- Provides structured country/language filters and source metadata.
- Better operational stability and predictable schema than scraped RSS.
- Trade-off: key management and tier rate limits.

### Candidate C — **NewsData.io** (free tier, API key)
- Provides entity/category filters and multilingual support.
- Useful backup to GDELT for quality resilience.
- Trade-off: key and quota constraints.

### Current implementation decision
- The code now uses **GDELT only** and intentionally raises an error if live news is unavailable.
- Static snippets and offline text fallbacks are removed from the live pipeline path.

---

## 2) Self-interpretable features with automatic semantics extraction (not hardcoded narrative text)

### Goal
Feature names and kernel narratives should emerge from data provenance + statistical attribution, not handcrafted story templates.

### Proposed automatic approach
1. **Attach provenance metadata at feature-write time**
   - Every block emits both:
     - numeric `features_for_ukt`
     - `feature_meta: dict[index, metadata]`
   - Metadata includes `block`, `entity`, `metric`, `source`, `time_window`, and optional `units`.

2. **Construct feature labels from metadata programmatically**
   - Label pattern: `{entity}_{metric}_{window}` (drop missing parts).
   - Example:
     - Graph block writes index 44 with metadata `{entity: Britain, metric: degree_centrality}`
     - UKT exposes label `Britain_degree_centrality`.

3. **Extract semantic phrases from top-loaded features**
   - For each kernel, pick top-k loadings by absolute value.
   - Group by `(entity, metric family)` and compute signed aggregate contribution.
   - Generate concise statements from these groups:
     - "Bridge-power concentration driven by Russia betweenness and Britain degree centrality".

4. **Cross-check with evidence, not static text**
   - Narrative lines cite values directly from loadings and region scores.
   - If evidence is weak (flat loadings), narrative reports uncertainty instead of forced interpretation.

### Common-sense examples
- If top loadings are `Britain_degree (+)` and `USA_pagerank (+)`, semantics should say the alliance core is central.
- If top loading is `India_betweenness (+)` with growing share, semantics should say India is acting as a bridge actor.
- If top loadings are near zero and diffuse, semantics should explicitly say "no dominant structural driver".

---

## 3) Realization plan (phased)

### Phase 1 (low risk)
- Add `feature_meta` plumbing through each pipeline block into UKT snapshots.
- Replace anonymous feature names with metadata-derived labels where present.
- Keep current SVD/regression logic unchanged.

### Phase 2 (medium risk)
- Add attribution grouping engine for kernel-level semantic extraction.
- Emit evidence-backed short/long narratives with confidence score.

### Phase 3 (higher value)
- Add consistency checks across steps:
  - same entity should maintain stable semantic identity across blocks.
  - flag drift where label identity changes unexpectedly.

---

## 4) Agent resources: GDP(PPP) as resource and Debt as anti-resource

### Proposed formulation
For country `i`:
- `resource_i = zscore(log(GDP_PPP_i))`
- `anti_resource_i = zscore(log(Debt_i))`
- `net_capacity_i = alpha * resource_i - beta * anti_resource_i`

Recommended defaults:
- `alpha = 1.0`
- `beta = 0.7` initially; tune with backtests.

Simulation use:
- Initialize `agent.resources` from affine transform of `net_capacity_i` to positive range (e.g., 20..200).
- Debt burden increases shock susceptibility and reduces outbound transfer efficiency.

### Common-sense examples
1. **High GDP, high debt (USA-like pattern)**
   - Strong initial capability, but high debt dampens long-run resilience.
2. **Lower GDP, low debt (India-like developing resilience pattern)**
   - Moderate capability with less debt drag, can improve relative standing over longer horizons.
3. **Commodity-heavy, debt-sensitive profile (Brazil/Russia-style scenario)**
   - Net capacity strongly reacts to financing conditions and alliance shocks.

### Data sources for realization
- GDP(PPP): World Bank or IMF WEO exports.
- Debt: IMF WEO gross debt (% GDP) + GDP(PPP) to convert to comparable scale, or direct sovereign debt datasets.

