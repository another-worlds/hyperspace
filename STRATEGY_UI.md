# Hyperspace v3.0 — Strategic UI & Feature Plan

> **Purpose**: Central planning document for UI strategy, UX improvements, and feature tracking.
> Organized by three lenses: **Visionary** (mission alignment), **Strategic** (navigation & control),
> **Tactical** (look, feel, timing).

---

## Table of Contents

1. [User Journey Walkthrough](#1-user-journey-walkthrough)
2. [Visionary Audit — Mission Alignment](#2-visionary-audit)
3. [Strategic Audit — Navigation & Control](#3-strategic-audit)
4. [Tactical Audit — Look, Feel, Timing](#4-tactical-audit)
5. [Per-Tab Deep Dive](#5-per-tab-deep-dive)
6. [Feature & Issue Tracker](#6-feature--issue-tracker)
7. [Caching & Performance Strategy](#7-caching--performance-strategy)

---

## 1. User Journey Walkthrough

Walking through the app as a first-time user, step by step:

### Landing Page
1. **Open the app** → wide layout, dark theme loads. Sidebar shows "Hyperspace" brand, "AI Accountability Infrastructure" subtitle, UKT/Governance status badges.
2. **See the landing dashboard** → Governance header card with mission statement, 3 metric cards (Feature Space: 80 dims, Pipeline Steps: 6, Geopolitical Nodes: 6), and 3 "accountability guarantee" cards.
3. **First problem**: The 3 metric cards are abstract numbers with no context. A new user sees "80 dims" and has no idea what that means or why it matters. The guarantee cards explain the *promise* but not the *mechanism*.
4. **Click "Launch Full Hyperspace Cycle"** → A `st.status` container appears with progress messages. The user sees flat text lines like "Fetching real data sources..." then "Running Finance block..." with no timing, no percentage, no indication of which APIs are being hit or how long remains.
5. **Wait 15-30 seconds** → Status updates are coarse-grained (one line per block). No progress bar. No per-source status. If an API is slow, the user just waits with no feedback.
6. **Pipeline completes** → The page reruns. Now 8 tabs appear. The user lands on **Tab 0: Mission Control**.

### Tab 0: Mission Control (dashboard.py renders results)
7. **See results** → Run ID watermark (tiny, barely visible at `#1e3348` on dark bg), data source badges (small green/red pills), 5 summary metrics in a row.
8. **Problem**: 5 metrics in a row is dense. "Active Kernels", "TFT Params", "Graph Density", "Active Concepts", "Recon Error" — these are all technical. A governance officer doesn't know what "Recon Error: 0.000342" means.
9. **Scroll down** → Governance Flags section. If flags exist, they appear as expanders with definitions. If none, a green "All checks passed" banner. This is well-designed.
10. **Continue scrolling** → Alignment Comparison (3 metrics), Interpretability Score Card (styled table), Semantic Narratives (green/blue boxes), Faithfulness Report, Temporal Drift Monitor, Kernel Evolution charts, Interpretability Contract, and finally Export section.
11. **Problem**: This is a *wall of content*. The user must scroll through 15+ sections to find what they care about. No table of contents, no anchors, no collapse mechanism for the entire Mission Control tab. The most actionable items (governance flags, scorecard) are buried among technical diagnostics (drift monitor, kernel evolution).
12. **Problem**: The two kernel evolution charts (reconstruction trend, importance stability) use auto-assigned Plotly colors with no connection to the domain color scheme (blue/orange/green/red/purple used elsewhere).

### Tab 1: Finance-Neural Block
13. **See title and 3 sliders** → Encoder Length (24-90), Prediction Length (6-30), Hidden Size (16-64). These are ML hyperparameters. A policy user has no idea what "Encoder Length: 48" means.
14. **See candlestick charts** → One per ticker. Well-rendered (cyan up, red down). But no volume bars, no moving averages, no annotations for regime changes.
15. **Click "Compute TFT Forecast"** → Spinner "Fitting TFT (3 epochs, CPU)..." appears. Takes 2-5s. Reasonable.
16. **See forecast chart** → 80% CI band + median line. Clean. But no actual-vs-predicted comparison. No per-ticker breakdown (aggregated across tickers). The caption explains v3.0 connection but uses jargon ("UKT indices 0–15").
17. **See correlation heatmap** → `px.imshow` with RdBu_r. Clean but small (350px). No hover detail beyond value. No ticker-pair analysis.
18. **See confidence summary** → Color-coded HIGH/MODERATE/LOW. This is effective governance communication.

### Tab 2: Informational Cluster Mapping
19. **Click "Fit Topic Model"** → Spinner for 3-10s. First-time load downloads the multilingual sentence transformer (~90 MB). No indication of model download vs. fitting time.
20. **See topic info table** → `st.dataframe` with topic ID, count, name. Standard but functional.
21. **See topic distribution bar chart** → Top 10 topics. But topic labels are just integer IDs (Topic 0, Topic 1). No human-readable labels from topic keywords.
22. **Problem**: The "Sample Documents by Topic" expander shows raw text snippets. No highlighting of why a document was assigned to that topic. No topic keyword display.
23. **Problem**: Coverage summary uses outlier ratio thresholds (>30% = LOW, >10% = MODERATE, ≤10% = HIGH) that are hardcoded and not explained to the user.

### Tab 3: Politics-Military Graph Engine
24. **Click "Build & Analyze Graph"** → Fetches political data + builds graph. Takes 2-10s depending on API availability.
25. **See geographic map** → Plotly scatter_geo with 6 nations. Nice but static markers (no animation, no edge overlay on the map itself).
26. **See relation graph** → Force-directed layout. The graph is small (6 nodes) but the visualization doesn't show edge weights or relationship types (alliance vs. competition).
27. **See centrality bar chart** → Grouped bars: 4 metrics × 6 nodes = 24 bars. This is hard to parse. The bars are narrow, labels overlap, and there's no way to filter by node or metric.
28. **See community structure** → UNIPOLAR/BIPOLAR/MULTIPOLAR text box. Effective but could link back to graph visualization.

### Tab 4: Agentic Simulation
29. **4 sliders** → Steps, Resource Flow, Alliance Fluidity, Shock Probability. Better-labeled than Finance sliders, but still no tooltip explaining what "Alliance Fluidity: 0.5" means in practice.
30. **Resource trajectories** → Line chart with 6 agents, colored by GEOPOLITICAL_NODES palette. Good use of color consistency.
31. **Final resource bar chart** → Clear. Colors match agents.
32. **Alliance matrix heatmap** → RdBu_r diverging scale. Clean.
33. **Power concentration** → Gini-based assessment (HIGH/MODERATE/LOW). Effective governance communication.
34. **Problem**: No ability to re-run simulation with different parameters *without* losing current results for comparison. No A/B comparison.

### Tab 5: Semantic Interpreter
35. **This is the most complex tab** (~510 lines). Contains: radar chart, narratives with evidence, concept activation heatmap, feature importance chart, SVD kernel expanders, stakeholder annotations, and a massive Advanced Diagnostics expander.
36. **Radar chart** → Scatterpolar with 12 dimensions. Visually compelling. Multiple traces (accumulated + per-block). But the 12 dimension labels wrap and overlap at the default Plotly polar layout size.
37. **Dominant themes** → Top 3 themes listed. Good summary.
38. **Narratives with evidence** → Green "System Summary" + blue "Reality Assessment" boxes, each with collapsible evidence trails. Excellent governance design.
39. **Concept activation heatmap** → Viridis color scale, concept labels on y-axis. But labels are truncated to 18 chars (unreadable for long concept names).
40. **Feature importance (Reality Regression)** → 80-bar chart colored by region. **Critical issue**: x-axis shows feature *indices* (0-79), not names. The FEATURE_NAMES list exists in config.py but is not used in the chart. Users see "Feature 37" instead of "China: Eigenvector Centrality".
41. **SVD Kernel expanders** → Each kernel has narrative + semantic interpretation + contest/annotate widgets. Expanded if importance > 0.2 (hardcoded). Good design but the 0.2 threshold should be configurable.
42. **Advanced Diagnostics expander** → Contains 8+ sub-sections (kernel matrix, importance, canvas trajectory, SAE metrics/loss, concept-kernel map, UVT coupling, per-head attention, USE encoding, step reports). **This is ~200 lines of UI in a single expander with no internal navigation**. ML engineers must scroll blindly.

### Tab 6: Hyperspace Pipeline
43. **Pipeline step status table** → Styled dataframe with Block, Data Source, Governance, Flags, Key Finding columns. Good accountability view.
44. **Problem**: The Governance column uses color styling (green for PASS, orange for FLAG) but the dataframe styling can fail silently (try/except at line 728).
45. **Kernel evolution chart** → Shows importance across pipeline steps. Useful for ML engineers.
46. **Export section** → 3 download buttons (Report, Metrics, Scorecard). Functional but not prominent. Buried at the bottom.

### Tab 7: Counterfactual
47. **Block removal selectbox** → Clean. User picks which data block to remove.
48. **Optional shock injection** → Collapsed expander with checkbox + slider. Clever progressive disclosure.
49. **Run counterfactual** → Recomputes SVD, SAE, stability. Takes 5-15s. **Problem**: Recomputes the original SVD even though it was already computed in the pipeline. Should cache the baseline.
50. **4 metric cards** → Original vs. counterfactual kernel counts, RR cosine, difference norm. Clear.
51. **RR Diff visualization** → 3-panel subplot (original, counterfactual, difference). **Problem**: 80 features × 3 panels = very wide. Difference panel uses teal/red coloring that's hard to distinguish at small bar sizes. Feature names not shown (same issue as Tab 5).
52. **Domain-level impact** → Color-coded per-region analysis. Effective governance output.

---

## 2. Visionary Audit

The grand vision: **"AI Accountability Infrastructure"** — a system that makes AI-driven geopolitical analysis transparent, contestable, and interpretable for non-technical stakeholders (policy officers, governance delegates, oversight bodies).

### What the UI Gets Right (Vision-Aligned)

| Element | Location | Why It Works |
|---------|----------|-------------|
| Policy Language Mode toggle | Sidebar | Directly addresses the dual-audience problem: ML engineers vs. policy officers |
| Governance Flags | Mission Control | Machine-readable accountability codes (GOV-001 to GOV-005) with human definitions |
| Interpretability Score Card | Mission Control | Quantified accountability (9 metrics with thresholds) |
| Contest/Annotate widgets | Interpreter Tab | Operationalizes contestability — stakeholders can challenge kernel interpretations |
| Counterfactual Engine | Tab 7 | "What if we remove this data source?" is a *governance question*, not just an ML exercise |
| Evidence trails | Interpreter Tab | Narratives backed by traceable feature provenance (index, region, loading, block) |
| Run ID watermark | Sidebar + Mission Control | Non-repudiation: every analysis run is uniquely identified |
| Source badges | Throughout | Data jurisdiction transparency (US-REGULATED, OPEN-PUBLIC, ACADEMIC-LICENSED) |
| Glossary | Sidebar expander | Non-technical definitions for all key terms |

### Where the UI Fails the Vision

| Issue | Impact on Vision | Severity |
|-------|-----------------|----------|
| **V-01**: Mission Control is a wall of 15+ sections with no hierarchy | A governance officer must scroll through kernel evolution charts and drift monitors to find the scorecard. The *accountability* content is diluted by *diagnostic* content. | HIGH |
| **V-02**: Feature indices instead of names in Reality Regression | The most important visualization (80-feature importance chart) shows "Feature 37" instead of "China: Eigenvector Centrality". This fundamentally breaks interpretability for non-technical users. | CRITICAL |
| **V-03**: Finance tab exposes ML hyperparameters | Sliders for "Encoder Length", "Hidden Size" are meaningless to policy users. Policy Language Mode doesn't hide or relabel these. | MEDIUM |
| **V-04**: No executive summary | There's no single-paragraph, plain-English summary of "what did the system find?" at the top of Mission Control. The closest is the Semantic Narratives section, but it's buried below 5 metrics + governance flags + alignment comparison + scorecard. | HIGH |
| **V-05**: Pipeline progress gives no governance context | During the 15-30s pipeline run, users see "Running Finance block..." — they don't know *why* each block matters or what it contributes to the analysis. | MEDIUM |
| **V-06**: Counterfactual tab doesn't explain *why* to use it | The tab opens with technical description ("removes one data block from the UKT"). A governance user needs to understand this as: "Test whether your conclusions change if we exclude one data source." | MEDIUM |
| **V-07**: Advanced Diagnostics is a black box | 200+ lines of charts/tables in a single collapsed expander. No table of contents, no navigation, no explanation of *who* each section is for. ML engineers can't find what they need; governance officers don't know what to ignore. | MEDIUM |
| **V-08**: No comparison between runs | The Kernel Memory and Drift Monitor track cross-run changes, but there's no UI for comparing Run A vs. Run B side-by-side. Governance officers can't see how analysis evolved over time. | LOW |
| **V-09**: Stakeholder annotations are append-only with no review workflow | Annotations are stored but there's no approval/rejection/discussion mechanism. In a real governance context, annotations need to be triaged. | LOW |

---

## 3. Strategic Audit

How the UI structure helps (or hinders) users in navigating features and achieving their goals.

### Navigation Architecture

```
Landing Page
    └─ "Launch Full Hyperspace Cycle" (single button)
         └─ 8 Tabs (flat, no hierarchy)
              ├─ Tab 0: Mission Control (15+ sections, scrollable)
              ├─ Tab 1: Finance (sliders + button + charts)
              ├─ Tab 2: Clusters (button + charts)
              ├─ Tab 3: Politics (button + charts)
              ├─ Tab 4: Agents (sliders + button + charts)
              ├─ Tab 5: Interpreter (button + 8+ sections, scrollable)
              ├─ Tab 6: Pipeline (table + charts + export)
              └─ Tab 7: Counterfactual (selectbox + button + charts)
```

### Strategic Issues

| Issue | Description | Impact |
|-------|-------------|--------|
| **S-01**: Flat tab structure with no grouping | 8 tabs at the same level. A user interested in "Finance" has no visual cue that Tabs 1-4 are *data blocks* while Tabs 5-7 are *meta-analysis*. There's no semantic grouping. | Users can't form a mental model of the system. |
| **S-02**: No cross-tab navigation | Interpreter tab shows feature importance by domain region, but there's no "Jump to Finance tab" link. Pipeline tab mentions "Finance block: PASS" but doesn't link to Tab 1. | Users must manually switch tabs to cross-reference. |
| **S-03**: Export buttons only in Pipeline tab | Download buttons for reports/metrics/scorecard are in Tab 6. The Counterfactual tab has its own export, but Mission Control (the governance hub) has no export. | Governance officers must navigate to Tab 6 to export. |
| **S-04**: No saved states or comparison | Each tab computation overwrites previous results. If a user runs TFT with Hidden Size 32, then runs it with 64, the first result is gone. No history, no comparison. | Users can't explore parameter sensitivity. |
| **S-05**: Sidebar is informational but not functional | Sidebar shows status badges, run ID, governance flags, provenance — all read-only. The only interactive elements are Policy Mode toggle and Ticker multiselect. | Sidebar could host quick actions (re-run block, toggle chart detail, export current view). |
| **S-06**: Tab-level buttons duplicate pipeline | Each tab has its own "Compute X" button. These run independently of the pipeline. If a user clicks "Compute TFT Forecast" after the pipeline, it *overwrites* the pipeline's finance result in session state with no warning. | Confusing state management. Users don't know if they're seeing pipeline results or manual results. |
| **S-07**: No user onboarding or guided tour | First-time users see a landing page with abstract metrics and must click a single button to start a 15-30s computation with no explanation of what will happen. | High bounce risk. Users don't know what they're getting into. |
| **S-08**: Counterfactual has no scenario memory | User runs counterfactual removing "Finance", then wants to compare with removing "Clusters". Must re-run; previous result is overwritten. | Can't build a portfolio of what-if scenarios. |

### User Personas & Goal Mapping

| Persona | Primary Goal | Current Path | Friction Points |
|---------|-------------|-------------|-----------------|
| **Policy Officer** | "What did the system find? Are there governance concerns?" | Landing → Launch → Mission Control → Scroll to Scorecard → Scroll to Flags | Must scroll past technical metrics; no executive summary; feature indices not names |
| **ML Engineer** | "How are the models performing? What's the kernel structure?" | Landing → Launch → Interpreter → Advanced Diagnostics | 200-line expander with no navigation; SAE loss curve buried; no model comparison |
| **Governance Auditor** | "Can I contest this analysis? What if we remove a data source?" | Landing → Launch → Counterfactual → Run → Compare | No scenario memory; must manually record results; export only for current scenario |
| **Data Analyst** | "What's the data quality? Which sources are live?" | Landing → Launch → Mission Control → Source badges | Source badges are tiny pills; no detailed data quality report; no per-source timing |

---

## 4. Tactical Audit

How the app looks, feels, and performs moment to moment.

### Visual Design

| Element | Assessment | Issue |
|---------|-----------|-------|
| **Dark theme** | Cohesive and professional. Gradient backgrounds, cyan accents, consistent across all elements. | Works well. |
| **Typography** | Inter (body) + JetBrains Mono (code/IDs). Clean hierarchy. | Caption text (`#2d4a66`) is too dark — nearly invisible on `#070d1a` background. Contrast ratio ~1.8:1, fails WCAG AA (requires 4.5:1). |
| **Metric cards** | Top-border cyan accent, gradient background, hover shadow. | The `#3d5673` label color is low contrast (~2.5:1). Metric names are hard to read. |
| **Tab bar** | Cyan bottom border on active, transition on hover. | Inactive tabs at `#3d5673` are hard to read (~2.5:1 contrast). |
| **Expanders** | 1px `#152030` border, `#567090` summary text. | Border is almost invisible. Expanders look like they're not interactive. Summary text at `#567090` is only ~3:1 contrast. |
| **Governance badges** | `.gov-flag` amber on dark brown, `.gov-pass` green on dark green. | Amber (`#fbbf24`) on `#1c0e00` is ~4.2:1 — barely passes WCAG AA. Could be brighter. |
| **Source badges** | Green cyan on dark green, red on dark red for fallback. | Small (`0.74em`) text makes these easy to miss. |
| **Buttons** | Gradient blue with cyan border, hover lift. | Only primary buttons are styled. Secondary/default buttons use Streamlit defaults (gray on dark = invisible). |

### Timing & Responsiveness

| Operation | Time | User Experience | Issue |
|-----------|------|----------------|-------|
| **Full pipeline** | 15-30s | Single `st.status` with 6 text lines | No progress bar, no per-block timing, no API status. User stares at text. |
| **TFT training** | 2-5s | `st.spinner` with one message | Acceptable, but no epoch progress. |
| **BERTopic fitting** | 3-10s | `st.spinner` with one message | First run downloads ~90 MB model with no download indicator. |
| **SAE training** (Interpreter) | 5-10s | `st.spinner` with one message | 100 epochs with no loss curve during training. Only shows curve *after*. |
| **Counterfactual SVD** | 5-15s | `st.spinner` with one message | Recomputes baseline SVD unnecessarily. |
| **UVT + USE** (optional) | 25-40s | Part of pipeline spinner | No separate indicator. User doesn't know this is running. |
| **Tab switching** | <0.5s | Instant (session state) | Good. All results pre-computed. |
| **Chart rendering** | <0.5s | Plotly renders quickly | Good. `use_container_width=True` ensures responsive sizing. |
| **Data API cascade** (news) | 5-20s | Hidden inside pipeline spinner | 11 sources tried sequentially. No per-source status or fallback indication during fetch. |

### Balance of Detail

| Section | Detail Level | Audience Match | Issue |
|---------|-------------|---------------|-------|
| Mission Control metrics (5 cards) | HIGH technical | Engineers only | Policy users lost |
| Governance Flags | RIGHT level | Both audiences | Well-calibrated |
| Interpretability Scorecard | RIGHT level | Both audiences | Well-calibrated |
| Kernel Evolution charts | HIGH technical | Engineers only | Buried in Mission Control (policy tab) |
| Semantic Narratives | RIGHT level | Both audiences | But buried too deep in the page |
| Reality Regression chart | WRONG detail | Neither audience served | Feature indices instead of names |
| Advanced Diagnostics | MAXIMUM technical | Engineers only | Massive, unnavigable expander |
| Counterfactual metrics | RIGHT level | Both audiences | Well-calibrated |
| Domain-Level Impact | RIGHT level | Both audiences | Good governance communication |

---

## 5. Per-Tab Deep Dive

### Tab 0: Mission Control

**Current**: 15+ sections in a single scroll. Mixes governance output (flags, scorecard, narratives) with technical diagnostics (drift, kernel evolution, contract).

**Issues**:
- T0-01: No executive summary at top
- T0-02: Kernel evolution charts use generic Plotly colors (not domain palette)
- T0-03: Alignment Comparison shows "Legacy vs Shared-Latent" — meaningless to non-ML users
- T0-04: Faithfulness Report shows raw confidence floats with no visual indicator
- T0-05: Drift Monitor alerts are text-only; should be visual (sparkline or gauge)
- T0-06: Export section is only in Pipeline tab, not here
- T0-07: 5 summary metrics in a row — too dense, labels too dark

**Proposed structure** (grouped sub-sections):
```
Mission Control
├─ Executive Summary (narrative + top 3 findings)
├─ Governance Status (flags + scorecard + faithfulness)
├─ Data Provenance (sources + jurisdictions + freshness)
├─ System Diagnostics (drift + kernel evolution + alignment) [collapsed by default]
└─ Export (all formats)
```

### Tab 1: Finance-Neural Block

**Issues**:
- T1-01: Sliders expose raw ML hyperparameters (encoder_length, hidden_size)
- T1-02: No volume bars on candlestick charts
- T1-03: No moving average overlay option
- T1-04: Forecast chart aggregates across tickers; no per-ticker breakdown
- T1-05: Correlation heatmap is small (350px) with no hover detail
- T1-06: Confidence summary thresholds (0.10, 0.30) are hardcoded magic numbers
- T1-07: No actual-vs-predicted validation chart
- T1-08: Caption text explains "UKT indices 0-15" — meaningless to most users

**Proposed improvements**:
- Policy Mode: Replace "Encoder Length" with "Analysis Depth", "Hidden Size" with "Model Complexity"
- Add per-ticker forecast option
- Add tooltip on each slider explaining what it controls in plain language
- Move threshold constants to config.py

### Tab 2: Informational Cluster Mapping

**Issues**:
- T2-01: Topic labels are integer IDs, not keyword summaries
- T2-02: First-time model download (~90 MB) has no progress indicator
- T2-03: Sample documents expander shows raw text, no keyword highlighting
- T2-04: Outlier ratio thresholds (30%, 10%) hardcoded, not explained
- T2-05: No topic coherence scores visible
- T2-06: No inter-topic distance map (BERTopic can generate this)
- T2-07: No language distribution chart (multilingual model, but no indication of languages seen)

**Proposed improvements**:
- Show top-3 keywords as topic label (BERTopic provides these)
- Add `st.progress` for model download
- Display topic coherence scores alongside distribution

### Tab 3: Politics-Military Graph Engine

**Issues**:
- T3-01: Geographic map has static markers; no edge overlay on the world map
- T3-02: Relation graph doesn't show edge weights or relationship types (alliance vs. competition)
- T3-03: Centrality bar chart: 24 bars (4 metrics × 6 nodes) — information overload
- T3-04: No way to filter centrality view by node or metric
- T3-05: Community detection result is text-only; should color-code graph by community
- T3-06: No edge tooltips showing relationship descriptions (available in GEOPOLITICAL_EDGES)

**Proposed improvements**:
- Add edge lines on geographic map (color by relationship type: alliance=cyan, competition=red)
- Split centrality into 4 small multiples (one per metric) instead of grouped bars
- Color graph nodes by community

### Tab 4: Agentic Simulation

**Issues**:
- T4-01: Slider tooltips don't explain mechanics (e.g., "Alliance Fluidity: how quickly agents form/break alliances")
- T4-02: No A/B comparison (re-running overwrites previous results)
- T4-03: Alliance matrix shows final state only; no animation of alliance evolution
- T4-04: Gini thresholds (0.50, 0.25) hardcoded
- T4-05: Simulation log is not surfaced in the UI (stored in session state but never displayed)
- T4-06: No sensitivity analysis ("try 100 runs and show distribution")

**Proposed improvements**:
- Add help text to each slider
- Surface simulation log in an expander
- Store previous sim results for comparison

### Tab 5: Semantic Interpreter

**Issues**:
- T5-01: Radar chart labels overlap at 12 dimensions in default Plotly polar layout
- T5-02: Concept activation heatmap truncates labels to 18 chars
- T5-03: Reality Regression shows feature indices, not names (CRITICAL — same as V-02)
- T5-04: Kernel expander threshold (importance > 0.2) is hardcoded
- T5-05: Advanced Diagnostics is ~200 lines in one expander, no navigation
- T5-06: Per-head attention heatmaps are small (4 in a row in columns) and hard to read
- T5-07: SAE training runs 100 epochs with no caching; re-click re-trains
- T5-08: USE and UVT sections in diagnostics have no explanation of when/why they matter

**Proposed improvements**:
- Use FEATURE_NAMES in reality regression hover/labels
- Add sub-tabs or accordion inside Advanced Diagnostics
- Cache SAE results by matrix hash
- Add "Zoom to Region" buttons for reality regression (show only temporal/semantic/etc.)

### Tab 6: Hyperspace Pipeline

**Issues**:
- T6-01: Status table Governance column color styling can fail silently (try/except swallows error)
- T6-02: No per-block timing information in the status table
- T6-03: Export buttons are 3 in a row at the bottom — not prominent
- T6-04: No "Re-run Pipeline" button (must use Reset at page bottom)
- T6-05: Kernel evolution chart uses auto-assigned colors, not domain palette
- T6-06: No filtering or sorting on the status table
- T6-07: Markdown report export doesn't include chart images (text-only)

**Proposed improvements**:
- Add timing column to status table
- Make export section more prominent (sticky or sidebar)
- Add domain palette to kernel evolution chart

### Tab 7: Counterfactual

**Issues**:
- T7-01: Baseline SVD recomputed every run (should cache from pipeline)
- T7-02: No scenario memory (each run overwrites previous)
- T7-03: RR diff 3-panel chart is very wide with 80 features; hard to scan
- T7-04: Feature indices instead of names in diff chart (same as V-02)
- T7-05: Shock injection slider range (32-47) is hardcoded to structural region; not configurable
- T7-06: No "compare two scenarios" view
- T7-07: Cosine similarity thresholds (0.95, 0.80) hardcoded
- T7-08: No "most affected features" highlight in the diff chart

**Proposed improvements**:
- Cache baseline SVD from pipeline in session state
- Add scenario history (store last 3 counterfactual results)
- Add top-10 most-changed features summary
- Use FEATURE_NAMES in diff chart

---

## 6. Feature & Issue Tracker

### Priority Levels
- **P0 — Critical**: Breaks core vision or renders output meaningless
- **P1 — High**: Significant UX friction or missing essential feature
- **P2 — Medium**: Polish, consistency, or nice-to-have
- **P3 — Low**: Future enhancement

### Issues

| ID | Priority | Category | Tab | Title | Description | Status |
|----|----------|----------|-----|-------|-------------|--------|
| V-02 | P0 | Visionary | 5,7 | Feature names in Reality Regression | Charts show indices 0-79 instead of FEATURE_NAMES. Breaks interpretability. | OPEN |
| V-01 | P0 | Visionary | 0 | Mission Control content hierarchy | 15+ sections in flat scroll. No executive summary, no grouping. | OPEN |
| V-04 | P0 | Visionary | 0 | No executive summary | No plain-English "here's what we found" at the top. | OPEN |
| S-01 | P1 | Strategic | All | Tab grouping/labels | 8 flat tabs with no semantic grouping (data blocks vs. meta-analysis). | OPEN |
| S-06 | P1 | Strategic | 1-4 | Tab buttons overwrite pipeline results | Manual "Compute X" silently overwrites pipeline state. | OPEN |
| T5-07 | P1 | Performance | 5 | SAE not cached | 100-epoch training re-runs on every button click. | OPEN |
| T7-01 | P1 | Performance | 7 | Counterfactual recomputes baseline SVD | Should reuse pipeline's SVD result. | OPEN |
| V-03 | P1 | Visionary | 1 | Finance sliders expose ML hyperparameters | "Encoder Length" meaningless to policy users. | OPEN |
| T4-CSS | P1 | Tactical | All | Caption text invisible | `#2d4a66` on `#070d1a` = ~1.8:1 contrast. WCAG AA requires 4.5:1. | OPEN |
| T4-CSS-2 | P1 | Tactical | All | Metric card labels hard to read | `#3d5673` label color = ~2.5:1 contrast. | OPEN |
| T4-CSS-3 | P1 | Tactical | All | Inactive tab text hard to read | `#3d5673` = ~2.5:1 contrast. | OPEN |
| T4-CSS-4 | P1 | Tactical | All | Expanders nearly invisible | 1px `#152030` border, `#567090` summary text. | OPEN |
| S-02 | P2 | Strategic | All | No cross-tab navigation | Can't jump from Interpreter to Finance. | OPEN |
| S-03 | P2 | Strategic | 0 | No export on Mission Control | Export only in Pipeline tab. | OPEN |
| S-04 | P2 | Strategic | 1-4 | No parameter comparison | Re-running overwrites; no history. | OPEN |
| T0-02 | P2 | Tactical | 0 | Kernel evolution wrong colors | Uses auto Plotly palette, not domain colors. | OPEN |
| T1-02 | P2 | Tactical | 1 | No volume bars on candlestick | Standard finance chart expectation missing. | OPEN |
| T1-04 | P2 | Tactical | 1 | Forecast aggregated across tickers | No per-ticker breakdown available. | OPEN |
| T2-01 | P2 | Tactical | 2 | Topic labels are integers | Should show top-3 keywords as label. | OPEN |
| T3-01 | P2 | Tactical | 3 | Geographic map has no edge overlay | Static markers only. | OPEN |
| T3-03 | P2 | Tactical | 3 | Centrality bar chart overloaded | 24 bars; should be small multiples or filterable. | OPEN |
| T5-01 | P2 | Tactical | 5 | Radar chart label overlap | 12 dimensions crowd the polar layout. | OPEN |
| T5-05 | P2 | Tactical | 5 | Advanced Diagnostics unnavigable | ~200 lines, no table of contents. | OPEN |
| T6-02 | P2 | Tactical | 6 | No per-block timing in pipeline table | Users can't see which block is slow. | OPEN |
| T7-02 | P2 | Strategic | 7 | No scenario memory | Each counterfactual overwrites previous. | OPEN |
| T7-03 | P2 | Tactical | 7 | RR diff chart too wide | 80 features × 3 panels. | OPEN |
| V-05 | P2 | Visionary | N/A | Pipeline progress lacks context | "Running Finance block..." — no explanation of why. | OPEN |
| V-06 | P2 | Visionary | 7 | Counterfactual intro is technical | Needs plain-English framing for governance users. | OPEN |
| T0-05 | P2 | Tactical | 0 | Drift alerts text-only | Should be visual (sparkline/gauge). | OPEN |
| T4-05 | P3 | Tactical | 4 | Simulation log not surfaced | Log exists in session state but never rendered. | OPEN |
| T2-06 | P3 | Tactical | 2 | No inter-topic distance map | BERTopic can generate; currently unused. | OPEN |
| T3-06 | P3 | Tactical | 3 | No edge tooltips on graph | Relationship descriptions available but not shown. | OPEN |
| T4-02 | P3 | Strategic | 4 | No A/B simulation comparison | Can't compare two parameter sets. | OPEN |
| S-07 | P3 | Strategic | N/A | No onboarding or guided tour | First-time users get no explanation. | OPEN |
| V-08 | P3 | Visionary | 0 | No cross-run comparison | Drift tracked but no side-by-side UI. | OPEN |
| V-09 | P3 | Visionary | 5 | Annotations lack review workflow | Append-only, no triage mechanism. | OPEN |

### Hardcoded Thresholds to Centralize

| Current Location | Value | Proposed Config Name | Description |
|-----------------|-------|---------------------|-------------|
| finance_tab.py:143 | 0.10 | `FORECAST_CONFIDENCE_HIGH` | High confidence band threshold |
| finance_tab.py:148 | 0.30 | `FORECAST_CONFIDENCE_MODERATE` | Moderate confidence band threshold |
| clusters_tab.py:105 | 0.30 | `OUTLIER_RATIO_CRITICAL` | Topic outlier critical threshold |
| clusters_tab.py:111 | 0.10 | `OUTLIER_RATIO_MODERATE` | Topic outlier moderate threshold |
| agents_tab.py:140 | 0.50 | `GINI_CONCENTRATION_HIGH` | High power concentration |
| agents_tab.py:147 | 0.25 | `GINI_CONCENTRATION_MODERATE` | Moderate power concentration |
| interpreter_tab.py:282 | 0.20 | `KERNEL_EXPANDER_THRESHOLD` | Auto-expand kernel if importance above |
| counterfactual_tab.py:345 | 0.95 | `CF_STABILITY_HIGH` | Counterfactual stable conclusion |
| counterfactual_tab.py:351 | 0.80 | `CF_STABILITY_MODERATE` | Counterfactual moderate sensitivity |
| counterfactual_tab.py:257 | 32-47 | `STRUCTURAL_REGION_BOUNDS` | Shock injection feature range |

---

## 7. Caching & Performance Strategy

### Current State

| Component | Cached? | Method | TTL | Issue |
|-----------|---------|--------|-----|-------|
| OHLCV data | YES | `@st.cache_data` | 1h | Good |
| Macro features | YES | `@st.cache_data` | 24h | Good |
| News corpus | YES | `@st.cache_data` | 30m | Good |
| Political data | YES | `@st.cache_data` | 24h | Good; streams 400MB file |
| BERTopic model | YES | `@st.cache_resource` | Hash-keyed | Good |
| TFT model | YES | `@st.cache_data` | Hash-keyed | Good |
| SAE training | NO | — | — | **Re-trains 100 epochs per click** |
| Counterfactual SVD | NO | — | — | **Recomputes baseline every time** |
| UVT training | NO | — | — | 120 epochs, 10-20s per run |
| USE training | NO | — | — | 150 epochs, 15-25s per run |
| Intermediate DataFrames | NO | — | — | `.corr()`, `.pivot()` recomputed |
| Kernel evolution viz | NO | — | — | Rebuilds from snapshots each render |

### Proposed Caching Additions

| Component | Strategy | Expected Savings |
|-----------|----------|-----------------|
| **SAE results** | Cache in `session_state` keyed by input matrix hash (SHA-256 of flattened bytes). Only re-train if matrix changes. | 5-10s per re-click |
| **Counterfactual baseline** | Read `ukt_snapshots[-1]` SVD decomposition directly from pipeline results. Only recompute the *counterfactual* (reduced) SVD. | 3-8s per counterfactual run |
| **UVT/USE models** | Cache in `session_state` keyed by UKT snapshot hash. | 25-40s on re-run |
| **Correlation matrices** | `@st.cache_data` on pivot+corr operation, keyed by OHLCV hash. | <1s, but avoids recompute on tab switch |
| **Plotly figures** | Don't cache (fast to render). But *do* cache the data transformations that feed them. | — |
| **Data freshness indicator** | Add `last_fetched` timestamp per source to session state. Display in sidebar. | UX improvement, no speed change |

### Pipeline Timing Improvements

| Change | Implementation | Impact |
|--------|---------------|--------|
| Per-block progress bar | Add `st.progress(step/total)` inside pipeline loop | Users see 1/6, 2/6... instead of text |
| Per-source status | Add source-level callbacks during data cascade (news: "Trying GDELT... Trying BBC RSS...") | Users understand which APIs are responsive |
| Block timing | Store `time.time()` delta per block in session state; display in Pipeline tab status table | Transparency for auditors |
| Lazy diagnostics | Wrap Advanced Diagnostics content in `if expander_open:` check (use session state flag) | Faster Interpreter tab initial render |

---

*Last updated: 2025-03-18 — Hyperspace v3.0 Prototype*
