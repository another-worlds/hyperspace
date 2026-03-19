# Documentation Guidelines — Hyperspace

> **This is the single source of operational truth for how documentation is managed.**
> Read this file FIRST, every session, before touching any code.

---

## 1. Mandatory Reading Order

Every session must begin by reading these files in this exact order:

1. **`docs/DOCS_GUIDELINES.md`** (this file) — operational rules
2. **`docs/VISION.md`** — philosophical north star
3. **`docs/ARCHITECTURE.md`** — current system design

Only after loading all three are you permitted to read code or consider tactical work.

---

## 2. Allowed Documentation Files

The project may contain **only** these documentation files:

| File | Location | Purpose |
|------|----------|---------|
| `DOCS_GUIDELINES.md` | `/docs/` | Documentation management rules (this file) |
| `VISION.md` | `/docs/` | Immutable philosophical core |
| `ARCHITECTURE.md` | `/docs/` | Living strategic blueprint |
| `SESSION_PROGRESS.md` | `/docs/` | Temporary tactical scratchpad |
| `CLAUDE.md` | `/` (root) | Project instructions for AI assistant |

No other `.md`, `.txt`, or documentation-like files are permitted anywhere in the project unless they are code comments, licenses, or `requirements.txt`.

---

## 3. Three-Layer Hierarchy

### Layer 1: VISION.md — The Philosophical Core

**Changes:** Extremely rare. Only for fundamental shifts in project identity or purpose.

**Contains:**
- Grand "why" — the mission and reason for existence
- Ideal end-state description
- Core invariants that must never be violated
- Philosophical principles and design north star
- What the system IS and what it is NOT

**Must never contain:**
- File paths, variable names, or code snippets
- Implementation timelines or version numbers
- Bug reports, error logs, or tactical issues
- Specific library names or dependency versions

### Layer 2: ARCHITECTURE.md — The Living Blueprint

**Changes:** Slow and deliberate. Updated when system design evolves meaningfully.

**Contains:**
- High-level system design and component diagrams
- Key technical decisions and their rationale
- Known major issues and technical debt (abstract, not tactical)
- Medium-to-long-term roadmap
- Data source inventory and integration points
- Component boundaries and responsibilities

**Must never contain:**
- Session-specific notes or partial discoveries
- Debugging logs or stack traces
- Line-number references or transient file locations
- Unvalidated experiments or speculative ideas

### Layer 3: SESSION_PROGRESS.md — The Tactical Scratchpad

**Changes:** Continuous within a session. Regenerated fresh at every session start.

**Contains:**
- Today's goals and implementation notes
- Bugs being investigated, experiments in progress
- Partial discoveries and working hypotheses
- Design alternatives being evaluated
- End-of-session migration checklist (filled before closing)

**Lifecycle:**
- Regenerated from template at session start (previous content discarded)
- Actively updated throughout the session
- At session end, distill and migrate insights upward
- Does NOT persist between sessions

---

## 4. Semantic Interface Protocol — Upward Migration

### Rules

1. **Tactical details never enter VISION.md or ARCHITECTURE.md directly.**
   Insights must be validated and distilled before moving upward.

2. **Migration direction is always upward:**
   `SESSION_PROGRESS.md` → `ARCHITECTURE.md` → `VISION.md` (very rarely)

3. **When migrating content:**
   - Use clear, concise, abstract language
   - Reference components by their stable names (e.g., "the Pipeline", "the UKT framework", "the Semantic Canvas")
   - Strip implementation specifics (file names, variable names, code snippets) unless they became architecturally significant
   - Summarize patterns, not incidents

4. **Migration triggers:**
   - A new component or subsystem was created → ARCHITECTURE.md
   - A key technical decision was made → ARCHITECTURE.md
   - A significant issue or constraint was discovered → ARCHITECTURE.md
   - A fundamental insight about the project's purpose emerged → VISION.md (rare)

### End-of-Session Checklist

Before closing a session, answer these questions:

- [ ] What changed in the system that should be remembered forever?
- [ ] Were any new components, subsystems, or interfaces created?
- [ ] Were any key technical decisions made? What was the rationale?
- [ ] Did any architectural constraints or trade-offs become apparent?
- [ ] Is there new technical debt that should be tracked?
- [ ] Did anything challenge or refine the project's core vision?

For each "yes", write a clean, distilled entry in the appropriate document.

---

## 5. Templates

### VISION.md Template

```markdown
# Vision — Hyperspace

## Mission
[One-paragraph statement of purpose]

## Core Philosophy
[Numbered list of fundamental principles]

## The [System Name] Promise
[What the system guarantees to its users/stakeholders]

## Critical Invariants
[Numbered list of properties that must never be violated]

## Ideal End-State
[Description of what the fully realized system looks like]

## What Hyperspace Is NOT
[Explicit boundaries and anti-patterns]
```

### ARCHITECTURE.md Template

```markdown
# Architecture — Hyperspace

## System Overview
[High-level description + ASCII diagram]

## Component Map
[Table or diagram of major components and their responsibilities]

## Data Flow
[How data moves through the system]

## Key Technical Decisions
[Decision log with rationale]

## Known Issues & Technical Debt
[Abstract descriptions of current limitations]

## Roadmap
[Planned features and improvements, ordered by priority]
```

### SESSION_PROGRESS.md Template

```markdown
# Session Progress — [DATE]

## Session Goals
- [ ] Goal 1
- [ ] Goal 2

## Implementation Notes
[Running log of what was done and why]

## Issues Encountered
[Bugs, blockers, unexpected behavior]

## Design Experiments
[Alternatives explored, with outcomes]

## Discoveries
[Insights that may need upward migration]

## Migration Checklist
- [ ] What changed that should be remembered forever?
- [ ] New components or interfaces created?
- [ ] Key technical decisions made?
- [ ] New constraints or trade-offs discovered?
- [ ] New technical debt to track?
- [ ] Vision refinements needed?
```

---

## 6. Formatting Conventions

- **Headings:** Use `##` for top-level sections, `###` for subsections. Reserve `#` for the document title only.
- **Tables:** Use Markdown tables for structured comparisons. Align columns.
- **Diagrams:** Use ASCII art for architecture diagrams. Keep them under 30 lines.
- **Lists:** Use `-` for unordered, `1.` for ordered. Nest with 2-space indent.
- **Emphasis:** Use `**bold**` for key terms on first use. Use `*italic*` sparingly.
- **Code references:** Use backticks for component names (e.g., `SharedProjection`). Avoid file paths in VISION.md; use them sparingly in ARCHITECTURE.md.
- **Cross-references:** Reference other docs by name (e.g., "See VISION.md §Core Philosophy"), not by file path.
