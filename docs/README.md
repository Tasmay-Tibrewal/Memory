# Documentation

This directory contains comprehensive documentation for the memory-augmented transformer project.

---

## Structure

```
docs/
├── README.md              # This file - documentation overview
├── architecture.md        # Detailed architecture explanation (~370 lines)
├── design.md              # Design decisions and rationale (~188 lines)
├── context.md             # Summary for handoffs (~390 lines)
├── philosophy.md          # Development philosophy and style guide (~400 lines)
├── prompt.md              # Agent onboarding prompt (~270 lines)
└── meta_artifacts/        # Session artifacts for context management
    ├── README.md          # Meta artifacts overview
    ├── session_summary.md # Consolidated session summaries
    └── session1/          # Session 1 detailed artifacts
        ├── implementation_plan.md
        ├── task.md
        ├── session.md
        └── walkthrough.md
```

---

## Document Descriptions

### `architecture.md` - Architecture Deep Dive

**Purpose**: Detailed technical explanation of all architectural components.

**Contents**:
- Overall system architecture diagram
- Memory bank implementations (Standard, Factorized, ReducedDim)
- Memory cross-attention mechanism with equations
- Transformer block variants (A and B)
- Chapter-based routing (MoE-style) with loss functions
- Memory adapter injection mechanism
- Low-rank compression options
- Implementation mapping (which file has what)
- Parameter count calculations
- Computational complexity analysis
- Configuration quick reference
- Related work comparison

**Audience**: Developers needing to understand inner workings or extend the architecture.

---

### `design.md` - Design Decisions

**Purpose**: Document all design choices, compromises, and areas for future improvement.

**Contents**:
- Training library selection rationale (PyTorch + Accelerate)
- Block variant default choice (Variant A)
- Routing strategy limitations (sequence-level only)
- Zero-initialization reasoning (W_o = 0)
- Hook-based adapter injection rationale
- Known limitations and workarounds
- Compromises made during implementation
- Future improvement areas (prioritized)
- Configuration recommendations
- Debugging tips

**Audience**: Developers debugging issues, making architectural changes, or understanding trade-offs.

---

### `philosophy.md` - Development Philosophy

**Purpose**: Document the coding, architecture, and documentation philosophy for consistency across sessions.

**Contents**:
- Core principles (flexibility, explicitness, modularity)
- Architecture philosophy (component organization, dependencies)
- Implementation philosophy (approach, error handling, performance)
- Coding style (Python style, naming, type hints, docstrings)
- Configuration philosophy (structure, defaults, naming)
- Documentation philosophy (principles, levels, writing style)
- Project structure philosophy (organization, naming)
- Session management philosophy (context preservation, handoffs)

**Audience**: Future developers, agents, or anyone continuing work on the project.

---

### `context.md` - Handoff Summary

**Purpose**: Complete project summary for session handoffs, context compaction, or onboarding.

**Contents**:
- Project overview and status
- Quick start for new sessions
- Complete file structure with line counts
- Every file's purpose and size
- All 5 key design decisions
- What's NOT implemented and why
- Complete configuration flags reference (50+ flags)
- File dependencies graph
- Running commands for all scenarios
- Next steps for continuation

**Audience**: Anyone picking up the project, including AI agents or new developers.

---

### `meta_artifacts/` - Session Artifacts

**Purpose**: Historical records of development sessions for context management.

**Structure**:
```
meta_artifacts/
├── README.md              # Meta artifacts overview and usage
├── session_summary.md     # Consolidated summaries of ALL sessions
└── session1/              # Session 1 artifacts
    ├── implementation_plan.md   # Approved implementation plan
    ├── task.md                  # Task tracking checklist
    ├── session.md               # Detailed session log (~580 lines)
    └── walkthrough.md           # Verification results
```

**Usage**:
- Read `session_summary.md` for quick context on all sessions
- Dive into `sessionN/session.md` for detailed history
- Check `implementation_plan.md` for original requirements

**Audience**: Developers continuing work, auditing decisions, or managing context across sessions.

---

## Quick Reference

| Need | Document |
|------|----------|
| Understand architecture | `architecture.md` |
| Understand why decisions were made | `design.md` |
| Quick context on project state | `context.md` |
| Session history overview | `meta_artifacts/session_summary.md` |
| Detailed session history | `meta_artifacts/session1/session.md` |
| Using the code | Package READMEs |

---

## Package READMEs

Each folder has its own README with detailed documentation:

| README | Lines | Contents |
|--------|-------|----------|
| [`memory_transformer/README.md`](../memory_transformer/README.md) | ~300 | All 11 core modules documented |
| [`training/README.md`](../training/README.md) | ~200 | Trainer, data loading, losses |
| [`inference/README.md`](../inference/README.md) | ~180 | Generation, routing strategies |
| [`scripts/README.md`](../scripts/README.md) | ~200 | CLI scripts with all arguments |
| [`configs/README.md`](../configs/README.md) | ~300 | Complete config reference |

---

## Documentation Hierarchy

```
Project Documentation
│
├── Main README.md (root)
│   └── Quick start, installation, features, troubleshooting
│
├── Package READMEs (usage documentation)
│   ├── memory_transformer/README.md  → Core module documentation
│   ├── training/README.md            → Training infrastructure
│   ├── inference/README.md           → Generation utilities
│   ├── scripts/README.md             → CLI usage
│   └── configs/README.md             → Configuration reference
│
├── Deep Dive Docs (this directory)
│   ├── architecture.md  → Technical architecture
│   ├── design.md        → Design rationale
│   ├── philosophy.md    → Development philosophy
│   └── context.md       → Project summary
│
└── Historical Records (meta_artifacts/)
    ├── session_summary.md  → Session overviews
    └── sessionN/           → Detailed session artifacts
```

---

## Relationship Between Docs

| Document | Purpose | Update Frequency |
|----------|---------|------------------|
| `context.md` | What the project IS now | Every major change |
| `session_summary.md` | How the project GOT here | End of each session |
| `session.md` | Detailed "how" for each session | During session |
| `architecture.md` | Technical deep dive | When architecture changes |
| `design.md` | Decision rationale | When decisions made |

---

## Contributing to Documentation

When adding new features, update documentation in this order:

1. **Update session's `session.md`**: Log what you're doing
2. **Update `context.md`**: Reflect current project state
3. **Update Package README**: Document new modules/functions
4. **Update `design.md`**: Document any design decisions
5. **Update `architecture.md`**: If architectural changes
6. **Update root README**: If user-facing features changed
7. **Update `session_summary.md`**: At end of session

---

## Total Documentation Stats

| Category | Files | Lines |
|----------|-------|-------|
| Deep Dive Docs | 5 | ~1,500 |
| Meta Artifacts | 3+ | ~800+ |
| Package READMEs | 5 | ~1,180 |
| Root README | 1 | ~335 |
| Config README | 1 | ~300 |
| **Total** | **15+** | **~4,115+** |
