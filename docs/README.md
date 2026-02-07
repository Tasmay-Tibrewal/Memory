# Documentation

This directory contains comprehensive documentation for the memory-augmented transformer project.

---

## Structure

```
docs/
â”œâ”€â”€ README.md              # This file - documentation overview
â”œâ”€â”€ architecture.md        # Detailed architecture explanation (~286 lines)
â”œâ”€â”€ design.md              # Design decisions and rationale (~120 lines)
â”œâ”€â”€ context.md             # Summary for handoffs (~321 lines)
â”œâ”€â”€ philosophy.md          # Development philosophy and style guide (~316 lines)
â”œâ”€â”€ prompt.md              # Agent onboarding prompt (~202 lines)
â””â”€â”€ meta_artifacts/        # Session artifacts for context management
    â”œâ”€â”€ README.md          # Meta artifacts overview
    â”œâ”€â”€ session_summary.md # Consolidated session summaries
    â””â”€â”€ session1/          # Session 1 historical artifacts
        â”œâ”€â”€ implementation_plan.md
        â”œâ”€â”€ task.md
        â”œâ”€â”€ session.md
        â””â”€â”€ walkthrough.md
    `-- session9/
        `-- session.md
    `-- session10/
        `-- session.md
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
â”œâ”€â”€ README.md              # Meta artifacts overview and usage
â”œâ”€â”€ session_summary.md     # Consolidated summaries of ALL sessions
â””â”€â”€ session1/              # Session 1 historical artifacts
    â”œâ”€â”€ implementation_plan.md   # Approved implementation plan
    â”œâ”€â”€ task.md                  # Task tracking checklist
    â”œâ”€â”€ session.md               # Detailed session log (historical)
    â””â”€â”€ walkthrough.md           # Verification results
`-- session10/             # Latest session artifacts
    `-- session.md
```

**Usage**:
- Read `session_summary.md` for quick context on all sessions
- Read `session10/session.md` for latest detailed history (use session1 for historical depth)
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
| Detailed session history | `meta_artifacts/session10/session.md` |
| Using the code | Package READMEs |

---

## Package READMEs

Each folder has its own README with detailed documentation:

| README | Lines | Contents |
|--------|-------|----------|
| [`memory_transformer/README.md`](../memory_transformer/README.md) | ~259 | All 11 core modules documented |
| [`training/README.md`](../training/README.md) | ~228 | Trainer, data loading, losses |
| [`inference/README.md`](../inference/README.md) | ~257 | Generation, merge, routing strategies |
| [`scripts/README.md`](../scripts/README.md) | ~225 | CLI scripts with all arguments |
| [`configs/README.md`](../configs/README.md) | ~299 | Complete config reference |

---

## Documentation Hierarchy

```
Project Documentation
â”‚
â”œâ”€â”€ Main README.md (root)
â”‚   â””â”€â”€ Quick start, installation, features, troubleshooting
â”‚
â”œâ”€â”€ Package READMEs (usage documentation)
â”‚   â”œâ”€â”€ memory_transformer/README.md  â†’ Core module documentation
â”‚   â”œâ”€â”€ training/README.md            â†’ Training infrastructure
â”‚   â”œâ”€â”€ inference/README.md           â†’ Generation utilities
â”‚   â”œâ”€â”€ scripts/README.md             â†’ CLI usage
â”‚   â””â”€â”€ configs/README.md             â†’ Configuration reference
â”‚
â”œâ”€â”€ Deep Dive Docs (this directory)
â”‚   â”œâ”€â”€ architecture.md  â†’ Technical architecture
â”‚   â”œâ”€â”€ design.md        â†’ Design rationale
â”‚   â”œâ”€â”€ philosophy.md    â†’ Development philosophy
â”‚   â””â”€â”€ context.md       â†’ Project summary
â”‚
â””â”€â”€ Historical Records (meta_artifacts/)
    â”œâ”€â”€ session_summary.md  â†’ Session overviews
    â””â”€â”€ sessionN/           â†’ Detailed session artifacts
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
| Deep Dive Docs (`docs/*.md`) | 6 | ~1,420 |
| Meta Artifacts (`docs/meta_artifacts/**/*.md`) | 7 | ~2,700+ |
| Package READMEs | 5 | ~1,250+ |
| Root README | 1 | ~342 |
| Configs (`configs/*.yaml`) | 5 | ~560 |
| **Total** | **23** | **~6,000+** |



