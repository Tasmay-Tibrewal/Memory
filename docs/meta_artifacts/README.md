# Meta Artifacts

This directory contains development artifacts organized by session for context management and handoffs.

---

## Purpose

The `meta_artifacts/` folder serves as a repository for:
- Development session logs and summaries
- Implementation plans and task tracking
- Verification walkthroughs
- Context for session handoffs and agent transitions

---

## Structure

```
meta_artifacts/
├── README.md              # This file
├── session_summary.md     # Consolidated summaries of ALL sessions
└── session1/              # Session 1 artifacts
    ├── implementation_plan.md   # Approved implementation plan
    ├── task.md                  # Task tracking checklist
    ├── session.md               # Detailed session log
    └── walkthrough.md           # Verification results
```

---

## Files

### `session_summary.md`
**Purpose**: Consolidated, exhaustive summaries of all development sessions.

Use this file to:
- Get quick context on what was done in each session
- Understand key decisions made
- Find session-specific details without reading full logs

### Session Folders (`sessionN/`)
Each session folder contains detailed artifacts:

| File | Purpose |
|------|---------|
| `implementation_plan.md` | Approved plan from planning phase |
| `task.md` | Task tracking with checkboxes |
| `session.md` | Complete detailed log of all work |
| `walkthrough.md` | Verification results and proof of work |

---

## Session Index

| Session | Date | Status | Summary |
|---------|------|--------|---------|
| [Session 1](session1/) | Feb 5, 2026 | ✅ Complete | Full codebase implementation |

---

## Usage

### For New Sessions
1. Create a new folder: `session{N}/`
2. Copy the template from previous session
3. Log work in `session.md`
4. Update `session_summary.md` with summary

### For Context Recovery
1. Read `session_summary.md` for quick overview
2. If more detail needed, read specific `session{N}/session.md`
3. Check `implementation_plan.md` for original requirements

### For Agent Handoffs
1. Point agent to `session_summary.md`
2. Agent can dive into specific session folders as needed
3. `docs/context.md` provides current project state

---

## Relationship to Other Docs

```
docs/
├── context.md          → Current project state (actively maintained)
├── architecture.md     → Technical architecture
├── design.md           → Design decisions
├── philosophy.md       → Development philosophy and style
└── meta_artifacts/     → Historical session records
    ├── session_summary.md  → Session summaries (append-only)
    └── session{N}/         → Detailed session artifacts
```

- **context.md**: What the project IS now
- **philosophy.md**: How to maintain consistency
- **session_summary.md**: How the project GOT here
- **session.md**: Detailed "how" for each session
