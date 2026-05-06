# Meta Artifacts

This directory stores session-level development records for context management and handoffs.

---

## Purpose

The `meta_artifacts/` folder contains:
- session logs and summaries
- implementation/task artifacts
- verification notes
- handoff context for future contributors and agents

---

## Structure

```text
meta_artifacts/
|-- README.md                  # This file
|-- session_summary.md         # Consolidated summaries for all sessions
|-- session1/                  # Session 1 historical artifacts
|   |-- implementation_plan.md
|   |-- task.md
|   |-- session.md
|   `-- walkthrough.md
`-- session9/                  # Prior session artifacts
|   `-- session.md
`-- session10/                 # Latest session artifacts
    `-- session.md
```

---

## Session Index

| Session | Date | Status | Summary |
| :------ | :--- | :----- | :------ |
| [Session 1](session1/) | Feb 5, 2026 | Complete | Initial implementation and early continuations |
| Sessions 2–8 | Feb 5, 2026 | Complete | Fixes, verification, and documentation refresh (see `session_summary.md`) |
| [Session 9](session9/) | Feb 7, 2026 | Complete | Memory-attention head override support + documentation / session audit |
| [Session 10](session10/) | Feb 7, 2026 | Complete | Shared chapter routing + routed scaling + wandb step / router / memory metrics |
| Session 11 | May 6, 2026 | Complete | Documentation overhaul on the `paper` branch — root README rewrite (ICLR 2026 NFAM acceptance, architecture diagram, headline results), v4/v5 kernel docs, mojibake cleanup, run-2 / IFT configs in the index. See `session_summary.md` (no per-folder artefact). |

---

## Usage

### For New Sessions
1. Create a new folder `session{N}/`.
2. Add/update that session's `session.md` while working.
3. Append a concise session summary to `session_summary.md`.

### For Context Recovery
1. Read [`session_summary.md`](session_summary.md) for the fastest overview, including the most recent Session 11 documentation overhaul.
2. Read [`session10/session.md`](session10/session.md) for the latest implementation-level detailed work (shared chapter routing + wandb metrics).
3. Use [`session1/session.md`](session1/session.md) for deeper historical context on the initial implementation.

### For Agent Handoffs
1. Share `session_summary.md` first.
2. Share specific `sessionN/session.md` files as needed.
3. Use `docs/context.md` for current project state.

---

## Relationship to Other Docs

- `docs/context.md`: current project state
- `docs/philosophy.md`: development conventions
- `docs/meta_artifacts/session_summary.md`: high-level chronological history
- `docs/meta_artifacts/sessionN/session.md`: detailed per-session logs
