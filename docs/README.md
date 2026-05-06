# Documentation

This directory contains the deep, technical, and historical documentation for the Memory-Augmented Transformer / Mixture of Chapters project.

For the user-facing overview, paper link, and quick start, see the [root `README.md`](../README.md).

---

## Structure

```text
docs/
├── README.md             # This file — documentation map
├── architecture.md       # Detailed technical architecture (~450 lines)
├── design.md             # Design decisions and rationale (~270 lines)
├── context.md            # Snapshot project state for handoffs (~450 lines)
├── philosophy.md         # Development philosophy and style guide (~400 lines)
├── prompt.md             # Onboarding prompt for new sessions / agents (~300 lines)
└── meta_artifacts/       # Session-level history
    ├── README.md
    ├── session_summary.md          # Cumulative session summary
    ├── session1/                   # Initial-implementation deep log + plan
    │   ├── implementation_plan.md
    │   ├── task.md
    │   ├── session.md
    │   └── walkthrough.md
    ├── session9/
    │   └── session.md
    └── session10/
        └── session.md
```

---

## Document Descriptions

### `architecture.md` — Architecture Deep Dive

Detailed technical explanation of all architectural components, including the workshop-paper reference configuration.

**Contents**

- Overall system diagram and reference to [`idea/MoC Arch Diagram Excalidraw.png`](../idea/MoC%20Arch%20Diagram%20Excalidraw.png)
- Memory bank implementations (Standard, Factorized, ReducedDim)
- Memory cross-attention with equations (full, low-rank, reduced-dim modes)
- Block variants A and B
- Chapter-based routing (MoE-style) with loss functions
- Memory adapter injection mechanism (persistent hooks, GC-safe)
- Token-level routing path (shared dense + routed sparse, MoE-weighted)
- Sparse kernel set (v1–v5 in `kernels-final/`)
- Implementation mapping (which file holds what)
- Parameter count + complexity analysis
- Configuration quick reference

**Audience:** developers extending the architecture, paper readers cross-checking the implementation.

---

### `design.md` — Design Decisions

All design choices with rationale, the compromises made, the known limitations, and the prioritised list of future improvements.

**Contents**

- Training library selection (PyTorch + Accelerate)
- Default block variant (A)
- Sequence-level vs token-level routing strategies and why both exist
- Zero-init `W_o` reasoning
- Persistent-hook adapter injection rationale (and the gradient-checkpointing trap it solves)
- Known limitations and workarounds
- Configuration recommendations and debugging tips

---

### `context.md` — Handoff Snapshot

The most up-to-date single-file project summary. Designed to bootstrap a new session, agent, or contributor in one read.

**Contents**

- Project status, dates, sessions completed
- File-count and line-count snapshot
- Implementation overview by package
- Key design decisions and what is **not** implemented
- The full configuration surface (every flag)
- Running commands for training, evaluation, and inference

---

### `philosophy.md` — Development Philosophy

Why the code, configs, and docs look the way they do. Useful for ensuring future contributions stay coherent with the existing structure.

**Contents**

- Core principles (flexibility, explicitness, modularity)
- Architecture and dependency direction
- Implementation conventions (errors, performance, research-vs-production)
- Coding style (Python, naming, type hints, docstrings)
- Configuration philosophy (structure, defaults, validation)
- Documentation levels and update triggers
- Session management

---

### `prompt.md` — Agent Onboarding Prompt

The starting prompt used when handing this codebase to a new agent or contributor. Tells them what to read first, what to ask before implementing, and how the runnable commands are organised.

---

### `meta_artifacts/` — Per-Session History

```text
meta_artifacts/
├── README.md             # Index and usage notes
├── session_summary.md    # Consolidated summaries of every session
├── session1/             # Initial implementation deep log + plan + walkthrough
├── session9/             # Audit & memory-attn head-override support
└── session10/            # Shared-chapter routing + wandb metrics expansion
```

Use these for long-form audit/debug context and for understanding *why* specific decisions appear in `design.md`.

---

## Quick Reference — Where to Look

| Need | Document |
| :--- | :------- |
| Understand the architecture | [`architecture.md`](architecture.md) |
| Understand why a decision was made | [`design.md`](design.md) |
| Quick context on current project state | [`context.md`](context.md) |
| Cumulative session history overview | [`meta_artifacts/session_summary.md`](meta_artifacts/session_summary.md) |
| Latest detailed session log | [`meta_artifacts/session10/session.md`](meta_artifacts/session10/session.md) |
| How to use the code | Package READMEs (see below) |
| Reproduce the workshop paper | [`../README.md`](../README.md) (Reference Configuration) + [`../configs/base_small_run2.yaml`](../configs/base_small_run2.yaml) |

---

## Package READMEs

Each subfolder has its own README:

| README | Contents |
| :----- | :------- |
| [`memory_transformer/README.md`](../memory_transformer/README.md) | Core modules: bank, attention, blocks, router, model, adapter |
| [`training/README.md`](../training/README.md) | Trainer, dataset loaders, router-loss aggregation |
| [`inference/README.md`](../inference/README.md) | Generation, routing strategies, merge / quantisation utilities |
| [`scripts/README.md`](../scripts/README.md) | Every CLI script — train, eval, benchmarks, inference, FLOPs estimator |
| [`configs/README.md`](../configs/README.md) | Complete YAML configuration reference |
| [`kernels-final/README.md`](../kernels-final/README.md) | Stable v1–v5 sparse routing kernels and benchmarking |
| [`kernels/README.md`](../kernels/README.md) | Exploratory kernel workspace, FSA lineage, NSA notes |

---

## Documentation Hierarchy

```text
Project Documentation
│
├── README.md (root)                          User-facing overview, paper, quick start, citation
│
├── Package READMEs (usage)
│   ├── memory_transformer/README.md          Core module API
│   ├── training/README.md                    Training infrastructure
│   ├── inference/README.md                   Generation utilities
│   ├── scripts/README.md                     CLI usage
│   ├── configs/README.md                     Configuration reference
│   ├── kernels-final/README.md               Stable kernel set
│   └── kernels/README.md                     Engineering workspace
│
├── Deep dive (this directory)
│   ├── architecture.md                       Technical architecture
│   ├── design.md                             Decision rationale
│   ├── philosophy.md                         Coding / docs conventions
│   ├── context.md                            Current project snapshot
│   └── prompt.md                             Agent onboarding
│
└── meta_artifacts/                           Session history
    ├── session_summary.md                    Chronological overviews
    └── sessionN/                             Detailed per-session artefacts
```

---

## Relationship Between Docs

| Document | Purpose | Update frequency |
| :------- | :------ | :--------------- |
| `context.md` | What the project **is now** | Every major change |
| `session_summary.md` | How the project **got here** | End of each session |
| `sessionN/session.md` | Detailed "how" for a single session | During that session |
| `architecture.md` | Technical deep dive | When architecture changes |
| `design.md` | Decision rationale | When decisions are made |
| Root `README.md` | User-facing overview | When user-visible features or results change |

---

## Contributing to Documentation

When adding a new feature or fixing a bug, update documentation in this order:

1. **Update the in-progress `session.md`** in `meta_artifacts/sessionN/`.
2. **Update `context.md`** to reflect current project state.
3. **Update the relevant package README** with the new API or behaviour.
4. **Update `design.md`** if a non-trivial decision was made.
5. **Update `architecture.md`** for any architectural change.
6. **Update the root `README.md`** if user-facing behaviour or results change.
7. **Append to `session_summary.md`** at the end of the session.
