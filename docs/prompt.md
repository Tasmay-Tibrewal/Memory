# Agent Onboarding Prompt

> **Use this file as your starting prompt when beginning any new session on this codebase.**

---

## Your Role

You are an expert AI coding assistant helping to develop, maintain, and extend a **Memory-Augmented Transformer** research project. This is an academic/research codebase implementing a novel approach to scaling language model memory beyond the context window.

**Before doing ANY implementation work, you MUST thoroughly read and understand the documentation files listed below.**

---

## The Task We're Solving

### Problem Statement

Large Language Models (LLMs) are fundamentally limited by their context window size. Once the context is full, older information is lost. This creates significant limitations for:
- Long-form reasoning and document analysis
- Multi-turn conversations with extensive history
- Knowledge-intensive tasks requiring large reference material

### Our Solution: Learnable Memory Bank

We implement a **learnable external memory bank** that the model can attend to via cross-attention. Key aspects:

1. **Memory Bank**: A fixed set of learnable tokens (e.g., 1024-100K tokens) that encode compressed knowledge
2. **Cross-Attention**: At each transformer layer, the model can query the memory bank
3. **Chapter Routing**: For large memory banks, MoE-style routing selects relevant "chapters" (subsets of memory)
4. **Adapter Integration**: Memory can be added to any pretrained model (Qwen, Llama, etc.) without fine-tuning the base model

### Why This Approach

- **Constant attention cost**: O(L × M) instead of O((L+M)²) if memory was in context
- **Learned compression**: Memory tokens learn to encode useful information
- **Modular**: Can be added to any pretrained transformer
- **Scalable**: Chapter routing enables 100K+ memory tokens efficiently

---

## Required Reading (Do This First!)

You MUST read these files before implementation. They contain critical design decisions and context:

### Core Documentation (in `docs/`)

| File | Purpose | Priority |
|------|---------|----------|
| **`docs/context.md`** | Exhaustive project summary, all files, all config flags, running commands. **Read this first for quick orientation.** | 🔴 Critical |
| **`docs/architecture.md`** | Detailed technical architecture with diagrams. How components connect. | 🔴 Critical |
| **`docs/design.md`** | Design decisions, trade-offs, known limitations. Why we made certain choices. | 🟡 Important |
| **`docs/philosophy.md`** | Development philosophy and coding style. How to write code for this project. | 🟡 Important |

### Package READMEs (in each subfolder)

| File | Purpose |
|------|---------|
| `memory_transformer/README.md` | Core module documentation - memory bank, attention, blocks |
| `training/README.md` | Training infrastructure - data, losses, trainer |
| `inference/README.md` | Generation and inference utilities |
| `scripts/README.md` | CLI scripts for training/eval/inference |
| `configs/README.md` | Complete configuration reference with all 50+ flags |

### Session Context (in `docs/meta_artifacts/`)

| File | Purpose |
|------|---------|
| `session_summary.md` | Summaries of all development sessions |
| `session1/session.md` | Detailed log of Session 1 work |

---

## Project Structure

```
Memory/
├── memory_transformer/     # Core implementation
│   ├── config.py          # All configuration dataclasses (50+ options)
│   ├── memory_bank.py     # Memory bank variants (standard, factorized, reduced-dim)
│   ├── memory_attention.py # Cross-attention to memory
│   ├── memory_block.py    # Transformer blocks (Variant A/B integration)
│   ├── model.py           # Full model for from-scratch training
│   ├── adapter.py         # Memory adapter for pretrained models
│   ├── router.py          # Chapter routing (MoE-style)
│   ├── lora.py            # LoRA implementation for comparison
│   ├── quantization.py    # 4/8-bit memory quantization
│   └── utils.py           # Utilities
│
├── training/              # Training infrastructure
│   ├── data.py           # Dataset loading (any HF dataset)
│   ├── losses.py         # Router auxiliary losses
│   └── trainer.py        # Accelerate-based trainer
│
├── inference/            # Inference utilities
│   ├── generate.py       # Text generation
│   └── routing_strategies.py # Inference routing strategies
│
├── scripts/              # CLI entry points
│   ├── train.py         # Training script
│   ├── eval.py          # Evaluation script
│   └── inference.py     # Inference script
│
├── configs/              # YAML configurations
│   ├── base_small.yaml  # Small from-scratch config
│   ├── adapter_qwen2.5_1.5b.yaml # Qwen adapter config
│   └── ...
│
├── docs/                 # Documentation
├── idea/                 # Original research idea files
└── requirements.txt      # Dependencies
```

---

## Key Architectural Concepts

### 1. Memory Bank

```
StandardMemoryBank: M ∈ ℝ^(N_m × d)     # Full learnable parameters
FactorizedMemoryBank: M = A @ B^T       # Low-rank factorization
ReducedDimMemoryBank: M ∈ ℝ^(N_m × r)   # Attention in r-dim space
```

### 2. Memory Cross-Attention

```
Input: Hidden states H ∈ ℝ^(B × L × d)
Memory: M ∈ ℝ^(N_m × d)

Q = H @ W_q
K = M @ W_k
V = M @ W_v
Output = softmax(Q @ K^T / √d_k) @ V @ W_o
```

### 3. Block Integration Variants

```
Variant A: Self-Attn → Memory Cross-Attn → MLP
Variant B: Self-Attn → MLP → Memory Cross-Attn → MLP
```

### 4. Chapter Routing

For large memory banks, divide into chapters and route:
```
router_logits = hidden_states.mean(dim=1) @ W_router
selected_chapters = top_k(softmax(router_logits), k=20)
```

---

## Configuration System

The codebase uses a hierarchical YAML configuration system with three main sections:

1. **`model:`** - Base transformer architecture
2. **`memory:`** - Memory bank and cross-attention settings
3. **`training:`** - Training hyperparameters

All 50+ configuration options are documented in `configs/README.md`.

---

## Development Philosophy (Key Points)

From `docs/philosophy.md`:

1. **Modular over Monolithic**: Each component is self-contained
2. **Configuration over Code Changes**: Experiments via YAML, not code edits
3. **Explicit over Implicit**: Named parameters, clear documentation
4. **Adapter-First Design**: Works on pretrained models
5. **Research-Oriented**: Easy experimentation, clear baselines

---

## Before You Start Implementation

### Checklist

- [ ] Read `docs/context.md` completely
- [ ] Read `docs/architecture.md` for system understanding
- [ ] Read `docs/design.md` for rationale behind decisions
- [ ] Skim `docs/philosophy.md` for coding style
- [ ] Check relevant package README for the area you're working on
- [ ] Review `docs/meta_artifacts/session_summary.md` for recent work

### Ask Questions If...

You should ask clarifying questions before proceeding if:

1. **The task is unclear**: What exactly needs to be done?
2. **Design decisions needed**: Multiple valid approaches exist
3. **Potential breaking changes**: Modifications that might affect other components
4. **Missing context**: Something from the user's intent is ambiguous
5. **Conflict with existing design**: Task seems to contradict documented philosophy
6. **Performance concerns**: Implementation might have scaling issues

**Always ask before making assumptions that could lead to significant rework.**

---

## Running the Code

### Training
```bash
# From-scratch
python scripts/train.py --config configs/base_small.yaml

# Adapter on pretrained
accelerate launch scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml
```

### Evaluation
```bash
python scripts/eval.py --config configs/base_small.yaml --checkpoint outputs/final_model
```

### Inference
```bash
python scripts/inference.py --config configs/adapter_qwen2.5_1.5b.yaml --checkpoint outputs/final_model --prompt "Hello, world"
```

---

## What NOT Implemented (By Design)

These are explicitly NOT in scope:

1. **Dynamic context extension** - We use fixed memory, not retrieval
2. **Memory updates during inference** - Memory is frozen after training
3. **Per-token memory attention** - Too expensive during training
4. **Custom kernels** - We use Flash Attention when available

---

## Session Management

This project uses a session-based development log:

- Each major work session is logged in `docs/meta_artifacts/session{N}/`
- Summaries are in `docs/meta_artifacts/session_summary.md`
- Update these files at the end of your work

---

## Starting Your Work

1. **State your understanding** of the task before implementing
2. **Reference specific files** you plan to modify
3. **Explain your approach** if doing anything non-trivial
4. **Ask questions** at any point if something is unclear
5. **Update documentation** for any significant changes
6. **Update session files** at the end of your work

---

## Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run with vanilla mode (no memory, for control experiments)
python scripts/train.py --config configs/vanilla_control.yaml

# Check model info without training
python -c "from memory_transformer.config import load_config; from memory_transformer.model import MemoryTransformer; m = MemoryTransformer(load_config('configs/base_small.yaml')); print(f'{sum(p.numel() for p in m.parameters()):,} params')"
```

---

## Final Notes

- This is a **research codebase** - clarity and experimentation speed matter more than micro-optimizations
- All major design decisions are documented - check docs before asking "why"
- Configuration drives behavior - code should be stable, experiments via YAML
- When in doubt, **ask questions first**

**Welcome to the project! Start by reading `docs/context.md`.**
