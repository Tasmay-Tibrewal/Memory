# Agent Onboarding Prompt

> **Use this file as your starting prompt when beginning any new session on this codebase.**

---

## Your Role

You are an expert AI coding assistant helping to develop, maintain, and extend a **Memory-Augmented Transformer** research project — branded as **Mixture of Chapters (MoC)** in the workshop paper. This is an academic / research codebase implementing a novel approach to scaling language model memory beyond the context window via a learned, addressable memory bank with sparse chapter routing.

The work has been **accepted at the ICLR 2026 Workshop on New Frontiers in Associative Memory** ([`idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf`](../idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf)).

**Before doing ANY implementation work, you MUST thoroughly read and understand the documentation files listed below.**

---

## The Task We're Solving

### Problem Statement

Large Language Models (LLMs) are fundamentally limited by their context window size. Once the context is full, older information is lost. Standard transformers also store all knowledge implicitly in dense parameters — there is no addressable memory you can scale, edit, freeze, or anchor against forgetting. This creates significant limitations for:

- Long-form reasoning and document analysis
- Multi-turn conversations with extensive history
- Knowledge-intensive tasks requiring large reference material
- Continued training without catastrophic forgetting

### Our Solution: Learnable Memory Bank with Chapter Routing

We implement a **learnable external memory bank** that the model can attend to via cross-attention. Key aspects:

1. **Memory Bank**: A fixed set of learnable tokens (workshop config: 262,208 tokens) that encode compressed knowledge.
2. **Cross-Attention**: At selected transformer layers, the model can query the memory bank.
3. **Chapter Routing (Mixture of Chapters)**: For large memory banks, MoE-style routing selects relevant "chapters" — subsets of memory tokens (workshop config: top-64 of 4,097 chapters, with 1 always-on shared chapter).
4. **Adapter Integration**: Memory can be added to any pretrained model (Qwen, Llama, Mistral) without fine-tuning the base model.

### Why This Approach

- **Constant attention cost**: `O(L · k · T)` per memory layer rather than `O((L+M)²)` if memory were in context.
- **Learned compression**: Memory tokens learn to encode useful information end-to-end.
- **Modular**: Can be added to any pretrained transformer (adapter mode) or trained from scratch.
- **Scalable**: Chapter routing enables hundreds of thousands of memory tokens efficiently.
- **Robust to post-training drift**: Knowledge benchmarks remain stable under heavy IFT; the bank can even be frozen during IFT with no measurable degradation.

---

## Required Reading (Do This First!)

You MUST read these files before implementation. They contain critical design decisions and context.

### Core Documentation (in `docs/`)

| File | Purpose | Priority |
| :--- | :------ | :------- |
| **`docs/context.md`** | Exhaustive project summary, all files, all config flags, running commands. **Read this first for quick orientation.** | Critical |
| **`docs/architecture.md`** | Detailed technical architecture with diagrams. How components connect. | Critical |
| **`docs/design.md`** | Design decisions, trade-offs, known limitations. Why we made certain choices. | Important |
| **`docs/philosophy.md`** | Development philosophy and coding style. How to write code for this project. | Important |

### Package READMEs (in each subfolder)

| File | Purpose |
| :--- | :------ |
| `memory_transformer/README.md` | Core module documentation — memory bank, attention, blocks, router |
| `training/README.md` | Training infrastructure — data, losses, trainer |
| `inference/README.md` | Generation, routing strategies, merge / quantisation utilities |
| `scripts/README.md` | CLI scripts for training / eval / inference, plus the benchmark suite tooling |
| `configs/README.md` | Complete configuration reference with all 50+ flags |
| `kernels-final/README.md` | Stable curated v1–v5 sparse routing kernels and benchmarking |
| `kernels/README.md` | Engineering workspace (exploratory variants, FSA lineage, NSA notes) |

### Session Context (in `docs/meta_artifacts/`)

| File | Purpose |
| :--- | :------ |
| `session_summary.md` | Cumulative summaries of all development sessions |
| `session10/session.md` | Latest detailed session log (shared chapters + wandb metrics) |
| `session1/session.md` | Historical deep log from initial implementation |

### Research artefacts (in `idea/`)

| File | Purpose |
| :--- | :------ |
| `idea.txt` | Original conceptual draft |
| `proposal.md` | Long-form proposal |
| `main.tex` | LaTeX paper source |
| `Mixture_of_Chapters_ICLR_Workshop_Paper.pdf` | Workshop paper (read this for the headline results) |
| `Presentation_MoC_BTP.pdf` | BTP presentation slides |
| `MoC Arch Diagram Excalidraw.png` | Architecture diagram |

---

## Project Structure

```text
Memory/
├── memory_transformer/         # Core implementation
│   ├── config.py               # All configuration dataclasses (50+ options)
│   ├── memory_bank.py          # Memory bank variants (standard, factorized, reduced-dim, chaptered)
│   ├── memory_attention.py     # Cross-attention to memory (dense + token-routed paths)
│   ├── memory_block.py         # Transformer blocks (Variant A/B), GQA self-attention, RoPE, MLP
│   ├── model.py                # Full model for from-scratch training
│   ├── adapter.py              # Memory adapter for pretrained models (persistent hooks)
│   ├── router.py               # Chapter routing (MoE-style)
│   ├── token_routing_kernel.py # Loader for v1–v5 sparse routing kernels
│   ├── lora.py                 # LoRA implementation for comparison
│   ├── quantization.py         # 4/8-bit memory quantisation
│   └── utils.py                # Utilities
│
├── training/                   # Training infrastructure
│   ├── data.py                 # Dataset loading (any HF dataset)
│   ├── losses.py               # Router auxiliary losses
│   └── trainer.py              # Accelerate-based trainer
│
├── inference/                  # Inference utilities
│   ├── generate.py             # Text generation
│   ├── routing_strategies.py   # Inference routing strategies
│   └── merge.py                # LoRA merge / model & memory quantisation / GGUF helper
│
├── scripts/                    # CLI entry points
│   ├── train.py                # Training script
│   ├── eval.py                 # Perplexity evaluation
│   ├── eval_mmlu.py            # MMLU accuracy
│   ├── eval_mcq_benchmark.py   # Generic MCQ benchmark evaluator
│   ├── eval_{hellaswag,arc,winogrande,boolq,openbookqa,triviaqa}.py
│   ├── eval_pretrain_suite.py  # Run benchmark suite + aggregate
│   ├── generate_ifeval_jsonl.py# IFEval-format predictions
│   ├── inference.py            # Inference script
│   └── estimate_flops.py       # Analytic FLOPs estimator
│
├── configs/                    # YAML configurations (workshop reproduction + ablations)
├── kernels-final/              # Curated stable sparse routing kernels (v1–v5)
├── kernels/                    # Engineering workspace (exploratory)
├── docs/                       # Documentation
├── idea/                       # Original research idea, paper, slides, diagram
└── requirements.txt            # Dependencies
```

---

## Key Architectural Concepts

### 1. Memory Bank

```text
StandardMemoryBank:    M ∈ ℝ^(N_m × d)        # Full learnable parameters
FactorizedMemoryBank:  M = A · B^T            # Low-rank factorisation
ReducedDimMemoryBank:  M ∈ ℝ^(N_m × r)        # Attention in r-dim space
ChapteredMemoryBank:   wraps any of the above # Chapter-indexed accessors
```

### 2. Memory Cross-Attention

```text
Input:     H ∈ ℝ^(B × L × d)
Memory:    M ∈ ℝ^(N_m × d)

Q = H · W_q
K = M · W_k
V = M · W_v
Output = softmax(Q · K^T / √d_k) · V · W_o    # W_o initialised to zero (critical)
```

### 3. Block Integration Variants

```text
Variant A (default):  Self-Attn → Memory Cross-Attn → MLP
Variant B:            Self-Attn → MLP → Memory Cross-Attn → MLP
```

### 4. Chapter Routing

For large memory banks, divide into chapters and route per input:

```python
router_logits = hidden_states.mean(dim=1) @ W_router    # (B, num_chapters)
selected_chapters = top_k(softmax(router_logits), k=64) # (B, top_k)
```

Workshop paper config: 4,097 chapters, top-k = 64, with 1 always-on shared chapter and routed scaling factor 2.5×.

### 5. Token-Level Routing (sparse kernels)

When `routing_strategy_*: token` is set:

1. Shared chapters → dense attention (FlashAttention if available, else PyTorch).
2. Routed chapters → sparse attention via `kernels-final/kernel_v{1..5}.py` (default `v2`).
3. Combined as `shared + routed_scaling_factor × routed`, then projected by `W_o`.

`v4` is the exact MoE-weighted fused kernel (forward + dQ/dK/dV/dW). `v5` is a joint-bias approximation. Falls back to an emulated PyTorch sparse path when the kernel is unavailable.

---

## Configuration System

The codebase uses a hierarchical YAML configuration system with three main sections:

1. **`model:`** — Base transformer architecture
2. **`memory:`** — Memory bank and cross-attention settings
3. **`training:`** — Training hyperparameters

All 50+ configuration options are documented in [`configs/README.md`](../configs/README.md) and the full surface is shown in [`configs/reference_all_options.yaml`](../configs/reference_all_options.yaml).

---

## Development Philosophy (Key Points)

From [`docs/philosophy.md`](philosophy.md):

1. **Modular over Monolithic**: Each component is self-contained.
2. **Configuration over Code Changes**: Experiments via YAML, not code edits.
3. **Explicit over Implicit**: Named parameters, clear documentation.
4. **Adapter-First Design**: Works on pretrained models.
5. **Research-Oriented**: Easy experimentation, clear baselines.

---

## Before You Start Implementation

### Checklist

- [ ] Read [`docs/context.md`](context.md) completely
- [ ] Read [`docs/architecture.md`](architecture.md) for system understanding
- [ ] Read [`docs/design.md`](design.md) for rationale behind decisions
- [ ] Skim [`docs/philosophy.md`](philosophy.md) for coding style
- [ ] Check the relevant package README for the area you're working on
- [ ] Review [`docs/meta_artifacts/session_summary.md`](meta_artifacts/session_summary.md) for recent work
- [ ] Browse [`idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf`](../idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf) for the headline results

### Ask Questions If...

You should ask clarifying questions before proceeding if:

1. **The task is unclear**: What exactly needs to be done?
2. **Design decisions needed**: Multiple valid approaches exist.
3. **Potential breaking changes**: Modifications that might affect other components.
4. **Missing context**: Something from the user's intent is ambiguous.
5. **Conflict with existing design**: Task seems to contradict documented philosophy.
6. **Performance concerns**: Implementation might have scaling issues.

**Always ask before making assumptions that could lead to significant rework.**

---

## Running the Code

### Training

```bash
# Reproduce the workshop paper config (DDP × 8)
accelerate launch --num_processes 8 scripts/train.py --config configs/base_small_run2.yaml

# iso-FLOP dense baseline
accelerate launch --num_processes 8 scripts/train.py --config configs/vanilla_control_run2.yaml

# Memory adapter on Qwen
accelerate launch --num_processes 4 scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml

# Instruction fine-tune from a pretrained MoC checkpoint
accelerate launch --num_processes 4 scripts/train.py --config configs/ift_base_model.yaml
```

### Evaluation

```bash
python scripts/eval.py --config configs/base_small_run2.yaml --checkpoint outputs/.../final_model

# Benchmark suite
python scripts/eval_pretrain_suite.py --config configs/base_small_run2.yaml --checkpoint outputs/.../final_model
```

### Inference

```bash
python scripts/inference.py --config configs/adapter_qwen2.5_1.5b.yaml \
    --checkpoint outputs/.../final_model --prompt "Hello, world"
```

### Analytic FLOPs

```bash
python scripts/estimate_flops.py --config configs/base_small_run2.yaml
```

---

## What is NOT Implemented (By Design)

These are explicitly NOT in scope for the current workshop release:

1. **Dynamic context bank** — inference-time memory updates with VAE compression / clustering / merging. Outlined in `idea/proposal.md` as future work.
2. **Memory updates during inference** — the bank is frozen after training.
3. **Broad token-routing benchmark / policy retuning** — core token-routing path is implemented; broad shape / GPU tuning is still ongoing.
4. **Training-time memory QAT** — basic post-training quantisation exists, but quantisation-aware memory training is not implemented.

---

## Session Management

This project uses a session-based development log:

- Each major work session is logged in `docs/meta_artifacts/session{N}/`.
- Cumulative summaries live in `docs/meta_artifacts/session_summary.md`.
- Update these files at the end of your work.

---

## Starting Your Work

1. **State your understanding** of the task before implementing.
2. **Reference specific files** you plan to modify.
3. **Explain your approach** if doing anything non-trivial.
4. **Ask questions** at any point if something is unclear.
5. **Update documentation** for any significant changes.
6. **Update session files** at the end of your work.

---

## Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Vanilla mode (no memory) for control experiments
python scripts/train.py --config configs/vanilla_control.yaml

# Check model info without training
python -c "from memory_transformer.config import load_config; from memory_transformer.model import MemoryTransformer; m = MemoryTransformer(load_config('configs/base_small.yaml')); print(f'{sum(p.numel() for p in m.parameters()):,} params')"
```

---

## Final Notes

- This is a **research codebase** — clarity and experimentation speed matter more than micro-optimisations.
- All major design decisions are documented — check `docs/` before asking "why".
- Configuration drives behaviour — code should be stable, experiments via YAML.
- When in doubt, **ask questions first**.

**Welcome to the project! Start by reading [`docs/context.md`](context.md).**
