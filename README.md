# Mixture of Chapters (MoC)

### Scaling Learnt Memory in Transformers

[![ICLR 2026 NFAM Workshop](https://img.shields.io/badge/ICLR%202026-NFAM%20Workshop-1f6feb?logo=academia&logoColor=white)](https://iclr.cc/virtual/2026/workshop/10000782)
[![Paper](https://img.shields.io/badge/Paper-PDF-d62728)](idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](#license)

> A learned, addressable memory bank for transformers — accessed via cross-attention, scaled to **262,208** memory tokens by partitioning into **4,097 chapters** with MoE-style top-k routing, and trained end-to-end. **Accepted at the ICLR 2026 Workshop on New Frontiers in Associative Memory.**

**Authors:** Tasmay Pankaj Tibrewal, Pritish Saha, Ankit Meda, Kunal Singh, Pradeep Moturi
**Affiliations:** IIT Kharagpur · Fractal AI Research
**Code:** this repository · **Paper:** [`idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf`](idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf) · **Slides:** [`idea/Presentation_MoC_BTP.pdf`](idea/Presentation_MoC_BTP.pdf)

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Headline Results](#headline-results)
3. [Architecture](#architecture)
4. [Reference Configuration](#reference-configuration)
5. [Why this works](#why-this-works)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [Repository Layout](#repository-layout)
9. [Configuration System](#configuration-system)
10. [Comparison Experiments](#comparison-experiments)
11. [Token-Level Routing & Sparse Kernels](#token-level-routing--sparse-kernels)
12. [Documentation Index](#documentation-index)
13. [Citation](#citation)
14. [License & Acknowledgements](#license--acknowledgements)

---

## TL;DR

Standard transformers store knowledge implicitly in dense parameters — there is no addressable memory you can scale, edit, freeze, or anchor against forgetting. We add a **learned bank of latent memory tokens** that every memory layer reads via cross-attention, and we scale it to hundreds of thousands of tokens by partitioning the bank into chapters and training a lightweight router to select a small top-k subset per input.

- **Explicit, end-to-end-learned memory** integrated as a third sublayer in selected transformer blocks (after self-attention).
- **Sparse access via Mixture of Chapters routing.** Memory attention cost shifts from `O(L · N_m)` to `O(L · k · T)` per memory layer.
- **Beats compute-matched dense baselines** on validation loss and **resists catastrophic forgetting** under heavy instruction fine-tuning.
- **Bank can be frozen during post-training** with no measurable degradation — clean separation between backbone task adaptation and memory-anchored knowledge.

---

## Headline Results

**Pretraining validation loss** (9.6B-token run, sequence length 1024, iso-FLOP):

| Model                              | Val loss ↓ |
| :--------------------------------- | :--------: |
| Vanilla transformer (backbone-only)|   2.92     |
| Vanilla transformer (iso-FLOP)     |   2.86     |
| **Mixture of Chapters (MoC)**      | **2.79**   |

**Knowledge retention under heavy IFT** (2 epochs on 230M tokens, context 1024 → 2048; Δ = IFT − pretrain, lower is better):

| Benchmark    | Vanilla (iso-FLOP) Δ pp ↓ | **MoC** Δ pp ↓ |
| :----------- | :-----------------------: | :------------: |
| MMLU         |          −0.99            |   **−0.35**    |
| ARC-Challenge|          −6.69            |   **−2.68**    |
| BoolQ        |          −6.24            |   **+0.24**    |
| OpenBookQA   |          −2.00            |     −2.00      |

**Beyond the workshop paper** — with tuned hyperparameters and bank activation, the loss gap to iso-FLOP **doubles** to 0.135; with 1.5× the training tokens it widens to 0.090. The gap is still widening at the end of every run we have done; we have not seen it saturate. See [the paper](idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf) for full tables and curves.

---

## Architecture

<p align="center">
  <img src="idea/MoC%20Arch%20Diagram%20Excalidraw.png" alt="Mixture of Chapters architecture" width="780">
</p>

A single MoC transformer block (Variant A): **Self-Attention → Memory Cross-Attention → SwiGLU MLP**, each with its own RMSNorm and residual.

- The **token stream** produces queries.
- The **memory bank** — `M ∈ ℝ^(N_m × d)`, randomly initialised, trained end-to-end — provides keys and values.
- A **router** scores chapters from the (mean-pooled) sequence representation, selects the top-k routed chapters, and the model attends only to those (plus a small always-on shared prefix). Per-chapter outputs are mixed MoE-style with router weights, so gradients flow back to the router.

For full details see [`docs/architecture.md`](docs/architecture.md).

---

## Reference Configuration

The configuration used in the workshop paper (and reproduced by [`configs/base_small_run2.yaml`](configs/base_small_run2.yaml)):

| Component | Setting |
| :-------- | :------ |
| Backbone | 16 layers, hidden dim 768, 12 query / 4 KV heads (GQA), SwiGLU MLP `d_ff=2304`, RoPE θ=100,000, RMSNorm, tied embeddings, vocab 49,152 (SmolLM2 tokenizer) |
| Memory placement | Cross-attention at layers `{2, 6, 10, 14}` |
| Memory bank | **262,208** latent tokens, **shared across all memory layers** |
| Chapter routing | **4,097 chapters** (1 shared + 4,096 routed); chapter size **T = 64**; **top-k = 64**; routed scaling factor **2.5×** |
| Memory attention | 12 query / 12 KV heads (no GQA on memory side); `wo_init_zero=True` |
| Routing strategy | Sequence-level (mean-pooled hidden states → softmax → top-k) |
| Regularisation | Load-balance loss 0.01, z-loss 0.001 |
| Parameters | **371.29M total** = 147.87M backbone + 22.04M memory layers + 201.38M memory bank |
| Pretraining | 9,600 steps × ~1M tokens/step ≈ **9.6B tokens**, bf16, DDP × 8 GPUs, WSD scheduler |
| IFT | UltraChat 200K, 2 epochs (3,200 steps), context 2048, DDP × 4 GPUs |

---

## Why this works

- **Explicit, addressable memory.** The bank is a separate substrate complementing implicit parametric capacity. It has its own learning rate, can be frozen independently, and can be inspected per-chapter.
- **Sparse access scales.** Dense cross-attention over a 262K-token bank would dominate compute; MoC touches only ~1.6% of the bank per sequence (k·T = 64 · 64 = 4,096 tokens + 1 shared chapter), making memory attention's cost roughly linear in *activated* memory rather than total memory.
- **Anchors knowledge across phase transitions.** Under a deliberately heavy 2-epoch IFT on 230M tokens — long enough to crush the dense baseline on knowledge benchmarks (ARC-Challenge drops to near-random) — MoC stays stable. **Freezing the bank during IFT performs on par with updating it**, evidence that the bank does the knowledge anchoring while the backbone adapts to the IFT distribution.
- **Independent softmax per chapter.** When router weights are applied, each chapter's cross-attention is computed with its own softmax and outputs are mixed by router probabilities. This keeps router weights in control of relative chapter contribution rather than letting Q·K similarity override them.

---

## Installation

### Core install

```bash
git clone https://github.com/Tasmay-Tibrewal/Memory.git
cd Memory
pip install -r requirements.txt
```

### Optional dependencies

```bash
# Flash Attention (Linux + CUDA 11.8+)
pip install flash-attn --no-build-isolation

# 4/8-bit quantisation
pip install bitsandbytes

# Triton (required for the v1–v5 sparse routing kernels on token-level routing)
pip install triton

# Experiment tracking
pip install wandb
```

### Verify the install (no HF download required)

```python
from memory_transformer.config import load_config
from memory_transformer.model import MemoryTransformer
import torch

cfg = load_config("configs/base_small.yaml")
# Shrink for a fast CPU run
cfg.model.num_layers = 2
cfg.model.hidden_dim = 64
cfg.model.num_heads = 4
cfg.model.intermediate_dim = 256
cfg.model.max_seq_len = 64
cfg.memory.num_memory_tokens = 128
cfg.memory.num_chapters = 8
cfg.memory.top_k_chapters = 2
cfg.memory.routing_strategy_inference = "hybrid"

model = MemoryTransformer(cfg).eval()
x = torch.randint(0, cfg.model.vocab_size, (1, 8))
print("Smoke OK:", model(input_ids=x, use_cache=True)["logits"].shape)
```

---

## Quick Start

### 1. Reproduce the paper config

```bash
# Pretrain MoC at the workshop-paper config (DDP × 8)
accelerate launch --num_processes 8 scripts/train.py --config configs/base_small_run2.yaml

# iso-FLOP dense baseline
accelerate launch --num_processes 8 scripts/train.py --config configs/vanilla_control_run2.yaml

# Instruction fine-tune from a pretrained MoC checkpoint
accelerate launch --num_processes 4 scripts/train.py --config configs/ift_base_model.yaml
```

The IFT configs use `init_from_checkpoint` (weights-only restart) so the optimizer/scheduler/global step are reset cleanly when transitioning pretrain → IFT.

### 2. From-scratch on a small budget

```bash
python scripts/train.py --config configs/base_small.yaml
```

### 3. Memory adapter on a pretrained model

```bash
python scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml
# Multi-GPU
accelerate launch --num_processes 4 scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml
```

### 4. Evaluate

```bash
# Validation perplexity (uses the dataset from the config)
python scripts/eval.py --config configs/base_small_run2.yaml --checkpoint outputs/base_small_run5/final_model

# MMLU
python scripts/eval_mmlu.py --config configs/base_small_run2.yaml --checkpoint outputs/.../final_model \
    --shots 5 --fewshot_mode manual --scoring_mode choice_text

# Full benchmark suite (MMLU + HellaSwag + ARC-C/E + WinoGrande + BoolQ + OpenBookQA + TriviaQA)
python scripts/eval_pretrain_suite.py --config configs/base_small_run2.yaml --checkpoint outputs/.../final_model

# IFEval JSONL for external scoring
python scripts/generate_ifeval_jsonl.py --config configs/ift_base_model.yaml \
    --checkpoint outputs/.../final_model --output outputs/ifeval/predictions.jsonl
```

### 5. Inference

```bash
python scripts/inference.py --checkpoint outputs/.../final_model --prompt "Explain associative memory."
```

### 6. Analytic FLOPs estimate

```bash
python scripts/estimate_flops.py --config configs/base_small_run2.yaml
```

---

## Repository Layout

```
Memory/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── memory_transformer/         # Core implementation
│   ├── config.py               # Centralised config dataclasses (50+ knobs)
│   ├── memory_bank.py          # Standard / Factorized / ReducedDim / Chaptered banks
│   ├── memory_attention.py     # Memory cross-attention (dense + token-routed paths)
│   ├── memory_block.py         # Variant A/B blocks, SelfAttention (GQA + RoPE), SwiGLU MLP
│   ├── router.py               # ChapterRouter, TokenLevelRouter, RollingRouter, losses
│   ├── token_routing_kernel.py # Loader for v1–v5 sparse routing kernels
│   ├── lora.py                 # LoRA implementation for comparisons
│   ├── model.py                # MemoryTransformer (from-scratch)
│   ├── adapter.py              # MemoryAdapter (persistent-hook injection into HF models)
│   ├── quantization.py         # 4/8-bit memory bank quantisation
│   └── utils.py                # Schedulers, parameter breakdown, helpers
│
├── training/                   # Training infrastructure
│   ├── trainer.py              # Accelerate-based loop (DDP/FSDP, WSD, eval, resume, init-from)
│   ├── data.py                 # Pretrain / instruction datasets, assistant-only label masking
│   └── losses.py               # Router auxiliary loss aggregation
│
├── inference/                  # Generation & deployment
│   ├── generate.py             # KV-cached single + batch generation
│   ├── routing_strategies.py   # Sequence / SequenceRolling / Rolling / Token / Hybrid routers
│   └── merge.py                # LoRA merge, memory extraction, full-model & memory quantisation, GGUF helper
│
├── scripts/                    # CLI entry points
│   ├── train.py                # Training launcher
│   ├── eval.py                 # Perplexity (single + distributed)
│   ├── eval_mmlu.py            # MMLU multiple-choice accuracy
│   ├── eval_mcq_benchmark.py   # Generic MCQ evaluator
│   ├── eval_{hellaswag,arc,winogrande,boolq,openbookqa,triviaqa}.py
│   ├── eval_pretrain_suite.py  # Run + aggregate the benchmark suite
│   ├── generate_ifeval_jsonl.py# Produce IFEval-format predictions
│   ├── inference.py            # Generation
│   └── estimate_flops.py       # Analytic FLOPs estimator
│
├── configs/                    # YAML configs (paper reproduction + ablations)
│   ├── base_small.yaml             # 16k-bank from-scratch config
│   ├── base_small_run2.yaml        # **Paper config** (262k-bank, 4097 chapters, top-k 64)
│   ├── vanilla_control.yaml        # iso-FLOP-ish dense baseline
│   ├── vanilla_control_run2.yaml   # iso-FLOP dense baseline matching the paper run
│   ├── ift_base_model.yaml         # IFT from a pretrained MoC checkpoint
│   ├── ift_vanilla_model.yaml      # IFT from the iso-FLOP baseline
│   ├── ift_vanilla_model_small.yaml# IFT from the backbone-only baseline
│   ├── adapter_qwen2.5_1.5b.yaml   # Memory adapter on Qwen2.5-1.5B
│   ├── memory_lora_combined.yaml   # Memory + LoRA combined adapter
│   └── reference_all_options.yaml  # Documentation template (every config knob)
│
├── kernels-final/              # Stable curated sparse routing kernels
│   ├── kernel_v1.py            # Unoptimised reference baseline
│   ├── kernel_v2.py            # Older optimised — default (most stable)
│   ├── kernel_v3.py            # Old optimised alternate
│   ├── kernel_v4.py            # Exact MoE-weighted fused kernel (forward + dQ/dK/dV/dW)
│   ├── kernel_v5.py            # Joint-bias weighted approximation
│   ├── memory_cross_attn.py    # Earlier custom Triton chapter-routed cross-attention
│   └── benchmark_kernels_final.py  # Correctness + timing benchmark (unweighted & MoE-weighted)
│
├── kernels/                    # Engineering workspace (experiments, FSA lineage, reports)
│   ├── kernel-architecture.md  # Detailed kernel-method narrative
│   ├── FSA_LOCAL_OPTIMIZATION_REPORT.md
│   ├── TOKEN_LEVEL_MEMORY_ROUTING_HANDOFF.md
│   ├── TOKEN_LEVEL_ROUTING_NSA_REPORT.md
│   └── ...                     # Variants, benchmarks, notebook
│
├── docs/                       # Deep documentation
│   ├── architecture.md
│   ├── design.md
│   ├── context.md
│   ├── philosophy.md
│   ├── prompt.md
│   └── meta_artifacts/         # Per-session logs and summaries
│
└── idea/                       # Original research artefacts
    ├── idea.txt                # Conceptual draft
    ├── proposal.md             # Long-form proposal
    ├── main.tex                # LaTeX paper source
    ├── Mixture_of_Chapters_ICLR_Workshop_Paper.pdf
    ├── Presentation_MoC_BTP.pdf
    ├── Memory_Layer_in_Transformers.pdf
    └── MoC Arch Diagram Excalidraw.png
```

---

## Configuration System

Configs are YAML, grouped into three sections: `model:`, `memory:`, `training:`. The full surface (every knob) is documented in [`configs/reference_all_options.yaml`](configs/reference_all_options.yaml) and [`configs/README.md`](configs/README.md).

Highlights:

```yaml
memory:
  vanilla_mode: false          # true => disable memory entirely (control)
  use_memory_adapter: true

  # Bank
  num_memory_tokens: 262208
  memory_num_heads: 12         # null => fall back to model.num_heads
  memory_num_kv_heads: 12

  # Placement
  memory_layer_placement: custom
  memory_layer_indices: [2, 6, 10, 14]
  memory_sharing: shared       # shared / per_layer / every_k_layers
  memory_block_variant: A      # A: SA→Mem→MLP   B: SA→MLP→Mem→MLP

  # Chapter routing
  use_chapters: true
  num_chapters: 4097           # = 1 shared + 4096 routed
  top_k_chapters: 64
  num_shared_chapters: 1
  routed_scaling_factor: 2.5
  normalize_shared_routed_before_mixing: true
  shared_routed_norm_type: rms
  routing_strategy_train: sequence       # sequence / sequence-rolling / token
  routing_strategy_inference: sequence   # + rolling / hybrid for cached decode
  routing_window_size: 128
  token_routing_kernel_version: v2       # v1 / v2 / v3 / v4 / v5

  # Router losses
  use_load_balance_loss: true
  load_balance_coefficient: 0.01
  use_z_loss: true
  z_loss_coefficient: 0.001

  # Initialisation
  wo_init_zero: true           # Zero-init memory output projection (critical)
  memory_init_std: 0.02

training:
  # Separate learning rates per parameter group
  memory_lr: 6e-4
  memory_bank_lr: 6e-4         # null => falls back to memory_lr
  base_model_lr: 3e-4
  lora_lr: 3e-4

  scheduler: WSD               # cosine / linear / constant / wsd
  wsd_stable_ratio: 0.0
  decay_start_step: 8160
  min_lr_ratio: 0.1
  mixed_precision: bf16
  gradient_checkpointing: true

  # Resume vs. fresh restart
  resume_from_checkpoint: null # Restores model + optimizer + scheduler + step
  init_from_checkpoint: null   # Weights-only restart (global_step=0)
```

---

## Comparison Experiments

The experimental design is set up so that the four obvious variants share infrastructure:

| # | Setting                | How to enable                                                    |
| - | ---------------------- | ---------------------------------------------------------------- |
| 1 | Vanilla (no memory)    | `vanilla_mode: true` (e.g. [`configs/vanilla_control_run2.yaml`](configs/vanilla_control_run2.yaml)) |
| 2 | Memory only            | `use_memory_adapter: true`, `use_lora: false`                    |
| 3 | LoRA only              | `use_memory_adapter: false`, `use_lora: true`                    |
| 4 | Memory + LoRA combined | `use_both_memory_and_lora: true` (e.g. [`configs/memory_lora_combined.yaml`](configs/memory_lora_combined.yaml)) |

---

## Token-Level Routing & Sparse Kernels

Sequence-level routing is the default and matches the workshop paper. The repository also implements **token-level routing** end-to-end (model + adapter), where each token routes to its own top-k chapters. The attention path is split into:

1. **Shared chapters** — dense cross-attention (FlashAttention if available, else PyTorch fallback). Always weighted at 1.0.
2. **Routed chapters** — sparse top-k cross-attention via a Triton kernel from `kernels-final/`.
3. **Output mix** — `shared_output + routed_scaling_factor × routed_output`, then projected by `W_o`.

Five curated kernel variants live in `kernels-final/`:

| Version | Role                                                                                  |
| :-----: | :------------------------------------------------------------------------------------ |
| `v1`    | Unoptimised reference baseline                                                        |
| `v2`    | Older optimised path — **default**, most stable overall                               |
| `v3`    | Old optimised alternate                                                               |
| `v4`    | **Exact MoE-weighted fused kernel** — single launch with forward + dQ/dK/dV/dW       |
| `v5`    | Joint-bias weighted approximation — adds `log(weight)` bias to logits, single softmax |

Set `memory.token_routing_kernel_version: v1|v2|v3|v4|v5` in the YAML. When the kernel path is unavailable for the current device/dtype/shape, the model falls back to an emulated PyTorch sparse path for correctness.

`kernels-final/benchmark_kernels_final.py` verifies forward, dQ, dK, dV, and dW correctness against naive Python references, and times both unweighted (joint softmax) and MoE-weighted (per-chapter independent softmax with CUDA stream parallelism) modes. The historic exploration archive is in [`kernels/`](kernels/), with the engineering narrative in [`kernels/kernel-architecture.md`](kernels/kernel-architecture.md).

---

## Documentation Index

| Document | Purpose |
| :------- | :------ |
| [`docs/architecture.md`](docs/architecture.md)       | Detailed architecture with equations, shapes, and complexity analysis |
| [`docs/design.md`](docs/design.md)                   | Design decisions, compromises, known limitations |
| [`docs/context.md`](docs/context.md)                 | Project status snapshot for handoffs / agent onboarding |
| [`docs/philosophy.md`](docs/philosophy.md)           | Coding, configuration, and documentation philosophy |
| [`docs/prompt.md`](docs/prompt.md)                   | Onboarding prompt for new sessions / agents |
| [`docs/meta_artifacts/session_summary.md`](docs/meta_artifacts/session_summary.md) | Cumulative session log |
| [`memory_transformer/README.md`](memory_transformer/README.md) | Core package walkthrough |
| [`training/README.md`](training/README.md)           | Trainer, data loaders, losses |
| [`inference/README.md`](inference/README.md)         | Generation, routing strategies, merge utilities |
| [`scripts/README.md`](scripts/README.md)             | All CLI entry points |
| [`configs/README.md`](configs/README.md)             | Complete config reference |
| [`kernels-final/README.md`](kernels-final/README.md) | Stable kernel set: choices and benchmarking |
| [`kernels/README.md`](kernels/README.md)             | Engineering workspace (exploratory) |
| [`idea/proposal.md`](idea/proposal.md)               | Original long-form research proposal |

---

## Citation

If you use this code or build on this work, please cite the workshop paper:

```bibtex
@inproceedings{tibrewal2026moc,
  title     = {Mixture of Chapters: Scaling Learnt Memory in Transformers},
  author    = {Tibrewal, Tasmay Pankaj and Saha, Pritish and Meda, Ankit and Singh, Kunal and Moturi, Pradeep},
  booktitle = {New Frontiers in Associative Memory Workshop, ICLR},
  year      = {2026},
  url       = {https://iclr.cc/virtual/2026/workshop/10000782}
}
```

---

## License & Acknowledgements

Released under the **MIT License**.

Compute and tooling support: **Fractal AI**, **Modal**, **Neysa**.
Mentor: Kunal Kingkar. Supervisor: Prof. Pawan Goyal (IIT Kharagpur).
Thanks to the ICLR 2026 NFAM Workshop reviewers.
