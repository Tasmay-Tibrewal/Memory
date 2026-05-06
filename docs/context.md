# Memory-Augmented Transformer / Mixture of Chapters: Context for Handoffs

This document provides a comprehensive single-file snapshot of the project for future work, session handoffs, context compaction, or agent transitions. It is designed to give any reader the complete picture of the project state in one read.

For the user-facing pitch, paper reference, and architecture diagram, see the [root `README.md`](../README.md).

---

## Project Overview

| Field | Value |
| :---- | :---- |
| **Title** | Mixture of Chapters: Scaling Learnt Memory in Transformers |
| **Goal** | Memory-augmented transformers with learnable cross-attention memory banks, scaled to 262K tokens via MoE-style chapter routing |
| **Repository** | `Memory/` |
| **Status** | COMPLETE — implementation + verification + workshop paper accepted (ICLR 2026 NFAM) |
| **Sessions** | 1–10 (initial implementation, fixes, verification, doc passes, shared-chapter routing, wandb metrics, and the Session 11 documentation overhaul) |
| **Workshop paper** | [`idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf`](../idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf) |
| **Architecture diagram** | [`idea/MoC Arch Diagram Excalidraw.png`](../idea/MoC%20Arch%20Diagram%20Excalidraw.png) |
| **Date of this snapshot** | February 2026 |

---

## Quick Start for New Sessions

```bash
# 1. Read this document for current state
# 2. Check meta_artifacts/session_summary.md for cumulative session history
# 3. Reproduce the workshop paper config:
accelerate launch --num_processes 8 scripts/train.py --config configs/base_small_run2.yaml

# 4. Run the iso-FLOP baseline:
accelerate launch --num_processes 8 scripts/train.py --config configs/vanilla_control_run2.yaml

# 5. Instruction-fine-tune a pretrained MoC checkpoint:
accelerate launch --num_processes 4 scripts/train.py --config configs/ift_base_model.yaml
```

---

## Headline Results (Workshop Paper)

Pretraining validation loss (9.6B tokens, iso-FLOP):

| Model | Val loss ↓ |
| :---- | :--------: |
| Vanilla (backbone-only) | 2.92 |
| Vanilla (iso-FLOP) | 2.86 |
| **Mixture of Chapters (MoC)** | **2.79** |

Knowledge-retention deltas under heavy IFT (lower magnitude is better):

| Benchmark | Vanilla Δ pp | **MoC Δ pp** |
| :-------- | :----------: | :----------: |
| MMLU | −0.99 | **−0.35** |
| ARC-Challenge | −6.69 | **−2.68** |
| BoolQ | −6.24 | **+0.24** |
| OpenBookQA | −2.00 | −2.00 |

---

## What's Implemented

### File Count Summary

| Directory | Files | Lines of Code (approx) |
| :-------- | :---: | :--------------------: |
| `memory_transformer/` | 12 | ~5,900 |
| `training/` | 4 | ~1,580 |
| `inference/` | 4 | ~1,070 |
| `scripts/` | 14 | ~4,300 |
| `configs/` | 11 (10 YAML + 1 README) | ~2,200 |
| `kernels-final/` | 7 | ~17,400 (Triton-heavy) |
| `docs/` | 6 (incl. README) | ~2,000 |
| `docs/meta_artifacts/` | 4+ folders | ~3,200+ |
| **Total** | **62+** | **~37,600+** |

### Core Architecture (`memory_transformer/`)

| File | Purpose |
| :--- | :------ |
| `__init__.py` | Package exports with lazy loading for HF-dependent modules |
| `config.py` | Configuration system (`MemoryConfig`, `ModelConfig`, `TrainingConfig`, `Config`, YAML loader) |
| `memory_bank.py` | Memory bank variants (Standard, Factorized, ReducedDim, Chaptered) |
| `memory_attention.py` | Memory cross-attention with full / low-rank / reduced-dim modes; token-routed dense+sparse path |
| `memory_block.py` | Variant A/B blocks, RMSNorm, RoPE, GQA SelfAttention, SwiGLU MLP, VanillaTransformerBlock |
| `router.py` | `ChapterRouter` (sequence + sequence-rolling + token), `TokenLevelRouter`, `RollingRouter`, MoE losses |
| `token_routing_kernel.py` | Loader for `kernels-final/kernel_v{1..5}.py` and weighted variants |
| `lora.py` | LoRA implementation for comparisons |
| `model.py` | `MemoryTransformer` for from-scratch training (with shared-chapter prepending and routed scaling) |
| `adapter.py` | `MemoryAdapter` for pretrained models (persistent forward hooks, GC-safe) |
| `quantization.py` | 4/8-bit memory bank quantisation (INT8 and packed INT4) |
| `utils.py` | Schedulers (cosine / linear / WSD), parameter breakdown, helpers |

### Training Infrastructure (`training/`)

| File | Purpose |
| :--- | :------ |
| `__init__.py` | Package init |
| `trainer.py` | Accelerate-based loop (DDP/FSDP, mixed precision, separate LR groups, eval, resume, init-from, early stopping, LR finder) |
| `data.py` | Pretrain + instruction datasets, tokenizer chat-template support, assistant-only label masking |
| `losses.py` | Router auxiliary loss aggregation (also lives in `router.py`; `losses.py` is a parallel utility) |

### Inference (`inference/`)

| File | Purpose |
| :--- | :------ |
| `__init__.py` | Package init |
| `generate.py` | KV-cached single + batch generation with explicit position_ids for right-padded batches |
| `merge.py` | LoRA merge, memory extraction, full-model quantisation (dynamic / fp16 / bf16 / fp8 / bnb 4-8bit), GGUF helper |
| `routing_strategies.py` | `Sequence` / `SequenceRolling` / `Rolling` / `Token` / `Hybrid` inference routers |

### Scripts (`scripts/`)

| File | Purpose |
| :--- | :------ |
| `train.py` | Training entry point (with `--resume` and `--init_from`) |
| `eval.py` | Validation perplexity (single + distributed) |
| `eval_mmlu.py` | MMLU multiple-choice accuracy (manual / dataset few-shot, label / choice_text scoring) |
| `eval_mcq_benchmark.py` | Generic MCQ evaluator |
| `eval_hellaswag.py` / `eval_arc.py` / `eval_winogrande.py` / `eval_boolq.py` / `eval_openbookqa.py` | Thin wrappers over `eval_mcq_benchmark.py` |
| `eval_triviaqa.py` | TriviaQA top-alias perplexity |
| `eval_pretrain_suite.py` | Run + aggregate the full benchmark suite |
| `generate_ifeval_jsonl.py` | Produce IFEval-format JSONL predictions for external scoring |
| `inference.py` | Generation script |
| `estimate_flops.py` | Analytic forward + backward + recompute FLOPs estimator (matches the paper's appendix calculation) |

### Configurations (`configs/`)

| File | Purpose |
| :--- | :------ |
| `base_small.yaml` | 16k-bank from-scratch config |
| `base_small_run2.yaml` | **Workshop paper config** (262k bank, 4097 chapters, top-k 64) |
| `vanilla_control.yaml` | Backbone-only-style baseline |
| `vanilla_control_run2.yaml` | iso-FLOP dense baseline matching the paper |
| `ift_base_model.yaml` | Instruction tuning from a pretrained MoC checkpoint |
| `ift_vanilla_model.yaml` | Instruction tuning from the iso-FLOP baseline |
| `ift_vanilla_model_small.yaml` | Instruction tuning from the backbone-only baseline |
| `adapter_qwen2.5_1.5b.yaml` | Memory adapter on Qwen2.5-1.5B |
| `memory_lora_combined.yaml` | Memory + LoRA combined adapter |
| `reference_all_options.yaml` | Documentation template — every config knob in one file |
| `README.md` | Full configuration reference |

### Documentation (`docs/`)

| File | Purpose |
| :--- | :------ |
| `README.md` | Documentation map |
| `architecture.md` | Detailed architecture with equations and diagrams |
| `design.md` | Design decisions and known limitations |
| `context.md` | This file |
| `philosophy.md` | Coding / configuration / documentation philosophy |
| `prompt.md` | Onboarding prompt for new sessions / agents |
| `meta_artifacts/` | Per-session logs and consolidated summary |

### Sparse Routing Kernels (`kernels-final/`)

| File | Purpose |
| :--- | :------ |
| `kernel_v1.py` | Unoptimised reference baseline |
| `kernel_v2.py` | Older optimised — **default**, most stable |
| `kernel_v3.py` | Old optimised alternate |
| `kernel_v4.py` | **Exact MoE-weighted fused kernel** — single launch, forward + dQ/dK/dV/dW |
| `kernel_v5.py` | Joint-bias weighted approximation (`logits += log(weight)`, single softmax) |
| `memory_cross_attn.py` | Earlier custom Triton chapter-routed cross-attention (forward + dKV strategies A/B/C/D) |
| `benchmark_kernels_final.py` | Correctness + timing benchmark (unweighted joint softmax & MoE-weighted independent softmax) |

The historic exploration archive lives in [`kernels/`](../kernels/).

---

## Key Design Decisions

### Decision 1: Training Library

- **Choice**: PyTorch + HuggingFace Transformers + Accelerate.
- **Rationale**: full control, FSDP/DDP support, HF model compatibility.

### Decision 2: Block Variant

- **Default**: Variant A (SA → Mem → MLP).
- **Configurable**: Variant B (SA → MLP → Mem → MLP).

### Decision 3: Routing Strategy

- **Default**: Sequence-level routing (mean-pool → softmax → top-k).
- **Optional**: Token-level routing for train and inference via `routing_strategy_*: token`.
- **Implementation**: Dense shared-chapter branch + sparse routed-chapter branch via the `kernels-final/` v1–v5 kernels (default `v2`).

### Decision 4: `W_o` Initialisation

- **Choice**: Zero initialisation for the memory-attention output projection.
- **Rationale**: model starts as if no memory exists; gradual learning of when/how to use memory; critical for stable training in **both** adapter and from-scratch paths.

### Decision 5: Adapter Injection

- **Choice**: Persistent PyTorch forward hooks registered lazily on the first `forward()` call.
- **Rationale**: non-invasive, works across HF architectures, survives `GradientCheckpointingLayer` backward recomputation.

### Decision 6: Shared Chapters + Routed Scaling (Session 10)

- **Choice**: Always include the first `num_shared_chapters` chapters (treated as 1.0-weighted shared knowledge) and scale routed chapter weights by `routed_scaling_factor` before mixing.
- **Rationale**: a 1-chapter shared prefix anchors common information; routed scaling (paper uses 2.5×) makes the model rely more on the sparse, content-specific routed chapters.

---

## What's NOT Implemented

| Feature | Reason | Future Priority |
| :------ | :----- | :-------------- |
| Dynamic context bank (inference-time updates) | Workshop scope; needs VAE + clustering + merging | Low (post-workshop) |
| Broad token-routing benchmark / policy retuning | Core path is implemented; broader shape/GPU validation pending | Medium |
| Unit tests | User requested to skip | Low |
| QAT for memory | Basic post-training quantisation only | Low |

---

## All Configuration Flags

### Memory Configuration (`memory:`)

```yaml
# Core toggles
vanilla_mode: bool = false           # Disable all memory
use_memory_adapter: bool = true      # Enable memory cross-attention
use_both_memory_and_lora: bool = false

# Memory bank
num_memory_tokens: int = 1024
memory_dim: int | null = null        # Default: model.hidden_dim
memory_num_heads: int | null = null  # Default: model.num_heads
memory_num_kv_heads: int | null = null  # Default: model.num_kv_heads

# Placement
memory_layer_placement: str = "all"  # all / first_k / last_k / every_n / custom / none
memory_layer_k: int = 5
memory_layer_n: int = 3
memory_layer_indices: list | null = null

# Sharing
memory_sharing: str = "shared"       # shared / per_layer / every_k_layers
memory_sharing_k: int = 2

# Block structure
memory_block_variant: str = "A"      # A or B
memory_dropout: float | null = null
memory_gradient_checkpointing: bool = true

# Low-rank
use_low_rank_memory: bool = false
memory_rank: int = 64
low_rank_mode: str = "factorized"    # factorized / reduced_dim
use_low_rank_projections: bool = false
projection_rank: int = 64

# Chapter routing
use_chapters: bool = false
num_chapters: int = 100
top_k_chapters: int = 20
num_shared_chapters: int = 0
routed_scaling_factor: float = 1.0
normalize_shared_routed_before_mixing: bool = false
shared_routed_norm_type: str = "rms"  # rms / layernorm
shared_routed_norm_eps: float = 1e-6
routing_strategy_train: str = "sequence"      # sequence / sequence-rolling / token
routing_strategy_inference: str = "sequence"  # + rolling / hybrid
routing_window_size: int = 128
token_routing_kernel_version: str = "v2"      # v1 / v2 / v3 / v4 / v5

# Router losses
use_load_balance_loss: bool = true
load_balance_coefficient: float = 0.01
use_auxiliary_loss: bool = false
auxiliary_loss_coefficient: float = 0.01
use_z_loss: bool = false
z_loss_coefficient: float = 0.001

# Quantisation
quantize_memory: bool = false
memory_quant_bits: int = 8

# Initialisation
wo_init_zero: bool = true
memory_init_std: float = 0.02

# LoRA
use_lora: bool = false
lora_rank: int = 16
lora_alpha: int = 32
lora_dropout: float = 0.05
lora_targets: list = ["q_proj", "v_proj"]
```

### Model Configuration (`model:`)

```yaml
hidden_dim: int = 768
num_heads: int = 12
num_kv_heads: int | null = null      # GQA KV heads
num_layers: int = 12
intermediate_dim: int = 3072
vocab_size: int = 32000
max_seq_len: int = 8192
max_position_embeddings: int | null = null  # HF-style alias

tokenizer_name: str | null = null
bos_token_id: int | null = null
eos_token_id: int | null = null
pad_token_id: int | null = null

use_rope: bool = true
rope_theta: float = 10000.0
dropout: float = 0.0
attention_dropout: float = 0.0
hidden_activation: str = "swiglu"    # swiglu / silu / relu / gelu / sigmoid / tanh
initializer_range: float = 0.02
self_attn_wo_init_std: float | null = null   # Optional override
mlp_down_proj_init_std: float | null = null  # Optional override

norm_eps: float = 1e-6
use_rms_norm: bool = true

base_model_name: str | null = null   # For adapter mode
freeze_base_model: bool = true
use_flash_attention: bool = true
tie_embeddings: bool = true
```

### Training Configuration (`training:`)

```yaml
# Learning rates
memory_lr: float = 1e-4
memory_bank_lr: float | null = null  # null => use memory_lr
lora_lr: float = 1e-4
base_model_lr: float = 1e-5

# Mode
training_mode: str = "instruction_finetuning"  # or "pretraining"

# Dataset
dataset_name: str
dataset_subset: str | null = null
dataset_split: str = "train"
eval_split: str = "test"
text_field: str | list = "messages"
max_length: int = 8192

# Distributed
distributed_strategy: str = "ddp"    # ddp / fsdp
num_gpus: int = 1
fsdp_sharding_strategy: str = "FULL_SHARD"

# Hyperparameters
batch_size: int = 4
gradient_accumulation_steps: int = 4
num_epochs: int | null = null
max_steps: int = 10000
warmup_steps: int = 100
warmup_ratio: float | null = null

# Optimizer
optimizer: str = "adamw"
weight_decay: float = 0.01
adam_beta1: float = 0.9
adam_beta2: float = 0.95
adam_epsilon: float = 1e-8
max_grad_norm: float = 1.0

# Scheduler
scheduler: str = "cosine"            # cosine / linear / constant / wsd
min_lr_ratio: float = 0.1
decay_start_step: int | null = null
decay_start_ratio: float | null = null
wsd_stable_steps: int | null = null
wsd_stable_ratio: float = 0.0

# Mixed precision
mixed_precision: str = "bf16"        # no / fp16 / bf16
save_precision: str | null = null    # fp32 / fp16 / bf16

# Checkpointing
gradient_checkpointing: bool = true
save_steps: int = 500
eval_steps: int = 500
save_total_limit: int | null = 3     # null => keep all
save_best_model: bool = true

# Early stopping
early_stopping: bool = false
early_stopping_patience: int = 5
early_stopping_threshold: float = 0.0

# Logging
logging_steps: int = 10
log_to_wandb: bool = false
wandb_project: str = "memory-transformer"
wandb_run_name: str | null = null

# Output
output_dir: str = "./outputs"
resume_from_checkpoint: str | null = null  # Full state restore
init_from_checkpoint: str | null = null    # Weights-only fresh start
```

---

## File Dependencies Graph

```text
memory_transformer/
├── config.py              ← No internal deps
├── memory_bank.py         ← No internal deps
├── token_routing_kernel.py← No internal deps (loads kernels-final/* lazily)
├── memory_attention.py    ← token_routing_kernel
├── router.py              ← No internal deps
├── lora.py                ← No internal deps
├── quantization.py        ← No internal deps
├── memory_block.py        ← memory_attention
├── model.py               ← config, memory_bank, memory_block, router, token_routing_kernel
├── adapter.py             ← config, memory_bank, memory_attention, router, lora, token_routing_kernel
└── utils.py               ← No internal deps

training/
├── data.py                ← No internal deps
├── losses.py              ← No internal deps
└── trainer.py             ← config, model, adapter, utils, data

inference/
├── generate.py            ← No internal deps
├── merge.py               ← memory_transformer (lora, quantization, memory_bank)
└── routing_strategies.py  ← No internal deps

scripts/
├── train.py               ← config, trainer
├── eval.py                ← config, model, adapter, data
├── eval_mmlu.py           ← config, model, adapter, tokenizer, HF datasets
├── eval_mcq_benchmark.py  ← config, model, adapter, tokenizer, HF datasets
├── eval_pretrain_suite.py ← orchestrates other eval_* scripts via subprocess
├── eval_triviaqa.py       ← config, model, adapter, tokenizer, HF datasets
├── generate_ifeval_jsonl.py ← config, model, adapter, generate
├── estimate_flops.py      ← config (analytic only, no model load)
└── inference.py           ← config, model, adapter, generate
```

---

## Running Commands

### Training

```bash
# Single GPU
python scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml

# DDP × 8 (paper config)
accelerate launch --num_processes 8 scripts/train.py --config configs/base_small_run2.yaml

# FSDP
accelerate launch --num_processes 4 --use_fsdp scripts/train.py --config configs/base_small.yaml

# Resume full state
python scripts/train.py --config ... --resume outputs/.../checkpoint-1000

# Weights-only restart (e.g., pretrain → IFT)
python scripts/train.py --config configs/ift_base_model.yaml --init_from outputs/base_small_run5/final_model
```

### Evaluation

```bash
python scripts/eval.py --config configs/... --checkpoint outputs/final_model

python scripts/eval_mmlu.py --config configs/... --checkpoint outputs/final_model \
    --shots 5 --fewshot_mode manual --scoring_mode choice_text

python scripts/eval_pretrain_suite.py --config configs/... --checkpoint outputs/final_model
```

### Inference

```bash
python scripts/inference.py --checkpoint outputs/final_model --prompt "Your prompt"
```

### Analytic FLOPs estimate

```bash
python scripts/estimate_flops.py --config configs/base_small_run2.yaml
```

---

## Comparison Experiments

The four obvious adapter variants share infrastructure:

1. [`configs/vanilla_control.yaml`](../configs/vanilla_control.yaml) — no-memory baseline.
2. [`configs/adapter_qwen2.5_1.5b.yaml`](../configs/adapter_qwen2.5_1.5b.yaml) — memory adapter only.
3. Modify a config: `use_lora=true, use_memory_adapter=false` — LoRA only.
4. [`configs/memory_lora_combined.yaml`](../configs/memory_lora_combined.yaml) — Memory + LoRA combined.

For from-scratch comparisons (matching the paper):

1. [`configs/vanilla_control_run2.yaml`](../configs/vanilla_control_run2.yaml) — iso-FLOP dense baseline.
2. [`configs/base_small_run2.yaml`](../configs/base_small_run2.yaml) — MoC.

---

## Next Steps

1. Run training experiments with provided configs.
2. Compare vanilla vs memory vs LoRA perplexity and benchmark accuracy.
3. Tune hyperparameters based on results.
4. Run the benchmark suite and log results (`scripts/eval_pretrain_suite.py`, plus per-benchmark scripts).
5. Document results in `walkthrough.md` for the relevant session.
6. **Future**: token-routing benchmark/policy tuning, dynamic context bank, larger-bank scaling laws.

---

## Session History

See [`docs/meta_artifacts/session_summary.md`](meta_artifacts/session_summary.md) for cumulative summaries, [`docs/meta_artifacts/session10/session.md`](meta_artifacts/session10/session.md) for the latest detailed log, and [`docs/meta_artifacts/session1/session.md`](meta_artifacts/session1/session.md) for earlier historical depth, including:

- All decisions made and rationale
- Every file created with descriptions
- Issues encountered and resolutions
- Complete timeline of work

---

## References

| Document | Purpose |
| :------- | :------ |
| [`idea/idea.txt`](../idea/idea.txt) | Original conceptual explanation |
| [`idea/proposal.md`](../idea/proposal.md) | Long-form research proposal |
| [`idea/main.tex`](../idea/main.tex) | LaTeX paper draft |
| [`idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf`](../idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf) | Workshop paper (NFAM @ ICLR 2026) |
| [`idea/Presentation_MoC_BTP.pdf`](../idea/Presentation_MoC_BTP.pdf) | BTP presentation slides |
| [`idea/MoC Arch Diagram Excalidraw.png`](../idea/MoC%20Arch%20Diagram%20Excalidraw.png) | Architecture diagram |
| [`docs/architecture.md`](architecture.md) | Detailed architecture explanation |
| [`docs/design.md`](design.md) | Design decisions and limitations |
| [`docs/philosophy.md`](philosophy.md) | Development philosophy and style |
| [`docs/meta_artifacts/session_summary.md`](meta_artifacts/session_summary.md) | Cumulative session summaries |
