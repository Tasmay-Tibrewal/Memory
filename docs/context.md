# Memory-Augmented Transformer: Context for Handoffs

This document provides a comprehensive summary for future work, session handoffs, context compaction, or agent transitions. It is designed to give any reader the complete picture of the project state.

---

## Project Overview

| Field | Value |
|-------|-------|
| **Goal** | Implement memory-augmented transformers with learnable cross-attention memory banks |
| **Repository** | `Memory/` |
| **Status** | ✅ **COMPLETE** - All requirements implemented, ready for experiments |
| **Session** | Sessions 1â€“7 (implementation + fixes + verification + doc refresh) |
| **Date** | February 5, 2026 |

---

## Quick Start for New Sessions

```bash
# 1. Read this document for context
# 2. Check meta_artifacts/session_summary.md for session history
# 3. Run training with:
python scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml

# Or multi-GPU:
accelerate launch scripts/train.py --config configs/base_small.yaml
```

---

## What's Implemented

### File Count Summary
| Directory | Files | Lines of Code (approx) |
|-----------|-------|------------------------|
| `memory_transformer/` | 11 | ~3,550 |
| `training/` | 4 | ~975 |
| `inference/` | 4 | ~840 |
| `scripts/` | 3 | ~450 |
| `configs/` | 5 | ~670 |
| `docs/` | 5 | ~1,420 |
| `docs/meta_artifacts/` | 3+ folders | ~2,400+ |
| **Total** | **35+** | **~10,700+** |

### Core Architecture (`memory_transformer/`)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 40 | Package exports with lazy loading |
| `config.py` | 330 | Configuration system (50+ options) |
| `memory_bank.py` | 306 | Memory bank variants (Standard, Factorized, ReducedDim, Chaptered) |
| `memory_attention.py` | 502 | Cross-attention for memory access |
| `memory_block.py` | 475 | Transformer blocks (Variant A/B, Vanilla) |
| `router.py` | 309 | Chapter routing with MoE losses |
| `lora.py` | 248 | Standard LoRA implementation |
| `model.py` | 438 | Full MemoryTransformer |
| `adapter.py` | 540 | MemoryAdapter for pretrained models |
| `quantization.py` | 167 | Memory bank quantization |
| `utils.py` | 198 | Utilities and helpers |

### Training Infrastructure (`training/`)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 6 | Package init |
| `trainer.py` | 698 | Training loop with Accelerate |
| `data.py` | 215 | Dataset loading (any HF dataset) |
| `losses.py` | 54 | Router auxiliary losses |

### Inference (`inference/`)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 9 | Package init |
| `generate.py` | 218 | Text generation utilities |
| `merge.py` | 377 | Model merging and quantization |
| `routing_strategies.py` | 234 | Inference routing (sequence/rolling/token/hybrid) |

### Scripts (`scripts/`)

| File | Lines | Purpose |
|------|-------|---------|
| `train.py` | 45 | Training entry point |
| `eval.py` | 261 | Evaluation (perplexity) |
| `inference.py` | 142 | Generation script |

### Configurations (`configs/`)

| File | Purpose |
|------|---------|
| `README.md` | Complete config reference (~300 lines) |
| `base_small.yaml` | From-scratch small model (768d, 12L) |
| `adapter_qwen2.5_1.5b.yaml` | Qwen adapter with dataset suggestions |
| `vanilla_control.yaml` | Control experiment (no memory) |
| `memory_lora_combined.yaml` | Memory + LoRA combined |

### Documentation (`docs/`)

| File | Lines | Purpose |
|------|-------|---------|
| `README.md` | 225 | Documentation overview |
| `architecture.md` | 370 | Detailed architecture with diagrams |
| `design.md` | 188 | Design decisions and rationale |
| `context.md` | This file | Handoff summary |
| `philosophy.md` | 400 | Development philosophy and style |
| `meta_artifacts/session_summary.md` | 240+ | Session summaries |

---

## Key Design Decisions

### Decision 1: Training Library
- **Choice**: PyTorch + HuggingFace Transformers + Accelerate
- **Rationale**: Full control, FSDP/DDP support, HF model compatibility
- **Alternative rejected**: Unsloth (too opinionated, limited custom attention)

### Decision 2: Block Variant
- **Default**: Variant A (SA → Mem → MLP)
- **Configurable**: Variant B (SA → MLP → Mem → MLP)
- **Rationale**: A is simpler, matches cross-attention patterns

### Decision 3: Routing Strategy
- **Training**: Sequence-level only (mean-pool)
- **Inference**: Sequence/Rolling/Token/Hybrid options
- **Rationale**: Token-level during prefill is memory prohibitive (~150TB)

### Decision 4: W_o Initialization
- **Choice**: Zero initialization
- **Rationale**: Model starts as if no memory, gradual learning
- **Critical for**: Adapter and from-scratch stability

### Decision 5: Adapter Injection
- **Choice**: PyTorch hooks
- **Rationale**: Non-invasive, works across architectures

---

## What's NOT Implemented

| Feature | Reason | Future Priority |
|---------|--------|-----------------|
| Dynamic context bank | Post-workshop (VAE, clustering needed) | Low |
| Token-level routing (prefill) | Needs custom CUDA kernel | Medium |
| Unit tests | User requested to skip | Low |
| QAT for memory | Basic quantization only | Low |

---

## All Configuration Flags

### Memory Configuration (`memory:`)
```yaml
# Core toggles
vanilla_mode: bool = false           # Disable all memory
use_memory_adapter: bool = true      # Enable memory cross-attention

# Memory bank
num_memory_tokens: int = 1024        # Memory size
memory_dim: int = null               # Default: hidden_dim

# Placement
memory_layer_placement: str = "all"  # all/first_k/last_k/every_n/custom
memory_layer_k: int = 5              # For first_k/last_k
memory_layer_n: int = 3              # For every_n
memory_layer_indices: list = null    # For custom

# Sharing
memory_sharing: str = "shared"       # shared/per_layer/every_k_layers
memory_sharing_k: int = 2            # For every_k_layers

# Block structure
memory_block_variant: str = "A"      # A or B
memory_dropout: float = null         # Memory cross-attn dropout (null => model.dropout)

# Low-rank
use_low_rank_memory: bool = false
memory_rank: int = 64
low_rank_mode: str = "factorized"    # factorized/reduced_dim
use_low_rank_projections: bool = false
projection_rank: int = 64

# Chapters
use_chapters: bool = false
num_chapters: int = 100
# tokens_per_chapter is auto-calculated as num_memory_tokens // num_chapters
top_k_chapters: int = 20
routing_strategy_train: str = "sequence"
routing_strategy_inference: str = "sequence"  # sequence/rolling/token/hybrid
routing_window_size: int = 128                # For rolling/hybrid inference

# Router losses
use_load_balance_loss: bool = true
load_balance_coefficient: float = 0.01
use_auxiliary_loss: bool = false
auxiliary_loss_coefficient: float = 0.01
use_z_loss: bool = false
z_loss_coefficient: float = 0.001

# Memory gradient checkpointing
memory_gradient_checkpointing: bool = true

# Quantization
quantize_memory: bool = false
memory_quant_bits: int = 8

# Initialization
wo_init_zero: bool = true
memory_init_std: float = 0.02

# LoRA
use_lora: bool = false
lora_rank: int = 16
lora_alpha: int = 32
lora_dropout: float = 0.05
lora_targets: list = ["q_proj", "v_proj"]
use_both_memory_and_lora: bool = false
```

### Model Configuration (`model:`)
```yaml
hidden_dim: int = 768
num_heads: int = 12
num_layers: int = 12
intermediate_dim: int = 3072
vocab_size: int = 32000
max_seq_len: int = 8192
tokenizer_name: str = null          # Optional override; used for from-scratch too
use_rope: bool = true
rope_theta: float = 10000.0
dropout: float = 0.0
attention_dropout: float = 0.0
norm_eps: float = 1e-6
use_rms_norm: bool = true
base_model_name: str = null          # For adapter mode
freeze_base_model: bool = true
use_flash_attention: bool = true
```

### Training Configuration (`training:`)
```yaml
# Learning rates
memory_lr: float = 1e-4
lora_lr: float = 1e-4
base_model_lr: float = 1e-5

# Training mode
training_mode: str = "instruction_finetuning"  # or "pretraining"

# Dataset
dataset_name: str = "..."
dataset_subset: str = null
dataset_split: str = "train"
eval_split: str = "test"
text_field: str = "messages"
max_length: int = 8192

# Distributed
distributed_strategy: str = "ddp"    # ddp/fsdp
num_gpus: int = 1
fsdp_sharding_strategy: str = "FULL_SHARD"

# Hyperparameters
batch_size: int = 4
gradient_accumulation_steps: int = 4
max_steps: int = 10000
warmup_steps: int = 100

# Optimizer
optimizer: str = "adamw"
weight_decay: float = 0.01
max_grad_norm: float = 1.0

# Scheduler
scheduler: str = "cosine"
min_lr_ratio: float = 0.1

# Mixed precision
mixed_precision: str = "bf16"
save_precision: str = null       # Optional: fp32/fp16/bf16

# Checkpointing
gradient_checkpointing: bool = true
save_steps: int = 500
eval_steps: int = 500
save_total_limit: int = 3
save_best_model: bool = true

# Early stopping
early_stopping: bool = false
early_stopping_patience: int = 5
early_stopping_threshold: float = 0.0

# Logging
logging_steps: int = 10
log_to_wandb: bool = false
wandb_run_name: str = null

# Output
output_dir: str = "./outputs"
resume_from_checkpoint: str = null
```

---

## File Dependencies Graph

```
memory_transformer/
├── config.py              ← No internal deps
├── memory_bank.py         ← No internal deps
├── memory_attention.py    ← No internal deps
├── router.py              ← No internal deps
├── lora.py                ← No internal deps
├── quantization.py        ← No internal deps
├── memory_block.py        ← memory_attention
├── model.py               ← config, memory_bank, memory_block, router
├── adapter.py             ← config, memory_bank, memory_attention, router, lora
└── utils.py               ← No internal deps

training/
├── data.py                ← No internal deps
├── losses.py              ← No internal deps
└── trainer.py             ← config, model, adapter, utils, data

inference/
├── generate.py            ← No internal deps
└── routing_strategies.py  ← No internal deps

scripts/
├── train.py               ← config, trainer
├── eval.py                ← config, model, adapter, data
└── inference.py           ← config, model, adapter, generate
```

---

## Running Commands

### Training
```bash
# Single GPU
python scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml

# Multi-GPU (DDP)
accelerate launch --num_processes 4 scripts/train.py --config configs/base_small.yaml

# Multi-GPU (FSDP)
accelerate launch --num_processes 4 --use_fsdp scripts/train.py --config configs/base_small.yaml

# Resume from checkpoint
python scripts/train.py --config ... --resume outputs/checkpoint-1000
```

### Evaluation
```bash
python scripts/eval.py --config configs/... --checkpoint outputs/final_model
```

### Inference
```bash
python scripts/inference.py --checkpoint outputs/final_model --prompt "Your prompt"
```

---

## Comparison Experiments

Run these 4 configs to compare approaches:
1. `vanilla_control.yaml` - No memory baseline
2. `adapter_qwen2.5_1.5b.yaml` - Memory adapter only
3. Modify config: `use_lora=true, use_memory_adapter=false` - LoRA only
4. `memory_lora_combined.yaml` - Both combined

---

## Next Steps for Continuation

1. **Run training experiments** with provided configs
2. **Compare** vanilla vs memory vs LoRA perplexity
3. **Tune hyperparameters** based on results
4. **Add benchmarks** (specific eval tasks)
5. **Document results** in walkthrough.md
6. **Future**: Token-level routing CUDA kernel, dynamic context bank

---

## Session History

See `docs/meta_artifacts/session_summary.md` for session summaries and `docs/meta_artifacts/session1/session.md` for detailed development logs including:
- All decisions made and rationale
- Every file created with descriptions
- Issues encountered and resolutions
- Complete timeline of work

---

## References

| Document | Purpose |
|----------|---------|
| `idea/idea.txt` | Original conceptual explanation |
| `idea/main.tex` | LaTeX paper draft |
| `docs/architecture.md` | Detailed architecture explanation |
| `docs/design.md` | Design decisions and limitations |
| `docs/philosophy.md` | Development philosophy and style |
| `docs/meta_artifacts/session_summary.md` | Session summaries |
