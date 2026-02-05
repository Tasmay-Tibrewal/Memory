# Configuration Files

This directory contains example configuration files for training memory-augmented transformers.

## Available Configs

| Config File | Purpose | Model Type |
|-------------|---------|------------|
| `base_small.yaml` | From-scratch pretraining | MemoryTransformer |
| `adapter_qwen2.5_1.5b.yaml` | Memory adapter on Qwen | MemoryAdapter |
| `vanilla_control.yaml` | Control experiment (no memory) | MemoryTransformer |
| `memory_lora_combined.yaml` | Memory + LoRA combined | MemoryAdapter |

---

## Config Structure

Each config file has three main sections:

```yaml
model:      # Model architecture settings
memory:     # Memory bank configuration  
training:   # Training hyperparameters and dataset
```

---

## Complete Configuration Reference

### Model Configuration (`model:`)

```yaml
model:
  # === Architecture (for from-scratch) ===
  hidden_dim: 768              # Hidden dimension
  num_heads: 12                # Number of attention heads
  num_layers: 12               # Number of transformer layers
  intermediate_dim: 3072       # MLP intermediate dimension
  vocab_size: 32000            # Vocabulary size
  max_seq_len: 8192            # Maximum sequence length
  
  # === Positional Encoding ===
  use_rope: true               # Use Rotary Position Embedding
  rope_theta: 10000.0          # RoPE theta parameter
  
  # === Regularization ===
  dropout: 0.0                 # Dropout rate
  attention_dropout: 0.0       # Attention dropout
  
  # === Normalization ===
  norm_eps: 1e-6               # Normalization epsilon
  use_rms_norm: true           # RMSNorm vs LayerNorm
  
  # === For Adapter Mode ===
  base_model_name: null        # HuggingFace model name (e.g., "Qwen/Qwen2.5-1.5B")
  freeze_base_model: true      # Freeze base model parameters
  
  # === Performance ===
  use_flash_attention: true    # Use Flash Attention if available
```

---

### Memory Configuration (`memory:`)

```yaml
memory:
  # === Main Toggle ===
  vanilla_mode: false          # Set true to disable ALL memory (control experiment)
  
  # === Memory Bank Settings ===
  num_memory_tokens: 1024      # Number of memory tokens (N_m)
  memory_dim: null             # Memory dimension (null = use hidden_dim)
  
  # === Layer Placement ===
  memory_layer_placement: all  # Where to put memory layers
                               # Options: "all", "first_k", "last_k", "every_n", "custom", "none"
  memory_layer_k: 5            # For first_k/last_k: number of layers
  memory_layer_n: 3            # For every_n: interval
  memory_layer_indices: null   # For custom: explicit list [0, 1, 5, 10]
  
  # === Memory Sharing ===
  memory_sharing: shared       # How layers share memory banks
                               # Options: "shared", "per_layer", "every_k_layers"
  memory_sharing_k: 2          # For every_k_layers: group size
  
  # === Block Integration ===
  memory_block_variant: A      # Block structure
                               # "A": Self-Attn → Memory → MLP
                               # "B": Self-Attn → MLP → Memory → MLP
  
  # === Low-Rank Memory ===
  use_low_rank_memory: false   # Enable low-rank memory bank
  memory_rank: 64              # Rank for low-rank decomposition
  low_rank_mode: factorized    # "factorized" (M=AB^T) or "reduced_dim"
  
  use_low_rank_projections: false  # Low-rank Q,K,V projections
  projection_rank: 64              # Projection rank
  
  # === Chapter Routing (MoE-style) ===
  use_chapters: false          # Enable chapter-based routing
  num_chapters: 100            # Number of chapters
  tokens_per_chapter: null     # Auto-calculated if null
  top_k_chapters: 20           # Chapters to select per forward pass
  
  routing_strategy_train: sequence      # Training: always sequence-level
  routing_strategy_inference: sequence  # Inference: "sequence", "rolling", "token"
  
  # === Router Losses ===
  use_load_balance_loss: true        # Load balancing loss
  load_balance_coefficient: 0.01     # Loss coefficient
  
  use_auxiliary_loss: false          # Auxiliary loss
  auxiliary_loss_coefficient: 0.01
  
  use_z_loss: false                  # Z-loss (router regularization)
  z_loss_coefficient: 0.001
  
  # === Quantization ===
  quantize_memory: false       # Quantize memory bank
  memory_quant_bits: 8         # Quantization bits (4 or 8)
  
  # === Initialization ===
  wo_init_zero: true           # Zero-init output projection (critical for stable training; adapter and from-scratch)
  memory_init_std: 0.02        # Memory token initialization std
  
  # === LoRA Settings ===
  use_lora: false              # Enable standard LoRA
  lora_rank: 16                # LoRA rank
  lora_alpha: 32               # LoRA alpha
  lora_dropout: 0.05           # LoRA dropout
  lora_targets:                # Layers to apply LoRA
    - q_proj
    - v_proj
  
  # === Mode Flags ===
  use_memory_adapter: true           # Enable memory cross-attention
  use_both_memory_and_lora: false    # Enable both for combined experiments
```

---

### Training Configuration (`training:`)

```yaml
training:
  # === Learning Rates (separate for components) ===
  memory_lr: 1e-4              # Memory parameters LR
  lora_lr: 1e-4                # LoRA parameters LR
  base_model_lr: 1e-5          # Base model LR (if not frozen)
  
  # === Training Mode ===
  training_mode: instruction_finetuning  # or "pretraining"
  
  # === Dataset ===
  dataset_name: HuggingFaceH4/ultrachat_200k  # HuggingFace dataset
  dataset_subset: null         # Dataset config (e.g., "en" for C4)
  dataset_split: train         # Training split
  eval_split: test             # Evaluation split
  text_field: messages         # Field containing text (string or list)
  max_length: 8192             # Max sequence length
  
  # === Distributed Training ===
  distributed_strategy: ddp    # "ddp" or "fsdp"
  num_gpus: 1                  # Number of GPUs
  fsdp_sharding_strategy: FULL_SHARD  # FSDP sharding
  
  # === Training Hyperparameters ===
  batch_size: 4                # Per-GPU batch size
  gradient_accumulation_steps: 4
  num_epochs: null             # If set, overrides max_steps
  max_steps: 10000             # Training steps
  warmup_steps: 100            # Warmup steps
  warmup_ratio: null           # If set, overrides warmup_steps
  
  # === Optimizer ===
  optimizer: adamw             # "adamw", "adam", "sgd"
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1e-8
  max_grad_norm: 1.0           # Gradient clipping
  
  # === Scheduler ===
  scheduler: cosine            # "cosine", "linear", "constant"
  min_lr_ratio: 0.1            # Min LR as ratio of peak
  
  # === Mixed Precision ===
  mixed_precision: bf16        # "no", "fp16", "bf16"
  save_precision: fp32         # Optional: "fp32", "fp16", "bf16"
  
  # === Checkpointing ===
  gradient_checkpointing: true # Reduce memory usage
  save_steps: 500              # Save checkpoint every N steps
  eval_steps: 500              # Evaluate every N steps
  save_total_limit: 3          # Keep only N recent checkpoints
  save_best_model: true        # Save model with best eval loss
  
  # === Early Stopping ===
  early_stopping: false        # Stop if no improvement
  early_stopping_patience: 3   # Evals to wait
  early_stopping_threshold: 0.0 # Min improvement required
  
  # === Logging ===
  logging_steps: 10            # Log every N steps
  log_to_wandb: false          # Enable Weights & Biases
  wandb_project: memory-transformer
  wandb_run_name: null         # Auto-generated if null
  
  # === Output ===
  output_dir: ./outputs        # Output directory
  resume_from_checkpoint: null # Path to resume from
```

---

## Recommended Configurations

### For Quick Experiments (Small Model)
```yaml
model:
  hidden_dim: 512
  num_heads: 8
  num_layers: 6

memory:
  num_memory_tokens: 512
  use_chapters: false

training:
  max_steps: 1000
  batch_size: 8
```

### For Adapter Fine-tuning
```yaml
model:
  base_model_name: Qwen/Qwen2.5-1.5B
  freeze_base_model: true

memory:
  num_memory_tokens: 2048
  memory_layer_placement: custom
  memory_layer_indices: [0, 1, 2, 3, 4, 23, 24, 25, 26, 27]
  use_low_rank_memory: true
  memory_rank: 256
```

### For Large Memory Banks (100k+ tokens)
```yaml
memory:
  num_memory_tokens: 100000
  use_chapters: true
  num_chapters: 1000
  top_k_chapters: 10
  use_load_balance_loss: true
```

### For Comparison Experiments
```yaml
# Config 1: Vanilla (no memory)
memory:
  vanilla_mode: true

# Config 2: Memory only
memory:
  use_memory_adapter: true
  use_lora: false

# Config 3: LoRA only
memory:
  use_memory_adapter: false
  use_lora: true

# Config 4: Both combined
memory:
  use_both_memory_and_lora: true
```

---

## Dataset Examples

### Pretraining Datasets
```yaml
# C4 (Common Crawl)
dataset_name: allenai/c4
dataset_subset: en
text_field: text
training_mode: pretraining

# SlimPajama
dataset_name: cerebras/SlimPajama-627B
text_field: text
training_mode: pretraining

# Wikipedia
dataset_name: wikipedia
dataset_subset: 20220301.en
text_field: text
training_mode: pretraining
```

### Instruction Tuning Datasets
```yaml
# UltraChat (general)
dataset_name: HuggingFaceH4/ultrachat_200k
dataset_split: train_sft
text_field: messages
training_mode: instruction_finetuning

# Code (CodeAlpaca)
dataset_name: sahil2801/CodeAlpaca-20k
text_field: ["instruction", "input", "output"]
training_mode: instruction_finetuning

# Math (R1 Distill)
dataset_name: open-r1/r1-distill-math
text_field: messages
training_mode: instruction_finetuning
```

---

## Usage

```bash
# Single GPU
python scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml

# Multi-GPU with Accelerate
accelerate launch scripts/train.py --config configs/base_small.yaml

# Evaluate
python scripts/eval.py --config configs/adapter_qwen2.5_1.5b.yaml --checkpoint outputs/final_model
```
