# Training Infrastructure

This package provides the training loop and data loading utilities for memory-augmented transformers.

## Module Overview

```
training/
├── __init__.py    # Package exports
├── trainer.py     # Main training loop with Accelerate
├── data.py        # Dataset loading and preprocessing
└── losses.py      # Router auxiliary losses
```

---

## Detailed Module Documentation

### `trainer.py` - Training Loop

**Purpose**: Complete training infrastructure with distributed training support.

**Class**: `Trainer`

**Features**:
- **Accelerate Integration**: Automatic DDP/FSDP handling
- **Mixed Precision**: bf16/fp16 with automatic scaling
- **Gradient Checkpointing**: Reduce memory usage
- **Separate Learning Rates**: Different LRs for memory, LoRA, base model
- **WandB Logging**: Optional experiment tracking
- **Checkpointing**: Resume from saved states
- **Eval During Training**: Periodic evaluation on validation set
- **Early Stopping**: Stop training when validation loss stops improving
- **Best Model Saving**: Automatically save model with best validation loss
- **Checkpoint Cleanup**: Keep only N most recent checkpoints
- **Learning Rate Finder**: Find optimal learning rate before training

**Constructor**:
```python
trainer = Trainer(
    config: Config,           # Full configuration
    model: Optional[nn.Module] = None,  # Use existing model or create new
)
```

**Training Flow**:
```python
# 1. Initialize
trainer = Trainer(config)

# 2. Train
trainer.train()  # Runs full training loop

# Training loop internals:
# - Loads batch from dataloader
# - Forward pass with gradient accumulation
# - Backward pass with gradient clipping
# - Optimizer step with scheduler
# - Logging and checkpointing
```

**Checkpoint Structure**:
```
outputs/checkpoint-1000/
├── model.pt            # Consolidated model weights (inference convenience)
├── config.yaml         # Full configuration
├── trainer_state.json  # Trainer metadata (step/epoch/best_loss)
├── model.safetensors   # Accelerate model weights (or FSDP variants like fsdp_model.bin)
├── optimizer.bin       # Optimizer state (FSDP-aware)
└── scheduler.bin       # Scheduler state
```

**Resume Training**:
```python
config.training.resume_from_checkpoint = "outputs/checkpoint-1000"
trainer = Trainer(config)
trainer.train()  # Continues from step 1000
```

---

### `data.py` - Dataset Loading

**Purpose**: Flexible dataset loading supporting any HuggingFace dataset.

**Class**: `TextDataset`

**Supported Modes**:

| Mode | Description | Data Format |
|------|-------------|-------------|
| `pretraining` | Raw text continuation | `{"text": "..."}` |
| `instruction_finetuning` | Chat/conversation format | `{"messages": [...]}` |

**Chat Format Support**:
- `messages` field: Standard chat format
- `conversations` field: Alternative format
- `prompt`/`response` fields: Simple Q/A format
- Automatic chat template application

**Factory Function**:
```python
from training.data import create_dataloader

dataloader = create_dataloader(
    dataset_name="HuggingFaceH4/ultrachat_200k",
    tokenizer=tokenizer,
    batch_size=4,
    max_length=4096,
    split="train_sft",
    subset=None,              # For datasets with configs
    text_field="messages",    # Can be string or list
    training_mode="instruction_finetuning",
    num_workers=4,
    shuffle=True,
    num_samples=None,         # Limit for testing
)
```

**Dataset Configuration Examples**:

```yaml
# Pretraining on C4
training:
  dataset_name: allenai/c4
  dataset_subset: en
  text_field: text
  training_mode: pretraining

# Instruction tuning on UltraChat
training:
  dataset_name: HuggingFaceH4/ultrachat_200k
  dataset_split: train_sft
  text_field: messages
  training_mode: instruction_finetuning

# Code instruction tuning
training:
  dataset_name: sahil2801/CodeAlpaca-20k
  text_field: ["instruction", "input", "output"]  # Multiple fields
  training_mode: instruction_finetuning
```

---

### `losses.py` - Router Losses

**Purpose**: Aggregate router auxiliary losses from multiple layers.

**Function**:
```python
from training.losses import compute_router_auxiliary_loss

total_loss = compute_router_auxiliary_loss(
    router_losses=[                    # List of dicts from each router
        {"load_balance": 0.01, "z_loss": 0.001},
        {"load_balance": 0.02, "z_loss": 0.002},
    ],
    load_balance_coef=0.01,           # Weight for load balance loss
    auxiliary_coef=0.0,               # Weight for auxiliary loss
    z_loss_coef=0.001,                # Weight for z-loss
)
```

**Loss Descriptions**:

| Loss | Formula | Purpose |
|------|---------|---------|
| Load Balance | `C × Σ(f_i × P_i)` | Uniform chapter usage |
| Auxiliary | `Σ(f_i - 1/C)²` | Penalize deviation from uniform |
| Z-Loss | `mean(log²(Σ exp(logits)))` | Regularize router outputs |

---

## Usage Examples

### Basic Training

```python
from memory_transformer import load_config
from training import Trainer

# Load config and train
config = load_config("configs/adapter_qwen2.5_1.5b.yaml")
trainer = Trainer(config)
trainer.train()
```

### Custom Model Training

```python
from memory_transformer import MemoryAdapter, load_config
from training import Trainer

# Create custom model
config = load_config("configs/adapter_qwen2.5_1.5b.yaml")
model = MemoryAdapter(config)

# Modify model if needed
# ...

# Train with custom model
trainer = Trainer(config, model=model)
trainer.train()
```

### Multi-GPU Training

```bash
# DDP (Data Distributed Parallel)
accelerate launch --num_processes 4 scripts/train.py --config configs/base_small.yaml

# FSDP (Fully Sharded Data Parallel)
accelerate launch --num_processes 4 \
    --use_fsdp \
    scripts/train.py --config configs/base_small.yaml
```

Tip: You can also set `training.distributed_strategy: fsdp` and
`training.fsdp_sharding_strategy: FULL_SHARD` in your config; you still need to
launch with `accelerate` for multi-process training.

---

## Configuration Reference

### Training Config Fields

```yaml
training:
  # Learning rates (separate for components)
  memory_lr: 2e-4
  lora_lr: 1e-4
  base_model_lr: 1e-5

  # Training mode
  training_mode: instruction_finetuning  # or "pretraining"

  # Dataset
  dataset_name: HuggingFaceH4/ultrachat_200k
  dataset_subset: null
  dataset_split: train_sft
  text_field: messages
  max_length: 4096

  # Distributed
  distributed_strategy: ddp  # or "fsdp"
  num_gpus: 1
  fsdp_sharding_strategy: FULL_SHARD

  # Hyperparameters
  batch_size: 4
  gradient_accumulation_steps: 4
  max_steps: 10000
  warmup_steps: 100

  # Optimizer
  optimizer: adamw
  weight_decay: 0.01
  max_grad_norm: 1.0

  # Scheduler
  scheduler: cosine  # or "linear", "constant"

  # === Mixed Precision ===
  mixed_precision: bf16        # "no", "fp16", "bf16"
  save_precision: fp32         # Optional: "fp32", "fp16", "bf16"

  # === Checkpointing ===
  gradient_checkpointing: true
  save_steps: 500              # Save checkpoint every N steps
  eval_steps: 500              # Evaluate every N steps
  save_total_limit: 3          # Keep only N recent checkpoints
  save_best_model: true        # Save model with best eval loss

  # === Early Stopping ===
  early_stopping: false        # Stop if no improvement
  early_stopping_patience: 3   # Evals to wait
  early_stopping_threshold: 0.0 # Min improvement required

  # Logging
  logging_steps: 10
  log_to_wandb: false

  # Output
  output_dir: ./outputs
```

---

## Dependencies

- `torch`: Training loop
- `accelerate`: Distributed training
- `datasets`: HuggingFace datasets
- `transformers`: Tokenizers and schedulers
- `tqdm`: Progress bars
- `wandb` (optional): Experiment tracking
