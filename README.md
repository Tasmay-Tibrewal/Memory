# Memory-Augmented Transformer

A PyTorch implementation of **memory-augmented transformers** with learnable cross-attention memory banks, designed for both from-scratch training and parameter-efficient fine-tuning of pretrained models.

---

## Overview

This project implements a novel memory-augmented transformer architecture where:
- **Learnable memory tokens** are stored in a memory bank
- Transformer layers access memory via **cross-attention** (queries from hidden states, keys/values from memory)
- **Chapter-based routing** (MoE-inspired) enables scaling to very large memory banks
- Memory layers can be added as **adapters** to pretrained models (Qwen, Llama, Mistral)

---

## Features

### Core Architecture
- **Learnable Memory Banks**: Cross-attention to persistent latent tokens learned during training
- **Multiple Memory Variants**: Standard, Factorized (M=AB^T), Reduced-dimension
- **Flexible Placement**: Memory in all layers, first/last k, every n-th, or custom list
- **Memory Sharing**: Shared bank, per-layer banks, or grouped sharing

### Efficient Scaling
- **Chapter-Based Routing**: MoE-style top-k selection for large memory banks (100k+ tokens)
- **Router Losses**: Load balancing, auxiliary, and z-loss from MoE literature
- **Low-Rank Compression**: Factorized memory, low-rank projections

### Training Infrastructure
- **Multi-GPU Training**: DDP/FSDP support via HuggingFace Accelerate
- **Mixed Precision**: bf16/fp16 training
- **Gradient Checkpointing**: Reduce memory usage (including for memory attention)
- **Separate Learning Rates**: Different LRs for memory, LoRA, and base model
- **Eval During Training**: Periodic evaluation on validation set
- **Early Stopping**: Stop training when validation loss stops improving
- **Best Model Saving**: Automatically save model with best validation loss
- **Resume from Checkpoint**: Continue training from saved state
- **Learning Rate Finder**: Find optimal learning rate before training

### Adapter Mode
- **Memory Adapters**: Add memory to frozen pretrained models
- **LoRA Integration**: Standard LoRA for comparison
- **Combined Mode**: Memory + LoRA together
- **Supported Models**: Qwen 2.5/3, Llama 2/3, Mistral

### Configuration
- **YAML-Based Config**: All 50+ options in config files
- **Example Configs**: Ready-to-use configurations with dataset suggestions
- **Vanilla Mode**: Disable memory for control experiments

---

## Installation

### Basic Installation
```bash
git clone <repository-url>
cd Memory
pip install -r requirements.txt
```

### Optional Dependencies
```bash
# For Flash Attention (Linux, CUDA 11.8+)
pip install flash-attn --no-build-isolation

# For 4/8-bit quantization
pip install bitsandbytes

# For experiment tracking
pip install wandb
```

### Verify Installation
```python
from memory_transformer import load_config
load_config("configs/base_small.yaml")
print("Installation successful!")
```

### Quick CPU Smoke Test (No HF Downloads)
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
cfg.memory.use_chapters = True
cfg.memory.num_chapters = 8
cfg.memory.top_k_chapters = 2
cfg.memory.routing_strategy_inference = "hybrid"

model = MemoryTransformer(cfg).eval()
x = torch.randint(0, cfg.model.vocab_size, (1, 8))
out = model(input_ids=x, use_cache=True)
print("Smoke OK:", out["logits"].shape)
```

---

## Quick Start

### 1. Training from Scratch (Small Model)
```bash
python scripts/train.py --config configs/base_small.yaml
```

### 2. Memory Adapter on Pretrained Model
```bash
python scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml
```

### 3. Multi-GPU Training
```bash
# DDP (recommended for most cases)
accelerate launch --num_processes 4 scripts/train.py --config configs/base_small.yaml

# FSDP (for very large models)
accelerate launch --num_processes 4 --use_fsdp scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml
```

### 4. Evaluation
```bash
python scripts/eval.py --config configs/adapter_qwen2.5_1.5b.yaml --checkpoint outputs/final_model
```

### 5. Inference
```bash
python scripts/inference.py --checkpoint outputs/final_model --prompt "Explain machine learning"
```

---

## Project Structure

```
Memory/
â”œâ”€â”€ README.md                 # This file
â”œâ”€â”€ requirements.txt          # Python dependencies
â”‚
â”œâ”€â”€ memory_transformer/       # Core implementation (11 modules)
â”‚   â”œâ”€â”€ README.md            # Package documentation
â”‚   â”œâ”€â”€ __init__.py          # Package exports
â”‚   â”œâ”€â”€ config.py            # Configuration system (50+ options)
â”‚   â”œâ”€â”€ memory_bank.py       # Memory bank implementations
â”‚   â”œâ”€â”€ memory_attention.py  # Cross-attention for memory
â”‚   â”œâ”€â”€ memory_block.py      # Transformer blocks with memory
â”‚   â”œâ”€â”€ router.py            # Chapter routing (MoE-style)
â”‚   â”œâ”€â”€ lora.py              # Standard LoRA implementation
â”‚   â”œâ”€â”€ model.py             # Full MemoryTransformer model
â”‚   â”œâ”€â”€ adapter.py           # Memory adapter for pretrained models
â”‚   â”œâ”€â”€ quantization.py      # Memory bank quantization
â”‚   â””â”€â”€ utils.py             # Utilities and helpers
â”‚
â”œâ”€â”€ training/                 # Training infrastructure
â”‚   â”œâ”€â”€ README.md            # Training documentation
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ trainer.py           # Training loop with Accelerate
â”‚   â”œâ”€â”€ data.py              # Dataset loading
â”‚   â””â”€â”€ losses.py            # Router auxiliary losses
â”‚
â”œâ”€â”€ inference/                # Inference utilities
â”‚   â”œâ”€â”€ README.md            # Inference documentation
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ generate.py          # Text generation
â”‚   â”œâ”€â”€ merge.py             # Model merging and quantization
â”‚   â””â”€â”€ routing_strategies.py # Inference routing (sequence/rolling/token)
â”‚
â”œâ”€â”€ scripts/                  # Entry point scripts
â”‚   â”œâ”€â”€ README.md            # Scripts documentation
â”‚   â”œâ”€â”€ train.py             # Training entry point
â”‚   â”œâ”€â”€ eval.py              # Evaluation (perplexity)
â”‚   â””â”€â”€ inference.py         # Generation script
â”‚
â”œâ”€â”€ configs/                  # Example configurations
â”‚   â”œâ”€â”€ README.md            # Complete config reference
â”‚   â”œâ”€â”€ base_small.yaml      # From-scratch small model
â”‚   â”œâ”€â”€ adapter_qwen2.5_1.5b.yaml  # Qwen adapter
â”‚   â”œâ”€â”€ vanilla_control.yaml # Control experiment
â”‚   â””â”€â”€ memory_lora_combined.yaml  # Memory + LoRA
│   └── reference_all_options.yaml # Full config reference
â”‚
â”œâ”€â”€ docs/                     # Comprehensive documentation
â”‚   â”œâ”€â”€ README.md            # Documentation overview
â”‚   â”œâ”€â”€ architecture.md      # Detailed architecture
â”‚   â”œâ”€â”€ design.md            # Design decisions
â”‚   â”œâ”€â”€ context.md           # Handoff summary
â”‚   â”œâ”€â”€ philosophy.md        # Development philosophy and style guide
â”‚   â”œâ”€â”€ prompt.md            # Agent onboarding prompt
â”‚   â””â”€â”€ meta_artifacts/      # Session artifacts for context management
â”‚       â”œâ”€â”€ README.md        # Meta artifacts overview
â”‚       â”œâ”€â”€ session_summary.md  # Consolidated session summaries
â”‚       â””â”€â”€ session1/        # Session 1 historical artifacts
│       └── session9/        # Latest detailed session artifacts
â”‚
â””â”€â”€ idea/                     # Original research documents
    â”œâ”€â”€ idea.txt             # Conceptual explanation
    â”œâ”€â”€ main.tex             # LaTeX paper draft
    â”œâ”€â”€ proposal.md          # Project proposal
    â””â”€â”€ Memory_Layer_in_Transformers.pdf  # Reference PDF
```

---

## Configuration

All settings are controlled via YAML config files. See [`configs/README.md`](configs/README.md) for complete reference.

### Config Structure
```yaml
model:      # Model architecture
memory:     # Memory bank settings
training:   # Training hyperparameters
```

### Key Configuration Options

#### Model Settings
```yaml
model:
  # Attention heads (set num_kv_heads < num_heads to enable GQA)
  num_heads: 12
  num_kv_heads: null
  max_position_embeddings: null
  hidden_activation: swiglu
  initializer_range: 0.02
  tie_embeddings: true

  # Tokenizer to use (must match vocab_size for from-scratch)
  tokenizer_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  vocab_size: 32000
  bos_token_id: null
  eos_token_id: null
  pad_token_id: null

  # Positional encoding + attention regularization
  use_rope: true
  attention_dropout: 0.0
```
 
#### Memory Settings
```yaml
memory:
  # Main toggles
  vanilla_mode: false          # Disable memory for control experiments
  use_memory_adapter: true     # Enable memory cross-attention
  
  # Memory bank
  num_memory_tokens: 2048      # Number of memory tokens
  memory_num_heads: null       # Optional memory-attn heads (null => model/base heads)
  memory_num_kv_heads: null    # Optional memory-attn KV heads (null => model/base KV heads)
  memory_layer_placement: all  # all/first_k/last_k/every_n/custom
  memory_sharing: shared       # shared/per_layer/every_k_layers
  memory_block_variant: A      # A: SAâ†’Memâ†’MLP, B: SAâ†’MLPâ†’Memâ†’MLP
  memory_dropout: null         # Memory cross-attn dropout (null => model.dropout)
  
  # Chapter routing
  use_chapters: true           # Enable MoE-style routing
  num_chapters: 16             # Number of chapters
  top_k_chapters: 4            # Chapters to select
  routing_strategy_inference: hybrid  # sequence/rolling/token/hybrid
  routing_window_size: 128            # Rolling/hybrid window size (tokens)
  
  # Low-rank options
  use_low_rank_memory: true    # Factorized memory bank
  memory_rank: 256             # Low-rank dimension
  
  # LoRA
  use_lora: false              # Enable LoRA
  use_both_memory_and_lora: false  # Combine both

  # Optional: quantize memory bank for inference/eval scripts
  quantize_memory: false
  memory_quant_bits: 8
```

#### Training Settings
```yaml
training:
  # Separate learning rates
  memory_lr: 2e-4
  lora_lr: 1e-4
  base_model_lr: 0             # 0 = frozen
  
  # Dataset
  dataset_name: HuggingFaceH4/ultrachat_200k
  training_mode: instruction_finetuning  # or pretraining
  
  # Distributed
  distributed_strategy: ddp    # ddp or fsdp
  fsdp_sharding_strategy: FULL_SHARD
  mixed_precision: bf16
  scheduler: wsd               # cosine/linear/constant/wsd
  wsd_stable_ratio: 0.3
  decay_start_ratio: null
  save_total_limit: 3          # null => keep all checkpoints
```

### Example Configurations

| Config | Use Case |
|--------|----------|
| `base_small.yaml` | From-scratch pretraining (100M params) |
| `adapter_qwen2.5_1.5b.yaml` | Memory adapter on Qwen2.5-1.5B |
| `vanilla_control.yaml` | Control experiment (no memory) |
| `memory_lora_combined.yaml` | Memory + LoRA combined |
| `reference_all_options.yaml` | Full config surface (documentation template) |

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/architecture.md`](docs/architecture.md) | Detailed architecture with diagrams |
| [`docs/design.md`](docs/design.md) | Design decisions, compromises, known issues |
| [`docs/context.md`](docs/context.md) | Quick summary for handoffs |
| [`docs/philosophy.md`](docs/philosophy.md) | Development philosophy and style guide |
| [`docs/meta_artifacts/session_summary.md`](docs/meta_artifacts/session_summary.md) | Session summaries |
| [`configs/README.md`](configs/README.md) | Complete configuration reference |

### Package Documentation
Each subfolder has its own README:
- [`memory_transformer/README.md`](memory_transformer/README.md) - Core modules
- [`training/README.md`](training/README.md) - Training infrastructure
- [`inference/README.md`](inference/README.md) - Generation utilities
- [`scripts/README.md`](scripts/README.md) - CLI scripts

---

## Comparison Experiments

Run these to compare different approaches:

```bash
# 1. Vanilla baseline (no memory)
python scripts/train.py --config configs/vanilla_control.yaml

# 2. Memory adapter only
python scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml

# 3. LoRA only (modify config: use_lora=true, use_memory_adapter=false)

# 4. Memory + LoRA combined
python scripts/train.py --config configs/memory_lora_combined.yaml

# Evaluate all
python scripts/eval.py --checkpoint outputs_vanilla/final_model
python scripts/eval.py --checkpoint outputs_memory/final_model
# ...
```

---

## Troubleshooting

### Out of Memory
```yaml
# Reduce batch size and use accumulation
training:
  batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true

# Or reduce memory bank size
memory:
  num_memory_tokens: 512
  use_low_rank_memory: true
  memory_rank: 128
```

### Slow Training
```yaml
model:
  use_flash_attention: true    # Requires flash-attn package
training:
  gradient_checkpointing: true
  mixed_precision: bf16
```

Adapter mode uses persistent hooks that are compatible with gradient checkpointing
(`GradientCheckpointingLayer` in `transformers>=4.35`). Hooks survive backward recompute;
side effects are suppressed via `_fwd_processed_layers`. One limitation: assumes one forward
per backward per micro-step. See `docs/design.md` for details.

### Model Not Learning
- Check `wo_init_zero: true` (critical for stable training â€” adapter and from-scratch)
- Enable `use_load_balance_loss: true` if router collapses
- Increase `memory_lr` relative to `base_model_lr`

---

## Future Work

The following features are planned for future development:

### Attention Visualization
- Visualize memory attention patterns
- Analyze which chapters are selected by the router
- Track router decisions over training

### Benchmarking Suite
- Throughput measurement scripts
- Latency profiling tools
- Memory usage tracking during training/inference

### Export & Deployment
- âœ… **Implemented**: Full model quantization (int8/4-bit) via `inference/merge.py`
- âœ… **Implemented**: Model merging and weight extraction
- ONNX export for production deployment
- TensorRT optimization

### Advanced Features
- Memory compression learning (distill documents into memory)
- Multi-tier memory with different granularities
- Retrieval-augmented hybrid approaches

---

## Citation

If you use this code, please cite:
```bibtex
@misc{memory-transformer,
  title={Memory-Augmented Transformer with Learnable Cross-Attention Memory Banks},
  author={...},
  year={2026}
}
```

---

## License

MIT License

