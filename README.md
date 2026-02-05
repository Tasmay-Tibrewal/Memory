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
from memory_transformer import MemoryConfig, load_config
print("Installation successful!")
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
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── memory_transformer/       # Core implementation (11 modules)
│   ├── README.md            # Package documentation
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration system (50+ options)
│   ├── memory_bank.py       # Memory bank implementations
│   ├── memory_attention.py  # Cross-attention for memory
│   ├── memory_block.py      # Transformer blocks with memory
│   ├── router.py            # Chapter routing (MoE-style)
│   ├── lora.py              # Standard LoRA implementation
│   ├── model.py             # Full MemoryTransformer model
│   ├── adapter.py           # Memory adapter for pretrained models
│   ├── quantization.py      # Memory bank quantization
│   └── utils.py             # Utilities and helpers
│
├── training/                 # Training infrastructure
│   ├── README.md            # Training documentation
│   ├── __init__.py
│   ├── trainer.py           # Training loop with Accelerate
│   ├── data.py              # Dataset loading
│   └── losses.py            # Router auxiliary losses
│
├── inference/                # Inference utilities
│   ├── README.md            # Inference documentation
│   ├── __init__.py
│   ├── generate.py          # Text generation
│   └── routing_strategies.py # Inference routing (sequence/rolling/token)
│
├── scripts/                  # Entry point scripts
│   ├── README.md            # Scripts documentation
│   ├── train.py             # Training entry point
│   ├── eval.py              # Evaluation (perplexity)
│   └── inference.py         # Generation script
│
├── configs/                  # Example configurations
│   ├── README.md            # Complete config reference
│   ├── base_small.yaml      # From-scratch small model
│   ├── adapter_qwen2.5_1.5b.yaml  # Qwen adapter
│   ├── vanilla_control.yaml # Control experiment
│   └── memory_lora_combined.yaml  # Memory + LoRA
│
├── docs/                     # Comprehensive documentation
│   ├── README.md            # Documentation overview
│   ├── architecture.md      # Detailed architecture
│   ├── design.md            # Design decisions
│   ├── context.md           # Handoff summary
│   ├── philosophy.md        # Development philosophy and style guide
│   ├── prompt.md            # Agent onboarding prompt
│   └── meta_artifacts/      # Session artifacts for context management
│       ├── README.md        # Meta artifacts overview
│       ├── session_summary.md  # Consolidated session summaries
│       └── session1/        # Session 1 detailed artifacts
│
└── idea/                     # Original research documents
    ├── idea.txt             # Conceptual explanation
    └── main.tex             # LaTeX paper draft
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

#### Memory Settings
```yaml
memory:
  # Main toggles
  vanilla_mode: false          # Disable memory for control experiments
  use_memory_adapter: true     # Enable memory cross-attention
  
  # Memory bank
  num_memory_tokens: 2048      # Number of memory tokens
  memory_layer_placement: all  # all/first_k/last_k/every_n/custom
  memory_sharing: shared       # shared/per_layer/every_k_layers
  memory_block_variant: A      # A: SA→Mem→MLP, B: SA→MLP→Mem→MLP
  
  # Chapter routing
  use_chapters: true           # Enable MoE-style routing
  num_chapters: 16             # Number of chapters
  top_k_chapters: 4            # Chapters to select
  
  # Low-rank options
  use_low_rank_memory: true    # Factorized memory bank
  memory_rank: 256             # Low-rank dimension
  
  # LoRA
  use_lora: false              # Enable LoRA
  use_both_memory_and_lora: false  # Combine both
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
  mixed_precision: bf16
```

### Example Configurations

| Config | Use Case |
|--------|----------|
| `base_small.yaml` | From-scratch pretraining (100M params) |
| `adapter_qwen2.5_1.5b.yaml` | Memory adapter on Qwen2.5-1.5B |
| `vanilla_control.yaml` | Control experiment (no memory) |
| `memory_lora_combined.yaml` | Memory + LoRA combined |

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
training:
  use_flash_attention: true    # Requires flash-attn package
  gradient_checkpointing: true
  mixed_precision: bf16
```

### Model Not Learning
- Check `wo_init_zero: true` (critical for stable training — adapter and from-scratch)
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
- ✅ **Implemented**: Full model quantization (int8/4-bit) via `inference/merge.py`
- ✅ **Implemented**: Model merging and weight extraction
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
