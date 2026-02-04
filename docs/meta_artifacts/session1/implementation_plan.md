# Memory-Augmented Transformer Implementation Plan

## Overview

Implement a memory-augmented transformer architecture with learnable cross-attention memory banks based on idea.txt and main.tex. The implementation will support both:
1. **From-scratch training** of memory-augmented models
2. **Memory adapters** for fine-tuning existing pretrained models
3. **Vanilla transformer** (control experiment with memory disabled)

## User Review Required

> [!IMPORTANT]
> **Key Architectural Decisions Requiring Your Input:**

### 1. Training Library Selection

I recommend **PyTorch + HuggingFace Transformers + Accelerate** for the following reasons:

| Option | Pros | Cons |
|--------|------|------|
| **Unsloth** | Fast, memory efficient | Heavily optimized for specific models, hard to inject custom memory layers |
| **PyTorch + Accelerate** | Full control, easy to add custom layers, good FSDP/DDP support | More boilerplate code |
| **Trainer (HF)** | Clean API, built-in distributed | Less flexibility for custom forward passes |
| **Lightning** | Good abstractions | Overhead for custom architectures |

**My Recommendation**: Use `torch` + `transformers` for model loading + `accelerate` for distributed training. This gives us:
- Full control over the custom memory cross-attention layers
- Straightforward FSDP/DDP integration via Accelerate
- Easy integration with existing HuggingFace models for the adapter approach

**Do you agree, or would you prefer a different approach?**

### 2. Router Loss for Chapters (MoE-Style) ✅ CONFIRMED

For chapter-based routing, using losses inspired by Switch Transformer and GShard:

1. **Load Balancing Loss**: Encourages uniform chapter usage
2. **Auxiliary Load Loss**: Penalize chapters that get too few/many tokens
3. **Z-Loss**: Regularize router logits to prevent divergence

**Configuration**: All losses implemented with config flags to enable/disable each.

### 3. Memory Bank Inside Block: Order Options

From idea.txt, two variants are proposed:
- **Variant A**: Self-Attn → Memory Cross-Attn → MLP
- **Variant B**: Self-Attn → MLP → Memory Cross-Attn → MLP (extra MLP)

I'll implement both with a config flag. **Which should be the default?** (I suggest Variant A as it's simpler and matches standard cross-attention patterns like in decoder-only models with encoder.)

### 4. Base Model for Adapter Experiments ✅ CONFIRMED

**Priority**: Qwen 2.5 and Qwen 3 series first, with extensible design for other models (Llama, Mistral, etc.)

---

## Proposed Changes

### Project Structure

```
Memory/
├── memory_transformer/
│   ├── __init__.py
│   ├── config.py                 # Configuration classes and loading
│   ├── memory_bank.py            # Memory bank implementations
│   ├── memory_attention.py       # Cross-attention layer for memory
│   ├── memory_block.py           # Transformer block with memory
│   ├── router.py                 # Chapter routing (MoE-style)
│   ├── lora.py                   # Standard LoRA implementation
│   ├── model.py                  # Full model (from-scratch + vanilla mode)
│   ├── adapter.py                # Memory adapter for pretrained models
│   ├── quantization.py           # Quantization utilities
│   └── utils.py                  # Utilities
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Training loop
│   ├── data.py                   # Dataset loading with flexibility
│   ├── distributed.py            # FSDP/DDP utilities
│   └── losses.py                 # Router losses, etc.
├── inference/
│   ├── __init__.py
│   ├── generate.py               # Generation with memory
│   └── routing_strategies.py     # Sequence-level, rolling, token-level
├── configs/
│   ├── base_small.yaml           # Small model from scratch
│   ├── base_medium.yaml          # Medium model from scratch
│   ├── adapter_qwen_1.5b.yaml    # Adapter for Qwen 1.5B
│   ├── adapter_comparison.yaml   # LoRA vs MemAdapter comparison
│   └── README.md                 # Config format documentation
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── eval.py                   # Evaluation script
│   └── inference.py              # Inference script
├── docs/
│   ├── architecture.md           # Architecture explanation
│   ├── design.md                 # Design choices and compromises
│   └── context.md                # Summary for handoffs
├── requirements.txt
└── README.md
```

---

### Core Components

#### [NEW] [config.py](file:///c:/Users/kesha/OneDrive/Desktop/Tasmay/UG/Memory/memory_transformer/config.py)

Configuration dataclass covering all flags from user requirements:

```python
@dataclass
class MemoryConfig:
    # Memory bank settings
    num_memory_tokens: int = 1024
    memory_dim: int = None  # Defaults to model dim if None
    
    # Memory layer placement
    memory_layer_placement: str = "all"  # "all", "first_k", "last_k", "every_n", "custom"
    memory_layer_k: int = 5              # Used for first_k, last_k
    memory_layer_n: int = 3              # Used for every_n
    memory_layer_custom: List[int] = None  # Custom layer indices
    
    # Memory bank sharing
    memory_sharing: str = "shared"  # "shared", "per_layer", "every_k_layers"
    memory_sharing_k: int = 2       # For every_k_layers
    
    # Low-rank options
    use_low_rank_memory: bool = False
    memory_rank: int = 64
    low_rank_mode: str = "factorized"  # "factorized" (M=AB^T) or "reduced_dim" (M in N_m x r)
    use_low_rank_projections: bool = False
    projection_rank: int = 64
    
    # Block integration
    memory_block_variant: str = "A"  # "A" or "B"
    
    # Chapter routing
    use_chapters: bool = False
    num_chapters: int = 100
    tokens_per_chapter: int = None  # Auto-calculated if None
    top_k_chapters: int = 20
    routing_strategy_train: str = "sequence"  # "sequence" only for now
    routing_strategy_inference: str = "sequence"  # "sequence", "rolling", "token"
    
    # Router loss
    use_load_balance_loss: bool = True
    load_balance_coefficient: float = 0.01
    use_z_loss: bool = False
    z_loss_coefficient: float = 0.001
    
    # Quantization
    quantize_memory: bool = False
    memory_quant_bits: int = 8
    
    # Initialization
    wo_init_zero: bool = True  # Initialize output projection to zero
    
    # LoRA settings (for comparison)
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: List[str] = None  # ["q_proj", "v_proj", etc.]
    
    # Combined mode
    use_memory_adapter: bool = True
    use_both_memory_and_lora: bool = False
    
    # Vanilla mode (disable all memory for control experiments)
    vanilla_mode: bool = False  # If True, completely disable memory layers

@dataclass
class TrainingConfig:
    # Learning rates
    memory_lr: float = 1e-4
    lora_lr: float = 1e-4
    base_model_lr: float = 1e-5
    
    # Training mode
    training_mode: str = "instruction_finetuning"  # "pretraining", "instruction_finetuning"
    
    # Dataset
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_subset: str = None
    dataset_split: str = "train"
    text_field: str = "text"
    
    # Distributed
    distributed_strategy: str = "ddp"  # "ddp", "fsdp"
    num_gpus: int = 1
    
    # Standard training params
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 10000
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Checkpointing
    gradient_checkpointing: bool = True
    save_steps: int = 500
    eval_steps: int = 500

@dataclass
class ModelConfig:
    # For from-scratch training
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    intermediate_dim: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 2048
    use_rope: bool = True
    
    # For adapter mode
    base_model_name: str = None  # e.g., "Qwen/Qwen2.5-1.5B"
    freeze_base_model: bool = True
```

---

#### [NEW] [memory_bank.py](file:///c:/Users/kesha/OneDrive/Desktop/Tasmay/UG/Memory/memory_transformer/memory_bank.py)

Memory bank implementations:

1. **StandardMemoryBank**: Full N_m × d learnable parameters
2. **FactorizedMemoryBank**: M = A × B^T with A: N_m × r, B: d × r
3. **ReducedDimMemoryBank**: M stored as N_m × r directly
4. **QuantizedMemoryBank**: Wrapper for quantized storage

---

#### [NEW] [memory_attention.py](file:///c:/Users/kesha/OneDrive/Desktop/Tasmay/UG/Memory/memory_transformer/memory_attention.py)

Cross-attention layer that:
- Takes hidden states H (B × L × d) as queries
- Takes memory bank M as keys and values
- Supports multi-head attention
- Supports low-rank projections (Wq: d→r, Wk: r→r, Wv: r→r, Wo: r→d)
- Initializes Wo to zero for stable adapter training

---

#### [NEW] [router.py](file:///c:/Users/kesha/OneDrive/Desktop/Tasmay/UG/Memory/memory_transformer/router.py)

Chapter-based routing:
- Router network: Linear(d, num_chapters)
- Mean-pool input sequence for routing decision
- Top-k chapter selection
- Compute load balancing loss
- Return selected memory tokens + routing weights

---

#### [NEW] [adapter.py](file:///c:/Users/kesha/OneDrive/Desktop/Tasmay/UG/Memory/memory_transformer/adapter.py)

Memory adapter for pretrained models:
- Load base model from HuggingFace
- Inject memory cross-attention layers at specified positions
- Freeze base model weights
- Configure separate optimizers for memory vs base

---

### Training Infrastructure

#### [NEW] [trainer.py](file:///c:/Users/kesha/OneDrive/Desktop/Tasmay/UG/Memory/training/trainer.py)

Training loop with:
- Multi-GPU support via Accelerate (DDP/FSDP)
- Separate parameter groups with different learning rates
- Gradient checkpointing
- Mixed precision training
- Logging and checkpointing

#### [NEW] [data.py](file:///c:/Users/kesha/OneDrive/Desktop/Tasmay/UG/Memory/training/data.py)

Flexible dataset loading:
- Support any HuggingFace dataset
- Configurable subset, split, and text field
- Preprocessing for pretraining vs instruction-finetuning
- Tokenization and batching

---

### Config Files

#### [NEW] [configs/base_small.yaml](file:///c:/Users/kesha/OneDrive/Desktop/Tasmay/UG/Memory/configs/base_small.yaml)

Example config for small from-scratch model:

```yaml
model:
  hidden_dim: 768
  num_heads: 12
  num_layers: 12
  # ...

memory:
  num_memory_tokens: 4096
  memory_layer_placement: "all"
  memory_sharing: "shared"
  use_chapters: true
  num_chapters: 16
  top_k_chapters: 4
  # ...

training:
  training_mode: "pretraining"
  memory_lr: 1e-4
  # ...
```

---

## Verification Plan

### Unit Tests

Since this is a new codebase, I'll create unit tests for:

1. **Memory Bank Tests** (`tests/test_memory_bank.py`)
   - Test shapes for all memory bank variants
   - Test forward pass produces correct output dimensions
   - Test quantization doesn't break forward

2. **Cross-Attention Tests** (`tests/test_memory_attention.py`)
   - Test attention output shape matches input
   - Test zero-initialized Wo produces zero output initially
   - Test attention weights sum to 1

3. **Router Tests** (`tests/test_router.py`)
   - Test top-k selection returns correct number of chapters
   - Test load balancing loss is computed correctly

4. **Integration Tests** (`tests/test_integration.py`)
   - Test full forward pass of memory-augmented model
   - Test adapter injection on a small model

**Run tests with**: `pytest tests/ -v`

### Manual Verification

1. **Training Smoke Test**:
   - Run training for 100 steps on tiny dataset
   - Verify loss decreases
   - Command: `python scripts/train.py --config configs/test_small.yaml`

2. **Adapter Loading Test**:
   - Load adapter on Qwen model
   - Run inference to verify output is coherent
   - Command: `python scripts/inference.py --config configs/adapter_qwen_1.5b.yaml --prompt "Hello, world"`

3. **Multi-GPU Test** (if you have multiple GPUs):
   - Run with `accelerate launch --num_processes 2 scripts/train.py`
   - Verify training works across GPUs

> [!NOTE]
> I'll create a `tests/` directory with pytest-compatible tests that can be run with `pytest tests/ -v`.

---

## Implementation Order

1. **Phase 1**: Project structure, config system, basic memory bank
2. **Phase 2**: Memory attention, transformer block integration
3. **Phase 3**: Chapter routing with losses
4. **Phase 4**: Low-rank variants, quantization
5. **Phase 5**: Adapter mechanism for pretrained models
6. **Phase 6**: LoRA implementation
7. **Phase 7**: Training infrastructure with Accelerate
8. **Phase 8**: Inference and evaluation scripts
9. **Phase 9**: Documentation (architecture.md, design.md, context.md, README)

---

## Remaining Questions

~~1. **Training library**: Do you agree with PyTorch + Accelerate?~~ → **Pending your confirmation** (I recommend PyTorch + Accelerate)

~~2. **Router losses**: All three with config flags~~ → ✅ **CONFIRMED**

~~3. **Default block variant**: Variant A or B?~~ → **Pending** (I suggest Variant A)

~~4. **Base models**: Qwen 2.5/3 priority, extensible~~ → ✅ **CONFIRMED**

5. **Sequence length**: What's the target max sequence length? (default: 2048)

6. **Dataset preferences**: Any specific datasets for initial configs?

7. **Test model size**: Use tiny models for unit tests? (I suggest yes for speed)
