# Scripts

Entry point scripts for training, evaluation, and inference.

## Available Scripts

```
scripts/
├── train.py      # Training entry point
├── eval.py       # Evaluation (perplexity)
└── inference.py  # Text generation
```

---

## Quick CPU Smoke Test (No HF Downloads)

If you want to validate the core `MemoryTransformer` forward pass without downloading tokenizers/models, run the "Quick CPU Smoke Test" snippet in the root `README.md`.

---

## `train.py` - Training Script

**Purpose**: Main entry point for training memory-augmented transformers.

### Usage

```bash
# Single GPU
python scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml

# Multi-GPU with Accelerate (recommended)
accelerate launch scripts/train.py --config configs/base_small.yaml

# Multi-GPU with specific settings
accelerate launch --num_processes 4 --mixed_precision bf16 \
    scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml

# Resume from checkpoint
python scripts/train.py --config configs/base_small.yaml \
    --resume outputs/checkpoint-1000
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to YAML config file |
| `--resume` | No | Path to checkpoint directory to resume from |

### What It Does

1. Loads configuration from YAML file
2. Creates model (MemoryTransformer or MemoryAdapter based on config)
3. Sets up distributed training via Accelerate
4. Creates optimizer with separate learning rates
5. Runs training loop with logging and checkpointing

### Output Structure

```
outputs/
├── checkpoint-500/
│   ├── model.pt
│   ├── config.yaml
│   ├── trainer_state.json
│   ├── model.safetensors
│   ├── optimizer.bin
│   └── scheduler.bin
├── checkpoint-1000/
│   └── ...
└── final_model/
    ├── model.pt
    ├── config.yaml
    ├── trainer_state.json
    ├── model.safetensors
    ├── optimizer.bin
    └── scheduler.bin
```

---

## `eval.py` - Evaluation Script

**Purpose**: Evaluate trained models on test datasets (computes perplexity).

Notes:
- Uses tokenizer from `model.tokenizer_name` (fallback: `model.base_model_name`, then TinyLlama).
- If `memory.quantize_memory: true`, quantizes the memory bank before evaluation.

### Usage

```bash
# Evaluate with same config as training
python scripts/eval.py --config configs/adapter_qwen2.5_1.5b.yaml \
    --checkpoint outputs/final_model

# Different dataset
python scripts/eval.py --config configs/base_small.yaml \
    --checkpoint outputs/final_model \
    --dataset wikitext \
    --split test

# Limit samples for quick test
python scripts/eval.py --config configs/base_small.yaml \
    --checkpoint outputs/final_model \
    --max_samples 1000

# Save results to JSON
python scripts/eval.py --config configs/base_small.yaml \
    --checkpoint outputs/final_model \
    --output results/eval_results.json
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | - | Path to YAML config file |
| `--checkpoint` | No | None | Path to checkpoint (uses untrained if not provided) |
| `--dataset` | No | From config | Override dataset name |
| `--split` | No | test | Dataset split to evaluate |
| `--batch_size` | No | 4 | Evaluation batch size |
| `--max_samples` | No | All | Limit number of samples |
| `--device` | No | cuda | Device to run on |
| `--output` | No | None | Path to save JSON results |

### Output

```
==================================================
Evaluation Results
==================================================
Dataset:     HuggingFaceH4/ultrachat_200k
Split:       test
Samples:     5000
Perplexity:  12.3456
Avg Loss:    2.5123
==================================================
```

### Output JSON Format

```json
{
  "perplexity": 12.3456,
  "avg_loss": 2.5123,
  "dataset": "HuggingFaceH4/ultrachat_200k",
  "split": "test",
  "checkpoint": "outputs/final_model",
  "num_samples": 5000
}
```

---

## `inference.py` - Generation Script

**Purpose**: Generate text using trained models.

Notes:
- Uses tokenizer from `model.tokenizer_name` (fallback: `model.base_model_name`, then TinyLlama).
- If `memory.quantize_memory: true`, quantizes the memory bank before generation.

### Usage

```bash
# Basic generation
python scripts/inference.py \
    --config configs/adapter_qwen2.5_1.5b.yaml \
    --prompt "What is machine learning?"

# From checkpoint
python scripts/inference.py \
    --checkpoint outputs/final_model \
    --prompt "Explain quantum computing"

# With sampling parameters
python scripts/inference.py \
    --checkpoint outputs/final_model \
    --prompt "Write a poem about AI" \
    --max_new_tokens 512 \
    --temperature 0.9 \
    --top_p 0.95
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | One of `--config` or `--checkpoint` | None | Path to YAML config |
| `--checkpoint` | One of `--config` or `--checkpoint` | None | Path to checkpoint directory |
| `--prompt` | Yes | - | Input prompt text |
| `--max_new_tokens` | No | 256 | Maximum tokens to generate |
| `--temperature` | No | 0.7 | Sampling temperature |
| `--top_p` | No | 0.9 | Nucleus sampling threshold |
| `--device` | No | cuda | Device to run on |

### Output

```
Loading model...
Prompt: What is machine learning?
--------------------------------------------------
Output: What is machine learning?

Machine learning is a subset of artificial intelligence that enables 
computers to learn from data without being explicitly programmed...
```

---

## Common Workflows

### Train and Evaluate

```bash
# 1. Train
accelerate launch scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml

# 2. Evaluate
python scripts/eval.py \
    --config configs/adapter_qwen2.5_1.5b.yaml \
    --checkpoint outputs/final_model

# 3. Generate
python scripts/inference.py \
    --checkpoint outputs/final_model \
    --prompt "Test prompt"
```

### Compare Configurations

```bash
# Train vanilla baseline
accelerate launch scripts/train.py --config configs/vanilla_control.yaml

# Train with memory
accelerate launch scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml

# Evaluate both
python scripts/eval.py --config configs/vanilla_control.yaml \
    --checkpoint outputs_vanilla/final_model

python scripts/eval.py --config configs/adapter_qwen2.5_1.5b.yaml \
    --checkpoint outputs_memory/final_model
```

### Quick Test Run

```bash
# Limited samples for testing
python scripts/train.py --config configs/base_small.yaml

# In config, set:
# training:
#   max_steps: 100
#   save_steps: 50
```

---

## Environment Setup

Before running scripts, ensure:

```bash
# Install dependencies
pip install -r requirements.txt

# For multi-GPU, configure Accelerate
accelerate config

# Optional: Login to HuggingFace for gated models
huggingface-cli login
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size in config
# training:
#   batch_size: 1
#   gradient_accumulation_steps: 16

# Or enable gradient checkpointing
# training:
#   gradient_checkpointing: true
```

### Model Loading Issues (Qwen)

```bash
# May need trust_remote_code
# This is handled automatically in the code
```

### Accelerate Issues

```bash
# Reset Accelerate config
accelerate config default

# Or specify everything manually
accelerate launch --num_processes 2 --mixed_precision bf16 ...
```
