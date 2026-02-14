# Scripts

Entry point scripts for training, evaluation, and inference.

## Available Scripts

```
scripts/
├── train.py               # Training entry point
├── eval.py                # Perplexity evaluation
├── eval_mmlu.py           # MMLU accuracy
├── eval_mcq_benchmark.py  # Generic MCQ benchmark evaluator
├── eval_hellaswag.py      # HellaSwag accuracy
├── eval_arc.py            # ARC-Challenge / ARC-Easy accuracy
├── eval_winogrande.py     # Winogrande accuracy
├── eval_boolq.py          # BoolQ accuracy
├── eval_openbookqa.py     # OpenBookQA accuracy
├── eval_triviaqa.py       # TriviaQA top-alias perplexity
├── eval_pretrain_suite.py # Run all benchmark scripts + aggregate
├── generate_ifeval_jsonl.py # Generate prompt/response JSONL for external IFEval scoring
├── inference.py           # Text generation
└── estimate_flops.py      # Analytical FLOPs estimator
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
- Applies optional `model.bos_token_id` / `model.eos_token_id` / `model.pad_token_id` overrides.
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

## Benchmark Accuracy Scripts

These scripts evaluate multiple-choice accuracy (not perplexity), useful for
base-model post-pretraining benchmarking.

```bash
# MMLU
python scripts/eval_mmlu.py --config configs/base_small.yaml --checkpoint outputs/final_model

# MMLU with default-style few-shot (5) and full-option-text scoring
python scripts/eval_mmlu.py --config configs/base_small.yaml --checkpoint outputs/final_model \
  --shots 5 --fewshot_mode manual --scoring_mode choice_text

# Other common MCQ benchmarks
python scripts/eval_hellaswag.py --config configs/base_small.yaml --checkpoint outputs/final_model
python scripts/eval_arc.py --variant challenge --config configs/base_small.yaml --checkpoint outputs/final_model
python scripts/eval_winogrande.py --config configs/base_small.yaml --checkpoint outputs/final_model
python scripts/eval_boolq.py --config configs/base_small.yaml --checkpoint outputs/final_model
python scripts/eval_openbookqa.py --config configs/base_small.yaml --checkpoint outputs/final_model
python scripts/eval_triviaqa.py --config configs/base_small.yaml --checkpoint outputs/final_model

# Generic one-benchmark CLI
python scripts/eval_mcq_benchmark.py --benchmark hellaswag --config configs/base_small.yaml --checkpoint outputs/final_model

# Explicit zero-shot label-only scoring (legacy style)
python scripts/eval_mcq_benchmark.py --benchmark hellaswag --config configs/base_small.yaml --checkpoint outputs/final_model \
  --shots 0 --scoring_mode label

# Master suite runner (runs all + aggregates)
python scripts/eval_pretrain_suite.py --config configs/base_small.yaml --checkpoint outputs/final_model

# Suite defaults now support manual few-shot + choice_text scoring
python scripts/eval_pretrain_suite.py --config configs/base_small.yaml --checkpoint outputs/final_model \
  --mmlu_shots 5 --mcq_shots 5 --mmlu_fewshot_mode manual --mcq_fewshot_mode manual \
  --scoring_mode choice_text
```

Notes:
- `eval_mmlu.py` and `eval_mcq_benchmark.py` support `--fewshot_mode {manual,dataset}`.
- `manual` uses handcrafted benchmark/task examples and shuffles option order per seed for label diversity.
- `--scoring_mode choice_text` scores full option text; `--scoring_mode label` scores `A/B/C/...` label tokens.
- `eval_triviaqa.py` uses question-only prompts (no context), samples few-shot examples from a non-test split by default (`--shots 5`), selects the highest full-sequence-probability alias per question, and reports top-alias perplexity (plus token-weighted corpus perplexity).
- `eval_pretrain_suite.py` defaults: `mmlu, hellaswag, arc_challenge, arc_easy, winogrande, boolq, openbookqa, triviaqa`.

---

## `generate_ifeval_jsonl.py` - IFEval JSONL Generator

Generates one JSONL line per prompt for external IFEval repos/tools.

### Usage

```bash
python scripts/generate_ifeval_jsonl.py \
  --config configs/ift_base_model.yaml \
  --checkpoint outputs/final_model \
  --batch_size 8 \
  --output outputs/ifeval/predictions.jsonl

# Multi-GPU (script shards by rank and merges into one ordered JSONL)
accelerate launch --num_processes 8 scripts/generate_ifeval_jsonl.py \
  --distributed \
  --config configs/ift_base_model.yaml \
  --checkpoint outputs/final_model \
  --batch_size 8 \
  --output outputs/ifeval/predictions.jsonl
```

### Notes

- Default dataset: `google/IFEval`, split `train`.
- Output rows contain: `key`, `prompt`, `response`, `model_id` (plus instruction metadata when present).
- Chat template application is enabled by default when tokenizer provides one (`--no-apply_chat_template` to disable).
- Chat template is strict by default (`--require_chat_template`); if template application is unavailable/fails, the script exits instead of silently falling back.
- Batched generation is enabled via `--batch_size` (default: `8`).
- In distributed mode, each rank writes a temporary part and rank 0 merges all parts in original prompt order.

---

## `inference.py` - Generation Script

**Purpose**: Generate text using trained models.

Notes:
- Uses tokenizer from `model.tokenizer_name` (fallback: `model.base_model_name`, then TinyLlama).
- Applies optional `model.bos_token_id` / `model.eos_token_id` / `model.pad_token_id` overrides.
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

## `estimate_flops.py` - FLOPs Estimator

**Purpose**: Estimate training FLOPs from a YAML config.

This script estimates forward/backward FLOPs for the from-scratch
`MemoryTransformer` path using the configured architecture and training setup.

### Usage

```bash
# Use config defaults (training.batch_size, training.max_length)
python scripts/estimate_flops.py --config configs/base_small.yaml

# Override per-device batch and sequence length
python scripts/estimate_flops.py --config configs/base_small.yaml \
    --batch_size 16 --seq_len 2048

# Override world-size math and grad-accum for what-if planning
python scripts/estimate_flops.py --config configs/base_small.yaml \
    --num_gpus 8 --gradient_accumulation_steps 2 --max_steps 1200
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to YAML config file |
| `--batch_size` | No | Override per-device micro-batch size |
| `--seq_len` | No | Override sequence length |
| `--num_gpus` | No | Override world size used for global FLOPs |
| `--gradient_accumulation_steps` | No | Override grad accumulation for optimizer-step totals |
| `--max_steps` | No | Override steps used for run-total FLOPs |
| `--flash_available` | No | Force flash-attn assumption: `auto`/`true`/`false` |

### Notes

- Includes both matmul and elementwise components:
  self-attn, memory-attn, MLP, norms, residuals, RoPE, router forward/loss,
  memory preprocessing, LM head, and CE softmax.
- Includes recompute FLOPs for both:
  `training.gradient_checkpointing` (whole-layer recompute) and
  `memory.memory_gradient_checkpointing` (memory-attn internal recompute).
- Uses `memory.routing_strategy_train` semantics (training path). Valid values
  here are `sequence`, `sequence-rolling`/`sequence_rolling`, and `token`.
- Includes factorized memory-bank materialization FLOPs when
  `memory.use_low_rank_memory: true` and `memory.low_rank_mode: factorized`.
- Matmul terms are exact for configured tensor shapes; non-matmul terms are
  analytic approximations. Optimizer-update FLOPs and communication overhead
  are not included.

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

