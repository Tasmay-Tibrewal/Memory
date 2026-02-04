# Inference Package

This package provides generation utilities and routing strategies for inference with memory-augmented transformers.

## Module Overview

```
inference/
├── __init__.py           # Package exports
├── generate.py           # Text generation utilities
└── routing_strategies.py # Different routing strategies for inference
```

---

## Detailed Module Documentation

### `generate.py` - Text Generation

**Purpose**: Generate text from memory-augmented models with various sampling strategies.

**Main Function**:
```python
from inference import generate

output_text = generate(
    model,
    tokenizer,
    prompt="What is machine learning?",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    device="cuda",
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | - | Memory transformer model |
| `tokenizer` | - | - | HuggingFace tokenizer |
| `prompt` | str | - | Input text |
| `max_new_tokens` | int | 256 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (higher = more random) |
| `top_p` | float | 0.9 | Nucleus sampling probability |
| `top_k` | int | 50 | Top-k filtering |
| `do_sample` | bool | True | Sample vs greedy decoding |
| `device` | str | "cuda" | Device to run on |

**Batch Generation**:
```python
from inference.generate import generate_batch

outputs = generate_batch(
    model,
    tokenizer,
    prompts=["Question 1?", "Question 2?", "Question 3?"],
    max_new_tokens=256,
    temperature=0.7,
    device="cuda",
)
# Returns list of generated strings
```

---

### `routing_strategies.py` - Inference Routing

**Purpose**: Different strategies for chapter selection during inference.

**Why Different Strategies?**

During training, we have the full sequence and can use sequence-level routing. During generation, we generate one token at a time, so we need adaptive strategies.

**Available Strategies**:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `SequenceLevelRouter` | Mean-pool full context | Prefill, short sequences |
| `RollingWindowRouter` | Use recent N tokens | Long generation |
| `TokenLevelRouter` | Per-token routing | Single token generation |
| `HybridRouter` | Sequence for prefill, rolling for generation | General use |

**Class Descriptions**:

#### `SequenceLevelRouter`
```python
from inference.routing_strategies import SequenceLevelRouter

router = SequenceLevelRouter(trained_router, top_k=4)
chapter_indices, weights = router.route(hidden_states)
```
- Uses mean-pooling over the full sequence
- Same behavior as training
- Good for short sequences

#### `RollingWindowRouter`
```python
from inference.routing_strategies import RollingWindowRouter

router = RollingWindowRouter(trained_router, top_k=4, window_size=128)
router.reset_cache()  # Reset at start of generation

# During generation
chapter_indices, weights = router.route(new_hidden_states, use_cache=True)
```
- Maintains a cache of recent hidden states
- Only considers last `window_size` tokens
- Avoids computing over entire (potentially very long) context

#### `TokenLevelRouter`
```python
from inference.routing_strategies import TokenLevelRouter

router = TokenLevelRouter(trained_router, top_k=4)
# Returns per-token routing
chapter_indices, weights = router.route(hidden_states)
# Shapes: (batch, seq_len, top_k) instead of (batch, top_k)
```
- Each token routes to different chapters
- Only efficient during generation (seq_len=1)
- **Not recommended during prefill** (memory prohibitive)

#### `HybridRouter`
```python
from inference.routing_strategies import HybridRouter

router = HybridRouter(trained_router, top_k=4, window_size=128)

# During prefill
prefill_chapters, weights = router.route(prefill_hidden_states)

# Start generation phase
router.start_generation(prefill_hidden_states)

# During generation
gen_chapters, weights = router.route(new_token_hidden_states)

# Reset for new sequence
router.reset()
```
- Combines sequence-level (prefill) with rolling (generation)
- Recommended for production use

**Factory Function**:
```python
from inference.routing_strategies import create_inference_router

router = create_inference_router(
    router=trained_router,
    strategy="hybrid",  # "sequence", "rolling", "token", "hybrid"
    top_k=4,
    window_size=128,
)
```

---

## Usage Examples

### Basic Generation

```python
from memory_transformer import MemoryAdapter, load_config
from inference import generate
from transformers import AutoTokenizer
import torch

# Load model
config = load_config("configs/adapter_qwen2.5_1.5b.yaml")
model = MemoryAdapter(config)
state_dict = torch.load("outputs/final_model/model.pt")
model.load_state_dict(state_dict)
model.cuda().eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# Generate
output = generate(
    model, 
    tokenizer, 
    "Explain the theory of relativity:",
    max_new_tokens=256,
)
print(output)
```

### Using Custom Routing Strategy

```python
from inference.routing_strategies import create_inference_router

# Get a router from your trained model
# (depends on your model structure)
trained_router = model.routers["0"]

# Create inference router
inf_router = create_inference_router(
    router=trained_router,
    strategy="hybrid",
    top_k=4,
    window_size=256,
)

# Use during manual generation loop
# ...
```

---

## Configuration Reference

Routing strategy can be set in config:

```yaml
memory:
  # Training routing (always sequence-level)
  routing_strategy_train: sequence
  
  # Inference routing
  routing_strategy_inference: sequence  # "sequence", "rolling", "token"
  
  # Affects rolling router
  # (window_size not in config yet, can be added)
```

---

## Performance Considerations

### `merge.py` - Model Merging & Quantization

**Purpose**: Utilities for merging LoRA weights, extracting memory components, and quantizing models for deployment.

**Main Functions**:

#### `quantize_full_model`
```python
from inference.merge import quantize_full_model

# Quantize full model (base + memory)
quantized_model = quantize_full_model(
    model,
    method="dynamic",  # Options below
    output_path="outputs/quantized_model.pt"
)
```
- **Methods**:
  - `fp16`, `bf16`, `fp32`: Standard precision conversion
  - `dynamic`: PyTorch int8 dynamic quantization (best for CPU)
  - `fp8`: Float8 conversion (requires torch 2.1+)
  - `bnb_8bit`: bitsandbytes 8-bit quantization (requires CUDA)
  - `bnb_4bit`: bitsandbytes 4-bit quantization (requires CUDA, best compression)
  - `bnb_fp8`: bitsandbytes fp8 mixed quantization

#### `export_to_gguf`
```python
from inference.merge import export_to_gguf

# Prepare model and run conversion (if llama.cpp available)
export_to_gguf(
    model,
    output_path="outputs/model.gguf",
    quantization_type="q4_k_m",  # q4_k_m, q8_0, f16, etc.
    llama_cpp_path="/path/to/llama.cpp"  # Optional
)
```
- **Note**: GGUF conversion requires `llama.cpp`. This function saves the model in a compatible format and attempts to run the conversion script.

#### `quantize_memory_for_deployment`
```python
from inference.merge import quantize_memory_for_deployment

# Quantize only learning memory bank (leaving model in fp16/bf16)
model = quantize_memory_for_deployment(
    model, 
    bits=4, 
    output_path="outputs/mem_quant_model.pt"
)
```

#### `merge_adapter_weights`
```python
from inference.merge import merge_adapter_weights

# Merge LoRA weights into base model
state_dict = merge_adapter_weights(
    adapter_model,
    output_path="outputs/merged_weights.pt"
)
```

#### `extract_memory_weights`
```python
from inference.merge import extract_memory_weights

# Extract only memory components (bank, attention, router)
# Useful for sharing adapters
mem_weights = extract_memory_weights(
    model,
    output_path="outputs/memory_only.pt"
)
```

---

## Performance Considerations

### Memory Usage by Strategy

| Strategy | Memory | Speed | Quality |
|----------|--------|-------|---------|
| Sequence | O(seq_len) | Fast | Best for short |
| Rolling | O(window_size) | Fast | Good for long |
| Token | O(1) per token | Slow | Variable |
| Hybrid | O(window_size) | Balanced | Recommended |

### Recommendations

1. **Short sequences (<1k tokens)**: Use `sequence`
2. **Long sequences (1k-8k tokens)**: Use `rolling` with window_size=256-512
3. **Very long sequences (>8k tokens)**: Use `hybrid` with smaller window
4. **Generation**: Use `hybrid` (sequence for prefill, rolling for generation)

---

## Dependencies

- `torch`: Tensor operations
- `transformers`: Tokenizers
- `bitsandbytes` (optional): For 4-bit/8-bit quantization
