# Design Decisions and Notes

This document records design choices, compromises, known issues, and areas for future improvement. The configuration that backs the published workshop results lives in [`configs/base_small_run2.yaml`](../configs/base_small_run2.yaml); see the [root `README.md`](../README.md) for headline numbers.

## Design Choices Made

### 1. Training Library: PyTorch + Accelerate

**Choice**: Raw PyTorch with HuggingFace Accelerate for distributed training.

**Rationale**:

- Full control over custom memory cross-attention layers
- Accelerate handles DDP/FSDP transparently
- Easy integration with HF models for adapter mode
- Unsloth would require extensive patching for custom attention

**Trade-off**: More boilerplate than using Trainer, but necessary for our custom architecture.

### 2. Block Variant A as Default

**Choice**: Self-Attention → Memory Cross-Attention → MLP

**Rationale**:

- Simpler architecture
- Matches encoder-decoder cross-attention patterns
- Variant B (extra MLP) can be enabled via config

### 3. Sequence-Level Default + Token-Level Kernel Path

**Choice**:

- Keep sequence-level routing as the default.
- Implement token-level routing when `routing_strategy_*: token` is selected.

**Rationale**:

- Token-level routing is more expressive and avoids sequence-level route leakage patterns.
- Naive token-level dense attention is memory-prohibitive during training/prefill.
- We therefore split token routing into:
  - dense shared-chapter attention (always weighted 1.0)
  - sparse routed-chapter attention via stable kernels (`kernels-final` v1/v2/v3 unweighted, v4 exact MoE-weighted fused, v5 joint-bias approximation; default `v2`)
- **Router weights are applied MoE-style**: each chapter runs through independent softmax attention, then outputs are weighted by router probabilities and summed. This ensures router weights directly control chapter contribution rather than being overridden by Q·K similarity (which would happen with joint-softmax pre-scaling).
- Per-chapter kernel calls use CUDA stream parallelism with event-based synchronization.
- Sequence-level remains a robust default for broad compatibility.

Kernel engineering notes:

- Experimentation and benchmarks live in `kernels/`
- Curated stable kernel set lives in `kernels-final/` (v1/v2/v3 unweighted; v4 exact MoE-weighted fused with full backward including dW; v5 single-softmax joint-bias approximation)

**Current**: Integrated in core model/adapter path. Remaining work is tuning and broader benchmark coverage.

### 4. Zero-Initialized Output Projection

**Choice**: Initialize W_o in memory cross-attention to zero.

**Rationale**:

- Ensures model behaves exactly like base model at initialization
- Memory contribution gradually increases as training progresses
- Prevents instability when injecting adapters into pretrained models

### 5. Persistent Hook Injection for Adapter Mode

**Choice**: Use persistent PyTorch forward hooks (registered lazily on first forward, never removed) to inject memory after each decoder layer.

**Rationale**:

- Non-invasive to pretrained model code
- Works across many model architectures
- Survives `GradientCheckpointingLayer` backward recomputation (hooks fire during recompute, producing correct memory-path gradients)
- Lazy registration ensures DDP/FSDP/compile wrapping is applied before hooks attach

**Trade-offs**:

- Slight overhead from hook mechanism.
- Per-forward state is stashed on instance attributes (`self._fwd_*`); hooks read these instead of closure variables.
- Side effects (router loss accumulation, routing-cache mutation) are suppressed during recompute via a `_fwd_processed_layers` set.
- Hooks must preserve the original output type (tuple vs tensor). HF decoder layers return 1-tuples like `(hidden_states,)`; returning a bare tensor would cause the model loop's `layer_outputs[0]` to index into the tensor's batch dimension instead of unpacking the tuple.
- Assumes one forward per backward per micro-step. Multiple forwards before a single backward is NOT supported when gradient checkpointing is active.
- The out-of-band guard (`_fwd_all_router_losses is None`) is best-effort: after `adapter.forward()` returns, `_fwd_*` state is intentionally kept alive for backward GC recompute, so direct `self.base_model(...)` calls will still trigger memory injection. Always use `adapter.forward()` as the API surface.

## Known Limitations

### 1. Quantized Memory Bank Is Inference-Oriented

**Status**: Implemented for inference/evaluation workflows.

**Current support**:

- Memory-bank quantization utilities are available (INT8 and packed INT4).
- Deployment helpers can quantize memory banks and full models for inference.

**Limitation**:

- Training-time quantization-aware updates are not implemented.

**Workaround**: Use low-rank factorization for training-time memory compression.

**Future**: Add QAT-style training path.

### 2. Token-Level Routing Cost / Fallback Behavior

As discussed in [`idea/proposal.md`](../idea/proposal.md) §8.4 and the workshop paper §4.4, token-level routing during prefill would require:

```
K tensor size = B × S × num_heads × routed_tokens × D
            = 250 × 10000 × 32 × 16000 × 128
            ≈ 150 TB  (infeasible)
```

**Current Solution**:

- Core path uses dense shared + sparse routed attention (kernel-backed).
- Fallback emulated sparse path exists for unsupported environments/shapes and is slower.

**Future Solution**:

- Expand benchmark coverage and policy tuning across larger shape/device matrices.

### 3. No Dynamic Memory Updates

The "context bank" for inference-time memory updates (described in [`idea/proposal.md`](../idea/proposal.md) §9 and the long-form paper draft §7) is intentionally **out of scope** for the workshop release. The workshop paper covers static end-to-end-trained memory with chapter routing; dynamic updates are explicit future work.

**Reason**: Requires VAE compression, clustering, importance-weighted merging — significant additional complexity that does not bear on the headline result that learned memory is a complementary scaling axis.

### 4. Limited Model Architecture Support

Currently tested with:

- Qwen 2.5 series
- Basic Llama/Mistral structure

Other architectures may need adapter.py modifications.

### 5. KV-Cache Attention Mask Requires Full-Length Mask

`SelfAttention` now requires full-length masks during cached decoding:

- When `past_kv` is provided, `attention_mask.shape[1]` must equal `kv_len`.
- Short step-only masks are rejected with a clear `ValueError`.

This avoids silently unmasking cached padded positions.

### 6. Inference Routing Helpers Are Not Pad-Aware

`inference/routing_strategies.py` uses unmasked mean pooling for sequence and rolling
routers. If hidden states include padding positions, routing logits/chapter selection can
be biased.

**Recommendation**:

- Use these helpers with hidden states that do not include padded positions, or
- Apply masked pooling before routing when padding is present.

### 7. Adapter Hooks + Gradient Checkpointing (Resolved)

**Previous issue**: `MemoryAdapter` used to register temporary forward hooks before
`self.base_model(...)` and remove them in a `finally` block. In `transformers>=4.35`,
decoder layers (Qwen2, Llama, Mistral, etc.) inherit `GradientCheckpointingLayer`, whose
`__call__` wraps forward in `_gradient_checkpointing_func` during backward recompute. With
temporary hooks, recompute ran without memory injection — producing incorrect gradients.

**Current fix**: Hooks are now **persistent** (registered lazily on first `forward()`, never
removed). Per-forward state is stashed on `self._fwd_*` instance attributes. Recompute is
detected via `self._fwd_processed_layers`; side effects (router loss append, routing cache
mutation) are suppressed on recompute, but memory injection always runs.

**Known limitations**:

- The design assumes one forward → one backward per micro-step (standard training with Accelerate). Multiple forwards before a single backward is **not supported** when gradient checkpointing is active on the base model.
- The out-of-band guard is best-effort (see Trade-offs in §5 above). Do not call `self.base_model(...)` directly; always go through `adapter.forward()`.

## Compromises

### 1. Memory Bank Size vs Compute

Large memory banks (the workshop paper uses 262K tokens) require chapter routing.

- With routing: lose some information access (only top-k chapters per input).
- Without routing: O(L × N_m) memory-attention cost dominates the budget.

**Compromise**: example configs span the spectrum — small from-scratch and adapter configs use 1K–16K tokens with light routing, while [`configs/base_small_run2.yaml`](../configs/base_small_run2.yaml) (the workshop paper config) uses 262K tokens with 4,097 chapters and top-k 64. Routing makes the larger bank feasible at iso-FLOP.

### 2. Shared vs Per-Layer Memory

Shared memory bank gives each layer access to all information, but layers may need different info.

**Compromise**: Default to shared (more capacity), but config allows per-layer.

### 3. Low-Rank vs Full Expressiveness

Low-rank memory reduces parameters but limits what each token can express.

**Compromise**:

- Default to full for from-scratch (expressiveness matters)
- Default to low-rank for adapters (parameter efficiency matters)

## Areas for Future Improvement

### High Priority

1. **Token-level routing tuning and benchmarks**: Expand validation/perf tuning for more shapes and GPUs.

2. **More efficient chapter selection**: Current implementation gathers all chapter tokens; could be optimized.

3. **Evaluation coverage expansion**: Core benchmark scripts now exist (MMLU + MCQ suite); next step is adding broader generative/reasoning benchmarks and standardized reporting.

### Medium Priority

4. **Quantization-aware training for memory banks**: post-training quantisation utilities exist (`memory_transformer/quantization.py`, `inference/merge.py`); QAT-style training-time updates are not yet implemented.

5. **Flash Attention 3**: integrate when broadly available.

6. **Scaling laws for learned memory**: extend pretraining beyond 9.6B tokens and sweep bank capacity to formalise the trend already observed in the post-paper runs (the gap to iso-FLOP keeps widening).

### Low Priority (Post-Workshop)

7. **Dynamic context bank**: VAE compression, clustering, memory consolidation.

8. **Kernel-path optimization**: Further optimize sparse routed path policies and fallback behavior.

9. **Mixed precision memory**: Store memory in fp16 but compute in bf16.

## Config Recommendations

### Workshop Paper Reproduction

Use [`configs/base_small_run2.yaml`](../configs/base_small_run2.yaml) for MoC and [`configs/vanilla_control_run2.yaml`](../configs/vanilla_control_run2.yaml) for the iso-FLOP dense baseline. For instruction tuning from a pretrained MoC checkpoint, use [`configs/ift_base_model.yaml`](../configs/ift_base_model.yaml) with `init_from_checkpoint` pointing at the pretrain output.

### For From-Scratch Pretraining (small budget)

```yaml
memory:
  num_memory_tokens: 4096-16384
  memory_sharing: shared
  use_chapters: true     # if N_m > 4K
  use_low_rank_memory: false  # full expressiveness when memory params fit
```

### For Adapter Fine-Tuning

```yaml
memory:
  num_memory_tokens: 1024-2048
  memory_layer_placement: custom        # e.g., first 5 + last 5 layers
  use_low_rank_memory: true
  memory_rank: 256
  use_low_rank_projections: true
  projection_rank: 128
```

### For Efficient Comparison

```yaml
# Compare these 4 configs:
# 1. vanilla_mode: true (no memory baseline)
# 2. use_memory_adapter: true, use_lora: false (memory only)
# 3. use_memory_adapter: false, use_lora: true (LoRA only)
# 4. use_both_memory_and_lora: true (combined)
```

## Debugging Tips

1. **Memory not helping?** Check W_o initialization is zero.
2. **Router collapse?** Enable load_balance_loss.
3. **OOM during training?** Reduce top_k_chapters or use low-rank.
4. **Slow training?** Enable gradient_checkpointing, use smaller batch with accumulation.

## Version History

- **v0.1.0** (Feb 5, 2026): Initial implementation with core features (Session 1).
- **v0.2.0** (Feb 7, 2026): Shared chapter routing, routed scaling factor, expanded wandb metrics (Session 10).
- **v0.3.0** (May 2026): Workshop release — accepted at the ICLR 2026 NFAM Workshop ([`idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf`](../idea/Mixture_of_Chapters_ICLR_Workshop_Paper.pdf)). Sparse routing kernel set extended with v4 (exact MoE-weighted fused) and v5 (joint-bias approximation).
