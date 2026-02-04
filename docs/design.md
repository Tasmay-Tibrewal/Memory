# Design Decisions and Notes

This document records design choices, compromises, known issues, and areas for future improvement.

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

### 3. Sequence-Level Routing Only

**Choice**: Route at sequence level (mean-pool), not token level.

**Rationale**:
- Token-level routing is memory-prohibitive during training/prefill
- Would require custom CUDA kernel for efficient implementation
- Sequence-level still provides reasonable chapter selection
- Token-level can be used during generation (seq_len=1)

**Future**: Custom CUDA kernel could enable token-level routing.

### 4. Zero-Initialized Output Projection

**Choice**: Initialize W_o in memory cross-attention to zero.

**Rationale**:
- Ensures model behaves exactly like base model at initialization
- Memory contribution gradually increases as training progresses
- Prevents instability when injecting adapters into pretrained models

### 5. Router Hooking for Adapter Mode

**Choice**: Use PyTorch hooks to inject memory after each layer.

**Rationale**:
- Non-invasive to pretrained model code
- Works across different model architectures
- Alternative (patching forward) would be more fragile

**Trade-off**: Slight overhead from hook mechanism.

## Known Limitations

### 1. No Quantized Memory Bank (Yet)

**Status**: Config option exists but not fully implemented.

**Issue**: Quantizing during training requires careful handling of gradients.

**Workaround**: Use low-rank factorization for memory compression instead.

**Future**: Implement QAT or post-training quantization.

### 2. Token-Level Routing Memory Issue

As discussed in idea.txt (lines 62-80), token-level routing during prefill would require:
```
K tensor size = B × S × num_heads × routed_tokens × D
            = 250 × 10000 × 32 × 16000 × 128
            ≈ 150 TB  (infeasible)
```

**Current Solution**: Sequence-level routing only.

**Future Solution**: Custom CUDA kernel that stores unique chapters once and uses index references.

### 3. No Dynamic Memory Updates

The "context bank" for inference-time memory updates (idea.txt lines 84-102) is NOT implemented for the ICLR workshop deadline.

**Reason**: Requires VAE compression, clustering, importance-weighted merging—significant additional complexity.

### 4. Limited Model Architecture Support

Currently tested with:
- Qwen 2.5 series
- Basic Llama/Mistral structure

Other architectures may need adapter.py modifications.

## Compromises

### 1. Memory Bank Size vs Compute

Large memory banks (100k+ tokens) require chapter routing.
- With routing: lose some information access
- Without routing: O(L × N_m) attention cost

**Compromise**: Default to moderate bank (2k-4k) with routing for larger.

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

1. **Token-level routing during generation**: Currently uses sequence-level even for single tokens.

2. **More efficient chapter selection**: Current implementation gathers all chapter tokens; could be optimized.

3. **Evaluation scripts**: Currently only train/inference, need proper eval benchmarks.

### Medium Priority

4. **Quantized memory banks**: Post-training quantization or QAT.

5. **Flash Attention 3**: When available, integrate for better performance.

6. **Gradient checkpointing for memory**: Currently checkpoints base model only.

### Low Priority (Post-Workshop)

7. **Dynamic context bank**: VAE compression, clustering, memory consolidation.

8. **Custom CUDA kernel**: For token-level routing during prefill.

9. **Mixed precision memory**: Store memory in fp16 but compute in bf16.

## Config Recommendations

### For From-Scratch Pretraining
```yaml
memory:
  num_memory_tokens: 4096-16384
  memory_sharing: shared
  use_chapters: true (if >4096 tokens)
  use_low_rank_memory: false  # Full expressiveness
```

### For Adapter Fine-tuning
```yaml
memory:
  num_memory_tokens: 1024-2048
  memory_layer_placement: custom  # First/last 5 layers
  use_low_rank_memory: true
  memory_rank: 256
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

- v0.1.0: Initial implementation with core features
