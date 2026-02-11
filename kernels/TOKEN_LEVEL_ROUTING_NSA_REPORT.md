# Token-Level Memory Routing via NSA: A Plain-English Report

This document explains two things in very simple terms:

1. What "sparse attention" means (vs normal/dense attention).
2. Why people sometimes reshape queries from `[B, L, H, D]` into `[1, B*L, H, D]` (or `[B*L, H, D]`) when using GPU kernels such as Native Sparse Attention (NSA).

It is written for a reader with minimal prior knowledge of attention kernels.

---

## Glossary (Minimal)

- `B`: batch size (number of sequences/examples processed at once)
- `L`: sequence length (tokens per sequence)
- `Q_total = B*L`: total number of tokens across the whole batch
- `H`: number of attention heads
- `D`: head dimension (so hidden size is typically `H*D`)
- `M`: number of memory tokens (external bank size)
- "KV": keys and values (the `K` and `V` tensors used by attention)
- "Self-attention": tokens attend to other tokens in the same sequence
- "Cross-attention": tokens attend to a different set of tokens (here: external memory)
- "Dense attention": each query attends to all keys
- "Sparse attention": each query attends to only some keys
- "Block": a contiguous chunk of keys (e.g., 64 tokens). Block-sparse means we select blocks, not arbitrary individual keys.

---

## 1) What Is Attention (First Principles)

For a single token (a single "query" vector `q`) and a set of keys/values (`k_j`, `v_j`), attention does:

1. Compute scores:
   - `s_j = (q dot k_j) / sqrt(D)`
2. Convert scores to probabilities with softmax:
   - `a_j = exp(s_j) / sum_j exp(s_j)`
3. Produce an output vector:
   - `o = sum_j a_j * v_j`

In a transformer layer, this is done for every token in the sequence, and for every head.

---

## 2) Dense (Normal) Attention vs Sparse Attention

### 2.1 Dense attention

If a query attends to *every* key, that is dense attention.

- Query count: `Q_total = B*L`
- Key count: for self-attention, keys are also `B*L` (per batch element, keys are length `L`)
- Key count: for memory cross-attention, keys are `M` (memory tokens)

Dense attention compute cost (rough intuition):

- Self-attention: `O(B * H * L * L * D)`
- Memory cross-attention (if dense): `O(B * H * L * M * D)`

Dense attention also tends to create (or conceptually involves) a big attention matrix of shape:

- Self-attention per head: `[L, L]`
- Cross-attention per head: `[L, M]`

Even if a "FlashAttention-style" kernel never materializes that full matrix, the *work* is still proportional to `L*L` or `L*M`.

### 2.2 Sparse attention

Sparse attention means each query token attends to only a small subset of keys.

Common sparse patterns:

- Sliding window: each token attends to only the last `W` tokens.
- Block-sparse: each token attends to a selected set of *blocks* of keys.
- Top-k routing: a router chooses the top-k blocks/chapters to attend to.

If each query attends to `K_sel` keys, sparse attention cost is closer to:

- `O(Q_total * K_sel * H * D)` (ignoring constants)

This is why sparse attention is attractive when `M` or `L` is very large.

---

## 3) Your Specific Setting: Token-Level Chapter Routing to Memory

You described:

- Memory bank has `M = C * Nc` tokens.
  - `C`: number of chapters
  - `Nc`: tokens per chapter (so chapter `c` corresponds to token range `[c*Nc, (c+1)*Nc)`).
- Each query token routes to `topk` chapters.

So each query attends to about:

- `K_sel = topk * Nc` memory tokens

If this routing is done per token, there are `Q_total = B*L` different routing decisions.

### Why the naive approach explodes

The naive implementation often does this for each token:

1. Gather the selected memory tokens into a per-token buffer:
   - `gathered_KV[t] = memory_KV[chapter_indices[t]]`
2. Run attention for that token on `gathered_KV[t]`.

But if you gather for every token, you create an intermediate KV tensor shaped like:

- `[Q_total, topk*Nc, H, D]` (or similar)

That is KV duplication: the same memory tokens get copied many times into many tokens' gathered buffers.

This is the "memory explosion" problem.

---

## 4) What NSA (Native Sparse Attention) Is Doing (Conceptually)

NSA is a Triton implementation of sparse attention where:

- Queries are tokens on a timeline.
- Keys/values are tokens on the same timeline.
- Each query selects some key blocks.
- The kernel computes attention without materializing a full `[T, T]` matrix.

Crucially, NSA's "selected" attention is *block-sparse*:

- You provide `block_indices` that say which blocks a query can attend to.
- Block size is `block_size` (like 32/64/128).
- If you pick `topk` blocks, each query attends to `topk * block_size` key tokens.

This is very close to your chapter routing idea, if you set:

- `block_size = Nc` (tokens per chapter)
- `block_indices = chapter_indices` (the chapter IDs)

So why can't we just feed your cross-attention directly?

Because NSA is written like self-attention on one timeline:

- It uses a single length `T` for q/k/v in the selected path.
- It applies causal-like checks comparing query position `t` to key positions (or key block starts).

So the kernel isn't "cross-attention to an external bank" by default.

---

## 5) Why Reshape `[B, L, H, D]` Into `[1, B*L, H, D]`?

This is the most important part of your question.

### 5.1 Why kernels care about shapes

GPU kernels (Triton/CUDA) usually index tokens in a very simple way:

- they iterate over token positions `t = 0..T-1`
- they may also iterate over batch `b = 0..B-1`
- and they use pointer arithmetic like:
  - "the pointer for token `(b, t)` is at base + (b*T + t) * stride"

So the kernel code often assumes:

- a token position index `t` that is valid within some `T`
- and a consistent mapping from `(b, t)` to memory addresses

When you reshape `[B, L, ...]` to `[1, B*L, ...]`, you are doing something simple:

- You are turning "B sequences of length L" into "one sequence of length B*L".

Mathematically, you are just changing the indexing:

- old indexing: token `(b, t)` where `0 <= b < B` and `0 <= t < L`
- new indexing: token `t_flat = b*L + t`, where `0 <= t_flat < B*L`

So:

- `[B, L, H, D]` and `[1, B*L, H, D]` contain the same number of tokens.
- It's just a different view of the same logical list of query tokens.

### 5.2 Why this helps for your case (shared memory across all tokens)

In your memory cross-attention, *all* tokens in the batch should attend into the *same* memory bank.

Many attention kernels are naturally "per sequence":

- They expect each batch element has its own keys/values of length `T`.
- They don't expect a single shared KV bank reused across the entire batch without duplication.

Flattening all tokens into one "sequence" makes it easier to express:

- "these are all queries that attend into the same KV region"

So the kernel can treat them as one set of queries, and the backward pass can naturally accumulate gradients into the shared KV memory.

### 5.3 Why not keep `[B, L, H, D]` and just run B sequences?

You can, but there is a problem:

- You need every sequence to use the same external memory KV.

If the kernel assumes K/V is per batch element, you'd need to either:

1. replicate the memory bank K/V across batch (wastes memory), or
2. modify the kernel to allow batch elements to share KV pointers, or
3. build a varlen layout where each "sequence" refers to the same KV region (often not supported).

Flattening to `[1, B*L, ...]` avoids needing per-batch KV replication in many designs.

### 5.4 Why the special `[1, ...]` batch size appears

Some implementations of "variable length sequences" (varlen) require:

- batch dimension is `1` and you pass `cu_seqlens` offsets to represent multiple sequences inside one packed buffer.

In the fla-org NSA code, there is an assertion in the high-level wrapper that when `cu_seqlens` is provided, `q.shape[0] == 1`.

So people often pack multiple sequences into a single "batch=1" buffer for varlen kernels.

Even if you don't use varlen, `[1, B*L, ...]` is a convenient way to say "one long packed sequence".

---

## 6) How This Connects to "Sparse Attention"

Now connect reshape + sparsity:

1. Flattening changes how we index queries.
2. Sparse attention needs a routing structure (block indices) for each query token.
3. If queries are `[1, B*L, ...]`, then routing indices should align:
   - `block_indices` shape naturally becomes `[1, B*L, ..., topk]` (plus heads depending on kernel).

So the reshape makes routing "per token" easier to represent as one array indexed by `t_flat`.

---

## 7) Path A vs Path B (How the Reshape Fits In)

### Path A (no kernel edits): prefix timeline

NSA selected kernels are causal and assume one timeline length `T`.

To use them for memory cross-attention:

1. Create a single fake timeline of length `T = M + Q_total`.
2. Put memory tokens as keys/values at the beginning (positions `0..M-1`).
3. Put real queries after that (positions `M..M+Q_total-1`).
4. Route each real query to memory blocks only.

This still uses the flattening idea because your real queries are usually formed by flattening `[B, L]` into `[Q_total]`.

### Path B (kernel edits): real cross-attention mode

You edit kernel so:

- queries have length `Tq = Q_total`
- keys/values have length `Tk = M`
- no causal constraint in selected mode for cross-attn

In Path B, flattening is still useful because you want to run one kernel over all queries:

- `q` is `[Q_total, H, D]` or `[1, Q_total, H, D]`

but now you do NOT need a fake prefix timeline.

---

## 8) Common Confusions (Quick Answers)

### "Does `[1, B*L, H, D]` mean I changed the model?"
No. It is only a different view/indexing of the same set of tokens.
You can always invert it:

- `t_flat = b*L + t`
- `b = t_flat // L`
- `t = t_flat % L`

### "Is sparse attention only about fewer keys?"
Yes. Sparse attention means each query attends to fewer keys than the full set.
The kernel uses your `block_indices` to only load those keys/values.

### "Why do we care about blocks?"
Blocks make GPU kernels faster:

- loading contiguous chunks is efficient
- indexing overhead is smaller
- many kernels are tuned for block sizes like 32/64/128/256

---

## 9) Practical Takeaway For Your Token-Level Routing Experiment

If you want to prototype token-level routed memory attention with an NSA-like kernel:

1. Flatten queries so routing is per-token in one dimension:
   - from `[B, L, H, D]` to `[1, B*L, H, D]`
2. Provide `block_indices` aligned to that flattened token index.
3. Choose `block_size = Nc` (or pad `Nc` to a kernel-friendly block size).
4. Decide Path A (prefix timeline, no kernel edits) or Path B (true cross-attn kernel edits).

---

## 10) Notes About This Repo Environment

On this machine, GPU execution isn't available (`torch.cuda.is_available() == False`), and Triton/FlashAttention aren't installed, so this report is conceptual and based on static reading of the NSA source code.

---

If you want, I can add a second document that is purely "numbers and shapes" for one concrete configuration (e.g., `B=128, L=2048, C=100, Nc=100, topk=10`) showing every tensor shape and index mapping step-by-step.

