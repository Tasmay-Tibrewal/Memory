# Learnable Memory Banks for Transformer Architectures

## Table of Contents

1. [Motivation](#1-motivation)
2. [Core Architecture](#2-core-architecture)
3. [Memory Bank Configuration](#3-memory-bank-configuration)
4. [Scaling via Chapter Routing](#4-scaling-via-chapter-routing)
5. [Memory Efficiency and Compression](#5-memory-efficiency-and-compression)
6. [Training and Fine-Tuning](#6-training-and-fine-tuning)
7. [Experimental Plan](#7-experimental-plan)
8. [Token-Level Routing](#8-token-level-routing)
9. [Dynamic Memory Update During Inference](#9-dynamic-memory-update-during-inference)
10. [Additional Technical Considerations](#10-additional-technical-considerations)

---

## 1. Motivation

Transformer-based models do not have an explicit memory mechanism in their architecture. Knowledge is stored implicitly in the model weights, learned during training, but there is no dedicated component for storing and retrieving information in a structured way.

There have been engineering-level solutions to this: saving information extracted during inference and feeding it back into the model's context on subsequent runs. These are applied solutions. They do not address the problem at the architectural level. What we are proposing is an architectural modification that gives transformers a learnable memory component, one that is trained alongside the rest of the model and can be queried through the standard attention mechanism.

---

## 2. Core Architecture

### 2.1 Memory Bank

The memory bank is a set of $N_m$ latent tokens, each of dimension $d$ (the model's embedding dimension). These tokens are learnable parameters, initialised randomly at the start of training and updated via backpropagation just like any other parameter in the model. The memory bank is not part of the input sequence; it is a fixed set of parameters that the model learns to use as a store of information.

### 2.2 Cross-Attention Mechanism

The model interacts with the memory bank through a cross-attention layer added to its transformer blocks. In this cross-attention:

- The hidden state tokens (from the input sequence) provide the **Query** vectors.
- The memory bank tokens provide the **Key** and **Value** vectors.

This is the same way self-attention allows tokens to refer to other tokens in the sequence; here, the input tokens attend to the memory bank to extract stored information. Suppose we have $N_m = 10{,}000$ memory tokens and a sequence of $S = 100$ tokens. The 100 input tokens will cross-attend to all 10,000 memory tokens, extracting whatever information is relevant to each token's representation. The MLP layer follows after this cross-attention.

### 2.3 Block Structure Variations

There are two configurations for a transformer block with memory:

**Variation 1:** Self-Attention → Memory Cross-Attention → MLP

**Variation 2:** Self-Attention → MLP → Memory Cross-Attention → MLP

Variation 2 has two MLP layers per block. Both configurations are repeated $N$ times to form the full model.

Memory cross-attention does not need to be in every block. It can be placed selectively: say in the first 5 blocks only, or every 3rd block, or only in the last 5 blocks. This gives flexibility in where memory retrieval happens within the network.

---

## 3. Memory Bank Configuration

### 3.1 Per-Layer vs Shared Memory Bank

Two options for how the memory bank is organised across layers:

1. **Per-layer memory banks:** Each memory-reference layer has its own dedicated memory bank. For example, 1,000 tokens per layer across 10 layers, for a total of 10,000 memory tokens across the model.

2. **Shared memory bank:** A single memory bank (say 10,000 tokens) is shared across all memory-reference layers. Every layer that has a memory cross-attention layer attends to the same set of memory tokens.

### 3.2 Manifold Alignment in Shared Memory

Each layer in a transformer operates in a slightly different vector space (manifold). If a single memory bank is shared, the Key and Value projections at each layer need to map memory tokens into that layer's vector space before attention can be computed meaningfully.

This is handled by the learned $W_k$ and $W_v$ matrices. At each memory-reference layer, $W_k$ and $W_v$ do not just learn the standard Key/Value projections; they also absorb the transformation needed to map from the memory bank's representation space into the current layer's space. A $d \times d$ transformation matrix is sufficient to map between any two spaces of dimension $d$. Since $W_k$ and $W_v$ are already $d \times d$, they can learn both the manifold transformation and the Key/Value projection simultaneously, with no increase in parameter count.

Formally, the effective weight matrices are:

$$W_k^* = W_k \cdot T_k, \quad W_v^* = W_v \cdot T_v$$

where $T_k$ and $T_v$ are the manifold transformation matrices, both $d \times d$. The product $W_k \cdot T_k$ is still $d \times d$, same size as $W_k$ alone. We have:

$$W_k^*.size = W_v^*.size = (W_k \cdot T_k).size = (W_v \cdot T_v).size = (d \times d) \cdot (d \times d) = d \times d = W_k.size = W_v.size$$

This shows that a $d \times d$ matrix is sufficient to learn both the layer-specific manifold transformation and the Key/Value weight matrices together. The model learns $W_k^*$ and $W_v^*$ end-to-end; no separate alignment step is needed.

The advantage of shared memory is that each layer gets access to the full pool of stored information. The potential downside is that different layers may benefit from different types of information (earlier layers might need syntactic structure, middle layers might need raw factual knowledge), and a shared bank could force layers to attend to tokens that are not relevant to them. Routing (Section 4) partially addresses this by letting each layer select only the relevant subset.

---

## 4. Scaling via Chapter Routing

### 4.1 The Scaling Problem

Cross-attention over $M$ memory tokens has computation that scales as $O(M)$. For large memory banks (say 100,000 tokens), attending to all of them at every layer is expensive. We need a way to select only the relevant subset of memory for each layer and each input.

### 4.2 Chapter-Based Routing

Inspired by Mixture of Experts (MoE), we divide the memory bank into **chapters**, just like sections in a textbook. For example, 100,000 memory tokens are split into 100 chapters of 1,000 tokens each.

At each memory-reference layer, a **router** is trained alongside the rest of the model. The router takes the current layer and the input sequence into account, and outputs an importance score for each chapter (analogous to expert importance in MoE). We then select the top-$k$ chapters (say top 20 out of 100) and attend only to those, reducing the number of tokens attended from 100,000 down to 20,000.

The routing can be quite sparse here. In MoE, each token is routed to a small but non-negligible fraction of experts. For memory chapters, the amount of information needed to answer any given query is much smaller and more specific than the total stored knowledge, so we can afford to be more selective than the typical sparsity level used with experts.

---

## 5. Memory Efficiency and Compression

### 5.1 Representation Density

Since the memory bank is learned during training, we hypothesise that it will store information in a more compressed and efficient form than raw textual memory or other vector-based memory implementations. The model is free to organise the memory bank in whatever way is most useful for retrieval, and gradient-based training will push it toward dense, information-rich representations.

### 5.2 Quantisation

To reduce the memory footprint of the memory bank on GPU, we can store the memory tokens in quantised form (say 4-bit or 8-bit), similar to how model weights are quantised for inference.

### 5.3 Low-Rank Factorisation

We can decompose the memory bank into a product of two lower-rank matrices, inspired by LoRA. If the memory bank $M$ has shape $N_m \times d$, we factorize it as:

$$M = A \cdot B, \quad A \in \mathbb{R}^{N_m \times r}, \quad B \in \mathbb{R}^{r \times d}$$

where $r \ll d$. Storage drops from $N_m \cdot d$ to $N_m \cdot r + r \cdot d$. The trade-off is that each memory token is constrained to a rank-$r$ subspace and can store less information per token.

---

## 6. Training and Fine-Tuning

### 6.1 Preventing Knowledge Loss During Fine-Tuning

When a pre-trained model is fine-tuned on a downstream task, there is generally some information loss as the model adapts to the new domain. The memory bank, which was populated with knowledge during pre-training, is at risk of being overwritten or degraded during fine-tuning.

To prevent this, we can either:

- **Freeze the memory bank** (and optionally $W_k$, $W_v$) after pre-training, so that the pre-trained knowledge stored in memory is locked in.
- Use a **much lower learning rate** for the memory bank during fine-tuning compared to the rest of the model.

Either approach preserves the pre-trained knowledge in memory while still allowing the rest of the model to adapt to the new domain.

---

## 7. Experimental Plan

### 7.1 From-Scratch Training

The full experimental setup involves training a small model (around 100M parameters) from scratch on 5 to 10 billion tokens for pre-training, followed by instruction fine-tuning on around 10 to 100 million tokens. We then compare a model with memory layers against a model of the same size and trained on the same data but without memory layers, across standard benchmarks.

### 7.2 Submission Target

We are considering submitting this work to the **ICLR 2026 Workshop: New Frontiers in Associative Memories** (https://iclr.cc/virtual/2026/workshop/10000782). The submission deadline is 14th February, which leaves roughly 3 weeks. Pre-training a model from scratch and running a full set of experiments in this window is tight, which motivates the adapter-based experiments described below.

### 7.3 Memory Adapters

Given the time constraint, we also propose a more immediately feasible set of experiments: using the memory mechanism as an **adapter** on top of an existing pre-trained model, rather than training from scratch.

We take an existing model (say Qwen 2.5 Math 1.5B or Qwen 2.5 1.5B) and fine-tune it on a downstream task under four conditions:

| Method | Description |
|---|---|
| Full Fine-Tuning | Standard full-model fine-tuning (baseline) |
| LoRA | Parameter-efficient fine-tuning via low-rank adaptation |
| Mem-Adapters | Memory cross-attention layers added as adapters (ours) |
| Mem-Adapters + LoRA | Both memory adapters and LoRA combined |

For the training data, we can use:

- **DeepSeek R1 reasoning traces** from the Open-R1 math dataset. It is not entirely clear how much impact memory will have on reasoning specifically, but reasoning patterns and problem-solving structure are the kind of thing that could be stored in a memory bank, so it is worth testing.
- **Medical or legal datasets**, which are more knowledge-heavy. Storing domain-specific factual information is a more natural fit for this architecture, so these may show a clearer signal than math reasoning.

### 7.4 Memory Adapter Configuration

When using memory as an adapter, we have several configuration choices:

- Place memory adapters in **selective layers** only, rather than all layers.
- Store the memory bank in **quantised** form.
- Use **low-rank** memory tokens.

For the projection matrices in the adapter, there are two options:

1. $W_q$, $W_k$, $W_v$ project to $d$ dimensions (standard $d \times d$ matrices), and the memory bank is stored in factorised form as $A \cdot B$ where $A \in \mathbb{R}^{N_m \times r}$ and $B \in \mathbb{R}^{r \times d}$. The full $d$-dimensional representation is reconstructed before attention is computed.

2. $W_q$, $W_k$, $W_v$ project to $r$ dimensions ($d \times r$ matrices), and the memory bank is stored directly as $N_m \times r$ (each token is $r$-dimensional). The entire attention computation runs in the reduced $r$-dimensional space.

All of these configuration choices (selective layers, quantisation, low-rank storage, the two projection options) also apply to the from-scratch training setup in Section 7.1. They are not limited to the adapter setting.

---

## 8. Token-Level Routing

This section describes a known issue with token-level chapter routing identified in the current codebase, the reformulation needed to support it, and why it is not yet practical.

### 8.1 Current Implementation: Sequence-Level Routing

In the current setup, routing to memory chapters is done **once per sequence**. The hidden states across the sequence are averaged, and this single averaged vector is fed to the router to select chapters. All tokens in the sequence then attend to the same set of memory chapters. During autoregressive generation, routing can be done per generated token since we process one token at a time, but during training and prefill the routing is sequence-level.

The tensor dimensions are:

- $Q$: $B \times H \times S \times D$
- $K$: $B \times H \times M_r \times D$

where $B$ is batch size, $H$ is number of heads, $S$ is sequence length, $D$ is head dimension ($D = d / H$, where $d$ is the embedding dimension), and $M_r$ is the total number of routed memory tokens (selected chapters $\times$ tokens per chapter).

Attention is computed as $Q \cdot K^T$, where $K^T$ transposes the last two dimensions of $K$. For this matmul to be valid in PyTorch, all dimensions except the last two must match between the two tensors (i.e., $B$ and $H$ must be the same), and the second-to-last dimension of the second tensor must equal the last dimension of the first tensor.

### 8.2 Why Token-Level Routing is Desirable

In MoE, routing happens at the **token level**: each token is independently routed to a different subset of experts. Ideally, we want the same for memory chapters. Different tokens in a sequence may need different information from the memory bank, and sequence-level routing is a coarse approximation that forces all of them to use the same chapters.

For token-wise routing, the $K$ matrix would need to be $B \times S \times H \times M_r \times D$ (each of the $B \times S$ tokens having its own routed memory). In terms of the routed token size: $M_r \times d = M_r \times H \times D = H \times M_r \times D$, where $d$ is the full embedding dimension. But this is not directly compatible with the standard $Q$ shape for matmul, which motivates the reformulation below.

### 8.3 Token-Level Routing Reformulation

To enable token-level routing, we reshape $Q$ so that each token is treated independently:

**Query transformation:**

$$Q: B \times H \times S \times D$$

$$\xrightarrow{\text{transpose dims 1 and 2}} B \times S \times H \times D$$

$$\xrightarrow{\text{view (merge dims 0 and 1)}} (B \cdot S) \times H \times D$$

$$\xrightarrow{\text{unsqueeze (expand 1 dim)}} (B \cdot S) \times H \times D \times 1$$

$$\xrightarrow{\text{transpose dims 2 and 3}} (B \cdot S) \times H \times 1 \times D$$

**Key (per-token routed):**

$$K: (B \cdot S) \times H \times M_r \times D$$

Now $Q$ and $K^T$ are compatible for matmul. The result has shape $(B \cdot S) \times H \times 1 \times M_r$.

To convert back to the standard attention shape:

$$(B \cdot S) \times H \times 1 \times M_r \xrightarrow{\text{view}} B \times S \times H \times 1 \times M_r \xrightarrow{\text{transpose dims 1 and 3}} B \times H \times S \times M_r$$

This is the same shape as the result from sequence-level routing. From here, softmax is applied, the result is multiplied with $V$ (which undergoes the same per-token reshaping), and the final output is converted back to $B \times S \times H \times D$ before being added to the residual embedding vector $X$.

### 8.4 Memory Requirement Analysis

The problem is the size of $K$ when expanded per-token. With practical values:

| Parameter | Value |
|---|---|
| $B$ (batch size) | 250 |
| $S$ (sequence length) | 10,000 |
| $H$ (num heads) | 32 |
| $D$ (head dim) | 128 |
| $M_r$ (routed memory tokens) | 16,000 |

The $K$ tensor has shape $(B \cdot S) \times H \times M_r \times D$. At full precision this comes out to roughly **150 TB**, which is far beyond what can be stored in GPU memory. Iterating over it would also break batch-wise efficiency.

### 8.5 Custom CUDA Kernel Approach

The key observation is that many tokens in the sequence will be routed to the **same** chapters. Rather than materialising a separate copy of each chapter for every token that uses it, we can write a custom matmul CUDA kernel that avoids this duplication.

Instead of loading $(B \cdot S) \times H \times M_r \times D$ into $K$, the kernel loads only the **unique** chapters:

$$K_{\text{unique}}: C \times H \times T_c \times D$$

where $C$ is the number of unique chapters (at most the total chapter count, say 1,000) and $T_c$ is tokens per chapter (say 1,000). For a 1M-token memory bank with embedding dimension 4096, this is at most around **4 GB**.

The kernel also maintains a **reference table** that records, for each of the $B \cdot S$ query tokens, which chapters it should attend to. In the previous formulation, $B \times S \times \text{num\_chapters\_routed}$ equalled 40 million (with 1,000 chapters of 1,000 tokens each). The reference table stores these 40 million mappings, taking roughly **40 MB** of additional storage.

During the matmul, instead of loading a chapter once for every query token that references it, the kernel reuses the same chapter data across all referencing queries. The $K$ matrix effectively has a different prefix dimension size than $Q$, and the matmul is handled not by direct dimension-matched multiplication, but by the kernel using the reference table to route each query to the correct chapter rows. One chapter is used by multiple query vectors rather than being duplicated for each one.

This brings the total memory requirement down from ~150 TB to ~4 GB + 40 MB, a reduction of roughly 40,000x. The number of flops is roughly the same as the naive approach (other than the overhead of referencing), but the irregular access pattern reduces the parallelism that GPUs can exploit, so there will be some slowdown compared to standard batched matmul. Given the memory savings, this is a worthwhile trade-off if an efficient enough implementation can be achieved.

### 8.6 Practical Limitation

Implementing this custom CUDA kernel requires significant CUDA expertise. For the current work and for the workshop submission, we will use **sequence-level routing only**. Token-level routing with the custom kernel is left as a future extension.

---

## 9. Dynamic Memory Update During Inference

*This part of the proposal is intended for a later stage and is unlikely to be included in the workshop submission due to time constraints.*

### 9.1 Context Bank Overview

The memory bank described above is **static** after training: it does not change once pre-training or fine-tuning is done. This section describes a second component, the **context bank**, which is updated at inference time to store new information as the model runs.

This is not limited to retaining information within a single long conversation. The context bank is meant to persist **across conversations**, similar to how humans carry knowledge from one interaction to the next. The memory bank (trained knowledge) and the context bank (runtime knowledge) are kept **separate**, so that inference-time updates do not corrupt the pre-trained memory.

### 9.2 Compression via VAE

To populate the context bank, we use a VAE-based autoencoder to compress input prompt tokens. Say a sequence of 10,000 tokens is compressed down to 100 latent tokens. These compressed tokens are appended to the context bank, and the bank grows over time as the model processes more inputs, until the maximum limit is reached (say 100,000 tokens).

The VAE is trained to compress and reconstruct the **concatenation of last-layer and first-layer hidden states**:

- **Last-layer embeddings** are included because by the final layer, information has been fully mixed across the sequence, giving a holistic representation of the input.
- **First-layer embeddings** are included to retain the raw semantic content of the original tokens, which the last-layer representations may have abstracted away.

Concatenating both gives the VAE a richer signal to learn compression from. An alternative is to train a time-series VAE that treats the sequence of hidden states as a temporal signal and compresses it accordingly.

### 9.3 Retrieval from the Context Bank

At inference time, we cannot train a router (there is no gradient). Instead, we divide the context bank into clusters using k-means or hierarchical clustering. To retrieve information relevant to the current input, we compute the dot product between the current query vector and each cluster centroid, and select the top-$k$ clusters. This dot product acts as the router. The model then attends only to the tokens in those selected clusters, keeping computation tractable even for large context banks (say up to 1M tokens).

### 9.4 Handling Maximum Capacity

The context bank has a maximum size. Once this limit is reached and we need to add new compressed tokens, we have to make room first.

The approach: suppose we want to add 100 new tokens but the bank is full. We find the 100 pairs of existing tokens that are **closest to each other** in embedding space, and add each pair element-wise. This turns 200 tokens into 100 (the sum of each pair), freeing up space for the 100 new tokens.

This addition-based merging is inspired by how retrieval-augmented generation (RAG) systems work. In RAG, hundreds or thousands of embeddings in a chunk are added together or averaged, and despite the compression, most of the semantic information is preserved for matching. By restricting merges to tokens that are already close in embedding space, we further reduce the information loss compared to merging arbitrary tokens.

### 9.5 Importance and Recency Weighting

The merging strategy above treats all tokens equally when deciding which pairs to merge. We can make it more selective by factoring in additional signals:

- **Importance:** How often a token has been attended to (referenced) during inference, and how much it contributed on average when it was referenced. Contribution per reference is measured by the token's softmax weight in the attention computation. A token that consistently gets high attention weights is storing information the model actively uses, and should be retained.

- **Age:** How many inference runs (context bank updates) have occurred since the token was added. Age is measured in terms of how many times the model has been run after training, where one run corresponds to one prompt being processed and the context bank being updated. Newer tokens have had fewer opportunities to be referenced, so they should be given the benefit of the doubt and retained longer. Older tokens that have also been referenced less are the best candidates for merging.

We can define a dynamic weighting function that combines recency, expected information loss from merging, and importance to decide which pairs of tokens to merge. This is roughly analogous to how biological memory systems tend to retain recent and frequently accessed information while letting older, less-used information fade.

### 9.6 Use Cases

- **Long-context tasks:** Models currently degrade after around 400,000 effective tokens. Techniques like context compression and important-memory documenting help, but will eventually run out for tasks that run for hours or days (autonomous agents, for example). A large context bank provides a much longer retention window with more efficient, precise representations.

- **Personal agents:** The model can retain information about a specific user across many conversations. Because this is an architectural solution, it can remember fine-grained details accurately. This is a step up from current approaches (like ChatGPT's memory feature) which store and retrieve raw text summaries from tool use.

- **Rapid knowledge updates:** Recent information (news, events) can be stored in the context bank without any retraining of the model.

All optimisations discussed for the memory bank (quantisation, low-rank storage, chapter routing) can also be applied to the context bank.

---

## 10. Additional Technical Considerations

### 10.1 Token-Level Routing During Autoregressive Generation

While token-level routing is not feasible during prefill or training (due to the memory constraints described in Section 8.4), it becomes practical during autoregressive generation. During generation, we process one token at a time ($S = 1$), which eliminates the sequence length factor from the memory requirement entirely.

The overhead of storing routed memory tokens in VRAM alongside the KV cache during generation is around **200 MB per sample** at full precision, assuming routing to 50,000 memory tokens with embedding dimension 4096. This overhead can be reduced further using the custom CUDA kernel approach from Section 8.5, where only the unique chapters need to be loaded rather than per-token copies.

### 10.2 Scalable Clustering for Large Context Banks

For very large context banks, standard k-means becomes expensive when adding new tokens or performing retrieval. More scalable alternatives:

- **Hierarchical clustering:** Builds a tree structure over the tokens, enabling logarithmic-time search for the relevant clusters.

- **Online k-means:** Updates cluster centroids incrementally as new tokens are added, avoiding full re-clustering of the entire bank each time.

- **IVF-PQ (Inverted File with Product Quantization):** Uses an inverted index structure with quantised token representations. This enables memory-efficient storage and fast approximate nearest-neighbor search, and scales well to millions of tokens.

### 10.3 Low-Rank Dimension Reduction

Section 5.3 describes low-rank **factorisation**, where the memory bank is decomposed into two matrices whose product reconstructs the full $d$-dimensional representation before attention is computed. There is a distinct alternative: store the memory tokens directly in a reduced dimension $r$ and run the entire cross-attention in that lower-dimensional space.

In this approach:

- Memory bank $M$: stored as $N_m \times r$ (reduced dimension, $r \ll d$, for example $r = 512$ instead of $d = 4096$)
- $W_q$: projects from $d$ to $r$ (maps queries into reduced space)
- $W_k$, $W_v$: map from $r$ to $r$ (operate within the reduced space)
- $W_o$: projects from $r$ to $d$ (maps output back to full dimension)

The cross-attention computation becomes:

$$Q_r = H \cdot W_q \quad \in \mathbb{R}^{L \times r}$$

$$K_r = M \cdot W_k \quad \in \mathbb{R}^{N_m \times r}$$

$$V_r = M \cdot W_v \quad \in \mathbb{R}^{N_m \times r}$$

$$\text{Attention} = \text{softmax}\!\left(\frac{Q_r \cdot K_r^T}{\sqrt{r}}\right) \cdot V_r \quad \in \mathbb{R}^{L \times r}$$

$$\text{Output} = \text{Attention} \cdot W_o \quad \in \mathbb{R}^{L \times d}$$

Storage for the memory bank drops by a factor of $d / r$ (8x for $r = 512$, $d = 4096$). Each memory token stores less information, but the reduction in storage allows for a much larger memory bank within the same memory budget.

The distinction from Section 5.3 is important: in factorisation, the full $d$-dimensional representation is reconstructed from the factor matrices before attention is computed. Here, the entire attention mechanism operates in the reduced $r$-dimensional space, and only the final output projection maps back to $d$.
