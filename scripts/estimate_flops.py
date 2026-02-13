#!/usr/bin/env python3
"""
Estimate training FLOPs from a YAML config.

This is a comprehensive analytic estimator for from-scratch MemoryTransformer
configs. It includes:
- Self-attention / memory-attention projections and attention matmuls
- LayerNorm/RMSNorm, MLP activations, residual adds, RoPE
- Router forward + router auxiliary-loss compute
- Memory-bank materialization cost for factorized banks
- Chapter-routing preprocessing (memory weighting / shared-routed normalization)
- LM head and cross-entropy softmax
- Forward + backward + recompute from gradient checkpointing

Notes:
- GEMM/matmul FLOPs are exact for the configured tensor shapes.
- Elementwise/nonlinear terms are principled approximations.
- Optimizer-update FLOPs, communication, and memory bandwidth are excluded.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_transformer.config import (
    get_memory_bank_assignments,
    get_memory_layer_indices,
    load_config,
)


SOFTMAX_FLOPS_PER_SCORE = 5.0
ACTIVATION_FLOPS = {
    "swiglu": 4.0,
    "silu": 4.0,
    "gelu": 8.0,
    "relu": 1.0,
    "sigmoid": 4.0,
    "tanh": 4.0,
}


def safe_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return int(value)


def linear_flops(tokens: int, in_dim: int, out_dim: int) -> int:
    return int(2 * tokens * in_dim * out_dim)


def fmt_flops(flops: float) -> str:
    return f"{int(flops):,} ({flops / 1e12:.3f} TFLOPs)"


def norm_flops(vectors: int, dim: int, use_rms_norm: bool) -> int:
    per_vec = (4 * dim + 4) if use_rms_norm else (8 * dim + 8)
    return int(vectors * per_vec)


def branch_norm_flops(vectors: int, dim: int, norm_type: str) -> int:
    nt = norm_type.lower()
    if nt == "rms":
        return norm_flops(vectors, dim, use_rms_norm=True)
    if nt == "layernorm":
        return norm_flops(vectors, dim, use_rms_norm=False)
    raise ValueError(f"Unsupported shared_routed_norm_type: {norm_type}")


def rope_flops(tokens: int, q_dim: int, k_dim: int, use_rope: bool) -> int:
    if not use_rope:
        return 0
    return int(3 * tokens * (q_dim + k_dim))


def attention_score_elementwise_flops(
    batch: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    *,
    include_mask_scale: bool,
    include_dropout: bool,
) -> int:
    scores = batch * num_heads * q_len * kv_len
    per_score = SOFTMAX_FLOPS_PER_SCORE
    if include_mask_scale:
        per_score += 2.0
    if include_dropout:
        per_score += 1.0
    return int(scores * per_score)


def attention_matmul_flops(batch: int, q_len: int, kv_len: int, attn_dim: int) -> int:
    return int(4 * batch * q_len * kv_len * attn_dim)


def mlp_activation_flops(tokens: int, intermediate_dim: int, hidden_activation: str) -> int:
    act = hidden_activation.lower()
    if act not in ACTIVATION_FLOPS:
        raise ValueError(
            f"Unknown hidden_activation '{hidden_activation}'. Supported: {sorted(ACTIVATION_FLOPS.keys())}"
        )
    if act == "swiglu":
        return int(tokens * intermediate_dim * (ACTIVATION_FLOPS["swiglu"] + 1.0))
    return int(tokens * intermediate_dim * ACTIVATION_FLOPS[act])


def router_aux_loss_flops(samples: int, num_chapters: int, top_k: int) -> float:
    s = float(samples)
    c = float(num_chapters)
    k = float(max(top_k, 1))

    f_ops = s * c * max(k - 1.0, 0.0) + c * max(s - 1.0, 0.0) + c
    p_ops = c * max(s - 1.0, 0.0) + c
    load_balance_ops = c + max(c - 1.0, 0.0) + 1.0

    p_sq_ops = s * c + c * max(s - 1.0, 0.0) + c
    mse_ops = c + c + max(c - 1.0, 0.0) + 1.0

    z_logits_ops = s * (5.0 * c)
    z_reduce_ops = s + max(s - 1.0, 0.0) + 1.0
    entropy_ops = s * c * 4.0

    return f_ops + p_ops + load_balance_ops + p_sq_ops + mse_ops + z_logits_ops + z_reduce_ops + entropy_ops


@dataclass
class Comp:
    matmul: float = 0.0
    elemwise: float = 0.0

    def total(self) -> float:
        return self.matmul + self.elemwise


@dataclass
class Breakdown:
    self_attn: Comp
    memory_attn: Comp
    mlp: Comp
    norms: Comp
    residuals: Comp
    rope: Comp
    router: Comp
    router_losses: Comp
    memory_preprocess: Comp
    memory_bank_materialization: Comp
    lm_head: Comp
    loss: Comp
    checkpoint_recompute: float
    memory_attn_internal_recompute: float

    def forward_matmul(self) -> float:
        return (
            self.self_attn.matmul
            + self.memory_attn.matmul
            + self.mlp.matmul
            + self.router.matmul
            + self.router_losses.matmul
            + self.memory_preprocess.matmul
            + self.memory_bank_materialization.matmul
            + self.lm_head.matmul
        )

    def forward_elemwise(self) -> float:
        return (
            self.self_attn.elemwise
            + self.memory_attn.elemwise
            + self.mlp.elemwise
            + self.norms.elemwise
            + self.residuals.elemwise
            + self.rope.elemwise
            + self.router.elemwise
            + self.router_losses.elemwise
            + self.memory_preprocess.elemwise
            + self.memory_bank_materialization.elemwise
            + self.loss.elemwise
        )

    def forward_total(self) -> float:
        return self.forward_matmul() + self.forward_elemwise()


def chapter_stats(config) -> Dict[str, int]:
    mem = config.memory
    if not mem.use_chapters:
        return {
            "num_chapters": 0,
            "tokens_per_chapter": 0,
            "shared_chapters": 0,
            "routed_chapters_selected": 0,
            "routed_chapters_available": 0,
            "shared_tokens": 0,
            "routed_tokens_selected": 0,
            "routed_tokens_available": 0,
            "selected_non_token_tokens": int(mem.num_memory_tokens),
        }

    num_chapters = safe_int("memory.num_chapters", int(mem.num_chapters))
    num_memory_tokens = safe_int("memory.num_memory_tokens", int(mem.num_memory_tokens))
    if num_memory_tokens % num_chapters != 0:
        raise ValueError(
            f"memory.num_memory_tokens ({num_memory_tokens}) must be divisible by "
            f"memory.num_chapters ({num_chapters})"
        )
    tpc = num_memory_tokens // num_chapters
    shared = max(0, min(int(mem.num_shared_chapters), num_chapters))
    available = num_chapters - shared
    routed_selected = min(int(mem.top_k_chapters), available)

    return {
        "num_chapters": num_chapters,
        "tokens_per_chapter": tpc,
        "shared_chapters": shared,
        "routed_chapters_selected": routed_selected,
        "routed_chapters_available": available,
        "shared_tokens": shared * tpc,
        "routed_tokens_selected": routed_selected * tpc,
        "routed_tokens_available": available * tpc,
        "selected_non_token_tokens": (shared + routed_selected) * tpc,
    }


def router_flops(
    batch: int,
    seq_len: int,
    hidden_dim: int,
    num_chapters: int,
    top_k: int,
    strategy: str,
) -> Dict[str, Comp]:
    st = strategy.lower()
    router = Comp()
    losses = Comp()

    if st == "token":
        samples = batch * seq_len
    elif st == "sequence":
        samples = batch
        router.elemwise += batch * hidden_dim * max(seq_len - 1, 0)
        router.elemwise += batch * hidden_dim
    elif st in {"sequence-rolling", "sequence_rolling"}:
        samples = batch
        router.elemwise += 4.0 * batch * seq_len * hidden_dim
    else:
        raise ValueError(
            "Unsupported routing_strategy_train for training FLOPs estimation: "
            f"'{strategy}'. Supported: sequence, sequence-rolling (or sequence_rolling), token."
        )

    router.matmul += linear_flops(samples, hidden_dim, num_chapters)
    router.elemwise += samples * num_chapters * SOFTMAX_FLOPS_PER_SCORE
    router.elemwise += samples * num_chapters * max(1, int(math.ceil(math.log2(max(top_k, 1)))))

    losses.elemwise += router_aux_loss_flops(samples, num_chapters, top_k)
    return {"router": router, "losses": losses}


def self_attn_layer_flops(
    batch: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    *,
    use_rope: bool,
    attention_dropout: float,
) -> Dict[str, float]:
    if hidden_dim % num_heads != 0:
        raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

    tokens = batch * seq_len
    head_dim = hidden_dim // num_heads
    kv_dim = num_kv_heads * head_dim

    matmul = 0.0
    elemwise = 0.0
    matmul += linear_flops(tokens, hidden_dim, hidden_dim)
    matmul += linear_flops(tokens, hidden_dim, kv_dim)
    matmul += linear_flops(tokens, hidden_dim, kv_dim)
    matmul += linear_flops(tokens, hidden_dim, hidden_dim)
    matmul += attention_matmul_flops(batch, seq_len, seq_len, hidden_dim)

    elemwise += attention_score_elementwise_flops(
        batch=batch,
        num_heads=num_heads,
        q_len=seq_len,
        kv_len=seq_len,
        include_mask_scale=True,
        include_dropout=attention_dropout > 0,
    )
    rope = rope_flops(tokens=tokens, q_dim=hidden_dim, k_dim=kv_dim, use_rope=use_rope)

    return {"matmul": matmul, "elemwise": elemwise, "rope": float(rope)}


def memory_projection_cost(
    tokens_q: int,
    mem_tokens_total: int,
    hidden_dim: int,
    memory_dim_in: int,
    memory_num_heads: int,
    memory_num_kv_heads: int,
    *,
    use_low_rank_projections: bool,
    projection_rank: int,
    reduced_dim_mode: bool,
    reduced_dim: int,
) -> Dict[str, float]:
    if hidden_dim % memory_num_heads != 0:
        raise ValueError(
            f"hidden_dim ({hidden_dim}) must be divisible by memory_num_heads ({memory_num_heads})"
        )
    if memory_num_heads % memory_num_kv_heads != 0:
        raise ValueError(
            f"memory_num_heads ({memory_num_heads}) must be divisible by memory_num_kv_heads ({memory_num_kv_heads})"
        )

    if reduced_dim_mode:
        r = safe_int("memory.memory_rank", int(reduced_dim))
        if r % memory_num_heads != 0:
            raise ValueError(
                f"memory.memory_rank ({r}) must be divisible by memory_num_heads ({memory_num_heads})"
            )
        reduced_head_dim = r // memory_num_heads
        kv_dim = memory_num_kv_heads * reduced_head_dim
        q = linear_flops(tokens_q, hidden_dim, r)
        k = linear_flops(mem_tokens_total, r, kv_dim)
        v = linear_flops(mem_tokens_total, r, kv_dim)
        o = linear_flops(tokens_q, r, hidden_dim)
        return {"q": q, "k": k, "v": v, "o": o, "attn_dim": float(r)}

    head_dim = hidden_dim // memory_num_heads
    kv_dim = memory_num_kv_heads * head_dim
    if use_low_rank_projections:
        rank = safe_int("memory.projection_rank", int(projection_rank))
        q = linear_flops(tokens_q, hidden_dim, rank) + linear_flops(tokens_q, rank, hidden_dim)
        k = linear_flops(mem_tokens_total, memory_dim_in, rank) + linear_flops(mem_tokens_total, rank, kv_dim)
        v = linear_flops(mem_tokens_total, memory_dim_in, rank) + linear_flops(mem_tokens_total, rank, kv_dim)
        o = linear_flops(tokens_q, hidden_dim, rank) + linear_flops(tokens_q, rank, hidden_dim)
        return {"q": q, "k": k, "v": v, "o": o, "attn_dim": float(hidden_dim)}

    q = linear_flops(tokens_q, hidden_dim, hidden_dim)
    k = linear_flops(mem_tokens_total, memory_dim_in, kv_dim)
    v = linear_flops(mem_tokens_total, memory_dim_in, kv_dim)
    o = linear_flops(tokens_q, hidden_dim, hidden_dim)
    return {"q": q, "k": k, "v": v, "o": o, "attn_dim": float(hidden_dim)}


def memory_attn_layer_flops(
    batch: int,
    seq_len: int,
    hidden_dim: int,
    memory_dim_in: int,
    memory_num_heads: int,
    memory_num_kv_heads: int,
    *,
    use_low_rank_projections: bool,
    projection_rank: int,
    reduced_dim_mode: bool,
    reduced_dim: int,
    attention_dropout: float,
    is_token_strategy: bool,
    shared_tokens: int,
    routed_tokens_selected: int,
    routed_tokens_available: int,
    selected_non_token_tokens: int,
    top_k_routed: int,
) -> Dict[str, float]:
    tokens_q = batch * seq_len
    matmul = 0.0
    elemwise = 0.0
    proj_out_only = 0.0

    if not is_token_strategy:
        mem_tokens = safe_int("selected_memory_tokens", int(selected_non_token_tokens))
        proj = memory_projection_cost(
            tokens_q=tokens_q,
            mem_tokens_total=batch * mem_tokens,
            hidden_dim=hidden_dim,
            memory_dim_in=memory_dim_in,
            memory_num_heads=memory_num_heads,
            memory_num_kv_heads=memory_num_kv_heads,
            use_low_rank_projections=use_low_rank_projections,
            projection_rank=projection_rank,
            reduced_dim_mode=reduced_dim_mode,
            reduced_dim=reduced_dim,
        )
        proj_out_only = float(proj["o"])
        matmul += proj["q"] + proj["k"] + proj["v"] + proj["o"]
        matmul += attention_matmul_flops(batch, seq_len, mem_tokens, int(proj["attn_dim"]))
        elemwise += attention_score_elementwise_flops(
            batch=batch,
            num_heads=memory_num_heads,
            q_len=seq_len,
            kv_len=mem_tokens,
            include_mask_scale=True,
            include_dropout=attention_dropout > 0,
        )
        total = matmul + elemwise
        return {
            "total": total,
            "matmul": matmul,
            "elemwise": elemwise,
            "proj_out_only": proj_out_only,
            "preproj_total": total - proj_out_only,
        }

    attn_dim_for_combined = None
    ref_proj = None

    if shared_tokens > 0:
        shared_proj = memory_projection_cost(
            tokens_q=tokens_q,
            mem_tokens_total=batch * shared_tokens,
            hidden_dim=hidden_dim,
            memory_dim_in=memory_dim_in,
            memory_num_heads=memory_num_heads,
            memory_num_kv_heads=memory_num_kv_heads,
            use_low_rank_projections=use_low_rank_projections,
            projection_rank=projection_rank,
            reduced_dim_mode=reduced_dim_mode,
            reduced_dim=reduced_dim,
        )
        matmul += shared_proj["q"] + shared_proj["k"] + shared_proj["v"]
        matmul += attention_matmul_flops(batch, seq_len, shared_tokens, int(shared_proj["attn_dim"]))
        elemwise += attention_score_elementwise_flops(
            batch=batch,
            num_heads=memory_num_heads,
            q_len=seq_len,
            kv_len=shared_tokens,
            include_mask_scale=True,
            include_dropout=attention_dropout > 0,
        )
        elemwise += batch * seq_len * int(shared_proj["attn_dim"])
        attn_dim_for_combined = int(shared_proj["attn_dim"])
        ref_proj = shared_proj

    if routed_tokens_available > 0 and routed_tokens_selected > 0:
        routed_proj = memory_projection_cost(
            tokens_q=tokens_q,
            mem_tokens_total=batch * routed_tokens_available,
            hidden_dim=hidden_dim,
            memory_dim_in=memory_dim_in,
            memory_num_heads=memory_num_heads,
            memory_num_kv_heads=memory_num_kv_heads,
            use_low_rank_projections=use_low_rank_projections,
            projection_rank=projection_rank,
            reduced_dim_mode=reduced_dim_mode,
            reduced_dim=reduced_dim,
        )
        matmul += routed_proj["q"] + routed_proj["k"] + routed_proj["v"]
        matmul += attention_matmul_flops(batch, seq_len, routed_tokens_selected, int(routed_proj["attn_dim"]))
        elemwise += attention_score_elementwise_flops(
            batch=batch,
            num_heads=memory_num_heads,
            q_len=seq_len,
            kv_len=routed_tokens_selected,
            include_mask_scale=True,
            include_dropout=attention_dropout > 0,
        )
        elemwise += 2.0 * batch * seq_len * int(routed_proj["attn_dim"]) * max(top_k_routed, 0)
        elemwise += 2.0 * batch * seq_len * int(routed_proj["attn_dim"])
        attn_dim_for_combined = int(routed_proj["attn_dim"])
        ref_proj = routed_proj

    if ref_proj is None or attn_dim_for_combined is None:
        raise ValueError("Token routing requires at least one active memory branch.")

    proj_out_only = float(ref_proj["o"])
    matmul += proj_out_only

    total = matmul + elemwise
    return {
        "total": total,
        "matmul": matmul,
        "elemwise": elemwise,
        "proj_out_only": proj_out_only,
        "preproj_total": total - proj_out_only,
    }


def memory_bank_materialization_flops(config, hidden_dim: int) -> float:
    mem = config.memory
    if not mem.use_low_rank_memory:
        return 0.0
    if str(mem.low_rank_mode).lower() != "factorized":
        return 0.0

    num_memory_tokens = safe_int("memory.num_memory_tokens", int(mem.num_memory_tokens))
    memory_rank = safe_int("memory.memory_rank", int(mem.memory_rank))
    full_dim = int(mem.memory_dim) if mem.memory_dim is not None else hidden_dim
    full_dim = safe_int("memory.memory_dim", full_dim)
    return float(linear_flops(num_memory_tokens, memory_rank, full_dim))


def estimate(
    config,
    batch_size: int,
    seq_len: int,
    flash_available: bool,
    *,
    num_gpus_override: Optional[int],
    grad_accum_override: Optional[int],
    max_steps_override: Optional[int],
) -> Dict[str, float]:
    if config.model.base_model_name is not None:
        raise NotImplementedError(
            "This estimator currently supports from-scratch MemoryTransformer configs only "
            "(model.base_model_name must be null)."
        )

    model = config.model
    mem = config.memory
    train = config.training

    hidden_dim = safe_int("model.hidden_dim", int(model.hidden_dim))
    num_heads = safe_int("model.num_heads", int(model.num_heads))
    num_kv_heads = int(model.num_kv_heads) if model.num_kv_heads is not None else num_heads
    num_kv_heads = safe_int("model.num_kv_heads", num_kv_heads)
    num_layers = safe_int("model.num_layers", int(model.num_layers))
    intermediate_dim = safe_int("model.intermediate_dim", int(model.intermediate_dim))
    vocab_size = safe_int("model.vocab_size", int(model.vocab_size))
    tokens = batch_size * seq_len

    # Mirror model-level config validation so estimator fails on the same invalid setups.
    if int(mem.num_shared_chapters) < 0:
        raise ValueError(
            f"memory.num_shared_chapters must be >= 0, got {mem.num_shared_chapters}"
        )
    if mem.use_chapters and int(mem.num_shared_chapters) > int(mem.num_chapters):
        raise ValueError(
            f"memory.num_shared_chapters ({mem.num_shared_chapters}) must be <= "
            f"memory.num_chapters ({mem.num_chapters})"
        )
    if float(mem.routed_scaling_factor) < 0:
        raise ValueError(
            f"memory.routed_scaling_factor must be >= 0, got {mem.routed_scaling_factor}"
        )
    if str(mem.shared_routed_norm_type) not in {"rms", "layernorm"}:
        raise ValueError(
            "memory.shared_routed_norm_type must be one of {'rms', 'layernorm'}, "
            f"got {mem.shared_routed_norm_type}"
        )
    if float(mem.shared_routed_norm_eps) <= 0:
        raise ValueError(
            f"memory.shared_routed_norm_eps must be > 0, got {mem.shared_routed_norm_eps}"
        )

    memory_layer_indices = set(get_memory_layer_indices(config))
    # Triggers memory_sharing / memory_sharing_k validation for active memory layers.
    _ = get_memory_bank_assignments(config)
    has_memory = (not mem.vanilla_mode) and len(memory_layer_indices) > 0
    chapter = chapter_stats(config) if has_memory and mem.use_chapters else {
        "num_chapters": 0,
        "tokens_per_chapter": 0,
        "shared_chapters": 0,
        "routed_chapters_selected": 0,
        "routed_chapters_available": 0,
        "shared_tokens": 0,
        "routed_tokens_selected": 0,
        "routed_tokens_available": 0,
        "selected_non_token_tokens": int(mem.num_memory_tokens) if has_memory else 0,
    }

    # Match create_memory_bank validation when memory banks are actually constructed.
    if has_memory and mem.use_low_rank_memory:
        if int(mem.memory_rank) <= 0:
            raise ValueError(
                f"memory.memory_rank must be > 0 when use_low_rank_memory=true, got {mem.memory_rank}"
            )
        if str(mem.low_rank_mode).lower() not in {"factorized", "reduced_dim"}:
            raise ValueError(f"Unknown low_rank_mode: {mem.low_rank_mode}")

    # Match ChapterRouter validation when routers are actually created.
    if has_memory and mem.use_chapters:
        if int(mem.top_k_chapters) <= 0:
            raise ValueError(
                f"memory.top_k_chapters must be > 0 when use_chapters=true, got {mem.top_k_chapters}"
            )
        if int(mem.top_k_chapters) > int(mem.num_chapters):
            raise ValueError(
                f"memory.top_k_chapters ({mem.top_k_chapters}) must be <= "
                f"memory.num_chapters ({mem.num_chapters})"
            )

    memory_dim_in = (
        int(mem.memory_rank)
        if (mem.use_low_rank_memory and str(mem.low_rank_mode).lower() == "reduced_dim")
        else (int(mem.memory_dim) if mem.memory_dim is not None else hidden_dim)
    )
    memory_dim_in = safe_int("memory_dim_in", memory_dim_in)

    memory_num_heads = int(mem.memory_num_heads) if mem.memory_num_heads is not None else num_heads
    memory_num_kv_heads = int(mem.memory_num_kv_heads) if mem.memory_num_kv_heads is not None else num_kv_heads
    memory_num_heads = safe_int("memory.memory_num_heads", memory_num_heads)
    memory_num_kv_heads = safe_int("memory.memory_num_kv_heads", memory_num_kv_heads)

    routing_strategy = str(mem.routing_strategy_train).lower()
    if has_memory and mem.use_chapters and routing_strategy not in {"sequence", "sequence-rolling", "sequence_rolling", "token"}:
        raise ValueError(
            "memory.routing_strategy_train must be one of "
            "{sequence, sequence-rolling, sequence_rolling, token} when use_chapters=true. "
            f"Got '{mem.routing_strategy_train}'."
        )
    is_token_strategy = bool(has_memory and mem.use_chapters and routing_strategy == "token")

    b = Breakdown(
        self_attn=Comp(),
        memory_attn=Comp(),
        mlp=Comp(),
        norms=Comp(),
        residuals=Comp(),
        rope=Comp(),
        router=Comp(),
        router_losses=Comp(),
        memory_preprocess=Comp(),
        memory_bank_materialization=Comp(),
        lm_head=Comp(),
        loss=Comp(),
        checkpoint_recompute=0.0,
        memory_attn_internal_recompute=0.0,
    )

    attn_dropout = float(model.attention_dropout)
    memory_attn_dropout = float(mem.memory_dropout if mem.memory_dropout is not None else model.dropout)
    norm_is_rms = bool(model.use_rms_norm)
    hidden_act = str(model.hidden_activation)
    block_variant = str(mem.memory_block_variant).upper()
    if has_memory and block_variant not in {"A", "B"}:
        raise ValueError(f"Unknown memory_block_variant: {block_variant}")
    if not has_memory:
        block_variant = "A"

    flash_effective = bool(model.use_flash_attention and flash_available)
    factorized_materialization = memory_bank_materialization_flops(config, hidden_dim)

    for layer_idx in range(num_layers):
        layer_total = 0.0
        layer_has_memory = layer_idx in memory_layer_indices and has_memory

        sa = self_attn_layer_flops(
            batch=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_rope=bool(model.use_rope),
            attention_dropout=attn_dropout,
        )
        b.self_attn.matmul += sa["matmul"]
        b.self_attn.elemwise += sa["elemwise"]
        b.rope.elemwise += sa["rope"]
        layer_total += sa["matmul"] + sa["elemwise"] + sa["rope"]

        ln_1 = norm_flops(tokens, hidden_dim, norm_is_rms)
        ln_2 = norm_flops(tokens, hidden_dim, norm_is_rms)
        b.norms.elemwise += ln_1 + ln_2
        layer_total += ln_1 + ln_2

        mlp_matmul = 0.0
        if hidden_act.lower() == "swiglu":
            mlp_matmul += linear_flops(tokens, hidden_dim, intermediate_dim)
            mlp_matmul += linear_flops(tokens, hidden_dim, intermediate_dim)
            mlp_matmul += linear_flops(tokens, intermediate_dim, hidden_dim)
        else:
            mlp_matmul += linear_flops(tokens, hidden_dim, intermediate_dim)
            mlp_matmul += linear_flops(tokens, intermediate_dim, hidden_dim)
        mlp_elem = mlp_activation_flops(tokens, intermediate_dim, hidden_act)
        b.mlp.matmul += mlp_matmul
        b.mlp.elemwise += mlp_elem
        layer_total += mlp_matmul + mlp_elem

        res_1 = tokens * hidden_dim
        res_2 = tokens * hidden_dim
        b.residuals.elemwise += res_1 + res_2
        layer_total += res_1 + res_2

        if layer_has_memory:
            if factorized_materialization > 0:
                b.memory_bank_materialization.matmul += factorized_materialization
                layer_total += factorized_materialization

            if mem.use_chapters:
                rc = router_flops(
                    batch=batch_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    num_chapters=chapter["num_chapters"],
                    top_k=max(1, int(mem.top_k_chapters)),
                    strategy=routing_strategy,
                )
                b.router.matmul += rc["router"].matmul
                b.router.elemwise += rc["router"].elemwise
                b.router_losses.elemwise += rc["losses"].elemwise
                layer_total += rc["router"].total() + rc["losses"].total()

                if is_token_strategy:
                    if (
                        mem.normalize_shared_routed_before_mixing
                        and chapter["shared_tokens"] > 0
                        and chapter["routed_tokens_available"] > 0
                    ):
                        prep_norm = branch_norm_flops(
                            vectors=safe_int("memory.num_memory_tokens", int(mem.num_memory_tokens)),
                            dim=memory_dim_in,
                            norm_type=str(mem.shared_routed_norm_type),
                        )
                        b.memory_preprocess.elemwise += prep_norm
                        layer_total += prep_norm
                else:
                    selected_non_token = chapter["selected_non_token_tokens"]
                    prep_weight = batch_size * selected_non_token * memory_dim_in
                    b.memory_preprocess.elemwise += prep_weight
                    layer_total += prep_weight
                    if (
                        mem.normalize_shared_routed_before_mixing
                        and chapter["shared_tokens"] > 0
                        and chapter["routed_tokens_selected"] > 0
                    ):
                        prep_norm = branch_norm_flops(
                            vectors=batch_size * selected_non_token,
                            dim=memory_dim_in,
                            norm_type=str(mem.shared_routed_norm_type),
                        )
                        b.memory_preprocess.elemwise += prep_norm
                        layer_total += prep_norm

            mem_attn = memory_attn_layer_flops(
                batch=batch_size,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                memory_dim_in=memory_dim_in,
                memory_num_heads=memory_num_heads,
                memory_num_kv_heads=memory_num_kv_heads,
                use_low_rank_projections=bool(mem.use_low_rank_projections),
                projection_rank=int(mem.projection_rank),
                reduced_dim_mode=bool(mem.use_low_rank_memory and str(mem.low_rank_mode).lower() == "reduced_dim"),
                reduced_dim=int(mem.memory_rank),
                attention_dropout=memory_attn_dropout,
                is_token_strategy=is_token_strategy,
                shared_tokens=chapter["shared_tokens"],
                routed_tokens_selected=chapter["routed_tokens_selected"],
                routed_tokens_available=chapter["routed_tokens_available"],
                selected_non_token_tokens=chapter["selected_non_token_tokens"],
                top_k_routed=chapter["routed_chapters_selected"],
            )
            b.memory_attn.matmul += mem_attn["matmul"]
            b.memory_attn.elemwise += mem_attn["elemwise"]
            layer_total += mem_attn["total"]

            mem_ln = norm_flops(tokens, hidden_dim, norm_is_rms)
            mem_res = tokens * hidden_dim
            b.norms.elemwise += mem_ln
            b.residuals.elemwise += mem_res
            layer_total += mem_ln + mem_res

            if block_variant == "B":
                post_mem_ln = norm_flops(tokens, hidden_dim, norm_is_rms)
                b.norms.elemwise += post_mem_ln
                layer_total += post_mem_ln

                mlp2_matmul = 0.0
                if hidden_act.lower() == "swiglu":
                    mlp2_matmul += linear_flops(tokens, hidden_dim, intermediate_dim)
                    mlp2_matmul += linear_flops(tokens, hidden_dim, intermediate_dim)
                    mlp2_matmul += linear_flops(tokens, intermediate_dim, hidden_dim)
                else:
                    mlp2_matmul += linear_flops(tokens, hidden_dim, intermediate_dim)
                    mlp2_matmul += linear_flops(tokens, intermediate_dim, hidden_dim)
                mlp2_elem = mlp_activation_flops(tokens, intermediate_dim, hidden_act)
                b.mlp.matmul += mlp2_matmul
                b.mlp.elemwise += mlp2_elem
                layer_total += mlp2_matmul + mlp2_elem

                mem_res_2 = tokens * hidden_dim
                b.residuals.elemwise += mem_res_2
                layer_total += mem_res_2

            if bool(mem.memory_gradient_checkpointing) and (not flash_effective):
                b.memory_attn_internal_recompute += mem_attn["preproj_total"]

        if bool(train.gradient_checkpointing):
            b.checkpoint_recompute += layer_total

    final_ln = norm_flops(tokens, hidden_dim, norm_is_rms)
    b.norms.elemwise += final_ln

    b.lm_head.matmul += linear_flops(tokens, hidden_dim, vocab_size)

    ce_tokens = batch_size * max(seq_len - 1, 0)
    b.loss.elemwise += ce_tokens * vocab_size * SOFTMAX_FLOPS_PER_SCORE

    forward_total = b.forward_total()
    backward_total = 2.0 * b.forward_matmul() + 2.0 * b.forward_elemwise()
    recompute_total = b.checkpoint_recompute + b.memory_attn_internal_recompute
    micro_train = forward_total + backward_total + recompute_total

    grad_accum = (
        safe_int("training.gradient_accumulation_steps", int(train.gradient_accumulation_steps))
        if grad_accum_override is None
        else safe_int("gradient_accumulation_steps", int(grad_accum_override))
    )
    world_size = (
        safe_int("training.num_gpus", int(train.num_gpus))
        if num_gpus_override is None
        else safe_int("num_gpus", int(num_gpus_override))
    )
    total_steps = (
        int(train.max_steps)
        if (train.num_epochs is None and max_steps_override is None)
        else (int(max_steps_override) if max_steps_override is not None else None)
    )
    total_run = None if total_steps is None else micro_train * grad_accum * world_size * total_steps

    return {
        "batch_size_per_device": batch_size,
        "seq_len": seq_len,
        "grad_accumulation_steps": grad_accum,
        "world_size": world_size,
        "memory_layers": len(memory_layer_indices),
        "routing_strategy_train": routing_strategy,
        "selected_memory_tokens_per_query": (
            chapter["selected_non_token_tokens"]
            if has_memory and mem.use_chapters
            else (int(mem.num_memory_tokens) if has_memory else 0)
        ),
        "shared_memory_tokens": chapter["shared_tokens"],
        "routed_memory_tokens_selected_per_query": chapter["routed_tokens_selected"],
        "routed_memory_tokens_available_for_projection": chapter["routed_tokens_available"],
        "forward_matmul": b.forward_matmul(),
        "forward_elemwise": b.forward_elemwise(),
        "forward_total": forward_total,
        "backward_total": backward_total,
        "checkpoint_recompute": b.checkpoint_recompute,
        "memory_attn_internal_recompute": b.memory_attn_internal_recompute,
        "recompute_total": recompute_total,
        "train_total_per_microstep_per_device": micro_train,
        "train_total_per_optimizer_step_per_device": micro_train * grad_accum,
        "train_total_per_optimizer_step_global": micro_train * grad_accum * world_size,
        "train_total_for_run_global": total_run,
        "total_steps_used": total_steps,
        "component_self_attn_total": b.self_attn.total(),
        "component_memory_attn_total": b.memory_attn.total(),
        "component_mlp_total": b.mlp.total(),
        "component_norm_total": b.norms.total(),
        "component_residual_total": b.residuals.total(),
        "component_rope_total": b.rope.total(),
        "component_router_total": b.router.total(),
        "component_router_losses_total": b.router_losses.total(),
        "component_memory_preprocess_total": b.memory_preprocess.total(),
        "component_memory_bank_materialization_total": b.memory_bank_materialization.total(),
        "component_lm_head_total": b.lm_head.total(),
        "component_loss_total": b.loss.total(),
    }


def detect_flash_available(mode: str) -> bool:
    m = mode.lower()
    if m == "true":
        return True
    if m == "false":
        return False
    if m != "auto":
        raise ValueError("--flash_available must be one of: auto, true, false")
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate training FLOPs from config")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override per-device micro-batch size (default: training.batch_size)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Override sequence length (default: training.max_length)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Override world size used for global FLOPs (default: training.num_gpus)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override grad accumulation (default: training.gradient_accumulation_steps)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max steps used for run-total estimate",
    )
    parser.add_argument(
        "--flash_available",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Assume flash-attn availability for memory-attn checkpoint modeling",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    batch_size = int(cfg.training.batch_size) if args.batch_size is None else int(args.batch_size)
    seq_len = int(cfg.training.max_length) if args.seq_len is None else int(args.seq_len)
    batch_size = safe_int("batch_size", batch_size)
    seq_len = safe_int("seq_len", seq_len)

    flash_available = detect_flash_available(args.flash_available)
    stats = estimate(
        cfg,
        batch_size=batch_size,
        seq_len=seq_len,
        flash_available=flash_available,
        num_gpus_override=args.num_gpus,
        grad_accum_override=args.gradient_accumulation_steps,
        max_steps_override=args.max_steps,
    )

    print("=" * 86)
    print("FLOPs Estimate (Comprehensive, Config-Driven)")
    print("=" * 86)
    print(f"Config:                                      {args.config}")
    print(f"Per-device batch size:                       {stats['batch_size_per_device']}")
    print(f"Sequence length:                             {stats['seq_len']}")
    print(f"Gradient accumulation steps:                 {stats['grad_accumulation_steps']}")
    print(f"World size (GPUs):                           {stats['world_size']}")
    print(f"Memory layers:                               {stats['memory_layers']}")
    print(f"Routing strategy (train):                    {stats['routing_strategy_train']}")
    print(
        "Selected memory tokens/query (attention):    "
        f"{stats['selected_memory_tokens_per_query']}"
    )
    print(
        "Shared memory tokens/query:                  "
        f"{stats['shared_memory_tokens']}"
    )
    print(
        "Routed tokens/query (selected):              "
        f"{stats['routed_memory_tokens_selected_per_query']}"
    )
    print(
        "Routed tokens for K/V projection:            "
        f"{stats['routed_memory_tokens_available_for_projection']}"
    )
    print(f"Assumed flash-attn available:                {flash_available}")
    print("-" * 86)
    print(f"Forward matmul / microstep / device:         {fmt_flops(stats['forward_matmul'])}")
    print(f"Forward elemwise / microstep / device:       {fmt_flops(stats['forward_elemwise'])}")
    print(f"Forward total / microstep / device:          {fmt_flops(stats['forward_total'])}")
    print(f"Backward total / microstep / device:         {fmt_flops(stats['backward_total'])}")
    print(f"Recompute total / microstep / device:        {fmt_flops(stats['recompute_total'])}")
    print(
        "  - model checkpoint recompute:              "
        f"{fmt_flops(stats['checkpoint_recompute'])}"
    )
    print(
        "  - memory-attn internal recompute:          "
        f"{fmt_flops(stats['memory_attn_internal_recompute'])}"
    )
    print(
        "Train total / microstep / device:            "
        f"{fmt_flops(stats['train_total_per_microstep_per_device'])}"
    )
    print(
        "Train total / optimizer-step / device:       "
        f"{fmt_flops(stats['train_total_per_optimizer_step_per_device'])}"
    )
    print(
        "Train total / optimizer-step / global:       "
        f"{fmt_flops(stats['train_total_per_optimizer_step_global'])}"
    )
    if stats["train_total_for_run_global"] is not None:
        print(
            "Train total / run / global:                  "
            f"{fmt_flops(stats['train_total_for_run_global'])}"
        )
        print(f"(using total_steps={stats['total_steps_used']})")
    else:
        print("Train total / run / global:                  N/A (num_epochs-based config)")
    print("-" * 86)
    print("Forward component breakdown (microstep / device)")
    print(f"  Self-attention:                            {fmt_flops(stats['component_self_attn_total'])}")
    print(f"  Memory attention:                          {fmt_flops(stats['component_memory_attn_total'])}")
    print(f"  MLP:                                       {fmt_flops(stats['component_mlp_total'])}")
    print(f"  Norms:                                     {fmt_flops(stats['component_norm_total'])}")
    print(f"  Residual adds:                             {fmt_flops(stats['component_residual_total'])}")
    print(f"  RoPE:                                      {fmt_flops(stats['component_rope_total'])}")
    print(f"  Router forward:                            {fmt_flops(stats['component_router_total'])}")
    print(f"  Router loss compute:                       {fmt_flops(stats['component_router_losses_total'])}")
    print(f"  Memory preprocess:                         {fmt_flops(stats['component_memory_preprocess_total'])}")
    print(
        "  Memory-bank materialization:               "
        f"{fmt_flops(stats['component_memory_bank_materialization_total'])}"
    )
    print(f"  LM head:                                   {fmt_flops(stats['component_lm_head_total'])}")
    print(f"  CE loss:                                   {fmt_flops(stats['component_loss_total'])}")
    print("=" * 86)
    print(
        "Assumptions: non-matmul terms (softmax/norm/activation/top-k/router losses) are approximated; "
        "matmul terms are exact for configured shapes."
    )


if __name__ == "__main__":
    main()
