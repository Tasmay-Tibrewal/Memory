"""
Transformer Block with Memory Cross-Attention.

Implements the memory-augmented transformer block in two variants:
- Variant A: Self-Attn -> Memory Cross-Attn -> MLP
- Variant B: Self-Attn -> MLP -> Memory Cross-Attn -> MLP
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_attention import MemoryCrossAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 8192,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_len: int,
        offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for positions.
        
        Args:
            x: Input tensor (for device)
            seq_len: Sequence length
            offset: Position offset (for generation)
            position_ids: Optional explicit position IDs. If provided, this overrides
                (seq_len, offset) and can support per-example positions for batched
                generation with padding.
            
        Returns:
            cos, sin tensors of shape:
            - (seq_len, dim) when using (seq_len, offset)
            - (batch_size, seq_len, dim) when using position_ids
        """
        if position_ids is not None:
            if position_ids.dim() not in (1, 2):
                raise ValueError(
                    f"position_ids must be 1D or 2D (got shape {tuple(position_ids.shape)})"
                )
            if position_ids.shape[-1] != seq_len:
                raise ValueError(
                    f"position_ids last dim ({position_ids.shape[-1]}) must match seq_len ({seq_len})"
                )
            pos = position_ids.to(device=self.cos_cached.device, dtype=torch.long)
            if pos.numel() == 0:
                # Degenerate but safe: return empty cos/sin with correct trailing dim.
                empty = self.cos_cached[:0]
                return empty, empty

            if (pos < 0).any():
                raise ValueError(
                    "position_ids must be non-negative (pads should be 0 and masked via attention_mask)."
                )

            max_pos = int(pos.max().item())
            if max_pos + 1 > self.cos_cached.shape[0]:
                self._build_cache(max_pos + 1)

            cos = self.cos_cached[pos]
            sin = self.sin_cached[pos]
            return cos, sin

        if seq_len + offset > self.cos_cached.shape[0]:
            self._build_cache(seq_len + offset)
        
        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to q and k."""
    # Reshape cos/sin for broadcasting.
    # - If cos/sin are (seq_len, dim): broadcast across batch and heads.
    # - If cos/sin are (batch, seq_len, dim): broadcast across heads only.
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, dim)
        sin = sin.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected cos/sin shape: {tuple(cos.shape)}")

    # Avoid mixed-precision upcasting (e.g., bf16/fp16 q,k multiplied by fp32 cos/sin).
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int = 8192,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # Bug 42 fix: Validate attention head sizing to avoid silent shape mismatches later.
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        if self.use_rope and (self.head_dim % 2 != 0):
            raise ValueError(
                f"RoPE requires even head_dim (got {self.head_dim}). "
                "Adjust hidden_dim/num_heads."
            )
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.rotary = None
        if use_rope:
            self.rotary = RotaryPositionalEmbedding(
                self.head_dim, 
                max_seq_len, 
                rope_theta
            )
        
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.attn_dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()
        self.use_flash_attention = use_flash_attention
        
        try:
            from flash_attn import flash_attn_func
            self._flash_attn = flash_attn_func
            self._has_flash = True
        except ImportError:
            self._has_flash = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            attention_mask: Optional mask
            position_offset: For generation
            position_ids: Optional per-token position IDs. When provided, overrides
                position_offset and supports padded batches with KV-cache.
            past_kv: Cached K,V from previous steps
            use_cache: Whether to return updated cache
            
        Returns:
            Output and optionally updated KV cache
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings (optional)
        if self.use_rope:
            assert self.rotary is not None
            cos, sin = self.rotary(
                q,
                seq_len,
                position_offset,
                position_ids=position_ids,
            )
            q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
            k = k.transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q = q.transpose(1, 2)  # Back to (batch, seq, heads, head_dim)
            k = k.transpose(1, 2)
        
        # Handle KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        new_kv = (k, v) if use_cache else None
        
        # Attention
        if self._has_flash and self.use_flash_attention:
            # Flash attention expects (batch, seq, heads, head_dim)
            # Bug 3 fix: Flash attention v2.x does not natively support key_padding_mask
            # for variable-length causal attention. For right-padded causal training this
            # is generally benign (padding comes after valid tokens). For left-padding or
            # correctness we fall back to standard attention when padding is detected.
            has_padding = attention_mask is not None and (attention_mask == 0).any()
            if has_padding:
                # Fall back to standard attention path for proper mask handling
                use_flash_here = False
            else:
                use_flash_here = True
            
            if use_flash_here:
                attn_output = self._flash_attn(
                    q, k, v,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    causal=True,
                )
            else:
                # Redirect to standard attention below
                pass
        else:
            use_flash_here = False
        
        if not (self._has_flash and self.use_flash_attention) or not use_flash_here:
            # Standard attention
            q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Causal mask
            kv_len = k.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=q.device, dtype=torch.bool),
                diagonal=kv_len - seq_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
            
            # Bug 1 fix: Use masked_fill to avoid NaN from 0.0 * -inf
            if attention_mask is not None:
                # attention_mask is (B, S) with 1=valid, 0=masked
                # Reshape for broadcasting: (B, 1, 1, kv_len)
                mask_len = attention_mask.shape[1]
                if mask_len != kv_len:
                    if past_kv is not None:
                        raise ValueError(
                            f"When using past_kv, attention_mask must cover full kv_len. "
                            f"Got mask_len={mask_len}, kv_len={kv_len}."
                        )
                    raise ValueError(
                        f"attention_mask length ({mask_len}) must match kv_len ({kv_len})."
                    )
                padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, kv_len)
                attn_weights = attn_weights.masked_fill(padding_mask, float("-inf"))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2)  # (batch, seq, heads, head_dim)
        
        # Reshape and project
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(attn_output)
        
        return output, new_kv


class MLP(nn.Module):
    """Feed-forward MLP with SwiGLU activation."""
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )


class MemoryTransformerBlock(nn.Module):
    """
    Transformer block with optional memory cross-attention.
    
    Variant A: Self-Attn -> Memory Cross-Attn -> MLP
    Variant B: Self-Attn -> MLP -> Memory Cross-Attn -> MLP
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        max_seq_len: int = 8192,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        memory_dropout: Optional[float] = None,
        attention_dropout: float = 0.0,
        use_rms_norm: bool = True,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
        # Memory settings
        has_memory: bool = True,
        memory_dim: Optional[int] = None,
        use_low_rank_projections: bool = False,
        projection_rank: Optional[int] = None,
        reduced_dim_mode: bool = False,
        reduced_dim: Optional[int] = None,
        wo_init_zero: bool = True,
        memory_block_variant: str = "A",
        gradient_checkpointing: bool = True,  # Memory gradient checkpointing
    ):
        super().__init__()
        
        self.has_memory = has_memory
        self.memory_block_variant = memory_block_variant
        
        # Normalization
        norm_cls = RMSNorm if use_rms_norm else nn.LayerNorm
        self.input_layernorm = norm_cls(hidden_dim, eps=norm_eps)
        self.post_attention_layernorm = norm_cls(hidden_dim, eps=norm_eps)
        
        # Self-attention
        self.self_attn = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            rope_theta=rope_theta,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_flash_attention=use_flash_attention,
        )
        
        # MLP
        self.mlp = MLP(hidden_dim, intermediate_dim, dropout)
        memory_attn_dropout = dropout if memory_dropout is None else memory_dropout
        
        # Memory cross-attention (optional)
        if has_memory:
            self.memory_layernorm = norm_cls(hidden_dim, eps=norm_eps)
            self.memory_attn = MemoryCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                memory_dim=memory_dim,
                use_low_rank_projections=use_low_rank_projections,
                projection_rank=projection_rank,
                reduced_dim_mode=reduced_dim_mode,
                reduced_dim=reduced_dim,
                wo_init_zero=wo_init_zero,
                dropout=memory_attn_dropout,
                use_flash_attention=use_flash_attention,
                gradient_checkpointing=gradient_checkpointing,
            )
            
            # Variant B has extra MLP after memory
            if memory_block_variant == "B":
                self.post_memory_layernorm = norm_cls(hidden_dim, eps=norm_eps)
                self.post_memory_mlp = MLP(hidden_dim, intermediate_dim, dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            memory: Memory bank tokens (optional)
            attention_mask: Optional attention mask
            position_offset: Position offset for generation
            position_ids: Optional per-token position IDs. When provided, overrides
                position_offset and supports padded batches with KV-cache.
            past_kv: Cached KV for generation
            use_cache: Whether to return updated KV cache
            
        Returns:
            Tuple of (output, optional_kv_cache)
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_offset=position_offset,
            position_ids=position_ids,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        if self.memory_block_variant == "A":
            # Variant A: Self-Attn -> Memory -> MLP
            
            # Memory cross-attention (if enabled)
            if self.has_memory and memory is not None:
                residual = hidden_states
                hidden_states = self.memory_layernorm(hidden_states)
                mem_output, _ = self.memory_attn(hidden_states, memory)
                hidden_states = residual + mem_output
            
            # MLP
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
        elif self.memory_block_variant == "B":
            # Variant B: Self-Attn -> MLP -> Memory -> MLP
            
            # First MLP
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            # Memory cross-attention (if enabled)
            if self.has_memory and memory is not None:
                residual = hidden_states
                hidden_states = self.memory_layernorm(hidden_states)
                mem_output, _ = self.memory_attn(hidden_states, memory)
                hidden_states = residual + mem_output
                
                # Second MLP
                residual = hidden_states
                hidden_states = self.post_memory_layernorm(hidden_states)
                hidden_states = self.post_memory_mlp(hidden_states)
                hidden_states = residual + hidden_states
        else:
            # Bug 8 fix: Unknown variant should raise error, not silently skip MLP
            raise ValueError(
                f"Unknown memory_block_variant: '{self.memory_block_variant}'. "
                "Supported variants are 'A' and 'B'."
            )
        
        return hidden_states, new_kv


class VanillaTransformerBlock(nn.Module):
    """
    Standard transformer block without memory (for control experiments).
    
    Just Self-Attn -> MLP.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        max_seq_len: int = 8192,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_rms_norm: bool = True,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        norm_cls = RMSNorm if use_rms_norm else nn.LayerNorm
        self.input_layernorm = norm_cls(hidden_dim, eps=norm_eps)
        self.post_attention_layernorm = norm_cls(hidden_dim, eps=norm_eps)
        
        self.self_attn = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            rope_theta=rope_theta,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_flash_attention=use_flash_attention,
        )
        
        self.mlp = MLP(hidden_dim, intermediate_dim, dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass."""
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_offset=position_offset,
            position_ids=position_ids,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, new_kv
