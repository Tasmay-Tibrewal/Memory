"""
Memory Cross-Attention layer for Memory-Augmented Transformer.

Implements cross-attention where:
- Queries come from the token sequence (hidden states)
- Keys and Values come from the memory bank

Supports:
- Multi-head attention
- Low-rank projections
- Zero initialization of output projection (for stable adapter training)
- Reduced dimension mode (attention in lower-dim space)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class MemoryCrossAttention(nn.Module):
    """
    Cross-attention layer for attending to memory bank.
    
    Given hidden states H and memory M:
    - Q = H @ W_q
    - K = M @ W_k  
    - V = M @ W_v
    - Output = softmax(Q @ K^T / sqrt(d_k)) @ V @ W_o
    
    Supports two modes:
    1. Full-dim: All projections are d -> d (standard)
    2. Reduced-dim: Q projects d -> r, K/V operate in r space, O projects r -> d
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        memory_dim: Optional[int] = None,
        use_low_rank_projections: bool = False,
        projection_rank: Optional[int] = None,
        reduced_dim_mode: bool = False,
        reduced_dim: Optional[int] = None,
        wo_init_zero: bool = True,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states (model dimension)
            num_heads: Number of attention heads
            memory_dim: Dimension of memory tokens (defaults to hidden_dim)
            use_low_rank_projections: If True, factorize W_q/W_k/W_v
            projection_rank: Rank for low-rank projections
            reduced_dim_mode: If True, memory is in reduced space, do attention there
            reduced_dim: Dimension of memory in reduced mode
            wo_init_zero: Initialize output projection to zero
            dropout: Attention dropout
            use_flash_attention: Use Flash Attention if available
            gradient_checkpointing: Use gradient checkpointing for memory savings
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim if memory_dim is not None else hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE
        self.gradient_checkpointing = gradient_checkpointing
        
        self.reduced_dim_mode = reduced_dim_mode
        self.reduced_dim = reduced_dim
        
        # Set flag unconditionally (Bug 11 fix)
        self.use_low_rank_projections = use_low_rank_projections
        
        # Initialize projections based on mode
        if reduced_dim_mode:
            assert reduced_dim is not None, "reduced_dim required for reduced_dim_mode"
            # Bug 9 fix: validate reduced_dim is divisible by num_heads
            assert reduced_dim % num_heads == 0, (
                f"reduced_dim ({reduced_dim}) must be divisible by num_heads ({num_heads})"
            )
            self._init_reduced_dim_projections(wo_init_zero)
        elif use_low_rank_projections:
            assert projection_rank is not None, "projection_rank required for low-rank"
            self._init_low_rank_projections(projection_rank, wo_init_zero)
        else:
            self._init_full_projections(wo_init_zero)
        
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def _init_full_projections(self, wo_init_zero: bool):
        """Initialize full-rank projection matrices."""
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.memory_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.memory_dim, self.hidden_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        if wo_init_zero:
            nn.init.zeros_(self.o_proj.weight)
    
    def _init_low_rank_projections(self, rank: int, wo_init_zero: bool):
        """
        Initialize low-rank factorized projections.
        
        Each W = W_down @ W_up where:
        - W_down: (in_dim, rank)
        - W_up: (rank, out_dim)
        """
        self.projection_rank = rank
        
        # Q projection: hidden_dim -> hidden_dim via rank
        self.q_down = nn.Linear(self.hidden_dim, rank, bias=False)
        self.q_up = nn.Linear(rank, self.hidden_dim, bias=False)
        
        # K projection: memory_dim -> hidden_dim via rank  
        self.k_down = nn.Linear(self.memory_dim, rank, bias=False)
        self.k_up = nn.Linear(rank, self.hidden_dim, bias=False)
        
        # V projection: memory_dim -> hidden_dim via rank
        self.v_down = nn.Linear(self.memory_dim, rank, bias=False)
        self.v_up = nn.Linear(rank, self.hidden_dim, bias=False)
        
        # O projection: hidden_dim -> hidden_dim via rank
        self.o_down = nn.Linear(self.hidden_dim, rank, bias=False)
        self.o_up = nn.Linear(rank, self.hidden_dim, bias=False)
        
        if wo_init_zero:
            nn.init.zeros_(self.o_up.weight)
        
        # Set flag
        self.use_low_rank_projections = True
    
    def _init_reduced_dim_projections(self, wo_init_zero: bool):
        """
        Initialize projections for reduced dimension mode.
        
        Memory is stored as (N_m, r) where r is reduced_dim.
        - W_q: hidden_dim -> reduced_dim (project queries down)
        - W_k: reduced_dim -> reduced_dim (operate in reduced space)
        - W_v: reduced_dim -> reduced_dim (operate in reduced space)
        - W_o: reduced_dim -> hidden_dim (project output back up)
        
        Attention happens in the reduced_dim space.
        """
        r = self.reduced_dim
        
        # Recalculate head_dim for reduced space
        self.reduced_head_dim = r // self.num_heads
        self.reduced_scale = self.reduced_head_dim ** -0.5
        
        self.q_proj = nn.Linear(self.hidden_dim, r, bias=False)
        self.k_proj = nn.Linear(r, r, bias=False)
        self.v_proj = nn.Linear(r, r, bias=False)
        self.o_proj = nn.Linear(r, self.hidden_dim, bias=False)
        
        if wo_init_zero:
            nn.init.zeros_(self.o_proj.weight)
        
        self.use_low_rank_projections = False
    
    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V tensors.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            memory: (num_memory_tokens, memory_dim) or 
                   (batch_size, num_memory_tokens, memory_dim)
        
        Returns:
            Q, K, V tensors
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Handle memory dimensions
        if memory.dim() == 2:
            # Shared memory: expand for batch
            memory = memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        num_mem_tokens = memory.shape[1]
        
        if hasattr(self, 'use_low_rank_projections') and self.use_low_rank_projections:
            # Low-rank projections
            q = self.q_up(self.q_down(hidden_states))
            k = self.k_up(self.k_down(memory))
            v = self.v_up(self.v_down(memory))
        else:
            # Standard or reduced-dim projections
            q = self.q_proj(hidden_states)
            k = self.k_proj(memory)
            v = self.v_proj(memory)
        
        # Determine dimensions for reshaping
        if self.reduced_dim_mode:
            head_dim = self.reduced_head_dim
            q_dim = self.reduced_dim
        else:
            head_dim = self.head_dim
            q_dim = self.hidden_dim
        
        # Reshape for multi-head attention
        # Q: (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, head_dim)
        
        # K, V: (batch, num_mem_tokens, num_heads, head_dim)
        k = k.view(batch_size, num_mem_tokens, self.num_heads, head_dim)
        v = v.view(batch_size, num_mem_tokens, self.num_heads, head_dim)
        
        return q, k, v
    
    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention.
        
        Args:
            q: (batch, seq_len, num_heads, head_dim)
            k: (batch, num_mem_tokens, num_heads, head_dim)
            v: (batch, num_mem_tokens, num_heads, head_dim)
            
        Returns:
            Attention output: (batch, seq_len, num_heads, head_dim)
        """
        if self.use_flash_attention:
            # Flash attention expects (batch, seq_len, num_heads, head_dim)
            # Cross-attention: use flash_attn_func with separate K, V
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,  # Cross-attention is not causal
            )
        else:
            # Standard attention
            # Transpose to (batch, num_heads, seq_len/mem_tokens, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            scale = self.reduced_scale if self.reduced_dim_mode else self.scale
            
            # Attention scores: (batch, num_heads, seq_len, num_mem_tokens)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # Attention output: (batch, num_heads, seq_len, head_dim)
            attn_output = torch.matmul(attn_weights, v)
            
            # Transpose back: (batch, seq_len, num_heads, head_dim)
            attn_output = attn_output.transpose(1, 2)
        
        return attn_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for memory cross-attention.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_dim)
            memory: Memory bank tokens (num_tokens, memory_dim) or
                   (batch_size, num_tokens, memory_dim) for batched routing
            return_attn_weights: Whether to return attention weights
            
        Returns:
            Tuple of:
            - Output tensor (batch_size, seq_len, hidden_dim)
            - Optional attention weights if return_attn_weights=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Use gradient checkpointing when enabled, training, and no flash attention
        # (Flash attention already has implicit checkpointing)
        use_checkpoint = (
            self.gradient_checkpointing 
            and self.training 
            and not self.use_flash_attention
        )
        
        if use_checkpoint:
            # Checkpoint the attention computation
            attn_output = torch.utils.checkpoint.checkpoint(
                self._forward_attention,
                hidden_states,
                memory,
                use_reentrant=False,
            )
        else:
            attn_output = self._forward_attention(hidden_states, memory)
        
        # Output projection (not checkpointed - small compared to attention)
        if hasattr(self, 'use_low_rank_projections') and self.use_low_rank_projections:
            output = self.o_up(self.o_down(attn_output))
        else:
            output = self.o_proj(attn_output)
        
        if return_attn_weights and not self.use_flash_attention:
            # Recompute weights for returning
            q, k, v = self._compute_qkv(hidden_states, memory)
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            scale = self.reduced_scale if self.reduced_dim_mode else self.scale
            attn_weights = F.softmax(
                torch.matmul(q_t, k_t.transpose(-2, -1)) * scale, 
                dim=-1
            )
            return output, attn_weights
        
        return output, None
    
    def _forward_attention(
        self, 
        hidden_states: torch.Tensor, 
        memory: torch.Tensor
    ) -> torch.Tensor:
        """Forward attention computation (for checkpointing)."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        q, k, v = self._compute_qkv(hidden_states, memory)
        
        # Compute attention
        attn_output = self._attention(q, k, v)
        
        # Reshape: (batch, seq_len, hidden_dim) or (batch, seq_len, reduced_dim)
        if self.reduced_dim_mode:
            attn_output = attn_output.view(batch_size, seq_len, self.reduced_dim)
        else:
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        return attn_output


class MemoryCrossAttentionWithRouting(nn.Module):
    """
    Memory cross-attention with chapter-based routing.
    
    Uses a router to select top-k chapters, then performs
    cross-attention only on the selected memory tokens.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_chapters: int,
        tokens_per_chapter: int,
        top_k: int,
        memory_dim: Optional[int] = None,
        use_low_rank_projections: bool = False,
        projection_rank: Optional[int] = None,
        reduced_dim_mode: bool = False,
        reduced_dim: Optional[int] = None,
        wo_init_zero: bool = True,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        self.num_chapters = num_chapters
        self.tokens_per_chapter = tokens_per_chapter
        self.top_k = top_k
        
        # The actual attention layer
        self.attention = MemoryCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            memory_dim=memory_dim,
            use_low_rank_projections=use_low_rank_projections,
            projection_rank=projection_rank,
            reduced_dim_mode=reduced_dim_mode,
            reduced_dim=reduced_dim,
            wo_init_zero=wo_init_zero,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        chapter_indices: torch.Tensor,
        chapter_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with chapter routing.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            memory: Full memory bank (num_tokens, dim) or 
                   already routed (batch_size, k*tokens_per_chapter, dim)
            chapter_indices: (batch_size, top_k) selected chapter indices
            chapter_weights: Optional (batch_size, top_k) routing weights
            
        Returns:
            Output tensor and optional attention weights
        """
        batch_size = hidden_states.shape[0]
        
        if memory.dim() == 2:
            # Select memory tokens based on chapter_indices
            # This is the sequence-level routing case
            selected_memory = self._select_chapters(memory, chapter_indices)
        else:
            # Memory already batched (e.g., from chaptered bank)
            selected_memory = memory
        
        # Bug 7 fix: Apply chapter_weights to scale memory tokens
        if chapter_weights is not None:
            # chapter_weights: (batch, top_k) -> expand to (batch, top_k*tpc, 1)
            w = chapter_weights.unsqueeze(-1)                          # (B, top_k, 1)
            w = w.repeat(1, 1, self.tokens_per_chapter)                # (B, top_k, tpc)
            w = w.reshape(batch_size, -1, 1)                           # (B, top_k*tpc, 1)
            selected_memory = selected_memory * w
        
        # Standard cross-attention on selected memory
        output, attn_weights = self.attention(
            hidden_states, 
            selected_memory,
            return_attn_weights=False,
        )
        
        return output, attn_weights
    
    def _select_chapters(
        self,
        memory: torch.Tensor,
        chapter_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select memory tokens for each batch item based on chapter indices.
        
        Args:
            memory: (num_tokens, dim)
            chapter_indices: (batch_size, top_k)
            
        Returns:
            Selected memory: (batch_size, top_k * tokens_per_chapter, dim)
        """
        batch_size, k = chapter_indices.shape
        dim = memory.shape[-1]
        
        # Create indices for gathering
        # For each chapter index, we need tokens_per_chapter consecutive tokens
        offsets = torch.arange(
            self.tokens_per_chapter, 
            device=chapter_indices.device
        )
        
        # Expand chapter indices: (batch_size, k, 1) * tokens_per_chapter
        # + offsets: (1, 1, tokens_per_chapter)
        # = (batch_size, k, tokens_per_chapter)
        all_indices = (
            chapter_indices.unsqueeze(-1) * self.tokens_per_chapter 
            + offsets.view(1, 1, -1)
        )
        
        # Flatten: (batch_size, k * tokens_per_chapter)
        all_indices = all_indices.view(batch_size, -1)
        
        # Gather from memory
        # Expand memory: (1, num_tokens, dim) -> (batch_size, num_tokens, dim)
        memory_expanded = memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Gather: need indices of shape (batch_size, k*tpc, dim)
        indices_expanded = all_indices.unsqueeze(-1).expand(-1, -1, dim)
        
        selected = torch.gather(memory_expanded, 1, indices_expanded)
        
        return selected
