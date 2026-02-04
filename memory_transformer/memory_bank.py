"""
Memory Bank implementations for Memory-Augmented Transformer.

Provides different memory bank variants:
- StandardMemoryBank: Full N_m x d learnable parameters
- FactorizedMemoryBank: M = A @ B^T with A: (N_m, r), B: (d, r)
- ReducedDimMemoryBank: M stored as (N_m, r), operates in reduced space
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """Base class for memory banks."""
    
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.init_std = init_std
    
    def get_memory(self) -> torch.Tensor:
        """
        Get the memory bank tokens.
        
        Returns:
            Memory tensor of shape (num_tokens, dim)
        """
        raise NotImplementedError
    
    def forward(self) -> torch.Tensor:
        """Alias for get_memory()."""
        return self.get_memory()


class StandardMemoryBank(MemoryBank):
    """
    Standard full-rank memory bank.
    
    Stores N_m learnable tokens, each of dimension d.
    Total parameters: N_m * d
    """
    
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        init_std: float = 0.02,
    ):
        super().__init__(num_tokens, dim, init_std)
        
        # Learnable memory tokens
        self.memory = nn.Parameter(torch.empty(num_tokens, dim))
        self._init_memory()
    
    def _init_memory(self):
        """Initialize memory tokens with normal distribution."""
        nn.init.normal_(self.memory, mean=0.0, std=self.init_std)
    
    def get_memory(self) -> torch.Tensor:
        """
        Get the memory bank tokens.
        
        Returns:
            Memory tensor of shape (num_tokens, dim)
        """
        return self.memory


class FactorizedMemoryBank(MemoryBank):
    """
    Low-rank factorized memory bank.
    
    Memory M = A @ B^T where:
    - A: (N_m, r) - token embeddings in reduced space
    - B: (d, r) - basis vectors
    
    Total parameters: (N_m + d) * r instead of N_m * d
    
    This approach learns the full d-dimensional representation
    but through a low-rank factorization.
    """
    
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        rank: int,
        init_std: float = 0.02,
    ):
        super().__init__(num_tokens, dim, init_std)
        self.rank = rank
        
        # Factorized components
        self.A = nn.Parameter(torch.empty(num_tokens, rank))
        self.B = nn.Parameter(torch.empty(dim, rank))
        
        self._init_memory()
    
    def _init_memory(self):
        """Initialize factors so M = A @ B^T has appropriate scale."""
        # Initialize such that the product has std ~= init_std
        # Var(AB^T) ~= rank * Var(A) * Var(B)
        # So set each to have std = init_std / sqrt(rank)
        factor_std = self.init_std / math.sqrt(self.rank)
        nn.init.normal_(self.A, mean=0.0, std=factor_std)
        nn.init.normal_(self.B, mean=0.0, std=factor_std)
    
    def get_memory(self) -> torch.Tensor:
        """
        Compute and return the memory bank tokens.
        
        Returns:
            Memory tensor of shape (num_tokens, dim)
        """
        # M = A @ B^T: (N_m, r) @ (r, d) -> (N_m, d)
        return self.A @ self.B.t()


class ReducedDimMemoryBank(MemoryBank):
    """
    Memory bank stored in reduced dimensions.
    
    Memory M is stored as (N_m, r) where r << d.
    All attention operations happen in the r-dimensional space.
    Projections handle the mapping between d and r dimensions.
    
    This is different from FactorizedMemoryBank:
    - Factorized: stores factors, reconstructs full d-dim for attention
    - ReducedDim: stores in r-dim, attention in r-dim, project output back
    
    Total memory parameters: N_m * r
    """
    
    def __init__(
        self,
        num_tokens: int,
        reduced_dim: int,
        init_std: float = 0.02,
    ):
        # Note: dim here is the reduced dimension
        super().__init__(num_tokens, reduced_dim, init_std)
        self.reduced_dim = reduced_dim
        
        # Memory stored in reduced space
        self.memory = nn.Parameter(torch.empty(num_tokens, reduced_dim))
        self._init_memory()
    
    def _init_memory(self):
        """Initialize memory tokens."""
        nn.init.normal_(self.memory, mean=0.0, std=self.init_std)
    
    def get_memory(self) -> torch.Tensor:
        """
        Get the memory bank tokens in reduced dimension.
        
        Returns:
            Memory tensor of shape (num_tokens, reduced_dim)
        """
        return self.memory


class ChapteredMemoryBank(nn.Module):
    """
    Memory bank organized into chapters for MoE-style routing.
    
    Wraps a base memory bank and provides methods to access
    specific chapters (subsets of tokens).
    """
    
    def __init__(
        self,
        base_bank: MemoryBank,
        num_chapters: int,
    ):
        super().__init__()
        self.base_bank = base_bank
        self.num_chapters = num_chapters
        
        # Verify divisibility
        if base_bank.num_tokens % num_chapters != 0:
            raise ValueError(
                f"num_tokens ({base_bank.num_tokens}) must be divisible by "
                f"num_chapters ({num_chapters})"
            )
        
        self.tokens_per_chapter = base_bank.num_tokens // num_chapters
    
    def get_memory(self) -> torch.Tensor:
        """Get full memory bank."""
        return self.base_bank.get_memory()
    
    def get_chapter(self, chapter_idx: int) -> torch.Tensor:
        """
        Get tokens from a specific chapter.
        
        Args:
            chapter_idx: Index of chapter (0 to num_chapters-1)
            
        Returns:
            Memory tensor of shape (tokens_per_chapter, dim)
        """
        memory = self.base_bank.get_memory()
        start = chapter_idx * self.tokens_per_chapter
        end = start + self.tokens_per_chapter
        return memory[start:end]
    
    def get_chapters(self, chapter_indices: torch.Tensor) -> torch.Tensor:
        """
        Get tokens from multiple chapters.
        
        Args:
            chapter_indices: Tensor of chapter indices, shape (k,)
            
        Returns:
            Memory tensor of shape (k * tokens_per_chapter, dim)
        """
        memory = self.base_bank.get_memory()
        
        # Gather all tokens from selected chapters
        all_indices = []
        for idx in chapter_indices:
            start = idx.item() * self.tokens_per_chapter
            end = start + self.tokens_per_chapter
            all_indices.extend(range(start, end))
        
        return memory[all_indices]
    
    def get_chapters_batched(
        self, 
        chapter_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get tokens from multiple chapters for batched routing.
        
        When different items in a batch route to different chapters,
        we need to handle this efficiently.
        
        Args:
            chapter_indices: Tensor of shape (batch_size, k) with k selected
                           chapter indices per batch item
            
        Returns:
            Tuple of:
            - Memory tensor of shape (batch_size, k * tokens_per_chapter, dim)
            - Flat indices for reconstruction
        """
        memory = self.base_bank.get_memory()  # (num_tokens, dim)
        batch_size, k = chapter_indices.shape
        
        # Create indices: (batch_size, k * tokens_per_chapter)
        # For each batch item, we need indices into the memory
        
        # Expand chapter_indices: (batch_size, k, 1)
        expanded = chapter_indices.unsqueeze(-1)
        
        # Create offset within chapter: (1, 1, tokens_per_chapter)
        offsets = torch.arange(
            self.tokens_per_chapter, 
            device=chapter_indices.device
        ).view(1, 1, -1)
        
        # Compute all indices: (batch_size, k, tokens_per_chapter)
        all_indices = expanded * self.tokens_per_chapter + offsets
        
        # Flatten to (batch_size, k * tokens_per_chapter)
        all_indices = all_indices.view(batch_size, -1)
        
        # Gather memory for all batch items
        # memory: (num_tokens, dim) -> expand to (1, num_tokens, dim)
        # all_indices: (batch_size, k * tokens_per_chapter) -> (batch_size, k * tokens_per_chapter, 1)
        memory_expanded = memory.unsqueeze(0).expand(batch_size, -1, -1)
        indices_expanded = all_indices.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        
        # Gather: (batch_size, k * tokens_per_chapter, dim)
        gathered = torch.gather(memory_expanded, 1, indices_expanded)
        
        return gathered, all_indices


def create_memory_bank(
    num_tokens: int,
    dim: int,
    use_low_rank: bool = False,
    rank: int = 64,
    low_rank_mode: str = "factorized",
    init_std: float = 0.02,
) -> MemoryBank:
    """
    Factory function to create appropriate memory bank.
    
    Args:
        num_tokens: Number of memory tokens
        dim: Dimension of each token (full model dimension)
        use_low_rank: Whether to use low-rank variant
        rank: Rank for low-rank variants
        low_rank_mode: "factorized" or "reduced_dim"
        init_std: Standard deviation for initialization
        
    Returns:
        MemoryBank instance
    """
    if not use_low_rank:
        return StandardMemoryBank(num_tokens, dim, init_std)
    
    if low_rank_mode == "factorized":
        return FactorizedMemoryBank(num_tokens, dim, rank, init_std)
    elif low_rank_mode == "reduced_dim":
        return ReducedDimMemoryBank(num_tokens, rank, init_std)
    else:
        raise ValueError(f"Unknown low_rank_mode: {low_rank_mode}")
