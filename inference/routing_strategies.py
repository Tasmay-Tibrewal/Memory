"""
Routing strategies for inference.

Different strategies for chapter selection during inference:
- SequenceLevelRouter: Use full context to select chapters (same as training)
- RollingWindowRouter: Use rolling window of recent tokens
- TokenLevelRouter: Per-token routing (for generation)
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceLevelRouter:
    """
    Sequence-level routing strategy.
    
    Same as training: mean-pool the full context to make routing decision.
    Best for prefill phase where full context is available.
    """
    
    def __init__(
        self,
        router: nn.Module,  # The trained router network
        top_k: int,
    ):
        self.router = router
        self.top_k = top_k
    
    def route(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select chapters based on full sequence.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            
        Returns:
            chapter_indices: (batch_size, top_k)
            chapter_weights: (batch_size, top_k)
        """
        # Mean-pool sequence
        pooled = hidden_states.mean(dim=1)
        
        # Get router logits
        logits = self.router.router(pooled)
        probs = F.softmax(logits, dim=-1)
        
        # Select top-k
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return indices, weights


class RollingWindowRouter:
    """
    Rolling window routing strategy.
    
    For generation: use a rolling window of recent tokens
    to determine chapter selection. Avoids using the full
    (potentially very long) context.
    """
    
    def __init__(
        self,
        router: nn.Module,
        top_k: int,
        window_size: int = 128,
    ):
        self.router = router
        self.top_k = top_k
        self.window_size = window_size
        self.cache = None
    
    def reset_cache(self):
        """Reset the rolling window cache."""
        self.cache = None
    
    def route(
        self,
        hidden_states: torch.Tensor,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select chapters using rolling window.
        
        Args:
            hidden_states: New hidden states (batch_size, new_len, hidden_dim)
            use_cache: Whether to use/update cache
            
        Returns:
            chapter_indices: (batch_size, top_k)
            chapter_weights: (batch_size, top_k)
        """
        batch_size = hidden_states.shape[0]
        
        if use_cache and self.cache is not None:
            # Concatenate with cache
            combined = torch.cat([self.cache, hidden_states], dim=1)
        else:
            combined = hidden_states
        
        # Keep only last window_size tokens
        if combined.shape[1] > self.window_size:
            combined = combined[:, -self.window_size:]
        
        # Update cache
        if use_cache:
            self.cache = combined.detach()
        
        # Mean pool over window
        pooled = combined.mean(dim=1)
        
        # Route
        logits = self.router.router(pooled)
        probs = F.softmax(logits, dim=-1)
        
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return indices, weights


class TokenLevelRouter:
    """
    Token-level routing strategy.
    
    For generation where each token may route to different chapters.
    Only efficient when generating one token at a time.
    
    WARNING: Using this during prefill will be very memory-intensive.
    """
    
    def __init__(
        self,
        router: nn.Module,
        top_k: int,
    ):
        self.router = router
        self.top_k = top_k
    
    def route(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select chapters per token.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            
        Returns:
            chapter_indices: (batch_size, seq_len, top_k)
            chapter_weights: (batch_size, seq_len, top_k)
        """
        # Per-token routing
        logits = self.router.router(hidden_states)  # (batch, seq, num_chapters)
        probs = F.softmax(logits, dim=-1)
        
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return indices, weights


class HybridRouter:
    """
    Hybrid routing strategy for generation.
    
    Uses sequence-level during prefill, then switches to
    rolling window during generation.
    """
    
    def __init__(
        self,
        router: nn.Module,
        top_k: int,
        window_size: int = 128,
    ):
        self.sequence_router = SequenceLevelRouter(router, top_k)
        self.rolling_router = RollingWindowRouter(router, top_k, window_size)
        self.is_generating = False
    
    def start_generation(self, prefill_hidden_states: torch.Tensor):
        """
        Start generation phase.
        
        Args:
            prefill_hidden_states: Hidden states from prefill
        """
        self.is_generating = True
        # Initialize rolling window cache with end of prefill
        if prefill_hidden_states.shape[1] > self.rolling_router.window_size:
            self.rolling_router.cache = prefill_hidden_states[:, -self.rolling_router.window_size:].detach()
        else:
            self.rolling_router.cache = prefill_hidden_states.detach()
    
    def reset(self):
        """Reset to prefill mode."""
        self.is_generating = False
        self.rolling_router.reset_cache()
    
    def route(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route based on current phase."""
        if self.is_generating:
            return self.rolling_router.route(hidden_states)
        else:
            return self.sequence_router.route(hidden_states)


def create_inference_router(
    router: nn.Module,
    strategy: str,
    top_k: int,
    window_size: int = 128,
):
    """
    Factory function to create inference router.
    
    Args:
        router: Trained router module
        strategy: "sequence", "rolling", "token", "hybrid"
        top_k: Number of chapters to select
        window_size: Window size for rolling strategy
        
    Returns:
        Router instance
    """
    if strategy == "sequence":
        return SequenceLevelRouter(router, top_k)
    elif strategy == "rolling":
        return RollingWindowRouter(router, top_k, window_size)
    elif strategy == "token":
        return TokenLevelRouter(router, top_k)
    elif strategy == "hybrid":
        return HybridRouter(router, top_k, window_size)
    else:
        raise ValueError(f"Unknown routing strategy: {strategy}")
