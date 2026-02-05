"""
Chapter Router for MoE-style memory access.

Implements routing to select top-k chapters from the memory bank,
with associated load balancing and auxiliary losses.
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChapterRouter(nn.Module):
    """
    Router for selecting top-k chapters from memory bank.
    
    Uses sequence-level routing: mean-pools the input sequence,
    then computes chapter importance scores.
    
    Inspired by MoE routing (Switch Transformer, GShard).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_chapters: int,
        top_k: int = 1,
        routing_strategy: str = "sequence",  # "sequence" or "token"
    ):
        """
        Args:
            hidden_dim: Input hidden dimension
            num_chapters: Number of chapters to route to
            top_k: Number of chapters to select
            routing_strategy: "sequence" (mean-pool) or "token" (per-token)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_chapters = num_chapters
        self.top_k = top_k
        self.routing_strategy = routing_strategy
        
        # Router network: simple linear projection
        self.router = nn.Linear(hidden_dim, num_chapters, bias=True)
        
        # Initialize router
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_losses: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute chapter routing.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_dim)
            return_losses: Whether to compute and return auxiliary losses
            
        Returns:
            Tuple of:
            - chapter_indices: (batch_size, top_k) selected chapter indices
            - chapter_weights: (batch_size, top_k) routing weights for selected
            - losses: Dict of auxiliary losses (empty if return_losses=False)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if self.routing_strategy == "sequence":
            # Mean-pool over sequence
            pooled = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
            router_logits = self.router(pooled)  # (batch_size, num_chapters)
        elif self.routing_strategy == "token":
            # Per-token routing (more expensive, primarily for generation)
            router_logits = self.router(hidden_states)  # (batch, seq, num_chapters)
            # For now, still use sequence-level selection
            router_logits = router_logits.mean(dim=1)  # (batch_size, num_chapters)
        else:
            raise ValueError(f"Unknown routing_strategy: {self.routing_strategy}")
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # (batch_size, num_chapters)
        
        # Select top-k chapters
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # Both: (batch_size, top_k)
        
        # Normalize weights of selected chapters
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary losses
        losses = {}
        if return_losses:
            losses = self._compute_losses(router_logits, router_probs, top_k_indices)
        
        return top_k_indices, top_k_weights, losses
    
    def _compute_losses(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for router training.
        
        Args:
            router_logits: Raw logits (batch_size, num_chapters)
            router_probs: Softmax probabilities (batch_size, num_chapters)
            selected_indices: Selected chapter indices (batch_size, top_k)
            
        Returns:
            Dict with loss tensors
        """
        losses = {}
        batch_size = router_probs.shape[0]
        
        # Load Balancing Loss (from Switch Transformer)
        # Encourages uniform chapter usage across the batch
        # L_balance = C * sum(f_i * P_i)
        # where f_i = fraction of batch routed to chapter i
        # P_i = average probability assigned to chapter i
        
        # Compute f_i: fraction of batch that selects each chapter (in top-k)
        # Create one-hot for selected chapters and sum
        one_hot = F.one_hot(
            selected_indices, 
            num_classes=self.num_chapters
        ).float()  # (batch, top_k, num_chapters)
        
        # f_i = mean over batch of whether chapter i was selected
        f = one_hot.sum(dim=1).mean(dim=0)  # (num_chapters,)
        
        # P_i = mean probability for each chapter
        P = router_probs.mean(dim=0)  # (num_chapters,)
        
        # Load balance loss
        load_balance_loss = self.num_chapters * (f * P).sum()
        losses["load_balance"] = load_balance_loss
        
        # Auxiliary Load Loss (penalize variance in chapter usage)
        # Bug 22 fix: Use soft probabilities for differentiable loss
        # The old code used one_hot(selected_indices) which has no gradient
        # Instead, use router_probs directly for a differentiable auxiliary loss
        P_squared = router_probs.pow(2).mean(dim=0)  # (num_chapters,)
        target_uniform = torch.ones_like(P_squared) / self.num_chapters
        auxiliary_loss = F.mse_loss(P_squared, target_uniform)
        losses["auxiliary"] = auxiliary_loss
        
        # Z-Loss (regularize router logits, prevent divergence)
        # L_z = mean(log^2(sum(exp(logits))))
        log_sum_exp = torch.logsumexp(router_logits, dim=-1)  # (batch_size,)
        z_loss = (log_sum_exp ** 2).mean()
        losses["z_loss"] = z_loss
        
        return losses


class TokenLevelRouter(nn.Module):
    """
    Token-level router for per-token chapter selection.
    
    Primarily useful during autoregressive generation where
    sequence length is 1 and token-level routing is feasible.
    
    WARNING: Using this during training/prefill with long sequences
    will be very memory intensive.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_chapters: int,
        top_k: int = 1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_chapters = num_chapters
        self.top_k = top_k
        
        self.router = nn.Linear(hidden_dim, num_chapters, bias=True)
        
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token routing.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            
        Returns:
            Tuple of:
            - chapter_indices: (batch_size, seq_len, top_k)
            - chapter_weights: (batch_size, seq_len, top_k)
        """
        # Router logits for each token
        router_logits = self.router(hidden_states)  # (batch, seq, num_chapters)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k per token
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # Both: (batch, seq, top_k)
        
        # Normalize
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_indices, top_k_weights


class RollingRouter(nn.Module):
    """
    Rolling window router for inference.
    
    During generation, uses a rolling window of recent tokens
    to compute routing decisions, providing some context
    while remaining efficient.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_chapters: int,
        top_k: int = 1,
        window_size: int = 64,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_chapters = num_chapters
        self.top_k = top_k
        self.window_size = window_size
        
        self.router = nn.Linear(hidden_dim, num_chapters, bias=True)
        
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cached_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing with rolling window.
        
        Args:
            hidden_states: Current states (batch_size, seq_len, hidden_dim)
            cached_states: Previous states from window (batch_size, window_len, hidden_dim)
            
        Returns:
            Tuple of:
            - chapter_indices: (batch_size, top_k)
            - chapter_weights: (batch_size, top_k)
            - updated_cache: Updated rolling window cache
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if cached_states is not None:
            # Concatenate with cache
            combined = torch.cat([cached_states, hidden_states], dim=1)
        else:
            combined = hidden_states
        
        # Keep only last window_size tokens
        if combined.shape[1] > self.window_size:
            combined = combined[:, -self.window_size:]
        
        # Mean pool over window
        pooled = combined.mean(dim=1)
        router_logits = self.router(pooled)
        router_probs = F.softmax(router_logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_indices, top_k_weights, combined


def compute_total_router_loss(
    losses: Dict[str, torch.Tensor],
    load_balance_coef: float = 0.01,
    auxiliary_coef: float = 0.0,
    z_loss_coef: float = 0.0,
) -> torch.Tensor:
    """
    Compute total router auxiliary loss.
    
    Args:
        losses: Dict from router forward pass
        load_balance_coef: Coefficient for load balancing loss
        auxiliary_coef: Coefficient for auxiliary loss
        z_loss_coef: Coefficient for z-loss
        
    Returns:
        Total auxiliary loss
    """
    total = torch.tensor(0.0, device=next(iter(losses.values())).device)
    
    if "load_balance" in losses and load_balance_coef > 0:
        total = total + load_balance_coef * losses["load_balance"]
    
    if "auxiliary" in losses and auxiliary_coef > 0:
        total = total + auxiliary_coef * losses["auxiliary"]
    
    if "z_loss" in losses and z_loss_coef > 0:
        total = total + z_loss_coef * losses["z_loss"]
    
    return total
