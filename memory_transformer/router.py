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
    
    Uses configurable sequence-level routing:
    - mean-pooled sequence ("sequence")
    - average of rolling-window means ("sequence-rolling")
    - optional token-level routing helper (`route_token_level`) for per-token
      chapter selection.
    
    Inspired by MoE routing (Switch Transformer, GShard).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_chapters: int,
        top_k: int = 1,
        routing_strategy: str = "sequence",  # "sequence", "sequence-rolling", or "token"
    ):
        """
        Args:
            hidden_dim: Input hidden dimension
            num_chapters: Number of chapters to route to
            top_k: Number of chapters to select
            routing_strategy:
                - "sequence": mean-pool over sequence
                - "sequence-rolling" (or "sequence_rolling"): average of per-position rolling means
                - "token": kept for backward compatibility in `forward()`;
                  use `route_token_level()` for true per-token chapter selection.
        """
        super().__init__()
        
        # Bug 43 fix: Validate chapter routing configuration early for clearer errors.
        if num_chapters <= 0:
            raise ValueError(f"num_chapters must be > 0, got {num_chapters}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if top_k > num_chapters:
            raise ValueError(
                f"top_k ({top_k}) must be <= num_chapters ({num_chapters})"
            )

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
        exclude_prefix_chapters: int = 0,
        rolling_window_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute chapter routing.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_dim)
            return_losses: Whether to compute and return auxiliary losses
            exclude_prefix_chapters: Exclude first N chapters from top-k selection
                (useful when those chapters are always included as shared chapters).
            rolling_window_size: Window size for "sequence-rolling" strategy.
                If None, defaults to full sequence length.
            
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
        elif self.routing_strategy in {"sequence-rolling", "sequence_rolling"}:
            pooled = self._sequence_rolling_pool(
                hidden_states,
                rolling_window_size=rolling_window_size,
            )  # (batch_size, hidden_dim)
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
        
        if exclude_prefix_chapters < 0 or exclude_prefix_chapters > self.num_chapters:
            raise ValueError(
                f"exclude_prefix_chapters must be in [0, {self.num_chapters}], "
                f"got {exclude_prefix_chapters}"
            )

        # Select top-k chapters (optionally excluding a shared prefix).
        if exclude_prefix_chapters > 0:
            available = self.num_chapters - exclude_prefix_chapters
            select_k = min(int(self.top_k), int(available))
            if select_k <= 0:
                top_k_indices = torch.empty(
                    (batch_size, 0),
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                top_k_weights = torch.empty(
                    (batch_size, 0),
                    dtype=router_probs.dtype,
                    device=hidden_states.device,
                )
            else:
                probs = router_probs.clone()
                probs[:, :exclude_prefix_chapters] = 0.0
                denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                probs = probs / denom
                top_k_weights, top_k_indices = torch.topk(probs, select_k, dim=-1)
                top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            top_k_weights, top_k_indices = torch.topk(
                router_probs, self.top_k, dim=-1
            )  # Both: (batch_size, top_k)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        
        # Compute auxiliary losses
        losses = {}
        if return_losses:
            losses = self._compute_losses(router_logits, router_probs, top_k_indices)
        
        return top_k_indices, top_k_weights, losses

    @staticmethod
    def _sequence_rolling_pool(
        hidden_states: torch.Tensor,
        rolling_window_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Pool hidden states by averaging rolling-window means across the sequence.

        For each position t, compute:
            mean(hidden_states[:, max(0, t-w+1):t+1, :])
        Then average these rolling means over t.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be 3D (B, T, D), got shape {tuple(hidden_states.shape)}"
            )
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if seq_len <= 0:
            raise ValueError("sequence length must be > 0")

        window = seq_len if rolling_window_size is None else int(rolling_window_size)
        if window <= 0:
            raise ValueError(f"rolling_window_size must be > 0, got {rolling_window_size}")
        window = min(window, seq_len)

        # Prefix sums for O(B*T*D) rolling-window means.
        cumsum = hidden_states.cumsum(dim=1)  # (B, T, D)
        zeros = hidden_states.new_zeros((batch_size, 1, hidden_dim))
        cumsum = torch.cat([zeros, cumsum], dim=1)  # (B, T+1, D), prefix with 0

        idx = torch.arange(seq_len, device=hidden_states.device)
        end = idx + 1
        start = torch.clamp(end - window, min=0)

        window_sums = cumsum[:, end, :] - cumsum[:, start, :]  # (B, T, D)
        lengths = (end - start).to(dtype=hidden_states.dtype).view(1, seq_len, 1)
        rolling_means = window_sums / lengths
        return rolling_means.mean(dim=1)  # (B, D)
    
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

        # Entropy metric (for monitoring router sharpness/collapse).
        # Not part of total router loss by default.
        entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-12))).sum(dim=-1).mean()
        losses["entropy"] = entropy
        
        return losses

    def route_token_level(
        self,
        hidden_states: torch.Tensor,
        return_losses: bool = True,
        exclude_prefix_chapters: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute token-level chapter routing.

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            return_losses: Whether to compute router auxiliary losses
            exclude_prefix_chapters: Exclude first N chapters from routed selection
                (used when those chapters are treated as always-on shared chapters).

        Returns:
            Tuple of:
            - chapter_indices: (batch_size, seq_len, top_k_selected)
            - chapter_weights: (batch_size, seq_len, top_k_selected)
            - losses: Router loss dict (empty when return_losses=False)
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be 3D (B, T, D), got shape {tuple(hidden_states.shape)}"
            )
        if exclude_prefix_chapters < 0 or exclude_prefix_chapters > self.num_chapters:
            raise ValueError(
                f"exclude_prefix_chapters must be in [0, {self.num_chapters}], "
                f"got {exclude_prefix_chapters}"
            )

        batch_size, seq_len, _ = hidden_states.shape
        logits = self.router(hidden_states)  # (B, T, C)
        probs = F.softmax(logits, dim=-1)

        available = self.num_chapters - exclude_prefix_chapters
        select_k = min(int(self.top_k), int(available))
        if select_k <= 0:
            empty_idx = torch.empty(
                (batch_size, seq_len, 0),
                dtype=torch.long,
                device=hidden_states.device,
            )
            empty_w = torch.empty(
                (batch_size, seq_len, 0),
                dtype=probs.dtype,
                device=hidden_states.device,
            )
            losses = {}
            if return_losses:
                flat_logits = logits.reshape(-1, self.num_chapters)
                flat_probs = probs.reshape(-1, self.num_chapters)
                flat_idx = torch.empty(
                    (batch_size * seq_len, 0),
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                losses = self._compute_losses(flat_logits, flat_probs, flat_idx)
            return empty_idx, empty_w, losses

        routed_probs = probs[..., exclude_prefix_chapters:]  # (B, T, C_shared_removed)
        routed_probs = routed_probs / routed_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        chapter_weights, chapter_indices_local = torch.topk(
            routed_probs,
            k=select_k,
            dim=-1,
        )
        chapter_weights = chapter_weights / chapter_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        chapter_indices = chapter_indices_local + int(exclude_prefix_chapters)

        losses = {}
        if return_losses:
            losses = self._compute_losses(
                logits.reshape(-1, self.num_chapters),
                probs.reshape(-1, self.num_chapters),
                chapter_indices.reshape(-1, select_k),
            )

        return chapter_indices, chapter_weights, losses


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
        if num_chapters <= 0:
            raise ValueError(f"num_chapters must be > 0, got {num_chapters}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if top_k > num_chapters:
            raise ValueError(
                f"top_k ({top_k}) must be <= num_chapters ({num_chapters})"
            )
        
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
        if num_chapters <= 0:
            raise ValueError(f"num_chapters must be > 0, got {num_chapters}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if top_k > num_chapters:
            raise ValueError(
                f"top_k ({top_k}) must be <= num_chapters ({num_chapters})"
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        
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
    reference_tensor: Optional[torch.Tensor] = None,
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
    # Bug 31 fix: Robust to empty loss dicts (can occur in some inference routing paths).
    if not losses:
        if reference_tensor is not None:
            return reference_tensor.new_tensor(0.0)
        return torch.tensor(0.0)

    first_loss = next(iter(losses.values()))
    total = first_loss.new_tensor(0.0)
    
    if "load_balance" in losses and load_balance_coef > 0:
        total = total + load_balance_coef * losses["load_balance"]
    
    if "auxiliary" in losses and auxiliary_coef > 0:
        total = total + auxiliary_coef * losses["auxiliary"]
    
    if "z_loss" in losses and z_loss_coef > 0:
        total = total + z_loss_coef * losses["z_loss"]
    
    return total
