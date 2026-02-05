"""
Full Memory-Augmented Transformer Model.

Combines all components into a complete model that can be:
1. Trained from scratch with memory
2. Used in vanilla mode (no memory) for control experiments
"""

import math
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, get_memory_layer_indices, get_memory_bank_assignments
from .memory_bank import (
    MemoryBank, 
    StandardMemoryBank, 
    FactorizedMemoryBank, 
    ReducedDimMemoryBank,
    ChapteredMemoryBank,
    create_memory_bank,
)
from .memory_block import (
    MemoryTransformerBlock, 
    VanillaTransformerBlock, 
    RMSNorm,
)
from .router import ChapterRouter, compute_total_router_loss


class MemoryTransformer(nn.Module):
    """
    Memory-Augmented Transformer for training from scratch.
    
    Supports:
    - Vanilla mode (no memory) for control experiments
    - Memory cross-attention with configurable placement
    - Chapter routing for large memory banks
    - Low-rank memory variants
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        self.model_config = config.model
        self.memory_config = config.memory
        
        hidden_dim = config.model.hidden_dim
        num_heads = config.model.num_heads
        num_layers = config.model.num_layers
        intermediate_dim = config.model.intermediate_dim
        vocab_size = config.model.vocab_size
        max_seq_len = config.model.max_seq_len
        
        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        
        # Compute which layers have memory
        self.memory_layer_indices = set(get_memory_layer_indices(config))
        self.memory_bank_assignments = get_memory_bank_assignments(config)
        
        # Create memory banks
        self.memory_banks = nn.ModuleDict()
        self._create_memory_banks()
        
        # Create routers (if using chapters)
        self.routers = nn.ModuleDict()
        self._create_routers()

        # Routing cache for rolling/hybrid inference strategies (per layer)
        self._routing_cache: Dict[str, torch.Tensor] = {}
        
        # Build transformer layers
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            has_memory = layer_idx in self.memory_layer_indices
            
            if config.memory.vanilla_mode or not has_memory:
                # Vanilla transformer block
                layer = VanillaTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    intermediate_dim=intermediate_dim,
                    max_seq_len=max_seq_len,
                    use_rope=config.model.use_rope,
                    rope_theta=config.model.rope_theta,
                    dropout=config.model.dropout,
                    attention_dropout=config.model.attention_dropout,
                    use_rms_norm=config.model.use_rms_norm,
                    norm_eps=config.model.norm_eps,
                    use_flash_attention=config.model.use_flash_attention,
                )
            else:
                # Memory-augmented block
                mem_cfg = config.memory
                layer = MemoryTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    intermediate_dim=intermediate_dim,
                    max_seq_len=max_seq_len,
                    use_rope=config.model.use_rope,
                    rope_theta=config.model.rope_theta,
                    dropout=config.model.dropout,
                    attention_dropout=config.model.attention_dropout,
                    use_rms_norm=config.model.use_rms_norm,
                    norm_eps=config.model.norm_eps,
                    use_flash_attention=config.model.use_flash_attention,
                    has_memory=True,
                    memory_dim=self._get_memory_dim(),
                    use_low_rank_projections=mem_cfg.use_low_rank_projections,
                    projection_rank=mem_cfg.projection_rank,
                    reduced_dim_mode=(
                        mem_cfg.use_low_rank_memory 
                        and mem_cfg.low_rank_mode == "reduced_dim"
                    ),
                    reduced_dim=mem_cfg.memory_rank if mem_cfg.low_rank_mode == "reduced_dim" else None,
                    wo_init_zero=mem_cfg.wo_init_zero,
                    memory_block_variant=mem_cfg.memory_block_variant,
                    gradient_checkpointing=mem_cfg.memory_gradient_checkpointing,
                )
            
            self.layers.append(layer)
        
        # Final norm and LM head
        norm_cls = RMSNorm if config.model.use_rms_norm else nn.LayerNorm
        self.norm = norm_cls(hidden_dim, eps=config.model.norm_eps)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        # Re-apply W_o zero init — apply() above clobbers it
        if self.memory_config.wo_init_zero:
            from memory_transformer.memory_attention import MemoryCrossAttention
            for module in self.modules():
                if isinstance(module, MemoryCrossAttention):
                    # Zero the output projection in whichever form it took:
                    if hasattr(module, 'o_proj'):          # full + reduced_dim paths
                        nn.init.zeros_(module.o_proj.weight)
                    if hasattr(module, 'o_up'):            # low_rank path
                        nn.init.zeros_(module.o_up.weight)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _get_memory_dim(self) -> int:
        """Get dimension of memory tokens."""
        mem_cfg = self.memory_config
        if mem_cfg.use_low_rank_memory and mem_cfg.low_rank_mode == "reduced_dim":
            return mem_cfg.memory_rank
        return mem_cfg.memory_dim or self.model_config.hidden_dim
    
    def _create_memory_banks(self):
        """Create memory bank(s) based on config."""
        if self.memory_config.vanilla_mode or not self.memory_layer_indices:
            return
        
        mem_cfg = self.memory_config
        hidden_dim = self.model_config.hidden_dim
        memory_dim = mem_cfg.memory_dim or hidden_dim
        
        # Determine how many banks we need
        if not self.memory_bank_assignments:
            return
        
        num_banks = max(self.memory_bank_assignments.values()) + 1
        
        for bank_idx in range(num_banks):
            bank = create_memory_bank(
                num_tokens=mem_cfg.num_memory_tokens,
                dim=memory_dim,
                use_low_rank=mem_cfg.use_low_rank_memory,
                rank=mem_cfg.memory_rank,
                low_rank_mode=mem_cfg.low_rank_mode,
                init_std=mem_cfg.memory_init_std,
            )
            
            # Wrap in chaptered bank if using chapters
            if mem_cfg.use_chapters:
                bank = ChapteredMemoryBank(bank, mem_cfg.num_chapters)
            
            self.memory_banks[str(bank_idx)] = bank
    
    def _create_routers(self):
        """Create routers for chapter selection."""
        if self.memory_config.vanilla_mode or not self.memory_config.use_chapters:
            return
        
        mem_cfg = self.memory_config
        hidden_dim = self.model_config.hidden_dim
        
        # One router per memory layer (or shared, depending on config)
        for layer_idx in self.memory_layer_indices:
            router = ChapterRouter(
                hidden_dim=hidden_dim,
                num_chapters=mem_cfg.num_chapters,
                top_k=mem_cfg.top_k_chapters,
                routing_strategy=mem_cfg.routing_strategy_train,
            )
            self.routers[str(layer_idx)] = router
    
    def get_memory_for_layer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get memory bank for a specific layer."""
        if layer_idx not in self.memory_bank_assignments:
            return None
        
        bank_idx = self.memory_bank_assignments[layer_idx]
        bank_key = str(bank_idx)
        if bank_key not in self.memory_banks:
            return None
        bank = self.memory_banks[bank_key]
        
        # Bug 14 fix: Simplified - isinstance check was a no-op since
        # both regular and ChapteredMemoryBank implement get_memory()
        return bank.get_memory()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            position_offset: Position offset for generation
            past_key_values: Cached KV for generation
            use_cache: Whether to return KV cache
            return_dict: Whether to return dict (always True)
            
        Returns:
            Dict with 'logits', 'loss' (if labels provided), 'past_key_values', 'router_losses'
        """
        batch_size, seq_len = input_ids.shape
        provided_past_key_values = past_key_values
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Track router losses
        all_router_losses = []
        
        # Initialize past_key_values if needed
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Reset rolling/hybrid routing cache at the start of a new cached sequence (prefill)
        if (
            not self.training
            and use_cache
            and provided_past_key_values is None
            and self.memory_config.use_chapters
            and self.memory_config.routing_strategy_inference in {"rolling", "hybrid"}
        ):
            self._routing_cache = {}
        
        new_key_values = []
        
        # Forward through layers
        for layer_idx, layer in enumerate(self.layers):
            past_kv = past_key_values[layer_idx]
            
            if layer_idx in self.memory_layer_indices and not self.memory_config.vanilla_mode:
                # Get memory for this layer
                memory = self.get_memory_for_layer(layer_idx)
                
                # Route if using chapters
                if self.memory_config.use_chapters and memory is not None:
                    router_key = str(layer_idx)
                    router = self.routers[router_key] if router_key in self.routers else None
                    if router is not None:
                        mem_cfg = self.memory_config
                        strategy = mem_cfg.routing_strategy_train if self.training else mem_cfg.routing_strategy_inference

                        # Use training routing during training; at inference we can optionally use rolling/hybrid
                        if self.training or (not use_cache) or strategy in {"sequence", "token"}:
                            router.routing_strategy = mem_cfg.routing_strategy_train if self.training else strategy
                            chapter_indices, chapter_weights, router_losses = router(
                                hidden_states, return_losses=self.training
                            )
                        elif strategy in {"rolling", "hybrid"}:
                            cache_key = str(layer_idx)
                            window_size = mem_cfg.routing_window_size

                            if strategy == "hybrid" and provided_past_key_values is None:
                                # Prefill: full-sequence routing decision, initialize rolling cache.
                                pooled = hidden_states.mean(dim=1)
                                cache_states = hidden_states
                                if cache_states.shape[1] > window_size:
                                    cache_states = cache_states[:, -window_size:]
                                self._routing_cache[cache_key] = cache_states.detach()
                            else:
                                # Generation: rolling window over recent hidden states.
                                cached = self._routing_cache.get(cache_key)
                                combined = (
                                    torch.cat([cached, hidden_states], dim=1)
                                    if cached is not None
                                    else hidden_states
                                )
                                if combined.shape[1] > window_size:
                                    combined = combined[:, -window_size:]
                                self._routing_cache[cache_key] = combined.detach()
                                pooled = combined.mean(dim=1)

                            logits = router.router(pooled)
                            probs = F.softmax(logits, dim=-1)
                            chapter_weights, chapter_indices = torch.topk(probs, router.top_k, dim=-1)
                            chapter_weights = chapter_weights / chapter_weights.sum(dim=-1, keepdim=True)
                            router_losses = {}
                        else:
                            raise ValueError(f"Unknown routing_strategy_inference: {strategy}")

                        all_router_losses.append(router_losses)
                        
                        # Get chaptered memory bank and select chapters
                        bank_idx = self.memory_bank_assignments[layer_idx]
                        chaptered_bank = self.memory_banks[str(bank_idx)]
                        memory, _ = chaptered_bank.get_chapters_batched(chapter_indices)
                        
                        # Bug 14 fix: Weight memory tokens by routing probabilities
                        # chapter_weights: (batch, top_k), each chapter contributes tokens_per_chapter tokens
                        tokens_per_chapter = mem_cfg.num_memory_tokens // mem_cfg.num_chapters
                        w = chapter_weights.unsqueeze(-1)                          # (B, top_k, 1)
                        w = w.repeat(1, 1, tokens_per_chapter)                     # (B, top_k, tpc)
                        w = w.reshape(memory.shape[0], -1, 1)                      # (B, top_k*tpc, 1)
                        memory = memory * w
                
                hidden_states, new_kv = layer(
                    hidden_states,
                    memory=memory,
                    attention_mask=attention_mask,
                    position_offset=position_offset,
                    past_kv=past_kv,
                    use_cache=use_cache,
                )
            else:
                # Vanilla layer (no memory argument)
                hidden_states, new_kv = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_offset=position_offset,
                    past_kv=past_kv,
                    use_cache=use_cache,
                )
            
            new_key_values.append(new_kv)
        
        # Final norm and LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Compute loss
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # Add router losses
            if all_router_losses and self.training:
                router_loss = self._aggregate_router_losses(all_router_losses)
                loss = loss + router_loss
        
        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": new_key_values if use_cache else None,
            "router_losses": all_router_losses,
        }
    
    def _aggregate_router_losses(
        self, 
        all_losses: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Aggregate router losses from all layers."""
        mem_cfg = self.memory_config
        total = torch.tensor(0.0, device=self.embed_tokens.weight.device)
        
        for losses in all_losses:
            layer_loss = compute_total_router_loss(
                losses,
                load_balance_coef=mem_cfg.load_balance_coefficient if mem_cfg.use_load_balance_loss else 0.0,
                auxiliary_coef=mem_cfg.auxiliary_loss_coefficient if mem_cfg.use_auxiliary_loss else 0.0,
                z_loss_coef=mem_cfg.z_loss_coefficient if mem_cfg.use_z_loss else 0.0,
            )
            total = total + layer_loss
        
        return total / len(all_losses) if all_losses else total
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_parameters(self) -> List[nn.Parameter]:
        """Get only memory-related parameters."""
        params = []
        
        # Memory banks
        for bank in self.memory_banks.values():
            params.extend(bank.parameters())
        
        # Routers
        for router in self.routers.values():
            params.extend(router.parameters())
        
        # Memory attention in blocks
        for layer in self.layers:
            if hasattr(layer, 'memory_attn'):
                params.extend(layer.memory_attn.parameters())
        
        return params
    
    def freeze_non_memory(self):
        """Freeze all non-memory parameters."""
        memory_params = set(self.get_memory_parameters())
        for param in self.parameters():
            if param not in memory_params:
                param.requires_grad = False


def create_memory_transformer(config: Config) -> MemoryTransformer:
    """Factory function to create MemoryTransformer."""
    return MemoryTransformer(config)
