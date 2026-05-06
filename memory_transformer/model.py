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
    SelfAttention,
    MLP,
)
from .router import ChapterRouter, compute_total_router_loss
from .token_routing_kernel import normalize_kernel_version


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
        self.initializer_range = float(config.model.initializer_range)
        if self.initializer_range <= 0:
            raise ValueError(
                f"model.initializer_range must be > 0, got {config.model.initializer_range}"
            )
        self.self_attn_wo_init_std = config.model.self_attn_wo_init_std
        if self.self_attn_wo_init_std is not None and self.self_attn_wo_init_std <= 0:
            raise ValueError(
                "model.self_attn_wo_init_std must be > 0 when set, "
                f"got {self.self_attn_wo_init_std}"
            )
        self.mlp_down_proj_init_std = config.model.mlp_down_proj_init_std
        if self.mlp_down_proj_init_std is not None and self.mlp_down_proj_init_std <= 0:
            raise ValueError(
                "model.mlp_down_proj_init_std must be > 0 when set, "
                f"got {self.mlp_down_proj_init_std}"
            )
        
        hidden_dim = config.model.hidden_dim
        num_heads = config.model.num_heads
        num_kv_heads = config.model.num_kv_heads
        num_layers = config.model.num_layers
        intermediate_dim = config.model.intermediate_dim
        vocab_size = config.model.vocab_size
        max_seq_len = (
            config.model.max_position_embeddings
            if config.model.max_position_embeddings is not None
            else config.model.max_seq_len
        )
        # Keep both fields synchronized for checkpoint/config clarity.
        config.model.max_seq_len = max_seq_len
        config.model.max_position_embeddings = max_seq_len

        # Validate shared-chapter routing configuration early.
        if self.memory_config.num_shared_chapters < 0:
            raise ValueError(
                f"memory.num_shared_chapters must be >= 0, got {self.memory_config.num_shared_chapters}"
            )
        if self.memory_config.use_chapters and self.memory_config.num_shared_chapters > self.memory_config.num_chapters:
            raise ValueError(
                f"memory.num_shared_chapters ({self.memory_config.num_shared_chapters}) "
                f"must be <= memory.num_chapters ({self.memory_config.num_chapters})"
            )
        if self.memory_config.routed_scaling_factor < 0:
            raise ValueError(
                f"memory.routed_scaling_factor must be >= 0, got {self.memory_config.routed_scaling_factor}"
            )
        if self.memory_config.shared_routed_norm_type not in {"rms", "layernorm"}:
            raise ValueError(
                "memory.shared_routed_norm_type must be one of {'rms', 'layernorm'}, "
                f"got {self.memory_config.shared_routed_norm_type}"
            )
        if self.memory_config.shared_routed_norm_eps <= 0:
            raise ValueError(
                f"memory.shared_routed_norm_eps must be > 0, got {self.memory_config.shared_routed_norm_eps}"
            )
        self.token_routing_kernel_version = normalize_kernel_version(
            self.memory_config.token_routing_kernel_version
        )
        
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
        mem_cfg = config.memory
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            has_memory = layer_idx in self.memory_layer_indices
            
            if config.memory.vanilla_mode or not has_memory:
                # Vanilla transformer block
                layer = VanillaTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    intermediate_dim=intermediate_dim,
                    max_seq_len=max_seq_len,
                    use_rope=config.model.use_rope,
                    rope_theta=config.model.rope_theta,
                    dropout=config.model.dropout,
                    hidden_activation=config.model.hidden_activation,
                    attention_dropout=config.model.attention_dropout,
                    use_rms_norm=config.model.use_rms_norm,
                    norm_eps=config.model.norm_eps,
                    use_flash_attention=config.model.use_flash_attention,
                )
            else:
                # Memory-augmented block
                layer = MemoryTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    memory_num_heads=mem_cfg.memory_num_heads,
                    memory_num_kv_heads=mem_cfg.memory_num_kv_heads,
                    intermediate_dim=intermediate_dim,
                    max_seq_len=max_seq_len,
                    use_rope=config.model.use_rope,
                    rope_theta=config.model.rope_theta,
                    dropout=config.model.dropout,
                    hidden_activation=config.model.hidden_activation,
                    memory_dropout=mem_cfg.memory_dropout,
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

        self.tie_embeddings = bool(config.model.tie_embeddings)
        
        # Initialize
        self.apply(self._init_weights)
        self._apply_targeted_init_overrides()

        # Optionally tie input embedding and LM head weights.
        if self.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
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

        # Bug 39 fix: Training-level gradient checkpointing support (enabled via Trainer when requested).
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for transformer layers (from-scratch model)."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for transformer layers (from-scratch model)."""
        self._gradient_checkpointing = False
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)

    def _apply_targeted_init_overrides(self):
        """
        Apply optional targeted init overrides for selected module weights.

        These overrides are only used in the from-scratch model path and only
        when explicitly set in config; otherwise global initializer_range stays
        in effect.
        """
        if self.self_attn_wo_init_std is not None:
            for module in self.modules():
                if isinstance(module, SelfAttention):
                    nn.init.normal_(
                        module.o_proj.weight,
                        mean=0.0,
                        std=self.self_attn_wo_init_std,
                    )

        if self.mlp_down_proj_init_std is not None:
            for module in self.modules():
                if isinstance(module, MLP):
                    nn.init.normal_(
                        module.down_proj.weight,
                        mean=0.0,
                        std=self.mlp_down_proj_init_std,
                    )
    
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

    @staticmethod
    def _select_routed_chapters_from_probs(
        probs: torch.Tensor,
        top_k: int,
        num_shared_chapters: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select routed chapters from router probabilities, excluding shared prefix."""
        batch_size, num_chapters = probs.shape
        shared = max(0, min(int(num_shared_chapters), int(num_chapters)))
        available = num_chapters - shared
        select_k = min(int(top_k), int(available))
        if select_k <= 0:
            return (
                torch.empty((batch_size, 0), dtype=torch.long, device=probs.device),
                torch.empty((batch_size, 0), dtype=probs.dtype, device=probs.device),
            )

        masked_probs = probs.clone()
        if shared > 0:
            masked_probs[:, :shared] = 0.0
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        chapter_weights, chapter_indices = torch.topk(masked_probs, select_k, dim=-1)
        chapter_weights = chapter_weights / chapter_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return chapter_indices, chapter_weights

    @staticmethod
    def _prepend_shared_chapters(
        chapter_indices: torch.Tensor,
        chapter_weights: torch.Tensor,
        *,
        num_chapters: int,
        num_shared_chapters: int,
        routed_scaling_factor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepend always-on shared chapters and normalize combined chapter weights."""
        shared = max(0, min(int(num_shared_chapters), int(num_chapters)))
        if shared <= 0:
            return chapter_indices, chapter_weights

        batch_size = chapter_indices.shape[0]
        shared_idx = torch.arange(
            shared,
            dtype=torch.long,
            device=chapter_indices.device,
        ).unsqueeze(0).expand(batch_size, -1)
        shared_weights = torch.ones(
            (batch_size, shared),
            dtype=chapter_weights.dtype,
            device=chapter_weights.device,
        )

        if chapter_indices.shape[1] > 0:
            routed_weights = chapter_weights * float(routed_scaling_factor)
            combined_indices = torch.cat([shared_idx, chapter_indices], dim=-1)
            combined_weights = torch.cat([shared_weights, routed_weights], dim=-1)
        else:
            combined_indices = shared_idx
            combined_weights = shared_weights

        combined_weights = combined_weights / combined_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return combined_indices, combined_weights

    @staticmethod
    def _normalize_shared_routed_memory_vectors(
        memory: torch.Tensor,
        *,
        num_shared_chapters: int,
        tokens_per_chapter: int,
        norm_type: str,
        eps: float,
    ) -> torch.Tensor:
        """
        Normalize shared and routed memory-token vectors separately before mixing.

        Expected memory layout is [shared chapters..., routed chapters...].
        """
        if memory.dim() != 3:
            raise ValueError(
                f"memory must be 3D (B, T, D), got shape {tuple(memory.shape)}"
            )
        if tokens_per_chapter <= 0:
            return memory

        shared_tokens = max(0, int(num_shared_chapters)) * int(tokens_per_chapter)
        shared_tokens = min(shared_tokens, int(memory.shape[1]))
        if shared_tokens <= 0 or shared_tokens >= int(memory.shape[1]):
            # No branch mixing in these cases.
            return memory

        shared = memory[:, :shared_tokens, :]
        routed = memory[:, shared_tokens:, :]

        if norm_type == "rms":
            shared = shared / shared.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
            routed = routed / routed.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        elif norm_type == "layernorm":
            shared = F.layer_norm(shared, (shared.shape[-1],), eps=eps)
            routed = F.layer_norm(routed, (routed.shape[-1],), eps=eps)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        return torch.cat([shared, routed], dim=1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
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
            position_ids: Optional per-token position IDs. When provided, overrides
                position_offset and supports padded batches with KV-cache.
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

        def _masked_mean(states: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
            """
            Mean-pool over sequence while ignoring padding (mask==0).

            mask is expected to be (B, T_mask). If T_mask != T_states, we align by
            taking the rightmost T_states positions (common for KV-cache decoding where
            attention_mask covers the full kv_len while states are a shorter window).
            """
            if mask is None:
                return states.mean(dim=1)
            if mask.dim() != 2:
                raise ValueError(f"attention_mask must be 2D (got shape {tuple(mask.shape)})")
            if mask.shape[0] != states.shape[0]:
                raise ValueError(
                    f"attention_mask batch ({mask.shape[0]}) must match states batch ({states.shape[0]})"
                )
            if mask.shape[1] < states.shape[1]:
                raise ValueError(
                    f"attention_mask length ({mask.shape[1]}) must be >= states length ({states.shape[1]})"
                )
            if mask.shape[1] != states.shape[1]:
                mask = mask[:, -states.shape[1]:]

            m = mask.to(dtype=states.dtype, device=states.device).unsqueeze(-1)  # (B, T, 1)
            denom = m.sum(dim=1).clamp(min=1.0)  # (B, 1)
            return (states * m).sum(dim=1) / denom

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
        use_gradient_checkpointing = (
            self._gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and not use_cache
        )
        checkpoint_fn = None
        if use_gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint as checkpoint_fn
        
        # Forward through layers
        for layer_idx, layer in enumerate(self.layers):
            past_kv = past_key_values[layer_idx]
            
            if layer_idx in self.memory_layer_indices and not self.memory_config.vanilla_mode:
                # Get memory for this layer
                memory = self.get_memory_for_layer(layer_idx)
                token_routing_state = None
                
                # Route if using chapters
                if self.memory_config.use_chapters and memory is not None:
                    router_key = str(layer_idx)
                    router = self.routers[router_key] if router_key in self.routers else None
                    if router is not None:
                        mem_cfg = self.memory_config
                        strategy = mem_cfg.routing_strategy_train if self.training else mem_cfg.routing_strategy_inference
                        bank_idx = self.memory_bank_assignments[layer_idx]
                        chaptered_bank = self.memory_banks[str(bank_idx)]
                        tokens_per_chapter = mem_cfg.num_memory_tokens // mem_cfg.num_chapters

                        if strategy == "token":
                            chapter_indices_global, chapter_weights, router_losses = router.route_token_level(
                                hidden_states=hidden_states,
                                return_losses=self.training,
                                exclude_prefix_chapters=mem_cfg.num_shared_chapters,
                            )
                            all_router_losses.append(router_losses)

                            full_memory = chaptered_bank.get_memory()  # [N_m, D]
                            num_shared = max(0, min(int(mem_cfg.num_shared_chapters), int(mem_cfg.num_chapters)))
                            shared_tokens = num_shared * tokens_per_chapter

                            if shared_tokens > 0:
                                shared_memory = full_memory[:shared_tokens].contiguous()
                            else:
                                shared_memory = None

                            if shared_tokens < full_memory.shape[0]:
                                routed_memory = full_memory[shared_tokens:].contiguous()
                            else:
                                routed_memory = None

                            if (
                                mem_cfg.normalize_shared_routed_before_mixing
                                and shared_memory is not None
                                and routed_memory is not None
                                and shared_memory.numel() > 0
                                and routed_memory.numel() > 0
                            ):
                                merged = torch.cat([shared_memory, routed_memory], dim=0).unsqueeze(0)
                                merged = self._normalize_shared_routed_memory_vectors(
                                    merged,
                                    num_shared_chapters=num_shared,
                                    tokens_per_chapter=tokens_per_chapter,
                                    norm_type=mem_cfg.shared_routed_norm_type,
                                    eps=float(mem_cfg.shared_routed_norm_eps),
                                ).squeeze(0)
                                shared_memory = merged[:shared_tokens].contiguous()
                                routed_memory = merged[shared_tokens:].contiguous()

                            if chapter_indices_global.numel() > 0:
                                chapter_indices_local = chapter_indices_global - int(num_shared)
                                if torch.any(chapter_indices_local < 0):
                                    raise RuntimeError(
                                        "Token-level routed chapter indices underflow after shared-prefix removal."
                                    )
                            else:
                                chapter_indices_local = chapter_indices_global

                            token_routing_state = {
                                "shared_memory": shared_memory,
                                "routed_memory": routed_memory,
                                "token_chapter_indices": chapter_indices_local.to(dtype=torch.int32),
                                "tokens_per_chapter": int(tokens_per_chapter),
                                "routed_scale": float(mem_cfg.routed_scaling_factor),
                                "kernel_version": self.token_routing_kernel_version,
                                # chapter_weights: (B, T, top_k) router-produced per-token
                                # importance weights for each selected chapter. Used for
                                # MoE-style weighted combination of per-chapter attention
                                # outputs. When present, each chapter's cross-attention is
                                # computed independently and mixed by these weights; when
                                # None, all chapters are attended jointly in a single pass.
                                "chapter_weights": chapter_weights,
                            }
                            memory = None

                        # Use router-native strategies during training and non-cached
                        # eval/prefill. rolling/hybrid require decode-time cache, so
                        # when use_cache=False we fall back to sequence routing.
                        elif (
                            self.training
                            or strategy in {"sequence", "sequence-rolling", "sequence_rolling"}
                            or ((not use_cache) and strategy in {"rolling", "hybrid"})
                        ):
                            if self.training:
                                effective_strategy = mem_cfg.routing_strategy_train
                            elif (not use_cache) and strategy in {"rolling", "hybrid"}:
                                effective_strategy = "sequence"
                            else:
                                effective_strategy = strategy
                            router.routing_strategy = effective_strategy
                            chapter_indices, chapter_weights, router_losses = router(
                                hidden_states,
                                return_losses=self.training,
                                exclude_prefix_chapters=mem_cfg.num_shared_chapters,
                                rolling_window_size=mem_cfg.routing_window_size,
                            )
                        elif strategy in {"rolling", "hybrid"}:
                            cache_key = str(layer_idx)
                            window_size = mem_cfg.routing_window_size

                            if strategy == "hybrid" and provided_past_key_values is None:
                                # Prefill: full-sequence routing decision, initialize rolling cache.
                                pooled = _masked_mean(hidden_states, attention_mask)
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
                                pooled = _masked_mean(combined, attention_mask)

                            logits = router.router(pooled)
                            probs = F.softmax(logits, dim=-1)
                            chapter_indices, chapter_weights = self._select_routed_chapters_from_probs(
                                probs,
                                top_k=router.top_k,
                                num_shared_chapters=mem_cfg.num_shared_chapters,
                            )
                            router_losses = {}
                        else:
                            raise ValueError(f"Unknown routing_strategy_inference: {strategy}")

                        if strategy != "token":
                            chapter_indices, chapter_weights = self._prepend_shared_chapters(
                                chapter_indices,
                                chapter_weights,
                                num_chapters=mem_cfg.num_chapters,
                                num_shared_chapters=mem_cfg.num_shared_chapters,
                                routed_scaling_factor=mem_cfg.routed_scaling_factor,
                            )

                            all_router_losses.append(router_losses)
                            
                            # Get chaptered memory bank and select chapters
                            memory, _ = chaptered_bank.get_chapters_batched(chapter_indices)
                            
                            # Bug 14 fix: Weight memory tokens by routing probabilities
                            # chapter_weights: (batch, top_k), each chapter contributes tokens_per_chapter tokens
                            if mem_cfg.normalize_shared_routed_before_mixing:
                                memory = self._normalize_shared_routed_memory_vectors(
                                    memory,
                                    num_shared_chapters=mem_cfg.num_shared_chapters,
                                    tokens_per_chapter=tokens_per_chapter,
                                    norm_type=mem_cfg.shared_routed_norm_type,
                                    eps=float(mem_cfg.shared_routed_norm_eps),
                                )
                            w = chapter_weights.unsqueeze(-1)                          # (B, top_k, 1)
                            w = w.repeat(1, 1, tokens_per_chapter)                     # (B, top_k, tpc)
                            w = w.reshape(memory.shape[0], -1, 1)                      # (B, top_k*tpc, 1)
                            memory = memory * w
                
                if use_gradient_checkpointing:
                    def _layer_forward(
                        h: torch.Tensor,
                        _layer: nn.Module = layer,
                        _memory: Optional[torch.Tensor] = memory,
                        _token_routing_state = token_routing_state,
                    ) -> torch.Tensor:
                        out, _ = _layer(
                            h,
                            memory=_memory,
                            token_routing_state=_token_routing_state,
                            token_routing_kernel_version=self.token_routing_kernel_version,
                            attention_mask=attention_mask,
                            position_offset=position_offset,
                            position_ids=position_ids,
                            past_kv=None,
                            use_cache=False,
                        )
                        return out

                    hidden_states = checkpoint_fn(_layer_forward, hidden_states, use_reentrant=False)
                    new_kv = None
                else:
                    hidden_states, new_kv = layer(
                        hidden_states,
                        memory=memory,
                        token_routing_state=token_routing_state,
                        token_routing_kernel_version=self.token_routing_kernel_version,
                        attention_mask=attention_mask,
                        position_offset=position_offset,
                        position_ids=position_ids,
                        past_kv=past_kv,
                        use_cache=use_cache,
                    )
            else:
                # Vanilla layer (no memory argument)
                if use_gradient_checkpointing:
                    def _layer_forward(
                        h: torch.Tensor,
                        _layer: nn.Module = layer,
                    ) -> torch.Tensor:
                        out, _ = _layer(
                            h,
                            attention_mask=attention_mask,
                            position_offset=position_offset,
                            position_ids=position_ids,
                            past_kv=None,
                            use_cache=False,
                        )
                        return out

                    hidden_states = checkpoint_fn(_layer_forward, hidden_states, use_reentrant=False)
                    new_kv = None
                else:
                    hidden_states, new_kv = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_offset=position_offset,
                        position_ids=position_ids,
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
                reference_tensor=total,
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
