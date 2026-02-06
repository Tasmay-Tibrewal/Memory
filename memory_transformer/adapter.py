"""
Memory Adapter for pretrained models.

Injects memory cross-attention layers into existing pretrained
transformers (e.g., Qwen, Llama, Mistral).
"""

import math
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config, get_memory_layer_indices, get_memory_bank_assignments
from .memory_bank import (
    create_memory_bank,
    ChapteredMemoryBank,
)
from .memory_attention import MemoryCrossAttention
from .router import ChapterRouter, compute_total_router_loss
from .lora import apply_lora_to_model, get_lora_parameters
from .memory_block import RMSNorm


class MemoryAdapterLayer(nn.Module):
    """
    A single memory adapter that wraps a pretrained transformer layer.
    
    Injects memory cross-attention after self-attention (Variant A)
    or after MLP (Variant B).
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
        use_flash_attention: bool = True,  # Bug 6 fix: Add parameter
        gradient_checkpointing: bool = True,  # Memory gradient checkpointing
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Pre-memory layer norm
        self.memory_layernorm = RMSNorm(hidden_dim)
        
        # Memory cross-attention
        self.memory_attn = MemoryCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            memory_dim=memory_dim,
            use_low_rank_projections=use_low_rank_projections,
            projection_rank=projection_rank,
            reduced_dim_mode=reduced_dim_mode,
            reduced_dim=reduced_dim,
            wo_init_zero=wo_init_zero,
            dropout=dropout,
            use_flash_attention=use_flash_attention,  # Bug 6 fix: Use parameter
            gradient_checkpointing=gradient_checkpointing,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply memory cross-attention.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            memory: Memory tokens
            
        Returns:
            Updated hidden states
        """
        residual = hidden_states
        hidden_states = self.memory_layernorm(hidden_states)
        mem_output, _ = self.memory_attn(hidden_states, memory)
        return residual + mem_output


class MemoryAdapter(nn.Module):
    """
    Memory adapter wrapper for pretrained models.
    
    Loads a pretrained model and injects memory cross-attention
    layers at specified positions.
    
    Supports:
    - Qwen 2.5 series
    - Qwen 3 series  
    - Extensible to other architectures
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        self.model_config = config.model
        self.memory_config = config.memory
        
        # Load pretrained model
        if config.model.base_model_name is None:
            raise ValueError("base_model_name must be set for adapter mode")
        
        # Bug 9 fix: Map mixed_precision config to correct torch dtype
        precision = config.training.mixed_precision
        if precision == "bf16":
            torch_dtype = torch.bfloat16
        elif precision == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        
        # Get model architecture info
        self._detect_architecture()
        
        # Freeze base model if configured
        if config.model.freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Apply LoRA if configured
        self.lora_modules = {}
        if config.memory.use_lora or config.memory.use_both_memory_and_lora:
            self.lora_modules = apply_lora_to_model(
                self.base_model,
                target_modules=config.memory.lora_targets,
                rank=config.memory.lora_rank,
                alpha=config.memory.lora_alpha,
                dropout=config.memory.lora_dropout,
            )
        
        # Set up memory components
        # Bug 24 fix: Sync config.model.num_layers with actual model before computing placement
        # This ensures every_n, last_k, etc. use the real layer count, not the default (12)
        config.model.num_layers = self.num_layers
        
        self.memory_layer_indices = set(get_memory_layer_indices(config))
        self.memory_bank_assignments = get_memory_bank_assignments(config)
        
        # Create memory banks
        self.memory_banks = nn.ModuleDict()
        self._create_memory_banks()
        
        # Create routers
        self.routers = nn.ModuleDict()
        self._create_routers()
        
        # Create memory adapter layers
        self.memory_adapters = nn.ModuleDict()
        self._create_memory_adapters()
        
        # Hook to capture intermediate states
        self._hooks = []
        self._hidden_states_cache = {}

        # Routing cache for rolling/hybrid inference strategies (per layer)
        self._routing_cache: Dict[str, torch.Tensor] = {}
    
    def _detect_architecture(self):
        """Detect pretrained model architecture."""
        model_type = self.base_model.config.model_type.lower()
        
        if "qwen" in model_type:
            self.architecture = "qwen"
            self.hidden_dim = self.base_model.config.hidden_size
            self.num_heads = self.base_model.config.num_attention_heads
            self.num_layers = self.base_model.config.num_hidden_layers
            self._get_layers_fn = lambda: self.base_model.model.layers
        elif "llama" in model_type:
            self.architecture = "llama"
            self.hidden_dim = self.base_model.config.hidden_size
            self.num_heads = self.base_model.config.num_attention_heads
            self.num_layers = self.base_model.config.num_hidden_layers
            self._get_layers_fn = lambda: self.base_model.model.layers
        elif "mistral" in model_type:
            self.architecture = "mistral"
            self.hidden_dim = self.base_model.config.hidden_size
            self.num_heads = self.base_model.config.num_attention_heads
            self.num_layers = self.base_model.config.num_hidden_layers
            self._get_layers_fn = lambda: self.base_model.model.layers
        else:
            # Try generic detection
            self.architecture = "generic"
            self.hidden_dim = getattr(self.base_model.config, 'hidden_size', 
                                     getattr(self.base_model.config, 'd_model', 768))
            self.num_heads = getattr(self.base_model.config, 'num_attention_heads',
                                    getattr(self.base_model.config, 'n_head', 12))
            self.num_layers = getattr(self.base_model.config, 'num_hidden_layers',
                                     getattr(self.base_model.config, 'n_layer', 12))
            self._get_layers_fn = lambda: self._find_layers()
    
    def _find_layers(self):
        """Try to find transformer layers in generic model."""
        # Common patterns
        if hasattr(self.base_model, 'model'):
            if hasattr(self.base_model.model, 'layers'):
                return self.base_model.model.layers
            if hasattr(self.base_model.model, 'decoder'):
                if hasattr(self.base_model.model.decoder, 'layers'):
                    return self.base_model.model.decoder.layers
        if hasattr(self.base_model, 'transformer'):
            if hasattr(self.base_model.transformer, 'h'):
                return self.base_model.transformer.h
        raise ValueError(f"Cannot find transformer layers in model of type {type(self.base_model)}")
    
    def _get_memory_dim(self) -> int:
        """Get dimension of memory tokens."""
        mem_cfg = self.memory_config
        if mem_cfg.use_low_rank_memory and mem_cfg.low_rank_mode == "reduced_dim":
            return mem_cfg.memory_rank
        return mem_cfg.memory_dim or self.hidden_dim
    
    def _create_memory_banks(self):
        """Create memory bank(s)."""
        if self.memory_config.vanilla_mode or not self.memory_layer_indices:
            return
        
        mem_cfg = self.memory_config
        memory_dim = mem_cfg.memory_dim or self.hidden_dim
        
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
            
            if mem_cfg.use_chapters:
                bank = ChapteredMemoryBank(bank, mem_cfg.num_chapters)
            
            self.memory_banks[str(bank_idx)] = bank
    
    def _create_routers(self):
        """Create chapter routers."""
        if self.memory_config.vanilla_mode or not self.memory_config.use_chapters:
            return
        
        mem_cfg = self.memory_config
        
        for layer_idx in self.memory_layer_indices:
            router = ChapterRouter(
                hidden_dim=self.hidden_dim,
                num_chapters=mem_cfg.num_chapters,
                top_k=mem_cfg.top_k_chapters,
                routing_strategy=mem_cfg.routing_strategy_train,
            )
            self.routers[str(layer_idx)] = router
    
    def _create_memory_adapters(self):
        """Create memory adapter layers."""
        if self.memory_config.vanilla_mode or not self.memory_layer_indices:
            return
        
        mem_cfg = self.memory_config
        
        for layer_idx in self.memory_layer_indices:
            adapter = MemoryAdapterLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                memory_dim=self._get_memory_dim(),
                use_low_rank_projections=mem_cfg.use_low_rank_projections,
                projection_rank=mem_cfg.projection_rank,
                reduced_dim_mode=(
                    mem_cfg.use_low_rank_memory 
                    and mem_cfg.low_rank_mode == "reduced_dim"
                ),
                reduced_dim=mem_cfg.memory_rank if mem_cfg.low_rank_mode == "reduced_dim" else None,
                wo_init_zero=mem_cfg.wo_init_zero,
                use_flash_attention=self.model_config.use_flash_attention,
                gradient_checkpointing=mem_cfg.memory_gradient_checkpointing,
            )
            self.memory_adapters[str(layer_idx)] = adapter
    
    def get_memory_for_layer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get memory for a layer."""
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
        use_cache: bool = False,
        past_key_values: Optional[tuple] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with memory adaptation.
        
        This implementation uses hooks to inject memory after each layer.
        
        Note: position_offset is ignored for adapter mode since HF models
        handle positional encoding internally with past_key_values.
        """
        # Filter out position_offset - HF models don't support it and handle
        # positions internally via past_key_values
        kwargs.pop('position_offset', None)

        # Reset rolling/hybrid routing cache at the start of a new cached sequence (prefill)
        if (
            not self.training
            and use_cache
            and past_key_values is None
            and self.memory_config.use_chapters
            and self.memory_config.routing_strategy_inference in {"rolling", "hybrid"}
        ):
            self._routing_cache = {}
        
        batch_size, seq_len = input_ids.shape
        all_router_losses = []
        
        # Create wrapper for forward with memory injection
        def create_hook(layer_idx):
            def hook(module, input, output):
                # output is usually (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    rest = output[1:]
                else:
                    hidden_states = output
                    rest = ()
                
                # Apply memory if this layer has it
                if (layer_idx in self.memory_layer_indices 
                    and not self.memory_config.vanilla_mode):
                    
                    memory = self.get_memory_for_layer(layer_idx)
                    
                    # Route if using chapters
                    if self.memory_config.use_chapters and memory is not None:
                        router_key = str(layer_idx)
                        router = self.routers[router_key] if router_key in self.routers else None
                        if router is not None:
                            mem_cfg = self.memory_config
                            strategy = mem_cfg.routing_strategy_train if self.training else mem_cfg.routing_strategy_inference

                            if self.training or (not use_cache) or strategy in {"sequence", "token"}:
                                router.routing_strategy = mem_cfg.routing_strategy_train if self.training else strategy
                                chapter_indices, chapter_weights, router_losses = router(
                                    hidden_states, return_losses=self.training
                                )
                            elif strategy in {"rolling", "hybrid"}:
                                cache_key = str(layer_idx)
                                window_size = mem_cfg.routing_window_size

                                if strategy == "hybrid" and past_key_values is None:
                                    pooled = hidden_states.mean(dim=1)
                                    cache_states = hidden_states
                                    if cache_states.shape[1] > window_size:
                                        cache_states = cache_states[:, -window_size:]
                                    self._routing_cache[cache_key] = cache_states.detach()
                                else:
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
                            
                            bank_idx = self.memory_bank_assignments[layer_idx]
                            chaptered_bank = self.memory_banks[str(bank_idx)]
                            memory, _ = chaptered_bank.get_chapters_batched(chapter_indices)
                            
                            # Bug 14 fix: Weight memory tokens by routing probabilities
                            tokens_per_chapter = mem_cfg.num_memory_tokens // mem_cfg.num_chapters
                            w = chapter_weights.unsqueeze(-1)                          # (B, top_k, 1)
                            w = w.repeat(1, 1, tokens_per_chapter)                     # (B, top_k, tpc)
                            w = w.reshape(memory.shape[0], -1, 1)                      # (B, top_k*tpc, 1)
                            memory = memory * w
                    
                    # Apply memory adapter
                    if memory is not None:
                        adapter = self.memory_adapters[str(layer_idx)]
                        hidden_states = adapter(hidden_states, memory)
                
                if rest:
                    return (hidden_states,) + rest
                return hidden_states
            
            return hook
        
        # Register hooks
        layers = self._get_layers_fn()
        handles = []
        for layer_idx, layer in enumerate(layers):
            handle = layer.register_forward_hook(create_hook(layer_idx))
            handles.append(handle)
        
        try:
            # Forward through base model with use_cache and past_key_values
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                past_key_values=past_key_values,
                **kwargs,
            )
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()
        
        # Add router losses to main loss
        loss = outputs.loss
        if loss is not None and all_router_losses and self.training:
            router_loss = self._aggregate_router_losses(all_router_losses)
            loss = loss + router_loss
        
        return {
            "logits": outputs.logits,
            "loss": loss,
            "past_key_values": getattr(outputs, 'past_key_values', None),
            "router_losses": all_router_losses,
        }
    
    def _aggregate_router_losses(
        self, 
        all_losses: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Aggregate router losses."""
        mem_cfg = self.memory_config
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        non_empty = 0
        
        for losses in all_losses:
            if not losses:
                continue
            non_empty += 1
            layer_loss = compute_total_router_loss(
                losses,
                load_balance_coef=mem_cfg.load_balance_coefficient if mem_cfg.use_load_balance_loss else 0.0,
                auxiliary_coef=mem_cfg.auxiliary_loss_coefficient if mem_cfg.use_auxiliary_loss else 0.0,
                z_loss_coef=mem_cfg.z_loss_coefficient if mem_cfg.use_z_loss else 0.0,
                reference_tensor=total,
            )
            total = total + layer_loss
        
        # Bug 32 fix: Divide by the number of non-empty loss dicts (we skip empties above).
        return total / non_empty if non_empty > 0 else total

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing on the underlying pretrained model (if supported)."""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()
        # HF models generally require disabling KV cache for checkpointing correctness.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "use_cache"):
            self.base_model.config.use_cache = False

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the underlying pretrained model (if supported)."""
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "use_cache"):
            self.base_model.config.use_cache = True
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        params = []
        
        # Memory banks
        for bank in self.memory_banks.values():
            params.extend(p for p in bank.parameters() if p.requires_grad)
        
        # Routers
        for router in self.routers.values():
            params.extend(p for p in router.parameters() if p.requires_grad)
        
        # Memory adapters
        for adapter in self.memory_adapters.values():
            params.extend(p for p in adapter.parameters() if p.requires_grad)
        
        # LoRA parameters
        params.extend(get_lora_parameters(self.base_model))
        
        return params
    
    def get_memory_parameters(self) -> List[nn.Parameter]:
        """Get memory-specific parameters (excluding LoRA)."""
        params = []
        
        for bank in self.memory_banks.values():
            params.extend(bank.parameters())
        
        for router in self.routers.values():
            params.extend(router.parameters())
        
        for adapter in self.memory_adapters.values():
            params.extend(adapter.parameters())
        
        return params
    
    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups with different learning rates."""
        mem_cfg = self.memory_config
        train_cfg = self.config.training
        
        groups = []
        
        # Memory parameters
        memory_params = self.get_memory_parameters()
        if memory_params:
            groups.append({
                "params": memory_params,
                "lr": train_cfg.memory_lr,
                "name": "memory",
            })
        
        # LoRA parameters
        lora_params = get_lora_parameters(self.base_model)
        if lora_params:
            groups.append({
                "params": lora_params,
                "lr": train_cfg.lora_lr,
                "name": "lora",
            })
        
        # Base model parameters (if not frozen)
        if not self.model_config.freeze_base_model:
            base_params = [
                p for p in self.base_model.parameters() 
                if p.requires_grad and p not in set(lora_params)
            ]
            if base_params:
                groups.append({
                    "params": base_params,
                    "lr": train_cfg.base_model_lr,
                    "name": "base_model",
                })
        
        return groups


def create_memory_adapter(config: Config) -> MemoryAdapter:
    """Factory function to create MemoryAdapter."""
    return MemoryAdapter(config)
