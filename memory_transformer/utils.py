"""
Utility functions for Memory-Augmented Transformer.
"""

import os
import random
import math
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import numpy as np


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(num_params: int) -> str:
    """Format parameter count nicely."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def estimate_bf16_size_mb(num_params: int) -> float:
    """Estimate parameter memory in MB if stored in bf16 (2 bytes/param)."""
    return (num_params * 2) / 1024 / 1024


def _is_memory_layer_parameter(name: str) -> bool:
    """Return True if parameter belongs to memory-layer logic (excluding banks)."""
    memory_layer_markers = (
        "memory_adapters.",          # Adapter mode memory layers
        ".memory_attn.",             # From-scratch memory attention
        ".memory_layernorm.",        # From-scratch memory norm
        ".post_memory_layernorm.",   # Variant B extra norm
        ".post_memory_mlp.",         # Variant B extra MLP
        "routers.",                  # Chapter router params
    )
    return any(marker in name for marker in memory_layer_markers)


def compute_parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """
    Compute parameter-count breakdown for training logs.

    Returns counts for:
    - total model
    - vanilla transformer part (no memory/adapters/LoRA)
    - LoRA parameters
    - memory bank parameters
    - memory-layer parameters without bank
    - memory-layer parameters with bank
    """
    total_params = 0
    lora_params = 0
    memory_bank_params = 0
    memory_layers_no_bank_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n

        if "lora_A" in name or "lora_B" in name:
            lora_params += n
            continue

        if "memory_banks." in name or "memory_bank." in name:
            memory_bank_params += n
            continue

        if _is_memory_layer_parameter(name):
            memory_layers_no_bank_params += n
            continue

    memory_layers_with_bank_params = memory_layers_no_bank_params + memory_bank_params
    vanilla_transformer_params = (
        total_params
        - lora_params
        - memory_bank_params
        - memory_layers_no_bank_params
    )

    return {
        "total_params": total_params,
        "vanilla_transformer_params": max(vanilla_transformer_params, 0),
        "lora_params": lora_params,
        "memory_bank_params": memory_bank_params,
        "memory_layers_no_bank_params": memory_layers_no_bank_params,
        "memory_layers_with_bank_params": memory_layers_with_bank_params,
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024


def compute_memory_stats(
    num_memory_tokens: int,
    memory_dim: int,
    num_chapters: Optional[int] = None,
    top_k: Optional[int] = None,
    precision: str = "bf16",
) -> Dict[str, Any]:
    """
    Compute memory bank statistics.
    
    Returns dict with:
    - total_parameters: Total params in memory bank
    - memory_size_mb: Size in MB
    - attention_cost_per_token: Relative attention cost
    """
    bytes_per_param = 2 if precision == "bf16" or precision == "fp16" else 4
    
    total_params = num_memory_tokens * memory_dim
    memory_size_mb = total_params * bytes_per_param / 1024 / 1024
    
    # Attention cost relative to standard self-attention
    # Self-attention: O(L^2) for sequence length L
    # Memory attention: O(L * M) or O(L * k * tokens_per_chapter) with routing
    if num_chapters and top_k:
        tokens_per_chapter = num_memory_tokens // num_chapters
        effective_tokens = top_k * tokens_per_chapter
    else:
        effective_tokens = num_memory_tokens
    
    return {
        "total_parameters": total_params,
        "memory_size_mb": memory_size_mb,
        "effective_memory_tokens": effective_tokens,
        "compression_ratio": num_memory_tokens / effective_tokens if num_chapters else 1.0,
    }


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    decay_start_step: Optional[int] = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create cosine annealing schedule with warmup.
    """
    if decay_start_step is None:
        decay_start_step = num_warmup_steps
    decay_start_step = max(num_warmup_steps, min(int(decay_start_step), int(num_training_steps)))

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < decay_start_step:
            return 1.0

        progress = float(current_step - decay_start_step) / float(
            max(1, num_training_steps - decay_start_step)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    decay_start_step: Optional[int] = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create linear schedule with warmup.
    """
    if decay_start_step is None:
        decay_start_step = num_warmup_steps
    decay_start_step = max(num_warmup_steps, min(int(decay_start_step), int(num_training_steps)))

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < decay_start_step:
            return 1.0

        return max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - decay_start_step)
            ),
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_wsd_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_stable_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create Warmup-Stable-Decay schedule.

    - Warmup: linear to peak LR
    - Stable: hold at peak LR
    - Decay: cosine decay from peak to min_lr_ratio
    """
    decay_start_step = max(
        num_warmup_steps,
        min(num_training_steps, num_warmup_steps + max(int(num_stable_steps), 0)),
    )
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
        decay_start_step=decay_start_step,
    )


def configure_tokenizer_special_ids(tokenizer: Any, model_config: Any) -> None:
    """
    Apply optional tokenizer special-token IDs from config.

    If pad_token_id is not specified and tokenizer has no pad token, fallback to eos.
    """
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(model_config, attr, None)
        if value is not None:
            setattr(tokenizer, attr, int(value))

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        elif eos_id is not None:
            tokenizer.pad_token_id = eos_id


def print_model_info(model: nn.Module, config: Optional[Any] = None):
    """Print model information."""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    model_size = get_model_size_mb(model)
    breakdown = compute_parameter_breakdown(model)
    
    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total Parameters:     {format_params(total_params)}")
    print(f"Trainable Parameters: {format_params(trainable_params)}")
    print(f"Trainable %:          {100 * trainable_params / total_params:.2f}%")
    print(f"Model Size:           {model_size:.2f} MB")
    print("-" * 60)
    print("Parameter Breakdown (params, bf16 estimate)")
    print("-" * 60)
    print(
        f"{'Total model':<33} "
        f"{breakdown['total_params']:,} "
        f"({estimate_bf16_size_mb(breakdown['total_params']):.2f} MB)"
    )
    print(
        f"{'Vanilla transformer (no adapters/memory)':<33} "
        f"{breakdown['vanilla_transformer_params']:,} "
        f"({estimate_bf16_size_mb(breakdown['vanilla_transformer_params']):.2f} MB)"
    )
    print(
        f"{'LoRA':<33} "
        f"{breakdown['lora_params']:,} "
        f"({estimate_bf16_size_mb(breakdown['lora_params']):.2f} MB)"
    )
    print(
        f"{'Memory bank':<33} "
        f"{breakdown['memory_bank_params']:,} "
        f"({estimate_bf16_size_mb(breakdown['memory_bank_params']):.2f} MB)"
    )
    print(
        f"{'Memory layers (without bank)':<33} "
        f"{breakdown['memory_layers_no_bank_params']:,} "
        f"({estimate_bf16_size_mb(breakdown['memory_layers_no_bank_params']):.2f} MB)"
    )
    print(
        f"{'Memory layers (with bank)':<33} "
        f"{breakdown['memory_layers_with_bank_params']:,} "
        f"({estimate_bf16_size_mb(breakdown['memory_layers_with_bank_params']):.2f} MB)"
    )
    
    if config is not None:
        mem_cfg = config.memory
        if not mem_cfg.vanilla_mode and mem_cfg.use_memory_adapter:
            print("-" * 60)
            print("Memory Configuration")
            print("-" * 60)
            print(f"Memory Tokens:        {mem_cfg.num_memory_tokens}")
            print(f"Memory Placement:     {mem_cfg.memory_layer_placement}")
            print(f"Memory Sharing:       {mem_cfg.memory_sharing}")
            if mem_cfg.use_chapters:
                print(f"Chapters:             {mem_cfg.num_chapters}")
                print(f"Top-K Chapters:       {mem_cfg.top_k_chapters}")
            if mem_cfg.use_low_rank_memory:
                print(f"Low-Rank Mode:        {mem_cfg.low_rank_mode}")
                print(f"Rank:                 {mem_cfg.memory_rank}")
    
    print("=" * 60)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    step: int,
    loss: float,
    path: str,
    config: Optional[Any] = None,
):
    """Save training checkpoint."""
    # Bug 23 fix: Guard makedirs for paths without directory component
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if config is not None:
        # Convert config to dict for saving
        checkpoint["config"] = {
            "model": config.model.__dict__,
            "memory": config.memory.__dict__,
            "training": config.training.__dict__,
        }
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return {
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "config": checkpoint.get("config"),
    }
