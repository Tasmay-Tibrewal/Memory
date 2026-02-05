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
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create cosine annealing schedule with warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create linear schedule with warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        return max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)
            ),
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def print_model_info(model: nn.Module, config: Optional[Any] = None):
    """Print model information."""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    model_size = get_model_size_mb(model)
    
    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total Parameters:     {format_params(total_params)}")
    print(f"Trainable Parameters: {format_params(trainable_params)}")
    print(f"Trainable %:          {100 * trainable_params / total_params:.2f}%")
    print(f"Model Size:           {model_size:.2f} MB")
    
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
    checkpoint = torch.load(path, map_location="cpu")
    
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
