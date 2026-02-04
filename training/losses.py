"""
Training losses including router auxiliary losses.
"""

from typing import Dict, List
import torch


def compute_router_auxiliary_loss(
    router_losses: List[Dict[str, torch.Tensor]],
    load_balance_coef: float = 0.01,
    auxiliary_coef: float = 0.0,
    z_loss_coef: float = 0.0,
) -> torch.Tensor:
    """
    Aggregate router auxiliary losses from multiple layers.
    
    Args:
        router_losses: List of loss dicts from each router
        load_balance_coef: Coefficient for load balancing loss
        auxiliary_coef: Coefficient for auxiliary loss
        z_loss_coef: Coefficient for z-loss
        
    Returns:
        Total auxiliary loss
    """
    if not router_losses:
        return torch.tensor(0.0)
    
    device = None
    for losses in router_losses:
        if losses:
            device = next(iter(losses.values())).device
            break
    
    if device is None:
        return torch.tensor(0.0)
    
    total = torch.tensor(0.0, device=device)
    count = 0
    
    for losses in router_losses:
        if not losses:
            continue
        
        if "load_balance" in losses and load_balance_coef > 0:
            total = total + load_balance_coef * losses["load_balance"]
        
        if "auxiliary" in losses and auxiliary_coef > 0:
            total = total + auxiliary_coef * losses["auxiliary"]
        
        if "z_loss" in losses and z_loss_coef > 0:
            total = total + z_loss_coef * losses["z_loss"]
        
        count += 1
    
    return total / max(count, 1)
