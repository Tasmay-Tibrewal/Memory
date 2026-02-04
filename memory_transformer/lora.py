"""
Standard LoRA (Low-Rank Adaptation) implementation.

Provides LoRA for comparison experiments and for
combined Memory + LoRA adapter mode.
"""

import math
from typing import Optional, List, Dict, Set
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Implements: output = x @ W + x @ A @ B * (alpha / r)
    where W is frozen original weights, A and B are trainable low-rank matrices.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout on LoRA path
            merge_weights: Whether to merge LoRA into base weights
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False
        
        # Original weight (will be set from pretrained)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # LoRA components
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_lora()
    
    def _init_lora(self):
        """Initialize LoRA matrices."""
        # A initialized with Kaiming, B initialized to zero
        # This means LoRA contribution starts at zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def merge(self):
        """Merge LoRA weights into base weights."""
        if not self.merged:
            # W' = W + B @ A * scaling
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from base weights."""
        if self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Output tensor (..., out_features)
        """
        if self.merged:
            return F.linear(x, self.weight)
        
        # Base output
        result = F.linear(x, self.weight)
        
        # LoRA adaptation
        lora_out = self.lora_dropout(x)
        lora_out = lora_out @ self.lora_A.t()  # (..., rank)
        lora_out = lora_out @ self.lora_B.t()  # (..., out_features)
        lora_out = lora_out * self.scaling
        
        return result + lora_out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """
        Create LoRALinear from existing nn.Linear.
        
        Args:
            linear: Existing linear layer to adapt
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout
            
        Returns:
            LoRALinear with copied weights
        """
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Copy original weights
        lora_linear.weight.data.copy_(linear.weight.data)
        
        # Freeze original weights
        lora_linear.weight.requires_grad = False
        
        return lora_linear


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
) -> Dict[str, LoRALinear]:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: Model to modify
        target_modules: List of module name patterns to target
                       e.g., ["q_proj", "v_proj", "k_proj", "o_proj"]
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        
    Returns:
        Dict mapping module names to their LoRALinear replacements
    """
    lora_modules = {}
    
    def should_apply(name: str) -> bool:
        return any(target in name for target in target_modules)
    
    # Find and replace matching linear layers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and should_apply(name):
            # Get parent module
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            # Create LoRA version
            lora_layer = LoRALinear.from_linear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            
            # Replace in parent
            setattr(parent, child_name, lora_layer)
            lora_modules[name] = lora_layer
    
    return lora_modules


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get only the LoRA parameters from a model.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        List of LoRA A and B parameters
    """
    lora_params = []
    
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(param)
    
    return lora_params


def freeze_non_lora_parameters(model: nn.Module):
    """
    Freeze all parameters except LoRA parameters.
    
    Args:
        model: Model with LoRA layers
    """
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False


def merge_lora_weights(model: nn.Module):
    """
    Merge all LoRA weights into base weights.
    
    Args:
        model: Model with LoRA layers
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module):
    """
    Unmerge all LoRA weights from base weights.
    
    Args:
        model: Model with LoRA layers
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def count_lora_parameters(model: nn.Module) -> int:
    """
    Count total LoRA parameters in model.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Number of LoRA parameters
    """
    total = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            total += module.lora_A.numel() + module.lora_B.numel()
    return total
