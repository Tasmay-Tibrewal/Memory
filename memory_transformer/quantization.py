"""
Quantization utilities for Memory Bank.

Provides quantization options for memory bank storage to reduce VRAM usage.
Currently supports:
- 8-bit quantization
- 4-bit quantization (experimental)

Note: Full implementation pending. Config flag exists but quantization
during training requires careful gradient handling.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedMemoryBank(nn.Module):
    """
    Memory bank with quantized storage.
    
    Stores memory tokens in lower precision and dequantizes on-the-fly
    during forward pass.
    
    WARNING: This is a basic implementation. For production use,
    consider using bitsandbytes or similar libraries for better
    quantization support.
    """
    
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        quant_bits: int = 8,
        init_std: float = 0.02,
    ):
        """
        Args:
            num_tokens: Number of memory tokens
            dim: Dimension of each token
            quant_bits: Quantization bits (4 or 8)
            init_std: Initialization std
        """
        super().__init__()
        
        self.num_tokens = num_tokens
        self.dim = dim
        self.quant_bits = quant_bits
        
        # Full precision memory for initialization
        memory = torch.empty(num_tokens, dim)
        nn.init.normal_(memory, mean=0.0, std=init_std)
        
        # Store quantized version
        if quant_bits == 8:
            self._quantize_8bit(memory)
        elif quant_bits == 4:
            self._quantize_4bit(memory)
        else:
            raise ValueError(f"Unsupported quant_bits: {quant_bits}")
    
    def _quantize_8bit(self, memory: torch.Tensor):
        """Quantize to INT8 with per-row scaling."""
        # Compute per-row scales
        abs_max = memory.abs().max(dim=1, keepdim=True).values
        scales = abs_max / 127.0
        scales = scales.clamp(min=1e-8)
        
        # Quantize
        quantized = torch.round(memory / scales).clamp(-128, 127).to(torch.int8)
        
        # Store
        self.register_buffer("quantized_memory", quantized)
        self.register_buffer("scales", scales.squeeze(1))
    
    def _quantize_4bit(self, memory: torch.Tensor):
        """Quantize to INT4 (packed as INT8) with per-row scaling."""
        # Compute per-row scales
        abs_max = memory.abs().max(dim=1, keepdim=True).values
        scales = abs_max / 7.0  # 4-bit signed: -8 to 7
        scales = scales.clamp(min=1e-8)
        
        # Quantize to 4-bit range
        quantized = torch.round(memory / scales).clamp(-8, 7).to(torch.int8)
        
        # Pack two 4-bit values into one 8-bit (for even dimensions)
        if self.dim % 2 != 0:
            raise ValueError("4-bit quantization requires even dimension")
        
        # Pack: take pairs of values, pack into single byte
        quantized = quantized.view(self.num_tokens, -1, 2)
        packed = (quantized[:, :, 0] & 0x0F) | ((quantized[:, :, 1] & 0x0F) << 4)
        packed = packed.to(torch.int8)
        
        self.register_buffer("quantized_memory", packed)
        self.register_buffer("scales", scales.squeeze(1))
    
    def _dequantize(self) -> torch.Tensor:
        """Dequantize memory to float."""
        if self.quant_bits == 8:
            # Simple scale multiplication
            return self.quantized_memory.float() * self.scales.unsqueeze(1)
        elif self.quant_bits == 4:
            # Unpack 4-bit values
            packed = self.quantized_memory
            low = (packed & 0x0F).to(torch.int8)
            # Sign extend 4-bit to 8-bit
            low = torch.where(low > 7, low - 16, low)
            high = ((packed >> 4) & 0x0F).to(torch.int8)
            high = torch.where(high > 7, high - 16, high)
            
            unpacked = torch.stack([low, high], dim=-1)
            unpacked = unpacked.view(self.num_tokens, self.dim)
            
            return unpacked.float() * self.scales.unsqueeze(1)
    
    def get_memory(self) -> torch.Tensor:
        """
        Get dequantized memory.
        
        Returns:
            Memory tensor of shape (num_tokens, dim) in float
        """
        return self._dequantize()
    
    def forward(self) -> torch.Tensor:
        """Alias for get_memory."""
        return self.get_memory()


def quantize_memory_bank(
    memory_tensor: torch.Tensor,
    quant_bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a memory tensor.
    
    Args:
        memory_tensor: Full precision memory (num_tokens, dim)
        quant_bits: Bits for quantization
        
    Returns:
        Tuple of (quantized_memory, scales)
    """
    abs_max = memory_tensor.abs().max(dim=1, keepdim=True).values
    
    if quant_bits == 8:
        scales = abs_max / 127.0
        scales = scales.clamp(min=1e-8)
        quantized = torch.round(memory_tensor / scales).clamp(-128, 127).to(torch.int8)
    elif quant_bits == 4:
        scales = abs_max / 7.0
        scales = scales.clamp(min=1e-8)
        quantized = torch.round(memory_tensor / scales).clamp(-8, 7).to(torch.int8)
    else:
        raise ValueError(f"Unsupported quant_bits: {quant_bits}")
    
    return quantized, scales.squeeze(1)


def dequantize_memory_bank(
    quantized_memory: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize a memory tensor.
    
    Args:
        quantized_memory: Quantized memory (int8)
        scales: Per-row scales
        
    Returns:
        Dequantized float tensor
    """
    return quantized_memory.float() * scales.unsqueeze(1)
