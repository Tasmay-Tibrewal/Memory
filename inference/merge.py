"""
Model merging and export utilities.

Provides functions for:
- Merging LoRA weights into base model
- Extracting only memory adapter weights
- Quantizing memory for deployment
- Full model quantization (int8/4-bit/fp8)
- GGUF export helper
"""

from typing import Optional, Dict
from pathlib import Path
import torch
import torch.nn as nn
import os
import subprocess

from memory_transformer.lora import LoRALinear, merge_lora_weights
from memory_transformer.quantization import QuantizedMemoryBank


def merge_adapter_weights(
    adapter_model: nn.Module,
    output_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Merge LoRA weights into base model and extract merged state dict.
    
    Args:
        adapter_model: MemoryAdapter with LoRA layers
        output_path: Optional path to save merged weights
        
    Returns:
        Merged state dict
    """
    # Merge LoRA if present
    merge_lora_weights(adapter_model)
    
    # Get merged state dict
    state_dict = adapter_model.state_dict()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, output_path)
        print(f"Saved merged weights to {output_path}")
    
    return state_dict


def extract_memory_weights(
    model: nn.Module,
    output_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract only memory-related weights from model.
    
    This is useful for sharing just the memory adapter without base model.
    
    Args:
        model: Model with memory components
        output_path: Optional path to save weights
        
    Returns:
        State dict with only memory weights
    """
    memory_keys = [
        "memory_bank",
        "memory_attn",
        "router",
        "memory_adapters",
    ]
    
    state_dict = model.state_dict()
    memory_state_dict = {}
    
    for key, value in state_dict.items():
        if any(mk in key for mk in memory_keys):
            memory_state_dict[key] = value
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(memory_state_dict, output_path)
        print(f"Saved memory weights to {output_path}")
        print(f"  {len(memory_state_dict)} tensors")
        total_params = sum(v.numel() for v in memory_state_dict.values())
        print(f"  {total_params:,} parameters")
    
    return memory_state_dict


def quantize_memory_for_deployment(
    model: nn.Module,
    bits: int = 8,
    output_path: Optional[str] = None,
) -> nn.Module:
    """
    Quantize memory bank for deployment.
    
    Replaces memory bank with quantized version for reduced memory usage.
    
    Args:
        model: Model with memory bank
        bits: Quantization bits (4 or 8)
        output_path: Optional path to save quantized model
        
    Returns:
        Model with quantized memory bank
    """
    from memory_transformer.memory_bank import ChapteredMemoryBank
    
    # Bug 2 fix: Banks live in model.memory_banks (ModuleDict), not memory_bank (singular)
    if not hasattr(model, 'memory_banks'):
        print("Warning: Model has no memory_banks attribute, skipping quantization")
        return model
    
    for key, bank in model.memory_banks.items():
        # Handle ChapteredMemoryBank wrapping
        inner = bank.base_bank if isinstance(bank, ChapteredMemoryBank) else bank
        
        if isinstance(inner, QuantizedMemoryBank):
            continue  # Already quantized
        
        # Get original memory using the universal get_memory() method
        original_memory = inner.get_memory().detach()
        
        # Create new quantized bank with correct attribute names (dim, not memory_dim)
        quantized = QuantizedMemoryBank(
            num_tokens=inner.num_tokens,
            dim=inner.dim,
            quant_bits=bits,
        )
        
        # Apply quantization by calling the appropriate method
        with torch.no_grad():
            if bits == 8:
                quantized._quantize_8bit(original_memory)
            elif bits == 4:
                quantized._quantize_4bit(original_memory)
        
        # Replace the bank
        if isinstance(bank, ChapteredMemoryBank):
            bank.base_bank = quantized
        else:
            model.memory_banks[key] = quantized
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        print(f"Saved quantized model to {output_path}")
    
    return model


def quantize_full_model(
    model: nn.Module,
    method: str = "dynamic",  # "dynamic", "fp16", "bf16", "bnb_4bit", etc.
    output_path: Optional[str] = None,
) -> nn.Module:
    """
    Quantize the full model (base + memory) for inference.
    
    Args:
        model: Full model to quantize
        method: Quantization method:
            - "dynamic": PyTorch dynamic quantization (int8, CPU efficient)
            - "fp16", "bf16", "fp32": Standard float conversion
            - "fp8": Float8 conversion (requires torch 2.1+)
            - "bnb_8bit": bitsandbytes 8-bit (requires CUDA)
            - "bnb_4bit": bitsandbytes 4-bit (requires CUDA)
            - "bnb_fp8": bitsandbytes fp8
        output_path: Optional path to save quantized model
        
    Returns:
        Quantized model
    """
    if method == "dynamic":
        # PyTorch dynamic quantization (works on CPU)
        print("Applying PyTorch dynamic quantization (int8)...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        model = quantized_model
        
    elif method in ["fp16", "bf16", "fp32"]:
        # Standard floating point conversion
        if method == "fp16":
            dtype = torch.float16
        elif method == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
            
        print(f"Converting model to {method}...")
        model = model.to(dtype)
        
    elif method == "fp8":
        # Float8 (requires Transformer Engine or compatible PyTorch)
        print("Converting model to fp8...")
        if hasattr(torch, "float8_e4m3fn"):
            model = model.to(torch.float8_e4m3fn)
        else:
             raise ImportError("Float8 dtype not available in this PyTorch version")

    elif method.startswith("bnb"):
        # bitsandbytes quantization (GPU only)
        try:
            import bitsandbytes as bnb
            from bitsandbytes.nn import Linear8bitLt, Linear4bit, LinearFP8Mixed
        except ImportError:
            raise ImportError(
                "bitsandbytes not installed. Install with: pip install bitsandbytes"
            )
        
        print(f"Applying bitsandbytes {method} quantization...")
        
        def replace_linear_layers(module, target_cls, **kwargs):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    if target_cls == Linear8bitLt:
                        new_layer = target_cls(
                            child.in_features,
                            child.out_features,
                            child.bias is not None,
                            has_fp16_weights=False,
                            threshold=6.0,
                        )
                    elif method == "bnb_fp8":
                        new_layer = target_cls(
                            child.in_features,
                            child.out_features,
                            child.bias is not None,
                        )
                    else:  # Linear4bit
                        new_layer = target_cls(
                            child.in_features,
                            child.out_features,
                            child.bias is not None,
                            quant_type="nf4",
                            compress_statistics=True,
                        )
                    
                    new_layer.weight.data = child.weight.data
                    if child.bias is not None:
                        new_layer.bias.data = child.bias.data
                    
                    setattr(module, name, new_layer)
                else:
                    replace_linear_layers(child, target_cls, **kwargs)
        
        if method == "bnb_8bit":
            replace_linear_layers(model, Linear8bitLt)
        elif method == "bnb_4bit":
            replace_linear_layers(model, Linear4bit)
        elif method == "bnb_fp8":
            replace_linear_layers(model, LinearFP8Mixed)
            
    else:
        raise ValueError(
            f"Unknown method based on: {method}\n"
            "Options: dynamic, fp16, bf16, fp32, fp8, bnb_8bit, bnb_4bit, bnb_fp8"
        )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        print(f"Saved model to {output_path}")
        
    return model


def export_to_gguf(
    model: nn.Module,
    output_path: str,
    quantization_type: str = "q4_k_m",
    llama_cpp_path: Optional[str] = None,
):
    """
    Prepare model for GGUF export and optionally run llama.cpp conversion.
    
    Since GGUF conversion usually requires the model on disk in standard format,
    this function helps bridge the gap.
    
    Args:
        model: Trained model
        output_path: Final GGUF output path
        quantization_type: GGUF quantization type (e.g. q4_k_m, q8_0, f16)
        llama_cpp_path: Path to llama.cpp root directory (optional)
    """
    print("-" * 50)
    print("GGUF Export Helper")
    print("-" * 50)
    
    # 1. Save intermediate model in fp16/fp32
    intermediate_path = Path(output_path).with_suffix(".pt")
    print(f"1. Saving intermediate FP16 model to {intermediate_path}...")
    
    quantize_full_model(model, method="fp16", output_path=str(intermediate_path))
    
    # 2. Instructions for conversion
    print("-" * 50)
    print("Model saved. To convert to GGUF, you typically need llama.cpp.")
    print("Run the following commands if you have llama.cpp installed:")
    print("")
    print(f"python {{llama_cpp_path}}/convert.py {intermediate_path} --outfile {output_path} --outtype {quantization_type}")
    print("")
    print("Note: Memory-Augmented transformers usually need custom C++ implementation")
    print("in llama.cpp to be fully supported. Standard conversion might fail")
    print("or ignore memory components if not strictly compatible with Llama arch.")
    print("-" * 50)
    
    # 3. Attempt automatic conversion if path provided
    if llama_cpp_path:
        convert_script = Path(llama_cpp_path) / "convert.py"
        if convert_script.exists():
            print(f"Found conversion script at {convert_script}")
            print("Attempting conversion (this may fail if custom arch not supported)...")
            try:
                cmd = [
                    "python", str(convert_script),
                    str(intermediate_path),
                    "--outfile", output_path,
                    "--outtype", quantization_type
                ]
                subprocess.run(cmd, check=True)
                print(f"Successfully converted to {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Conversion failed: {e}")
            except Exception as e:
                print(f"Error running conversion: {e}")
        else:
            print(f"Warning: convert.py not found at {convert_script}")


def get_model_memory_footprint(model: nn.Module) -> Dict[str, float]:
    """
    Calculate memory footprint of model components.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dict with memory usage in MB for each component
    """
    footprint = {
        "total": 0.0,
        "memory_bank": 0.0,
        "memory_attention": 0.0,
        "lora": 0.0,
        "base_model": 0.0,
        "other": 0.0,
    }
    
    for name, param in model.named_parameters():
        size_mb = param.numel() * param.element_size() / 1024 / 1024
        footprint["total"] += size_mb
        
        if "memory_bank" in name or "memory_tokens" in name:
            footprint["memory_bank"] += size_mb
        elif "memory_attn" in name or "memory_adapters" in name:
            footprint["memory_attention"] += size_mb
        elif "lora" in name:
            footprint["lora"] += size_mb
        elif "base_model" in name:
            footprint["base_model"] += size_mb
        else:
            footprint["other"] += size_mb
    
    return footprint


def print_model_memory_report(model: nn.Module):
    """Print detailed memory usage report."""
    footprint = get_model_memory_footprint(model)
    
    print("=" * 50)
    print("Model Memory Footprint")
    print("=" * 50)
    print(f"{'Component':<20} {'Size (MB)':<15} {'%':>10}")
    print("-" * 50)
    
    total = footprint["total"]
    for name, size in footprint.items():
        if name != "total" and size > 0:
            pct = 100 * size / total if total > 0 else 0
            print(f"{name:<20} {size:<15.2f} {pct:>10.1f}%")
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {total:<15.2f} {'100.0':>10}%")
    print("=" * 50)
