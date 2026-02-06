#!/usr/bin/env python3
"""
Evaluation script for Memory-Augmented Transformer.

Supports:
- Single GPU evaluation
- Multi-GPU distributed evaluation (via Accelerate)
- Various metrics (perplexity, accuracy)

Usage:
    # Single GPU
    python scripts/eval.py --config configs/adapter_qwen2.5_1.5b.yaml --checkpoint outputs/final_model
    
    # Multi-GPU (distributed)
    accelerate launch scripts/eval.py --distributed --config configs/base_small.yaml --checkpoint outputs/final_model
"""

import argparse
import sys
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_transformer.config import load_config
from memory_transformer.model import MemoryTransformer
from memory_transformer.adapter import MemoryAdapter
from training.data import create_dataloader
from transformers import AutoTokenizer

# Try to import accelerate for distributed eval
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


def load_model(config, checkpoint_path: str = None):
    """Load model from config and optionally checkpoint."""
    if config.model.base_model_name is not None:
        model = MemoryAdapter(config)
    else:
        model = MemoryTransformer(config)
    
    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path)
        model_path_pt = checkpoint_dir / "model.pt"
        if model_path_pt.exists():
            state_dict = torch.load(model_path_pt, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            # Fallback: support Accelerate/Transformers-style checkpoint files.
            safe_path = checkpoint_dir / "model.safetensors"
            bin_path = checkpoint_dir / "pytorch_model.bin"
            if safe_path.exists():
                from safetensors.torch import load_file

                state_dict = load_file(str(safe_path), device="cpu")
                model.load_state_dict(state_dict)
            elif bin_path.exists():
                state_dict = torch.load(bin_path, map_location="cpu")
                model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(
                    f"No supported model weights found in {checkpoint_dir} "
                    f"(expected {model_path_pt.name}, {safe_path.name}, or {bin_path.name})."
                )
    
    return model


def load_tokenizer(config):
    """Load tokenizer based on config."""
    tokenizer_name = config.model.tokenizer_name or config.model.base_model_name
    if tokenizer_name is None:
        tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def compute_perplexity_single(model, dataloader, device):
    """Compute perplexity on a dataset (single GPU)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs["loss"]
            # Bug 12 fix: Account for label shifting (model uses labels[..., 1:])
            shifted_labels = labels[..., 1:]
            num_tokens = (shifted_labels != -100).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss


def compute_perplexity_distributed(model, dataloader, accelerator):
    """Compute perplexity on a dataset (distributed)."""
    model.eval()
    total_loss_sum = torch.tensor(0.0, device=accelerator.device)
    total_tokens = torch.tensor(0, device=accelerator.device, dtype=torch.long)
    gather_for_metrics = getattr(accelerator, "gather_for_metrics", None)
    use_gather_for_metrics = callable(gather_for_metrics)
    
    progress_bar = tqdm(
        dataloader, 
        desc="Evaluating",
        disable=not accelerator.is_main_process
    )
    
    with torch.no_grad():
        for batch in progress_bar:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()

            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels)

            token_mask = shift_labels != -100
            per_sample_loss_sum = (per_token_loss * token_mask).sum(dim=1)
            per_sample_tokens = token_mask.sum(dim=1)

            if use_gather_for_metrics:
                per_sample_loss_sum = gather_for_metrics(per_sample_loss_sum)
                per_sample_tokens = gather_for_metrics(per_sample_tokens)

            total_loss_sum = total_loss_sum + per_sample_loss_sum.sum()
            total_tokens = total_tokens + per_sample_tokens.sum()

    if not use_gather_for_metrics:
        total_loss_sum = accelerator.reduce(total_loss_sum, reduction="sum")
        total_tokens = accelerator.reduce(total_tokens, reduction="sum")

    avg_loss = (total_loss_sum / total_tokens.clamp_min(1).to(total_loss_sum.dtype)).item()
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate Memory-Augmented Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint dir")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (ignored in distributed)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--distributed", action="store_true", help="Use distributed evaluation")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override dataset if specified
    if args.dataset:
        config.training.dataset_name = args.dataset
    
    # Check if we should use distributed
    use_distributed = args.distributed and ACCELERATE_AVAILABLE
    
    if use_distributed:
        # Distributed evaluation
        accelerator = Accelerator()
        
        if accelerator.is_main_process:
            print(f"Running distributed evaluation on {accelerator.num_processes} processes")
        
        # Load model
        model = load_model(config, args.checkpoint)

        # Optional: quantize memory bank for evaluation
        if getattr(config.memory, "quantize_memory", False):
            from inference.merge import quantize_memory_for_deployment
            model = quantize_memory_for_deployment(
                model,
                bits=config.memory.memory_quant_bits,
            )
        
        # Load tokenizer
        tokenizer = load_tokenizer(config)
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset_name=config.training.dataset_name,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=config.training.max_length,
            split=args.split,
            subset=config.training.dataset_subset,
            text_field=config.training.text_field,
            training_mode=config.training.training_mode,
            shuffle=False,
            num_samples=args.max_samples,
            drop_last=False,  # Bug 26 fix: Include all samples in eval
        )
        
        # Prepare with accelerator
        model, dataloader = accelerator.prepare(model, dataloader)
        
        # Evaluate
        perplexity, avg_loss = compute_perplexity_distributed(model, dataloader, accelerator)
        
        if accelerator.is_main_process:
            _print_results(config, args, perplexity, avg_loss, len(dataloader.dataset))
            if args.output:
                _save_results(config, args, perplexity, avg_loss, len(dataloader.dataset))
    
    else:
        # Single GPU evaluation
        print(f"Loading model from {args.checkpoint or 'config'}...")
        model = load_model(config, args.checkpoint)

        # Optional: quantize memory bank for evaluation
        if getattr(config.memory, "quantize_memory", False):
            from inference.merge import quantize_memory_for_deployment
            model = quantize_memory_for_deployment(
                model,
                bits=config.memory.memory_quant_bits,
            )

        model = model.to(args.device)
        model.eval()
        
        # Load tokenizer
        tokenizer = load_tokenizer(config)
        
        # Create eval dataloader
        print(f"Loading dataset {config.training.dataset_name} split={args.split}...")
        dataloader = create_dataloader(
            dataset_name=config.training.dataset_name,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=config.training.max_length,
            split=args.split,
            subset=config.training.dataset_subset,
            text_field=config.training.text_field,
            training_mode=config.training.training_mode,
            shuffle=False,
            num_samples=args.max_samples,
            drop_last=False,  # Bug 26 fix: Include all samples in eval
        )
        
        # Evaluate
        print("Computing perplexity...")
        perplexity, avg_loss = compute_perplexity_single(model, dataloader, args.device)
        
        _print_results(config, args, perplexity, avg_loss, len(dataloader.dataset))
        if args.output:
            _save_results(config, args, perplexity, avg_loss, len(dataloader.dataset))


def _print_results(config, args, perplexity, avg_loss, num_samples):
    """Print evaluation results."""
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Dataset:     {config.training.dataset_name}")
    print(f"Split:       {args.split}")
    print(f"Samples:     {num_samples}")
    print(f"Perplexity:  {perplexity:.4f}")
    print(f"Avg Loss:    {avg_loss:.4f}")
    print("=" * 50)


def _save_results(config, args, perplexity, avg_loss, num_samples):
    """Save evaluation results to JSON."""
    results = {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "dataset": config.training.dataset_name,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "num_samples": num_samples,
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
