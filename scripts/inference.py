#!/usr/bin/env python3
"""
Inference script for Memory-Augmented Transformer.

Usage:
    python scripts/inference.py --config configs/adapter_qwen2.5_1.5b.yaml --prompt "Hello"
    python scripts/inference.py --checkpoint outputs/final_model --prompt "What is AI?"
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_transformer.config import load_config
from memory_transformer.model import MemoryTransformer
from memory_transformer.adapter import MemoryAdapter
from transformers import AutoTokenizer


def load_model(config_path: str = None, checkpoint_path: str = None):
    """Load model from config or checkpoint."""
    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path)
        config = load_config(checkpoint_dir / "config.yaml")
    else:
        config = load_config(config_path)
    
    # Create model
    if config.model.base_model_name is not None:
        model = MemoryAdapter(config)
    else:
        model = MemoryTransformer(config)
    
    # Load weights if checkpoint provided
    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path)
        state_dict = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Inference with Memory-Augmented Transformer")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    if not args.config and not args.checkpoint:
        parser.error("Either --config or --checkpoint must be provided")
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.config, args.checkpoint)
    model = model.to(args.device)
    model.eval()
    
    # Load tokenizer
    if config.model.base_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.base_model_name,
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    
    with torch.no_grad():
        # Bug 15 + Bug 8 fix: Proper sampling with top_p and KV cache
        input_ids = inputs["input_ids"]
        past_key_values = None
        
        for _ in range(args.max_new_tokens):
            # Forward pass with KV cache
            if past_key_values is not None:
                model_input = input_ids[:, -1:]
                position_offset = input_ids.shape[1] - 1
            else:
                model_input = input_ids
                position_offset = 0
            
            outputs = model(
                input_ids=model_input,
                use_cache=True,
                past_key_values=past_key_values,
                position_offset=position_offset,
            )
            logits = outputs["logits"]
            past_key_values = outputs.get("past_key_values")
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / args.temperature
            
            # Apply top-p (nucleus) filtering - Bug 15 fix: actually use args.top_p
            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Output: {output_text}")


if __name__ == "__main__":
    main()
