"""
Generation utilities for memory-augmented transformer.
"""

from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    device: str = "cuda",
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: Memory-augmented transformer model
        tokenizer: Tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        top_k: Top-k filtering
        do_sample: Whether to sample or greedy decode
        device: Device to run on
        
    Returns:
        Generated text string
    """
    model.eval()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Bug 8 fix: Add KV cache support
    past_key_values = None
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Forward pass with KV cache
        if past_key_values is not None:
            # Only pass the last token, use cached KV
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
        
        if do_sample:
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Append (still needed for final decode)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return output_text


@torch.no_grad()
def generate_batch(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    device: str = "cuda",
) -> List[str]:
    """
    Generate text for a batch of prompts.
    
    Bug 15 fix: Added top_k, top_p, do_sample parameters with proper sampling logic.
    Bug 8 fix: Added KV cache support.
    
    Note: For simplicity, this pads all prompts to same length.
    For production, consider using dynamic batching.
    """
    model.eval()
    
    # Tokenize all prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    batch_size = input_ids.shape[0]
    finished = [False] * batch_size
    
    # Bug 8 fix: Add KV cache support
    past_key_values = None
    is_first_step = True  # Bug 21 fix: Track first step separately
    
    for _ in range(max_new_tokens):
        if all(finished):
            break
        
        # Forward pass with KV cache
        if past_key_values is not None:
            model_input = input_ids[:, -1:]
            position_offset = input_ids.shape[1] - 1
        else:
            model_input = input_ids
            position_offset = 0
        
        outputs = model(
            input_ids=model_input,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
            position_offset=position_offset,
        )
        logits = outputs["logits"]
        past_key_values = outputs.get("past_key_values")
        
        # Bug 21 fix: Find last real token position per sequence (not always -1)
        # With right-padded batches, shorter sequences have pad at position -1
        if is_first_step:
            # First step (prefill): find last non-pad position per sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
            batch_indices = torch.arange(batch_size, device=logits.device)
            next_token_logits = logits[batch_indices, seq_lengths, :]
            is_first_step = False
        else:
            # After first step, we're generating one token at a time
            next_token_logits = logits[:, -1, :]
        
        if do_sample:
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Bug 6 fix: Mask finished sequences - replace their next token with pad
        # to avoid appending garbage after EOS
        if tokenizer.pad_token_id is not None:
            pad_id = tokenizer.pad_token_id
        else:
            pad_id = tokenizer.eos_token_id  # Fallback to EOS if no pad
        
        for i, is_finished in enumerate(finished):
            if is_finished:
                next_tokens[i] = pad_id
        
        # Append (pad tokens for finished sequences won't affect decode with skip_special_tokens)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        ], dim=-1)
        
        # Check for EOS (only for non-finished sequences)
        for i, token in enumerate(next_tokens.squeeze(-1)):
            if not finished[i] and token.item() == tokenizer.eos_token_id:
                finished[i] = True
    
    # Decode all
    outputs = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in input_ids
    ]
    
    return outputs
