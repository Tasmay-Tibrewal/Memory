#!/usr/bin/env python3
"""
Generate JSONL predictions for IFEval-style external evaluators.

This script writes one JSON object per prompt with:
  - key
  - prompt
  - response

By default it loads `google/IFEval` and generates completions for `train`.

Usage:
  python scripts/generate_ifeval_jsonl.py \
    --config configs/ift_base_model.yaml \
    --checkpoint outputs/final_model \
    --output outputs/ifeval/predictions.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval_mmlu import load_model, load_tokenizer
from memory_transformer.config import load_config

try:
    from accelerate import Accelerator

    ACCELERATE_AVAILABLE = True
except Exception:
    ACCELERATE_AVAILABLE = False


def _apply_chat_template_if_enabled(
    tokenizer,
    prompt: str,
    apply_chat_template: bool,
    system_prompt: Optional[str],
    require_chat_template: bool,
    smollm_eot_chat_rewrite: bool,
) -> Tuple[str, bool]:
    if not apply_chat_template:
        return prompt, False

    if not hasattr(tokenizer, "apply_chat_template"):
        if require_chat_template:
            raise RuntimeError("Tokenizer has no apply_chat_template() but chat template was required")
        return prompt, False

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if smollm_eot_chat_rewrite:
            # Convert chat-control tokens to the training-style EOT token surface.
            # Handle both canonical and typo variants defensively.
            rendered = (
                rendered.replace("<|im_start|>", "<|endoftext|>")
                .replace("<|im_end|>", "<|endoftext|>")
                .replace("<|imstart|>", "<|endoftext|>")
                .replace("<|imend|>", "<|endoftext|>")
            )
        return rendered, True
    except Exception as e:
        if require_chat_template:
            raise RuntimeError(f"Failed to apply chat template: {e}") from e
        return prompt, False


def _select_next_token(
    next_token_logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    # Keep a cleaned copy for fallback argmax decisions.
    fallback_logits = torch.nan_to_num(
        next_token_logits,
        nan=-1e9,
        posinf=1e9,
        neginf=-1e9,
    )

    if (not do_sample) or temperature <= 0:
        return torch.argmax(fallback_logits, dim=-1, keepdim=True)

    logits = next_token_logits / temperature
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

    if top_k > 0:
        k = min(int(top_k), int(logits.shape[-1]))
        topk_threshold = torch.topk(logits, k)[0][..., -1, None]
        remove = logits < topk_threshold
        logits[remove] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = 0
        remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
        logits[remove] = float("-inf")

    # If filtering produced rows with no finite value, fall back to unfiltered logits.
    no_finite = ~torch.isfinite(logits).any(dim=-1, keepdim=True)
    if bool(torch.any(no_finite)):
        logits = torch.where(no_finite, fallback_logits, logits)

    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    # Guarantee valid multinomial rows. If a row is still invalid, convert it
    # to a one-hot distribution at greedy argmax.
    row_sums = probs.sum(dim=-1, keepdim=True)
    invalid = row_sums <= 0
    if bool(torch.any(invalid)):
        greedy_idx = torch.argmax(fallback_logits, dim=-1, keepdim=True)
        one_hot = torch.zeros_like(probs).scatter_(1, greedy_idx, 1.0)
        probs = torch.where(invalid, one_hot, probs)
        row_sums = probs.sum(dim=-1, keepdim=True)

    probs = probs / row_sums.clamp_min(1e-12)
    return torch.multinomial(probs, num_samples=1)


def _prepare_inputs(
    tokenizer,
    text: str,
    device: torch.device,
    max_input_tokens: Optional[int],
    add_special_tokens: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=bool(add_special_tokens),
        truncation=(max_input_tokens is not None),
        max_length=max_input_tokens,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompt_texts: Sequence[str],
    prompt_is_chat_templated: Sequence[bool],
    *,
    device: torch.device,
    max_input_tokens: Optional[int],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    use_cache: bool,
    stop_on_eos: bool,
    ban_special_tokens: bool,
) -> List[str]:
    if len(prompt_texts) == 0:
        return []
    if len(prompt_texts) != len(prompt_is_chat_templated):
        raise ValueError("prompt_texts and prompt_is_chat_templated must have the same length")

    templated = [bool(x) for x in prompt_is_chat_templated]
    if any(t != templated[0] for t in templated):
        raise ValueError(
            "Mixed templated/non-templated prompts in one batch are not supported. "
            "Batch them separately."
        )

    # Left padding keeps each sample's final prompt token aligned at -1.
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            list(prompt_texts),
            return_tensors="pt",
            add_special_tokens=not templated[0],
            padding=True,
            truncation=(max_input_tokens is not None),
            max_length=max_input_tokens,
        )
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    prompt_len = int(input_ids.shape[1])
    past_key_values = None
    batch_size = int(input_ids.shape[0])
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
    eos_id = tokenizer.eos_token_id
    special_ids = sorted(
        {
            int(i)
            for i in getattr(tokenizer, "all_special_ids", [])
            if isinstance(i, int) and 0 <= int(i) < int(tokenizer.vocab_size)
        }
    )
    finished_fill_id = (
        int(eos_id)
        if eos_id is not None
        else int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    )

    for _ in range(int(max_new_tokens)):
        if use_cache and past_key_values is not None:
            model_input = input_ids[:, -1:]
            # Use per-row position IDs from true (unmasked) sequence lengths.
            position_ids = (attention_mask.long().sum(dim=1) - 1).clamp(min=0).unsqueeze(1)
            position_offset = 0
            model_past = past_key_values
        else:
            model_input = input_ids
            position_ids = attention_mask.long().cumsum(dim=1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            position_offset = 0
            model_past = None

        outputs = model(
            input_ids=model_input,
            attention_mask=attention_mask,
            use_cache=bool(use_cache),
            past_key_values=model_past,
            position_offset=position_offset,
            position_ids=position_ids,
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        if use_cache:
            if isinstance(outputs, dict):
                past_key_values = outputs.get("past_key_values")
            else:
                past_key_values = getattr(outputs, "past_key_values", None)

        step_logits = logits[:, -1, :]
        if not bool(torch.all(unfinished)):
            # Stabilize completed rows: keep them on a fixed token and avoid
            # propagating potentially degenerate logits through sampling.
            safe_id = int(finished_fill_id)
            vocab = int(step_logits.shape[-1])
            if safe_id < 0 or safe_id >= vocab:
                safe_id = 0
            step_logits = step_logits.clone()
            step_logits[~unfinished] = float("-inf")
            step_logits[~unfinished, safe_id] = 0.0

        if ban_special_tokens and len(special_ids) > 0 and bool(torch.any(unfinished)):
            # Avoid empty outputs in greedy mode caused by repeatedly selecting
            # special tokens that are then stripped by skip_special_tokens=True.
            step_logits = step_logits.clone()
            row_idx = torch.nonzero(unfinished, as_tuple=False).flatten()
            col_idx = torch.tensor(special_ids, device=step_logits.device, dtype=torch.long)
            step_logits[row_idx.unsqueeze(1), col_idx.unsqueeze(0)] = float("-inf")

        next_token = _select_next_token(
            step_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Keep completed rows stable while other rows continue generating.
        if not bool(torch.all(unfinished)):
            fill = torch.full_like(next_token, int(finished_fill_id))
            next_token = torch.where(unfinished.unsqueeze(1), next_token, fill)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [
                attention_mask,
                unfinished.to(dtype=attention_mask.dtype).unsqueeze(1),
            ],
            dim=-1,
        )

        if stop_on_eos and eos_id is not None:
            just_finished = (next_token.squeeze(1) == int(eos_id)) & unfinished
            unfinished = unfinished & (~just_finished)
            if not bool(torch.any(unfinished)):
                break

    generated_ids = input_ids[:, prompt_len:]
    responses: List[str] = []
    for row in generated_ids:
        responses.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return responses


@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    prompt_text: str,
    prompt_is_chat_templated: bool,
    *,
    device: torch.device,
    max_input_tokens: Optional[int],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    use_cache: bool,
    stop_on_eos: bool,
    ban_special_tokens: bool,
) -> str:
    return generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=[prompt_text],
        prompt_is_chat_templated=[prompt_is_chat_templated],
        device=device,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        use_cache=use_cache,
        stop_on_eos=stop_on_eos,
        ban_special_tokens=ban_special_tokens,
    )[0]


def _infer_model_id(checkpoint: Optional[str], config_path: str) -> str:
    if checkpoint:
        return Path(checkpoint).name
    return Path(config_path).stem


def _rank_output_path(output_path: Path, rank: int) -> Path:
    return output_path.with_name(f"{output_path.name}.rank{rank}.tmp")


def _prefer_tokenizer_default_special_ids(tokenizer, cfg, *, enabled: bool, is_main: bool) -> None:
    """
    Override tokenizer BOS/EOS/PAD ids with tokenizer-native defaults.

    This is important when old training/checkpoint configs pinned special IDs
    to incompatible values (e.g., 0/0/0 for chat tokenizers), which can break
    EOS stopping behavior during generation.
    """
    if not enabled:
        return

    tokenizer_name = cfg.model.tokenizer_name or cfg.model.base_model_name
    if tokenizer_name is None:
        return

    try:
        default_tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception:
        return

    before = (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)

    if default_tok.bos_token_id is not None:
        tokenizer.bos_token_id = int(default_tok.bos_token_id)
    if default_tok.eos_token_id is not None:
        tokenizer.eos_token_id = int(default_tok.eos_token_id)
    if default_tok.pad_token_id is not None:
        tokenizer.pad_token_id = int(default_tok.pad_token_id)
    elif tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = int(tokenizer.eos_token_id)

    after = (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    if is_main and before != after:
        print(
            "Adjusted tokenizer special ids to tokenizer defaults: "
            f"bos/eos/pad {before} -> {after}"
        )


def _auto_fix_suspicious_special_ids(
    tokenizer,
    cfg,
    *,
    enabled: bool,
    prefer_tokenizer_default_special_ids: bool,
    is_main: bool,
) -> None:
    """
    Defensive fix for clearly suspicious special-ID overrides.

    Typical bad case: config forces bos/eos/pad to the same ID (often 0),
    while tokenizer-native IDs are distinct (e.g., SmolLM2: 1/2/2).
    """
    if not enabled:
        return

    tokenizer_name = cfg.model.tokenizer_name or cfg.model.base_model_name
    if tokenizer_name is None:
        return

    try:
        canonical = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception:
        return

    current = (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    canonical_ids = (canonical.bos_token_id, canonical.eos_token_id, canonical.pad_token_id)

    all_equal = (
        current[0] is not None
        and current[1] is not None
        and current[2] is not None
        and current[0] == current[1] == current[2]
    )
    zero_triplet = current == (0, 0, 0)
    canonical_differs = current != canonical_ids and all(x is not None for x in canonical_ids)

    should_fix = (not prefer_tokenizer_default_special_ids) and canonical_differs and (all_equal or zero_triplet)
    if not should_fix:
        if is_main and (not prefer_tokenizer_default_special_ids) and canonical_differs:
            print(
                "Warning: tokenizer special IDs differ from tokenizer defaults: "
                f"current={current}, tokenizer-default={canonical_ids}. "
                "This may degrade generation quality."
            )
        return

    tokenizer.bos_token_id = int(canonical_ids[0])
    tokenizer.eos_token_id = int(canonical_ids[1])
    tokenizer.pad_token_id = int(canonical_ids[2])
    if is_main:
        print(
            "Detected suspicious special-ID override and auto-corrected for generation: "
            f"bos/eos/pad {current} -> {canonical_ids}"
        )


def _load_tokenizer_defaults_ignoring_config_overrides(cfg, *, is_main: bool):
    """
    Load tokenizer directly from HF defaults (ignore config special-ID and chat-template overrides).
    """
    tokenizer_name = cfg.model.tokenizer_name or cfg.model.base_model_name
    if tokenizer_name is None:
        tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token_id = int(tok.eos_token_id)
        else:
            tok.add_special_tokens({"pad_token": "<pad>"})
    if is_main:
        print(
            "Loaded tokenizer defaults (ignoring config overrides): "
            f"name={tokenizer_name}, bos/eos/pad={tok.bos_token_id}/{tok.eos_token_id}/{tok.pad_token_id}"
        )
    return tok


def _load_effective_config(
    config_path: str,
    checkpoint_path: Optional[str],
    *,
    prefer_checkpoint_config: bool,
    is_main: bool,
):
    """
    Resolve config source:
    - explicit config file
    - checkpoint/config.yaml (if present)
    """
    explicit_cfg = load_config(config_path)
    if not checkpoint_path:
        return explicit_cfg

    ckpt_cfg_path = Path(checkpoint_path) / "config.yaml"
    if not ckpt_cfg_path.exists():
        return explicit_cfg

    ckpt_cfg = load_config(ckpt_cfg_path)
    if not prefer_checkpoint_config:
        return explicit_cfg

    if is_main and str(ckpt_cfg_path) != str(Path(config_path)):
        exp_m = explicit_cfg.model
        ck_m = ckpt_cfg.model
        exp_ids = (exp_m.bos_token_id, exp_m.eos_token_id, exp_m.pad_token_id)
        ck_ids = (ck_m.bos_token_id, ck_m.eos_token_id, ck_m.pad_token_id)
        if (
            exp_m.tokenizer_name != ck_m.tokenizer_name
            or exp_ids != ck_ids
            or exp_m.chat_template != ck_m.chat_template
        ):
            print(
                "Using checkpoint config.yaml for generation (detected possible config mismatch).\n"
                f"  explicit config:   {config_path}\n"
                f"  checkpoint config: {ckpt_cfg_path}\n"
                f"  tokenizer_name explicit/checkpoint: {exp_m.tokenizer_name} / {ck_m.tokenizer_name}\n"
                f"  bos/eos/pad explicit/checkpoint:    {exp_ids} / {ck_ids}"
            )
    return ckpt_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IFEval predictions JSONL")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory")
    parser.add_argument(
        "--prefer_checkpoint_config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When checkpoint/config.yaml exists, prefer it over --config for model/tokenizer settings "
            "(default: True)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="google/IFEval",
        help="HF dataset name (default: google/IFEval)",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="Prompt field in dataset rows")
    parser.add_argument("--key_field", type=str, default="key", help="Key/id field in dataset rows")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max number of prompts")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed when max_samples is set")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=None,
        help="Optional hard cap for input prompt tokens",
    )
    parser.add_argument(
        "--apply_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap each prompt in tokenizer chat template when available (default: True)",
    )
    parser.add_argument(
        "--require_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if chat template application is requested but unavailable/fails (default: True)",
    )
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt for chat-template mode")
    parser.add_argument(
        "--use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use KV cache during decoding (default: True)",
    )
    parser.add_argument(
        "--stop_on_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop generation when EOS token is emitted (default: True)",
    )
    parser.add_argument(
        "--ban_special_tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Mask tokenizer special tokens from next-token selection on unfinished rows. "
            "Useful when greedy decode collapses to empty outputs."
        ),
    )
    parser.add_argument("--do_sample", action="store_true", help="Use sampling (default: greedy decode)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 disables)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for batched decoding")
    parser.add_argument(
        "--debug_prompt_preview_rows",
        type=int,
        default=1,
        help="Print rendered prompt/token preview for first N rows on rank 0 (default: 1, 0 disables).",
    )
    parser.add_argument(
        "--smollm_eot_chat_rewrite",
        action="store_true",
        help=(
            "Use tokenizer default chat template (ignoring config chat-template/special-ID overrides), "
            "then rewrite <|im_start|>/<|im_end|> to <|endoftext|> before tokenization."
        ),
    )
    parser.add_argument(
        "--prefer_tokenizer_default_special_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use tokenizer-native BOS/EOS/PAD ids instead of config-overridden ids "
            "(default: True). Helps when old checkpoints used mismatched ids."
        ),
    )
    parser.add_argument(
        "--auto_fix_suspicious_special_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If special IDs are suspicious (e.g., bos=eos=pad=0) and differ from tokenizer defaults, "
            "auto-correct to tokenizer defaults for generation (default: True)."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (e.g., cuda, cuda:0, cpu)")
    parser.add_argument("--distributed", action="store_true", help="Use Accelerate multi-process generation")
    parser.add_argument(
        "--keep_rank_outputs",
        action="store_true",
        help="Keep temporary per-rank JSONL parts after merge (distributed mode only)",
    )
    parser.add_argument("--model_id", type=str, default=None, help="Optional model id to include in JSONL rows")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ifeval/predictions.jsonl",
        help="Output JSONL file",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError(f"--max_new_tokens must be > 0, got {args.max_new_tokens}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"--max_samples must be > 0 when set, got {args.max_samples}")
    if args.temperature < 0:
        raise ValueError(f"--temperature must be >= 0, got {args.temperature}")
    if args.top_k < 0:
        raise ValueError(f"--top_k must be >= 0, got {args.top_k}")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError(f"--top_p must be in (0, 1], got {args.top_p}")
    if args.do_sample and args.temperature <= 0:
        raise ValueError("--do_sample requires --temperature > 0")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be > 0, got {args.batch_size}")
    if args.smollm_eot_chat_rewrite and (not args.apply_chat_template):
        raise ValueError("--smollm_eot_chat_rewrite requires --apply_chat_template")

    accelerator = None
    if args.distributed:
        if not ACCELERATE_AVAILABLE:
            raise RuntimeError("accelerate is required for --distributed")
        accelerator = Accelerator()
        world_size = int(accelerator.num_processes)
        rank = int(accelerator.process_index)
        is_main = bool(accelerator.is_main_process)
        device = accelerator.device
    else:
        world_size = 1
        rank = 0
        is_main = True
        device = torch.device(args.device)

    # Re-resolve config with correct is_main once rank is known.
    cfg = _load_effective_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        prefer_checkpoint_config=bool(args.prefer_checkpoint_config),
        is_main=is_main,
    )
    max_input_tokens = (
        int(args.max_input_tokens) if args.max_input_tokens is not None else int(cfg.training.max_length)
    )

    if is_main:
        print(f"Loading model from {args.checkpoint or args.config} ...")
        if accelerator is not None:
            print(f"Distributed generation enabled: world_size={world_size}")
    model = load_model(cfg, args.checkpoint)
    if args.smollm_eot_chat_rewrite:
        # Explicitly ignore config tokenizer overrides for special IDs/template.
        tokenizer = _load_tokenizer_defaults_ignoring_config_overrides(cfg, is_main=is_main)
        if is_main:
            print(
                "smollm_eot_chat_rewrite enabled: config chat_template/bos/eos/pad overrides are ignored."
            )
    else:
        tokenizer = load_tokenizer(cfg)
        _prefer_tokenizer_default_special_ids(
            tokenizer,
            cfg,
            enabled=bool(args.prefer_tokenizer_default_special_ids),
            is_main=is_main,
        )
        _auto_fix_suspicious_special_ids(
            tokenizer,
            cfg,
            enabled=bool(args.auto_fix_suspicious_special_ids),
            prefer_tokenizer_default_special_ids=bool(args.prefer_tokenizer_default_special_ids),
            is_main=is_main,
        )
    model = model.to(device)
    model.eval()
    if is_main and bool(args.apply_chat_template):
        if getattr(tokenizer, "chat_template", None):
            print("Chat template detected; prompts will be rendered via apply_chat_template.")
        else:
            msg = "Chat template requested, but tokenizer has no chat_template."
            if bool(args.require_chat_template):
                raise RuntimeError(msg)
            print(msg + " Falling back to raw prompts.")

    if is_main:
        print(f"Loading dataset {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples is not None:
        rng = random.Random(int(args.seed))
        k = min(int(args.max_samples), len(ds))
        selected = rng.sample(range(len(ds)), k=k)
        ds = ds.select(selected)

    model_id = args.model_id or _infer_model_id(args.checkpoint, args.config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = len(ds)

    local_output_path = _rank_output_path(out_path, rank) if accelerator is not None else out_path
    local_indices = [i for i in range(total_rows) if i % world_size == rank]

    if is_main:
        print(f"Generating {total_rows} responses -> {out_path}")

    with open(local_output_path, "w", encoding="utf-8") as f:
        iterator = tqdm(
            range(0, len(local_indices), int(args.batch_size)),
            desc=f"IFEval-rank{rank}",
            disable=(accelerator is not None and not accelerator.is_main_process),
        )
        for start in iterator:
            batch_indices = local_indices[start : start + int(args.batch_size)]
            rows = [ds[int(idx)] for idx in batch_indices]
            prompts = [str(row[args.prompt_field]) for row in rows]

            prompt_texts: List[str] = []
            prompt_templated: List[bool] = []
            for prompt in prompts:
                prompt_text, is_templated = _apply_chat_template_if_enabled(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    apply_chat_template=bool(args.apply_chat_template),
                    system_prompt=args.system_prompt,
                    require_chat_template=bool(args.require_chat_template),
                    smollm_eot_chat_rewrite=bool(args.smollm_eot_chat_rewrite),
                )
                prompt_texts.append(prompt_text)
                prompt_templated.append(is_templated)

            if is_main and start == 0 and int(args.debug_prompt_preview_rows) > 0:
                n_preview = min(int(args.debug_prompt_preview_rows), len(prompt_texts))
                for j in range(n_preview):
                    rendered = prompt_texts[j]
                    tok_preview = tokenizer(
                        rendered,
                        add_special_tokens=not bool(prompt_templated[j]),
                        truncation=(max_input_tokens is not None),
                        max_length=max_input_tokens,
                    )["input_ids"]
                    print("-" * 80)
                    print(f"[debug] row={batch_indices[j]} templated={bool(prompt_templated[j])}")
                    print(f"[debug] raw prompt preview: {prompts[j][:240]!r}")
                    print(f"[debug] rendered preview:   {rendered[:320]!r}")
                    print(
                        "[debug] tokenized length="
                        f"{len(tok_preview)}, first_ids={tok_preview[:32]}"
                    )
                print("-" * 80)

            responses = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                prompt_is_chat_templated=prompt_templated,
                device=device,
                max_input_tokens=max_input_tokens,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                use_cache=bool(args.use_cache),
                stop_on_eos=bool(args.stop_on_eos),
                ban_special_tokens=bool(args.ban_special_tokens),
            )

            for idx, row, prompt, response in zip(batch_indices, rows, prompts, responses):
                payload: Dict = {
                    "key": row.get(args.key_field),
                    "prompt": prompt,
                    "response": response,
                    "model_id": model_id,
                }
                if "instruction_id_list" in row:
                    payload["instruction_id_list"] = row["instruction_id_list"]
                if "kwargs" in row:
                    payload["kwargs"] = row["kwargs"]
                if accelerator is not None:
                    payload["__idx"] = int(idx)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if accelerator is not None:
        accelerator.wait_for_everyone()
        if is_main:
            merged = []
            for r in range(world_size):
                rp = _rank_output_path(out_path, r)
                if not rp.exists():
                    raise FileNotFoundError(f"Missing distributed output part: {rp}")
                with open(rp, "r", encoding="utf-8") as part_f:
                    for line in part_f:
                        line = line.strip()
                        if not line:
                            continue
                        merged.append(json.loads(line))
            if len(merged) != total_rows:
                raise RuntimeError(
                    f"Merged row count mismatch: got {len(merged)}, expected {total_rows}"
                )
            merged.sort(key=lambda x: int(x["__idx"]))
            with open(out_path, "w", encoding="utf-8") as out_f:
                for item in merged:
                    item.pop("__idx", None)
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if not args.keep_rank_outputs:
                for r in range(world_size):
                    rp = _rank_output_path(out_path, r)
                    try:
                        os.remove(rp)
                    except OSError:
                        pass
            print(f"Done. Wrote {total_rows} rows to {out_path}")
    else:
        print(f"Done. Wrote {total_rows} rows to {out_path}")


if __name__ == "__main__":
    main()
