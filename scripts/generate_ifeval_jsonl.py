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
from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

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
) -> str:
    if not apply_chat_template:
        return prompt

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _select_next_token(
    next_token_logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if (not do_sample) or temperature <= 0:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    logits = next_token_logits / temperature

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

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _prepare_inputs(
    tokenizer,
    text: str,
    device: torch.device,
    max_input_tokens: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
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
def generate_completion(
    model,
    tokenizer,
    prompt_text: str,
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
) -> str:
    input_ids, attention_mask = _prepare_inputs(
        tokenizer=tokenizer,
        text=prompt_text,
        device=device,
        max_input_tokens=max_input_tokens,
    )
    prompt_len = int(input_ids.shape[1])
    past_key_values = None

    for _ in range(int(max_new_tokens)):
        if use_cache and past_key_values is not None:
            model_input = input_ids[:, -1:]
            position_offset = input_ids.shape[1] - 1
            model_past = past_key_values
        else:
            model_input = input_ids
            position_offset = 0
            model_past = None

        outputs = model(
            input_ids=model_input,
            attention_mask=attention_mask,
            use_cache=bool(use_cache),
            past_key_values=model_past,
            position_offset=position_offset,
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        if use_cache:
            if isinstance(outputs, dict):
                past_key_values = outputs.get("past_key_values")
            else:
                past_key_values = getattr(outputs, "past_key_values", None)

        next_token = _select_next_token(
            logits[:, -1, :],
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=-1,
        )

        if stop_on_eos and tokenizer.eos_token_id is not None:
            if int(next_token.item()) == int(tokenizer.eos_token_id):
                break

    generated_ids = input_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _infer_model_id(checkpoint: Optional[str], config_path: str) -> str:
    if checkpoint:
        return Path(checkpoint).name
    return Path(config_path).stem


def _rank_output_path(output_path: Path, rank: int) -> Path:
    return output_path.with_name(f"{output_path.name}.rank{rank}.tmp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IFEval predictions JSONL")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory")
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
    parser.add_argument("--do_sample", action="store_true", help="Use sampling (default: greedy decode)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 disables)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
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

    cfg = load_config(args.config)
    max_input_tokens = (
        int(args.max_input_tokens) if args.max_input_tokens is not None else int(cfg.training.max_length)
    )

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

    if is_main:
        print(f"Loading model from {args.checkpoint or args.config} ...")
        if accelerator is not None:
            print(f"Distributed generation enabled: world_size={world_size}")
    model = load_model(cfg, args.checkpoint)
    tokenizer = load_tokenizer(cfg)
    model = model.to(device)
    model.eval()

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
            local_indices,
            desc=f"IFEval-rank{rank}",
            disable=(accelerator is not None and not accelerator.is_main_process),
        )
        for idx in iterator:
            row = ds[int(idx)]
            prompt = str(row[args.prompt_field])
            prompt_text = _apply_chat_template_if_enabled(
                tokenizer=tokenizer,
                prompt=prompt,
                apply_chat_template=bool(args.apply_chat_template),
                system_prompt=args.system_prompt,
            )
            response = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                device=device,
                max_input_tokens=max_input_tokens,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                use_cache=bool(args.use_cache),
                stop_on_eos=bool(args.stop_on_eos),
            )

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
