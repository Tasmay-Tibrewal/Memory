#!/usr/bin/env python3
"""
Evaluate checkpoints on MMLU-style multiple-choice datasets.

This script computes accuracy (not perplexity) by scoring answer options with
 causal-LM log likelihood.

Default dataset:
    cais/mmlu

Usage:
    python scripts/eval_mmlu.py --config configs/base_small.yaml --checkpoint outputs/final_model
    accelerate launch --num_processes 8 scripts/eval_mmlu.py --distributed --config configs/base_small.yaml --checkpoint outputs/final_model
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_transformer.adapter import MemoryAdapter
from memory_transformer.config import load_config
from memory_transformer.model import MemoryTransformer
from memory_transformer.utils import configure_tokenizer_special_ids

try:
    from accelerate import Accelerator

    ACCELERATE_AVAILABLE = True
except Exception:
    ACCELERATE_AVAILABLE = False


@dataclass
class MCExample:
    question: str
    choices: List[str]
    answer_index: int
    subject: str


def load_model(config, checkpoint_path: Optional[str] = None):
    if config.model.base_model_name is not None:
        model = MemoryAdapter(config)
    else:
        model = MemoryTransformer(config)

    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path)
        model_path_pt = checkpoint_dir / "model.pt"
        if model_path_pt.exists():
            state_dict = torch.load(model_path_pt, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        else:
            safe_path = checkpoint_dir / "model.safetensors"
            bin_path = checkpoint_dir / "pytorch_model.bin"
            if safe_path.exists():
                from safetensors.torch import load_file

                state_dict = load_file(str(safe_path), device="cpu")
                model.load_state_dict(state_dict)
            elif bin_path.exists():
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(
                    f"No supported model weights found in {checkpoint_dir} "
                    f"(expected {model_path_pt.name}, {safe_path.name}, or {bin_path.name})."
                )
    return model


def load_tokenizer(config):
    tokenizer_name = config.model.tokenizer_name or config.model.base_model_name
    if tokenizer_name is None:
        tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    configure_tokenizer_special_ids(tokenizer, config.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def normalize_answer_index(answer, num_choices: int) -> int:
    if isinstance(answer, bool):
        answer = int(answer)
    if isinstance(answer, int):
        idx = int(answer)
        if 0 <= idx < num_choices:
            return idx
        raise ValueError(f"Integer answer index out of range: {idx} for {num_choices} choices")
    if isinstance(answer, str):
        s = answer.strip()
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < num_choices:
                return idx
        if len(s) == 1 and s.upper().isalpha():
            idx = ord(s.upper()) - ord("A")
            if 0 <= idx < num_choices:
                return idx
    raise ValueError(f"Unsupported answer format: {answer!r}")


def parse_example(raw: Dict, fallback_subject: str) -> MCExample:
    if "question" not in raw:
        raise ValueError(f"Missing 'question' in example keys: {sorted(raw.keys())}")
    question = str(raw["question"]).strip()

    if "choices" in raw:
        choices_raw = raw["choices"]
        if not isinstance(choices_raw, Sequence):
            raise ValueError("'choices' exists but is not a sequence")
        choices = [str(c).strip() for c in choices_raw]
    else:
        letter_keys = [k for k in ("A", "B", "C", "D", "E", "F") if k in raw]
        if len(letter_keys) < 2:
            raise ValueError(f"Could not infer choices from keys: {sorted(raw.keys())}")
        choices = [str(raw[k]).strip() for k in letter_keys]

    if len(choices) < 2:
        raise ValueError(f"Need at least 2 choices, got {len(choices)}")

    answer_key = None
    for key in ("answer", "label", "target"):
        if key in raw:
            answer_key = key
            break
    if answer_key is None:
        raise ValueError(f"Missing answer key in example keys: {sorted(raw.keys())}")

    answer_index = normalize_answer_index(raw[answer_key], len(choices))
    subject = str(raw.get("subject", fallback_subject))
    return MCExample(
        question=question,
        choices=choices,
        answer_index=answer_index,
        subject=subject,
    )


def format_mc_example(example: MCExample, include_answer: bool) -> str:
    lines = [f"Question: {example.question}"]
    for i, choice in enumerate(example.choices):
        label = chr(ord("A") + i)
        lines.append(f"{label}. {choice}")
    if include_answer:
        label = chr(ord("A") + example.answer_index)
        lines.append(f"Answer: {label}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def build_subject_prefix(subject: str, fewshot: List[MCExample]) -> str:
    subject_name = subject.replace("_", " ")
    header = f"The following are multiple choice questions (with answers) about {subject_name}.\n\n"
    if not fewshot:
        return header
    body = "\n\n".join(format_mc_example(ex, include_answer=True) for ex in fewshot)
    return header + body + "\n\n"


def _build_scoring_tensors(
    tokenizer,
    prompt: str,
    continuations: List[str],
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        bos = tokenizer.bos_token_id
        prompt_ids = [bos] if bos is not None else [tokenizer.eos_token_id]

    input_rows: List[List[int]] = []
    label_rows: List[List[int]] = []

    for cont in continuations:
        cont_ids = tokenizer.encode(cont, add_special_tokens=False)
        if not cont_ids:
            raise ValueError(f"Continuation produced empty tokenization: {cont!r}")

        max_prompt_tokens = max_length - len(cont_ids)
        if max_prompt_tokens < 1:
            raise ValueError(
                f"max_length={max_length} too small for continuation length={len(cont_ids)}"
            )
        used_prompt = prompt_ids[-max_prompt_tokens:]

        ids = used_prompt + cont_ids
        labels = ([-100] * len(used_prompt)) + cont_ids
        input_rows.append(ids)
        label_rows.append(labels)

    pad_id = tokenizer.pad_token_id
    max_seq = max(len(x) for x in input_rows)
    padded_inputs: List[List[int]] = []
    padded_labels: List[List[int]] = []
    for ids, labels in zip(input_rows, label_rows):
        pad = max_seq - len(ids)
        padded_inputs.append(ids + ([pad_id] * pad))
        padded_labels.append(labels + ([-100] * pad))

    input_ids = torch.tensor(padded_inputs, dtype=torch.long)
    labels = torch.tensor(padded_labels, dtype=torch.long)
    return input_ids, labels


@torch.no_grad()
def score_options(
    model,
    tokenizer,
    prompt: str,
    option_labels: List[str],
    device: torch.device,
    max_length: int,
    use_cache: bool = False,
) -> List[float]:
    continuations = [f" {label}" for label in option_labels]
    input_ids, labels = _build_scoring_tensors(
        tokenizer=tokenizer,
        prompt=prompt,
        continuations=continuations,
        max_length=max_length,
    )

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=bool(use_cache),
    )
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    per_token_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    token_mask = (shift_labels != -100).float()
    nll_sum = (per_token_nll * token_mask).sum(dim=1)
    n_tokens = token_mask.sum(dim=1).clamp_min(1.0)
    avg_nll = nll_sum / n_tokens
    return (-avg_nll).detach().cpu().tolist()


def list_subjects(dataset_name: str, subject_arg: Optional[str]) -> List[str]:
    if subject_arg:
        return [s.strip() for s in subject_arg.split(",") if s.strip()]

    cfgs = get_dataset_config_names(dataset_name)
    # For cais/mmlu, exclude aggregate/non-standard configs by default.
    blocked = {"all", "auxiliary_train"}
    subjects = [c for c in cfgs if c not in blocked]
    if subjects:
        return subjects
    return cfgs


def load_subject_split(
    dataset_name: str,
    subject: str,
    split: str,
) -> List[MCExample]:
    ds = load_dataset(dataset_name, subject, split=split)
    out: List[MCExample] = []
    for row in ds:
        out.append(parse_example(row, fallback_subject=subject))
    return out


def evaluate_subject(
    model,
    tokenizer,
    dataset_name: str,
    subject: str,
    split: str,
    dev_split: str,
    shots: int,
    max_length: int,
    max_samples: Optional[int],
    accelerator: Optional["Accelerator"],
    use_cache: bool,
) -> Dict[str, float]:
    eval_examples = load_subject_split(dataset_name, subject, split)
    if max_samples is not None:
        eval_examples = eval_examples[:max_samples]

    fewshot: List[MCExample] = []
    if shots > 0:
        dev_examples = load_subject_split(dataset_name, subject, dev_split)
        fewshot = dev_examples[: min(shots, len(dev_examples))]

    prefix = build_subject_prefix(subject, fewshot)
    world_size = accelerator.num_processes if accelerator is not None else 1
    rank = accelerator.process_index if accelerator is not None else 0

    local_correct = 0
    local_total = 0
    rows = list(enumerate(eval_examples))
    if accelerator is not None and world_size > 1:
        rows = [(i, ex) for i, ex in rows if i % world_size == rank]

    iterator = tqdm(
        rows,
        desc=f"{subject}",
        leave=False,
        disable=(accelerator is not None and not accelerator.is_main_process),
    )
    for _, ex in iterator:
        prompt = prefix + format_mc_example(ex, include_answer=False)
        labels = [chr(ord("A") + i) for i in range(len(ex.choices))]
        scores = score_options(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            option_labels=labels,
            device=(accelerator.device if accelerator is not None else next(model.parameters()).device),
            max_length=max_length,
            use_cache=use_cache,
        )
        pred_idx = int(max(range(len(scores)), key=lambda j: scores[j]))
        local_correct += int(pred_idx == ex.answer_index)
        local_total += 1

    if accelerator is not None:
        sums = torch.tensor([local_correct, local_total], device=accelerator.device, dtype=torch.long)
        sums = accelerator.reduce(sums, reduction="sum")
        correct = int(sums[0].item())
        total = int(sums[1].item())
    else:
        correct = local_correct
        total = local_total

    acc = float(correct) / float(max(total, 1))
    return {
        "subject": subject,
        "accuracy": acc,
        "correct": correct,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model accuracy on MMLU-style MCQ tasks")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--dataset", type=str, default="cais/mmlu", help="HF dataset name")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects")
    parser.add_argument("--split", type=str, default="test", help="Eval split")
    parser.add_argument("--dev_split", type=str, default="dev", help="Few-shot source split")
    parser.add_argument("--shots", type=int, default=5, help="Few-shot examples per subject")
    parser.add_argument("--max_samples_per_subject", type=int, default=None, help="Cap eval samples per subject")
    parser.add_argument("--max_subjects", type=int, default=None, help="Evaluate only first N subjects")
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Max sequence length for scoring (default: training.max_length)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device in non-distributed mode")
    parser.add_argument("--distributed", action="store_true", help="Use Accelerate distributed eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Pass use_cache=True to model forward during option scoring (default: False)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    max_length = int(cfg.training.max_length if args.max_length is None else args.max_length)
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")

    use_distributed = bool(args.distributed)
    if use_distributed and not ACCELERATE_AVAILABLE:
        raise RuntimeError("accelerate is required for --distributed")

    accelerator: Optional[Accelerator] = None
    if use_distributed:
        accelerator = Accelerator()
        if accelerator.is_main_process:
            print(f"Running distributed MMLU eval on {accelerator.num_processes} processes")

    if accelerator is None or accelerator.is_main_process:
        print(f"Loading model from {args.checkpoint or 'config'}...")
    model = load_model(cfg, args.checkpoint)
    tokenizer = load_tokenizer(cfg)

    if accelerator is not None:
        model = accelerator.prepare(model)
        device = accelerator.device
    else:
        device = torch.device(args.device)
        model = model.to(device)
    model.eval()

    subjects = list_subjects(args.dataset, args.subjects)
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
    if not subjects:
        raise ValueError("No subjects found to evaluate")

    if accelerator is None or accelerator.is_main_process:
        print(f"Dataset: {args.dataset}")
        print(f"Subjects: {len(subjects)}")
        print(f"Split: {args.split}")
        print(f"Shots: {args.shots}")
        print(f"Max length: {max_length}")

    results: List[Dict[str, float]] = []
    subject_iter = tqdm(
        subjects,
        desc="Subjects",
        disable=(accelerator is not None and not accelerator.is_main_process),
    )
    for subject in subject_iter:
        r = evaluate_subject(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            subject=subject,
            split=args.split,
            dev_split=args.dev_split,
            shots=max(int(args.shots), 0),
            max_length=max_length,
            max_samples=args.max_samples_per_subject,
            accelerator=accelerator,
            use_cache=bool(args.use_cache),
        )
        results.append(r)
        if accelerator is None or accelerator.is_main_process:
            subject_iter.set_postfix({"acc": f"{r['accuracy']:.3f}"})

    if accelerator is not None:
        accelerator.wait_for_everyone()

    # All ranks have identical reduced metrics; print/save on main only.
    is_main = True if accelerator is None else accelerator.is_main_process
    if not is_main:
        return

    total_correct = sum(int(r["correct"]) for r in results)
    total_count = sum(int(r["total"]) for r in results)
    weighted_acc = float(total_correct) / float(max(total_count, 1))
    macro_acc = sum(float(r["accuracy"]) for r in results) / float(len(results))

    print("\n" + "=" * 72)
    print("MMLU Evaluation Results")
    print("=" * 72)
    print(f"Weighted accuracy: {weighted_acc:.4f} ({total_correct}/{total_count})")
    print(f"Macro accuracy:    {macro_acc:.4f}")
    print("-" * 72)
    for r in results:
        print(f"{r['subject']:<36} acc={r['accuracy']:.4f}  n={int(r['total'])}")
    print("=" * 72)

    if args.output:
        payload = {
            "dataset": args.dataset,
            "split": args.split,
            "dev_split": args.dev_split,
            "shots": int(args.shots),
            "max_length": max_length,
            "weighted_accuracy": weighted_acc,
            "macro_accuracy": macro_acc,
            "total_correct": total_correct,
            "total_count": total_count,
            "subjects": results,
            "config": args.config,
            "checkpoint": args.checkpoint,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
