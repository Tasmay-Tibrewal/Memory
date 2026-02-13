#!/usr/bin/env python3
"""
Evaluate common pretraining MCQ benchmarks with option-label scoring.

Supported benchmarks:
- hellaswag
- arc_challenge
- arc_easy
- openbookqa
- winogrande
- boolq

This script measures multiple-choice accuracy by scoring:
    prompt + " A" / " B" / ...
and choosing the highest log-likelihood option.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval_mmlu import load_model, load_tokenizer, score_options
from memory_transformer.config import load_config

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


@dataclass
class BenchmarkSpec:
    dataset_name: str
    subset: Optional[str]
    eval_split: str
    fewshot_split: str
    instruction: str


SPECS: Dict[str, BenchmarkSpec] = {
    "hellaswag": BenchmarkSpec(
        dataset_name="Rowan/hellaswag",
        subset=None,
        eval_split="validation",
        fewshot_split="train",
        instruction="Choose the most plausible continuation.",
    ),
    "arc_challenge": BenchmarkSpec(
        dataset_name="ai2_arc",
        subset="ARC-Challenge",
        eval_split="validation",
        fewshot_split="train",
        instruction="Choose the correct answer.",
    ),
    "arc_easy": BenchmarkSpec(
        dataset_name="ai2_arc",
        subset="ARC-Easy",
        eval_split="validation",
        fewshot_split="train",
        instruction="Choose the correct answer.",
    ),
    "openbookqa": BenchmarkSpec(
        dataset_name="openbookqa",
        subset="main",
        eval_split="validation",
        fewshot_split="train",
        instruction="Choose the correct answer.",
    ),
    "winogrande": BenchmarkSpec(
        dataset_name="winogrande",
        subset="winogrande_xl",
        eval_split="validation",
        fewshot_split="train",
        instruction="Choose the option that best fills the blank.",
    ),
    "boolq": BenchmarkSpec(
        dataset_name="google/boolq",
        subset=None,
        eval_split="validation",
        fewshot_split="train",
        instruction="Answer the question with Yes or No.",
    ),
}


def _letter_to_index(value: str, n: int) -> int:
    s = str(value).strip()
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < n:
            return idx
        if 1 <= idx <= n:
            return idx - 1
    if len(s) == 1 and s.upper().isalpha():
        idx = ord(s.upper()) - ord("A")
        if 0 <= idx < n:
            return idx
    raise ValueError(f"Cannot parse answer index from {value!r} with n={n}")


def parse_row(benchmark: str, row: Dict) -> MCExample:
    b = benchmark.lower()
    if b == "hellaswag":
        question = f"{str(row['ctx']).strip()}\nWhat is the most plausible next sentence?"
        choices = [str(x).strip() for x in row["endings"]]
        answer = _letter_to_index(row["label"], len(choices))
        return MCExample(question=question, choices=choices, answer_index=answer)

    if b in {"arc_challenge", "arc_easy"}:
        question = str(row["question"]).strip()
        choices = [str(x).strip() for x in row["choices"]["text"]]
        answer = _letter_to_index(row["answerKey"], len(choices))
        return MCExample(question=question, choices=choices, answer_index=answer)

    if b == "openbookqa":
        question = str(row["question_stem"]).strip()
        choices = [str(x).strip() for x in row["choices"]["text"]]
        answer = _letter_to_index(row["answerKey"], len(choices))
        return MCExample(question=question, choices=choices, answer_index=answer)

    if b == "winogrande":
        sentence = str(row["sentence"]).strip()
        question = f"Complete the sentence:\n{sentence}"
        choices = [str(row["option1"]).strip(), str(row["option2"]).strip()]
        answer = _letter_to_index(row["answer"], len(choices))
        return MCExample(question=question, choices=choices, answer_index=answer)

    if b == "boolq":
        passage = str(row["passage"]).strip()
        question = str(row["question"]).strip()
        full_q = f"Passage: {passage}\nQuestion: {question}"
        choices = ["Yes", "No"]
        answer = 0 if bool(row["answer"]) else 1
        return MCExample(question=full_q, choices=choices, answer_index=answer)

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def format_example(ex: MCExample, include_answer: bool) -> str:
    lines = [f"Question: {ex.question}"]
    for i, choice in enumerate(ex.choices):
        label = chr(ord("A") + i)
        lines.append(f"{label}. {choice}")
    if include_answer:
        lines.append(f"Answer: {chr(ord('A') + ex.answer_index)}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def load_examples(
    benchmark: str,
    split: str,
    max_samples: Optional[int],
) -> List[MCExample]:
    spec = SPECS[benchmark]
    ds = load_dataset(spec.dataset_name, spec.subset, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))
    out: List[MCExample] = []
    for row in ds:
        out.append(parse_row(benchmark, row))
    return out


def evaluate(
    benchmark: str,
    model,
    tokenizer,
    shots: int,
    split: Optional[str],
    fewshot_split: Optional[str],
    max_samples: Optional[int],
    max_length: int,
    accelerator: Optional["Accelerator"],
) -> Dict[str, float]:
    spec = SPECS[benchmark]
    eval_split = split or spec.eval_split
    fs_split = fewshot_split or spec.fewshot_split

    eval_examples = load_examples(benchmark, eval_split, max_samples=max_samples)
    fewshot_examples = []
    if shots > 0:
        fewshot_examples = load_examples(benchmark, fs_split, max_samples=shots)

    prefix_parts = [
        "The following are multiple-choice questions.",
        spec.instruction,
        "",
    ]
    if fewshot_examples:
        prefix_parts.append("\n\n".join(format_example(ex, include_answer=True) for ex in fewshot_examples))
        prefix_parts.append("")
    prefix = "\n".join(prefix_parts).strip() + "\n\n"

    world_size = accelerator.num_processes if accelerator is not None else 1
    rank = accelerator.process_index if accelerator is not None else 0

    indexed = list(enumerate(eval_examples))
    if world_size > 1:
        indexed = [(i, ex) for i, ex in indexed if i % world_size == rank]

    iterator = tqdm(
        indexed,
        desc=benchmark,
        leave=False,
        disable=(accelerator is not None and not accelerator.is_main_process),
    )

    device = accelerator.device if accelerator is not None else next(model.parameters()).device
    local_correct = 0
    local_total = 0
    for _, ex in iterator:
        prompt = prefix + format_example(ex, include_answer=False)
        labels = [chr(ord("A") + i) for i in range(len(ex.choices))]
        scores = score_options(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            option_labels=labels,
            device=device,
            max_length=max_length,
        )
        pred = max(range(len(scores)), key=lambda j: scores[j])
        local_correct += int(pred == ex.answer_index)
        local_total += 1

    if accelerator is not None:
        sums = torch.tensor([local_correct, local_total], device=accelerator.device, dtype=torch.long)
        sums = accelerator.reduce(sums, reduction="sum")
        correct = int(sums[0].item())
        total = int(sums[1].item())
    else:
        correct, total = local_correct, local_total

    accuracy = float(correct) / float(max(total, 1))
    return {
        "benchmark": benchmark,
        "dataset_name": spec.dataset_name,
        "subset": spec.subset,
        "eval_split": eval_split,
        "fewshot_split": fs_split,
        "shots": int(shots),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def build_parser(default_benchmark: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one MCQ benchmark")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=default_benchmark,
        choices=sorted(SPECS.keys()),
        required=(default_benchmark is None),
        help="Benchmark name",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--shots", type=int, default=0, help="Few-shot examples from fewshot split")
    parser.add_argument("--split", type=str, default=None, help="Eval split override")
    parser.add_argument("--fewshot_split", type=str, default=None, help="Few-shot split override")
    parser.add_argument("--max_samples", type=int, default=None, help="Max eval samples")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length for scoring")
    parser.add_argument("--device", type=str, default="cuda", help="Device for non-distributed mode")
    parser.add_argument("--distributed", action="store_true", help="Use Accelerate distributed evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    return parser


def main(argv: Optional[List[str]] = None, default_benchmark: Optional[str] = None) -> None:
    parser = build_parser(default_benchmark=default_benchmark)
    args = parser.parse_args(argv)
    benchmark = str(args.benchmark).lower()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    max_length = int(cfg.training.max_length if args.max_length is None else args.max_length)
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")

    accelerator: Optional[Accelerator] = None
    if args.distributed:
        if not ACCELERATE_AVAILABLE:
            raise RuntimeError("accelerate is required for --distributed")
        accelerator = Accelerator()
        if accelerator.is_main_process:
            print(f"Running distributed eval on {accelerator.num_processes} processes")

    if accelerator is None or accelerator.is_main_process:
        print(f"Loading model from {args.checkpoint or 'config'}...")
    model = load_model(cfg, args.checkpoint)
    tokenizer = load_tokenizer(cfg)

    if accelerator is not None:
        model = accelerator.prepare(model)
    else:
        model = model.to(args.device)
    model.eval()

    result = evaluate(
        benchmark=benchmark,
        model=model,
        tokenizer=tokenizer,
        shots=max(int(args.shots), 0),
        split=args.split,
        fewshot_split=args.fewshot_split,
        max_samples=args.max_samples,
        max_length=max_length,
        accelerator=accelerator,
    )

    is_main = True if accelerator is None else accelerator.is_main_process
    if not is_main:
        return

    print("\n" + "=" * 64)
    print(f"{benchmark} results")
    print("=" * 64)
    print(f"Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
    print(
        f"Dataset: {result['dataset_name']}"
        + (f" / {result['subset']}" if result["subset"] is not None else "")
    )
    print(f"Split: {result['eval_split']} | Shots: {result['shots']}")
    print("=" * 64)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
