#!/usr/bin/env python3
"""
Evaluate TriviaQA with alias-based averaged perplexity.

Design choices (as requested):
- No context passages are used in prompts (question + answer format only).
- For each question, score all available answer aliases (including normalized aliases).
- Per-question metric = average perplexity across all aliases.
- Global metric = average of per-question perplexities.
- Few-shot is enabled by default with 5 examples sampled randomly from a non-test split.
"""

import argparse
import json
import math
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
class TriviaExample:
    question: str
    aliases: List[str]
    display_answer: str


def _normalize_text_key(text: str) -> str:
    """Normalization key for de-duplicating aliases."""
    return " ".join(str(text).strip().lower().split())


def _collect_aliases(answer: Dict) -> List[str]:
    """
    Collect aliases and normalized aliases (+ core value fields), de-duplicated.
    """
    raw_candidates: List[str] = []
    for key in ["aliases", "normalized_aliases"]:
        values = answer.get(key, [])
        if isinstance(values, list):
            raw_candidates.extend([str(v) for v in values])
    for key in [
        "value",
        "normalized_value",
        "matched_wiki_entity_name",
        "normalized_matched_wiki_entity_name",
    ]:
        value = str(answer.get(key, "")).strip()
        if value:
            raw_candidates.append(value)

    deduped: List[str] = []
    seen = set()
    for cand in raw_candidates:
        cleaned = " ".join(str(cand).strip().split())
        if not cleaned:
            continue
        k = _normalize_text_key(cleaned)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(cleaned)
    return deduped


def _select_display_answer(answer: Dict, aliases: List[str]) -> str:
    """
    Choose a canonical answer string for few-shot demonstrations.
    """
    for key in ["value", "normalized_value"]:
        v = str(answer.get(key, "")).strip()
        if v:
            return v
    if aliases:
        return aliases[0]
    return ""


def parse_trivia_row(row: Dict) -> TriviaExample:
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("TriviaQA row missing non-empty 'question'")

    answer = row.get("answer")
    if not isinstance(answer, dict):
        raise ValueError("TriviaQA row missing dict 'answer'")

    aliases = _collect_aliases(answer)
    if not aliases:
        raise ValueError("TriviaQA row has no usable answer aliases")

    display_answer = _select_display_answer(answer, aliases)
    return TriviaExample(question=question, aliases=aliases, display_answer=display_answer)


def load_trivia_examples(
    dataset_name: str,
    dataset_config: str,
    split: str,
    max_samples: Optional[int],
    random_seed: Optional[int] = None,
) -> List[TriviaExample]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples is not None:
        k = min(len(ds), int(max_samples))
        if random_seed is None:
            ds = ds.select(range(k))
        else:
            rng = random.Random(int(random_seed))
            indices = rng.sample(range(len(ds)), k=k)
            ds = ds.select(indices)

    examples: List[TriviaExample] = []
    for row in ds:
        examples.append(parse_trivia_row(row))
    return examples


def _format_qa(question: str, answer: Optional[str]) -> str:
    q = f"Question: {question}\nAnswer:"
    if answer is None:
        return q
    return q + f" {answer}"


def _build_prefix(fewshot_examples: List[TriviaExample]) -> str:
    lines = [
        "Answer the following trivia questions.",
        "",
    ]
    if fewshot_examples:
        for ex in fewshot_examples:
            lines.append(_format_qa(ex.question, ex.display_answer))
            lines.append("")
    return "\n".join(lines).strip() + "\n\n"


def evaluate_triviaqa(
    model,
    tokenizer,
    eval_examples: List[TriviaExample],
    fewshot_examples: List[TriviaExample],
    max_length: int,
    accelerator: Optional["Accelerator"],
    use_cache: bool,
) -> Dict[str, float]:
    prefix = _build_prefix(fewshot_examples)
    world_size = accelerator.num_processes if accelerator is not None else 1
    rank = accelerator.process_index if accelerator is not None else 0

    indexed = list(enumerate(eval_examples))
    if world_size > 1:
        indexed = [(i, ex) for i, ex in indexed if i % world_size == rank]

    iterator = tqdm(
        indexed,
        desc="triviaqa",
        leave=False,
        disable=(accelerator is not None and not accelerator.is_main_process),
    )

    device = accelerator.device if accelerator is not None else next(model.parameters()).device
    local_sum_question_ppl = 0.0
    local_questions = 0
    local_sum_aliases = 0.0

    for _, ex in iterator:
        prompt = prefix + _format_qa(ex.question, answer=None)
        scores = score_options(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            option_labels=None,
            option_texts=ex.aliases,
            scoring_mode="choice_text",
            device=device,
            max_length=max_length,
            use_cache=use_cache,
        )
        # score_options returns mean log-probability per continuation token.
        # Convert each alias score to perplexity via ppl = exp(avg_nll) = exp(-avg_logprob).
        alias_ppls = [math.exp(-float(s)) for s in scores]
        question_avg_ppl = sum(alias_ppls) / max(len(alias_ppls), 1)

        local_sum_question_ppl += question_avg_ppl
        local_questions += 1
        local_sum_aliases += float(len(alias_ppls))

    if accelerator is not None:
        sums = torch.tensor(
            [local_sum_question_ppl, float(local_questions), local_sum_aliases],
            device=accelerator.device,
            dtype=torch.float64,
        )
        sums = accelerator.reduce(sums, reduction="sum")
        global_sum_question_ppl = float(sums[0].item())
        global_questions = int(round(float(sums[1].item())))
        global_sum_aliases = float(sums[2].item())
    else:
        global_sum_question_ppl = local_sum_question_ppl
        global_questions = local_questions
        global_sum_aliases = local_sum_aliases

    avg_per_question_ppl = global_sum_question_ppl / max(global_questions, 1)
    avg_aliases_per_question = global_sum_aliases / max(global_questions, 1)

    return {
        "avg_per_question_ppl": float(avg_per_question_ppl),
        "num_questions": int(global_questions),
        "avg_aliases_per_question": float(avg_aliases_per_question),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TriviaQA alias-averaged perplexity")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--dataset", type=str, default="mandarjoshi/trivia_qa", help="HF dataset name")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="rc.nocontext",
        help="HF dataset config (default: rc.nocontext to avoid context usage)",
    )
    parser.add_argument("--split", type=str, default="validation", help="Eval split")
    parser.add_argument("--fewshot_split", type=str, default="train", help="Few-shot source split")
    parser.add_argument("--shots", type=int, default=5, help="Few-shot example count (0 for zero-shot)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max eval samples")
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
        help="Pass use_cache=True to model forward during alias scoring (default: False)",
    )
    args = parser.parse_args()

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
            print(f"Running distributed TriviaQA eval on {accelerator.num_processes} processes")

    if accelerator is None or accelerator.is_main_process:
        print(f"Loading model from {args.checkpoint or 'config'}...")
    model = load_model(cfg, args.checkpoint)
    tokenizer = load_tokenizer(cfg)

    if accelerator is not None:
        model = accelerator.prepare(model)
    else:
        model = model.to(args.device)
    model.eval()

    if accelerator is None or accelerator.is_main_process:
        print(
            f"Loading TriviaQA dataset={args.dataset}/{args.dataset_config} "
            f"split={args.split}, fewshot_split={args.fewshot_split}..."
        )

    eval_examples = load_trivia_examples(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
        random_seed=None,  # eval ordering remains deterministic head-of-split unless overridden
    )

    shots = max(int(args.shots), 0)
    fewshot_examples: List[TriviaExample] = []
    if shots > 0:
        fewshot_examples = load_trivia_examples(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.fewshot_split,
            max_samples=shots,
            random_seed=int(args.seed),
        )

    result = evaluate_triviaqa(
        model=model,
        tokenizer=tokenizer,
        eval_examples=eval_examples,
        fewshot_examples=fewshot_examples,
        max_length=max_length,
        accelerator=accelerator,
        use_cache=bool(args.use_cache),
    )

    is_main = True if accelerator is None else accelerator.is_main_process
    if not is_main:
        return

    print("\n" + "=" * 72)
    print("TriviaQA Alias-Averaged Perplexity")
    print("=" * 72)
    print(f"Dataset: {args.dataset} / {args.dataset_config}")
    print(f"Split: {args.split}")
    print(f"Questions: {result['num_questions']}")
    print(f"Shots: {shots} (from split={args.fewshot_split})")
    print(f"Average aliases/question: {result['avg_aliases_per_question']:.2f}")
    print(f"Average per-question perplexity: {result['avg_per_question_ppl']:.4f}")
    print("=" * 72)

    if args.output:
        payload = {
            "benchmark": "triviaqa",
            "dataset_name": args.dataset,
            "dataset_config": args.dataset_config,
            "eval_split": args.split,
            "fewshot_split": args.fewshot_split,
            "shots": shots,
            "max_length": max_length,
            "avg_per_question_ppl": result["avg_per_question_ppl"],
            "num_questions": result["num_questions"],
            "avg_aliases_per_question": result["avg_aliases_per_question"],
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
