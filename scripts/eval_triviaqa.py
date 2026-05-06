#!/usr/bin/env python3
"""
Evaluate TriviaQA with top-alias perplexity.

Design choices (as requested):
- No context passages are used in prompts (question + answer format only).
- For each question, score all available answer aliases (including normalized aliases).
- Select the alias with highest full-sequence probability.
- Per-question metric = perplexity of that selected alias.
- Global primary metric = average per-question top-alias perplexity.
- Also report token-weighted corpus perplexity over selected top aliases.
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
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval_mmlu import _build_scoring_tensors, load_model, load_tokenizer
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


@torch.no_grad()
def _score_aliases_detailed(
    model,
    tokenizer,
    prompt: str,
    aliases: List[str],
    device: torch.device,
    max_length: int,
    use_cache: bool,
) -> List[Dict[str, float]]:
    continuations = [f" {str(txt).strip()}" for txt in aliases]
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

    out: List[Dict[str, float]] = []
    for i in range(len(aliases)):
        out.append(
            {
                "nll_sum": float(nll_sum[i].item()),
                "n_tokens": float(n_tokens[i].item()),
                "avg_nll": float(avg_nll[i].item()),
                "seq_logprob": float((-nll_sum[i]).item()),
                "avg_logprob": float((-avg_nll[i]).item()),
            }
        )
    return out


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
    local_sum_top_alias_ppl = 0.0
    local_questions = 0
    local_sum_aliases = 0.0
    local_top_alias_nll_sum = 0.0
    local_top_alias_tokens = 0.0

    for _, ex in iterator:
        prompt = prefix + _format_qa(ex.question, answer=None)
        alias_stats = _score_aliases_detailed(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            aliases=ex.aliases,
            device=device,
            max_length=max_length,
            use_cache=use_cache,
        )
        # Select the alias with highest full-sequence probability (max sequence log-probability).
        best_idx = max(range(len(alias_stats)), key=lambda j: alias_stats[j]["seq_logprob"])
        best = alias_stats[best_idx]
        top_alias_ppl = math.exp(best["avg_nll"])

        local_sum_top_alias_ppl += top_alias_ppl
        local_questions += 1
        local_sum_aliases += float(len(alias_stats))
        local_top_alias_nll_sum += float(best["nll_sum"])
        local_top_alias_tokens += float(best["n_tokens"])

    if accelerator is not None:
        sums = torch.tensor(
            [
                local_sum_top_alias_ppl,
                float(local_questions),
                local_sum_aliases,
                local_top_alias_nll_sum,
                local_top_alias_tokens,
            ],
            device=accelerator.device,
            dtype=torch.float64,
        )
        sums = accelerator.reduce(sums, reduction="sum")
        global_sum_top_alias_ppl = float(sums[0].item())
        global_questions = int(round(float(sums[1].item())))
        global_sum_aliases = float(sums[2].item())
        global_top_alias_nll_sum = float(sums[3].item())
        global_top_alias_tokens = float(sums[4].item())
    else:
        global_sum_top_alias_ppl = local_sum_top_alias_ppl
        global_questions = local_questions
        global_sum_aliases = local_sum_aliases
        global_top_alias_nll_sum = local_top_alias_nll_sum
        global_top_alias_tokens = local_top_alias_tokens

    avg_top_alias_ppl = global_sum_top_alias_ppl / max(global_questions, 1)
    corpus_ppl_top_alias = math.exp(global_top_alias_nll_sum / max(global_top_alias_tokens, 1.0))
    avg_aliases_per_question = global_sum_aliases / max(global_questions, 1)

    return {
        # Primary metric: for each question, choose alias with highest sentence probability,
        # compute that alias perplexity, then average across questions.
        "avg_top_alias_ppl": float(avg_top_alias_ppl),
        # Standard token-weighted corpus perplexity over selected top-probability aliases.
        "corpus_ppl_top_alias": float(corpus_ppl_top_alias),
        # Backward-compatible key name (now equal to avg_top_alias_ppl).
        "avg_per_question_ppl": float(avg_top_alias_ppl),
        "num_questions": int(global_questions),
        "avg_aliases_per_question": float(avg_aliases_per_question),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TriviaQA top-alias perplexity")
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
    print("TriviaQA Top-Alias Perplexity")
    print("=" * 72)
    print(f"Dataset: {args.dataset} / {args.dataset_config}")
    print(f"Split: {args.split}")
    print(f"Questions: {result['num_questions']}")
    print(f"Shots: {shots} (from split={args.fewshot_split})")
    print(f"Average aliases/question: {result['avg_aliases_per_question']:.2f}")
    print(f"Average top-alias perplexity: {result['avg_top_alias_ppl']:.4f}")
    print(f"Corpus perplexity (top-alias, token-weighted): {result['corpus_ppl_top_alias']:.4f}")
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
            "avg_top_alias_ppl": result["avg_top_alias_ppl"],
            "corpus_ppl_top_alias": result["corpus_ppl_top_alias"],
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
