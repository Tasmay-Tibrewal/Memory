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

This script measures multiple-choice accuracy by scoring either:
    (a) option labels (" A" / " B" / ...)
or (b) full option text continuations,
and choosing the highest log-likelihood option.

Few-shot prompting is enabled by default (`--shots 5`) with handcrafted
benchmark-specific exemplars (`--fewshot_mode manual`).
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


def _shuffle_example_choices(
    example: MCExample,
    rng: random.Random,
    target_answer_index: Optional[int] = None,
) -> MCExample:
    """
    Return a copy with shuffled options and remapped answer index.

    If target_answer_index is provided, place the correct option at that index
    (mod num_choices) and shuffle distractors around it.
    """
    n = len(example.choices)
    if n <= 1:
        return MCExample(
            question=example.question,
            choices=list(example.choices),
            answer_index=example.answer_index,
        )

    correct = example.choices[example.answer_index]
    distractors = [c for i, c in enumerate(example.choices) if i != example.answer_index]
    rng.shuffle(distractors)
    if target_answer_index is None:
        target_answer_index = rng.randrange(n)
    target = int(target_answer_index) % n

    new_choices: List[str] = [""] * n
    new_choices[target] = correct
    d_ptr = 0
    for i in range(n):
        if i == target:
            continue
        new_choices[i] = distractors[d_ptr]
        d_ptr += 1
    new_answer = target
    return MCExample(question=example.question, choices=new_choices, answer_index=new_answer)


def _manual_mcq_fewshot_pool(benchmark: str) -> List[MCExample]:
    """Handcrafted benchmark-specific few-shot exemplars."""
    b = benchmark.lower()
    if b in {"arc_challenge", "arc_easy"}:
        return [
            MCExample(
                question="Which force pulls objects toward Earth?",
                choices=["Magnetism", "Gravity", "Friction", "Electric force"],
                answer_index=1,
            ),
            MCExample(
                question="Which process do plants use to make food using sunlight?",
                choices=["Respiration", "Digestion", "Photosynthesis", "Fermentation"],
                answer_index=2,
            ),
            MCExample(
                question="If an object travels 60 km in 3 hours, what is its average speed?",
                choices=["15 km/h", "20 km/h", "30 km/h", "180 km/h"],
                answer_index=1,
            ),
            MCExample(
                question="What is the primary gas in Earth's atmosphere?",
                choices=["Nitrogen", "Oxygen", "Carbon dioxide", "Hydrogen"],
                answer_index=0,
            ),
            MCExample(
                question="When a liquid changes to a gas at its surface, that process is called:",
                choices=["Condensation", "Freezing", "Melting", "Evaporation"],
                answer_index=3,
            ),
            MCExample(
                question="Which organ pumps blood through the human body?",
                choices=["Liver", "Heart", "Lung", "Kidney"],
                answer_index=1,
            ),
        ]

    if b == "openbookqa":
        return [
            MCExample(
                question="Which material is attracted to a magnet?",
                choices=["Iron nail", "Glass cup", "Rubber band", "Wooden spoon"],
                answer_index=0,
            ),
            MCExample(
                question="What helps prevent the spread of many infectious diseases?",
                choices=["Vaccination", "More sugar intake", "Less sleep", "Skipping handwashing"],
                answer_index=0,
            ),
            MCExample(
                question="Which simple machine is a sloped surface used to raise objects?",
                choices=["Pulley", "Lever", "Inclined plane", "Wheel and axle"],
                answer_index=2,
            ),
            MCExample(
                question="Which state of matter has a fixed volume but no fixed shape?",
                choices=["Solid", "Liquid", "Gas", "Plasma"],
                answer_index=1,
            ),
            MCExample(
                question="Why do metal spoons often feel colder than wooden spoons in the same room?",
                choices=[
                    "Metal has lower mass",
                    "Metal reflects all heat",
                    "Metal conducts heat away from your hand faster",
                    "Wood creates cold energy",
                ],
                answer_index=2,
            ),
            MCExample(
                question="Which source of energy is renewable?",
                choices=["Coal", "Natural gas", "Solar", "Diesel"],
                answer_index=2,
            ),
        ]

    if b == "winogrande":
        return [
            MCExample(
                question="Liam handed Noah the notebook because _ had finished writing.",
                choices=["Liam", "Noah"],
                answer_index=0,
            ),
            MCExample(
                question="Maya thanked Priya because _ helped carry the boxes.",
                choices=["Maya", "Priya"],
                answer_index=1,
            ),
            MCExample(
                question="The violin did not fit in the case because _ was too small.",
                choices=["violin", "case"],
                answer_index=1,
            ),
            MCExample(
                question="The plant near the window grew faster because _ got more sunlight.",
                choices=["plant", "window"],
                answer_index=0,
            ),
            MCExample(
                question="Jordan called Alex after class because _ had missed the assignment details.",
                choices=["Jordan", "Alex"],
                answer_index=1,
            ),
            MCExample(
                question="The laptop overheated before the desktop because _ had a blocked vent.",
                choices=["laptop", "desktop"],
                answer_index=0,
            ),
        ]

    if b == "boolq":
        return [
            MCExample(
                question=(
                    "Passage: Water boils at lower temperatures at higher elevations.\n"
                    "Question: Does water boil below 100C at high altitude?"
                ),
                choices=["Yes", "No"],
                answer_index=0,
            ),
            MCExample(
                question=(
                    "Passage: Penguins are flightless birds adapted for swimming.\n"
                    "Question: Can penguins fly long distances in the air?"
                ),
                choices=["Yes", "No"],
                answer_index=1,
            ),
            MCExample(
                question=(
                    "Passage: The Pacific Ocean is larger than the Atlantic Ocean.\n"
                    "Question: Is the Atlantic the largest ocean on Earth?"
                ),
                choices=["Yes", "No"],
                answer_index=1,
            ),
            MCExample(
                question=(
                    "Passage: Vaccines train the immune system to recognize pathogens.\n"
                    "Question: Do vaccines help the immune system prepare for infections?"
                ),
                choices=["Yes", "No"],
                answer_index=0,
            ),
            MCExample(
                question=(
                    "Passage: Gold is a chemical element with symbol Au.\n"
                    "Question: Is Au the symbol for gold?"
                ),
                choices=["Yes", "No"],
                answer_index=0,
            ),
            MCExample(
                question=(
                    "Passage: Sound travels faster in air than in steel.\n"
                    "Question: Does sound travel faster in steel than in air?"
                ),
                choices=["Yes", "No"],
                answer_index=0,
            ),
        ]

    if b == "hellaswag":
        return [
            MCExample(
                question="A chef places a pan on the stove and turns on the heat. What is the most plausible next step?",
                choices=[
                    "The chef adds oil to the pan.",
                    "The chef puts the pan in the sink and walks away.",
                    "The chef starts mowing the lawn.",
                    "The chef switches off all lights and leaves the kitchen.",
                ],
                answer_index=0,
            ),
            MCExample(
                question="A student opens a textbook and highlights key lines. What is the most plausible next step?",
                choices=[
                    "The student reviews notes and solves practice questions.",
                    "The student jumps into a swimming pool fully dressed.",
                    "The student throws the textbook out of the window.",
                    "The student starts painting a wall immediately.",
                ],
                answer_index=0,
            ),
            MCExample(
                question="A cyclist stops at a red traffic light in a busy intersection. What is the most plausible next step?",
                choices=[
                    "The cyclist waits until the light turns green.",
                    "The cyclist lies down in the middle of the road.",
                    "The cyclist removes both wheels and walks away.",
                    "The cyclist starts cooking dinner at the signal.",
                ],
                answer_index=0,
            ),
            MCExample(
                question="A programmer runs unit tests and sees one failing test. What is the most plausible next step?",
                choices=[
                    "The programmer reads the error message and debugs the code.",
                    "The programmer deletes the entire repository instantly.",
                    "The programmer goes outside to fly a kite in the rain.",
                    "The programmer prints random numbers and ships to production.",
                ],
                answer_index=0,
            ),
            MCExample(
                question="A person picks up a toothbrush and applies toothpaste. What is the most plausible next step?",
                choices=[
                    "They brush their teeth.",
                    "They place the brush in the freezer for an hour.",
                    "They throw toothpaste on the floor.",
                    "They write an email with the toothbrush.",
                ],
                answer_index=0,
            ),
            MCExample(
                question="A gardener digs a small hole and places a seed inside. What is the most plausible next step?",
                choices=[
                    "They cover the seed with soil and water it.",
                    "They put the seed in a blender.",
                    "They paint the seed blue and frame it.",
                    "They leave the garden and lock it forever.",
                ],
                answer_index=0,
            ),
        ]

    raise ValueError(f"No manual few-shot pool configured for benchmark: {benchmark}")


def _select_manual_mcq_fewshot(benchmark: str, shots: int, seed: int) -> List[MCExample]:
    if shots <= 0:
        return []
    pool = _manual_mcq_fewshot_pool(benchmark)
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)

    selected: List[MCExample] = []
    start_offset = rng.randrange(max(len(pool[0].choices), 1))
    while len(selected) < shots:
        for idx in order:
            n = len(pool[idx].choices)
            target = (start_offset + len(selected)) % max(n, 1)
            selected.append(
                _shuffle_example_choices(
                    pool[idx],
                    rng,
                    target_answer_index=target,
                )
            )
            if len(selected) >= shots:
                break
    return selected


def _stable_text_seed(text: str) -> int:
    """Deterministic text-to-int helper (avoids Python hash randomization)."""
    acc = 0
    for i, ch in enumerate(text):
        acc += (i + 1) * ord(ch)
    return acc


def _zero_based_index(value: object, n: int) -> int:
    s = str(value).strip()
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < n:
            return idx
    if len(s) == 1 and s.upper().isalpha():
        idx = ord(s.upper()) - ord("A")
        if 0 <= idx < n:
            return idx
    raise ValueError(f"Cannot parse zero-based answer index from {value!r} with n={n}")


def _one_based_index(value: object, n: int) -> int:
    s = str(value).strip()
    if s.isdigit():
        idx = int(s)
        if 1 <= idx <= n:
            return idx - 1
    if len(s) == 1 and s.upper().isalpha():
        idx = ord(s.upper()) - ord("A")
        if 0 <= idx < n:
            return idx
    raise ValueError(f"Cannot parse one-based answer index from {value!r} with n={n}")


def _index_from_choice_labels(answer_key: object, labels: List[object], n: int) -> int:
    key = str(answer_key).strip()
    normalized_labels = [str(x).strip() for x in labels]

    # Prefer exact label matching from dataset metadata (most reliable).
    for i, lbl in enumerate(normalized_labels):
        if key == lbl:
            return i
        if key.upper() == lbl.upper():
            return i

    # Fallbacks if labels are missing/unexpected.
    if normalized_labels and all(lbl.isdigit() for lbl in normalized_labels):
        return _one_based_index(key, n)
    return _zero_based_index(key, n)


def parse_row(benchmark: str, row: Dict) -> MCExample:
    b = benchmark.lower()
    if b == "hellaswag":
        question = f"{str(row['ctx']).strip()}\nWhat is the most plausible next sentence?"
        choices = [str(x).strip() for x in row["endings"]]
        answer = _zero_based_index(row["label"], len(choices))
        return MCExample(question=question, choices=choices, answer_index=answer)

    if b in {"arc_challenge", "arc_easy"}:
        question = str(row["question"]).strip()
        choices = [str(x).strip() for x in row["choices"]["text"]]
        choice_labels = row["choices"].get("label", [])
        answer = _index_from_choice_labels(row["answerKey"], choice_labels, len(choices))
        return MCExample(question=question, choices=choices, answer_index=answer)

    if b == "openbookqa":
        question = str(row["question_stem"]).strip()
        choices = [str(x).strip() for x in row["choices"]["text"]]
        choice_labels = row["choices"].get("label", [])
        answer = _index_from_choice_labels(row["answerKey"], choice_labels, len(choices))
        return MCExample(question=question, choices=choices, answer_index=answer)

    if b == "winogrande":
        sentence = str(row["sentence"]).strip()
        question = f"Complete the sentence:\n{sentence}"
        choices = [str(row["option1"]).strip(), str(row["option2"]).strip()]
        # WinoGrande answers are 1-based ("1"/"2"), not zero-based.
        answer = _one_based_index(row["answer"], len(choices))
        return MCExample(question=question, choices=choices, answer_index=answer)

    if b == "boolq":
        passage = str(row["passage"]).strip()
        question = str(row["question"]).strip()
        full_q = f"Passage: {passage}\nQuestion: {question}"
        choices = ["Yes", "No"]
        answer = 0 if bool(row["answer"]) else 1
        return MCExample(question=full_q, choices=choices, answer_index=answer)

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def format_example(ex: MCExample, include_answer: bool, answer_style: str = "label") -> str:
    lines = [f"Question: {ex.question}"]
    for i, choice in enumerate(ex.choices):
        label = chr(ord("A") + i)
        lines.append(f"{label}. {choice}")
    if include_answer:
        if answer_style == "choice_text":
            lines.append(f"Answer: {ex.choices[ex.answer_index]}")
        else:
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
    fewshot_mode: str,
    scoring_mode: str,
    seed: int,
    use_cache: bool,
) -> Dict[str, float]:
    spec = SPECS[benchmark]
    eval_split = split or spec.eval_split
    fs_split = fewshot_split or spec.fewshot_split

    eval_examples = load_examples(benchmark, eval_split, max_samples=max_samples)
    fewshot_examples = []
    if shots > 0:
        if fewshot_mode == "manual":
            fewshot_examples = _select_manual_mcq_fewshot(
                benchmark=benchmark,
                shots=min(shots, 64),
                seed=seed + (_stable_text_seed(benchmark) % 10_000),
            )
        elif fewshot_mode == "dataset":
            fewshot_examples = load_examples(benchmark, fs_split, max_samples=shots)
        else:
            raise ValueError(f"Unknown fewshot_mode: {fewshot_mode}")

    prefix_parts = [
        "The following are multiple-choice questions.",
        (
            f"{spec.instruction} Answer using the full option text."
            if scoring_mode == "choice_text"
            else f"{spec.instruction} Answer using only the option letter."
        ),
        "",
    ]
    if fewshot_examples:
        prefix_parts.append(
            "\n\n".join(
                format_example(ex, include_answer=True, answer_style=scoring_mode)
                for ex in fewshot_examples
            )
        )
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
        prompt = prefix + format_example(ex, include_answer=False, answer_style=scoring_mode)
        labels = [chr(ord("A") + i) for i in range(len(ex.choices))]
        scores = score_options(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            option_labels=labels,
            option_texts=ex.choices,
            scoring_mode=scoring_mode,
            device=device,
            max_length=max_length,
            use_cache=use_cache,
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
        "fewshot_mode": fewshot_mode,
        "scoring_mode": scoring_mode,
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
    parser.add_argument("--shots", type=int, default=5, help="Few-shot examples (set 0 for zero-shot)")
    parser.add_argument(
        "--fewshot_mode",
        type=str,
        default="manual",
        choices=["manual", "dataset"],
        help="Few-shot source: handcrafted benchmark exemplars (`manual`) or dataset split (`dataset`).",
    )
    parser.add_argument("--split", type=str, default=None, help="Eval split override")
    parser.add_argument("--fewshot_split", type=str, default=None, help="Few-shot split override")
    parser.add_argument("--max_samples", type=int, default=None, help="Max eval samples")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length for scoring")
    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="choice_text",
        choices=["label", "choice_text"],
        help="Option scoring target: option label token vs full option text.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Pass use_cache=True to model forward during option scoring (default: False)",
    )
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
        fewshot_mode=args.fewshot_mode,
        scoring_mode=args.scoring_mode,
        seed=int(args.seed),
        use_cache=bool(args.use_cache),
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
    print(
        f"Split: {result['eval_split']} | Shots: {result['shots']} | "
        f"Few-shot mode: {result['fewshot_mode']} | Scoring: {result['scoring_mode']}"
    )
    print("=" * 64)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
