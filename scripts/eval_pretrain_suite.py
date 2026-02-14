#!/usr/bin/env python3
"""
Run a pretraining benchmark suite and aggregate results.

Default suite:
- mmlu
- hellaswag
- arc_challenge
- winogrande
- boolq
- openbookqa
- triviaqa
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_BENCHMARKS = [
    "mmlu",
    "hellaswag",
    "arc_challenge",
    "winogrande",
    "boolq",
    "openbookqa",
    "triviaqa",
]


def _run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _build_launcher(distributed: bool, num_processes: int) -> List[str]:
    if not distributed:
        return [sys.executable]
    return [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(num_processes)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and aggregate pretraining benchmark evaluations")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(DEFAULT_BENCHMARKS),
        help=f"Comma-separated benchmarks from: {DEFAULT_BENCHMARKS}",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for non-distributed mode")
    parser.add_argument("--distributed", action="store_true", help="Run each benchmark with accelerate launch")
    parser.add_argument("--num_processes", type=int, default=1, help="Processes for --distributed")
    parser.add_argument("--mmlu_shots", type=int, default=5, help="Few-shot count for MMLU")
    parser.add_argument("--mcq_shots", type=int, default=5, help="Few-shot count for non-MMLU MCQ tasks")
    parser.add_argument("--triviaqa_shots", type=int, default=5, help="Few-shot count for TriviaQA")
    parser.add_argument(
        "--triviaqa_dataset_config",
        type=str,
        default="rc.nocontext",
        help="TriviaQA dataset config (default: rc.nocontext)",
    )
    parser.add_argument("--triviaqa_split", type=str, default="validation", help="TriviaQA eval split")
    parser.add_argument("--triviaqa_fewshot_split", type=str, default="train", help="TriviaQA few-shot source split")
    parser.add_argument(
        "--mmlu_fewshot_mode",
        type=str,
        default="manual",
        choices=["manual", "dataset"],
        help="Few-shot source mode for MMLU.",
    )
    parser.add_argument(
        "--mcq_fewshot_mode",
        type=str,
        default="manual",
        choices=["manual", "dataset"],
        help="Few-shot source mode for non-MMLU MCQ benchmarks.",
    )
    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="choice_text",
        choices=["label", "choice_text"],
        help="Option scoring target used by benchmark scripts.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Pass use_cache=True to model forward in child eval scripts (default: False)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples (MMLU: per subject; others: per benchmark)",
    )
    parser.add_argument("--mmlu_max_subjects", type=int, default=None, help="Optional cap on MMLU subjects")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/benchmark_suite",
        help="Directory for per-benchmark and summary JSON files",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running remaining benchmarks if one fails",
    )
    args = parser.parse_args()

    benchmarks = [b.strip().lower() for b in args.benchmarks.split(",") if b.strip()]
    supported = set(DEFAULT_BENCHMARKS)
    unknown = [b for b in benchmarks if b not in supported]
    if unknown:
        raise ValueError(f"Unknown benchmarks: {unknown}. Supported: {sorted(supported)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    launcher = _build_launcher(distributed=args.distributed, num_processes=args.num_processes)
    results: Dict[str, Dict] = {}
    failures: Dict[str, str] = {}

    for bench in benchmarks:
        output_path = out_dir / f"{bench}.json"
        if bench == "mmlu":
            cmd = launcher + [
                "scripts/eval_mmlu.py",
                "--config",
                args.config,
                "--shots",
                str(int(args.mmlu_shots)),
                "--fewshot_mode",
                args.mmlu_fewshot_mode,
                "--scoring_mode",
                args.scoring_mode,
                "--output",
                str(output_path),
            ]
            if args.checkpoint:
                cmd += ["--checkpoint", args.checkpoint]
            if args.max_samples is not None:
                cmd += ["--max_samples_per_subject", str(int(args.max_samples))]
            if args.mmlu_max_subjects is not None:
                cmd += ["--max_subjects", str(int(args.mmlu_max_subjects))]
            if args.distributed:
                cmd += ["--distributed"]
            else:
                cmd += ["--device", args.device]
            if args.use_cache:
                cmd += ["--use_cache"]
        elif bench == "triviaqa":
            cmd = launcher + [
                "scripts/eval_triviaqa.py",
                "--config",
                args.config,
                "--dataset_config",
                args.triviaqa_dataset_config,
                "--split",
                args.triviaqa_split,
                "--fewshot_split",
                args.triviaqa_fewshot_split,
                "--shots",
                str(int(args.triviaqa_shots)),
                "--output",
                str(output_path),
            ]
            if args.checkpoint:
                cmd += ["--checkpoint", args.checkpoint]
            if args.max_samples is not None:
                cmd += ["--max_samples", str(int(args.max_samples))]
            if args.distributed:
                cmd += ["--distributed"]
            else:
                cmd += ["--device", args.device]
            if args.use_cache:
                cmd += ["--use_cache"]
        else:
            cmd = launcher + [
                "scripts/eval_mcq_benchmark.py",
                "--benchmark",
                bench,
                "--config",
                args.config,
                "--shots",
                str(int(args.mcq_shots)),
                "--fewshot_mode",
                args.mcq_fewshot_mode,
                "--scoring_mode",
                args.scoring_mode,
                "--output",
                str(output_path),
            ]
            if args.checkpoint:
                cmd += ["--checkpoint", args.checkpoint]
            if args.max_samples is not None:
                cmd += ["--max_samples", str(int(args.max_samples))]
            if args.distributed:
                cmd += ["--distributed"]
            else:
                cmd += ["--device", args.device]
            if args.use_cache:
                cmd += ["--use_cache"]

        try:
            _run(cmd)
            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            results[bench] = payload
        except Exception as e:
            failures[bench] = str(e)
            if not args.continue_on_error:
                raise
            print(f"[WARN] {bench} failed: {e}")

    summary = {
        "benchmarks": benchmarks,
        "results": results,
        "failures": failures,
        "config": args.config,
        "checkpoint": args.checkpoint,
    }

    normalized_scores: Dict[str, float] = {}
    ppl_metrics: Dict[str, float] = {}
    aux_metrics: Dict[str, float] = {}
    for bench, payload in results.items():
        if bench == "mmlu":
            normalized_scores[bench] = float(payload.get("weighted_accuracy", 0.0))
        elif bench == "triviaqa":
            ppl_metrics[bench] = float(
                payload.get(
                    "avg_top_alias_ppl",
                    payload.get("avg_per_question_ppl", 0.0),
                )
            )
            aux_metrics["triviaqa_corpus_ppl_top_alias"] = float(
                payload.get("corpus_ppl_top_alias", 0.0)
            )
        else:
            normalized_scores[bench] = float(payload.get("accuracy", 0.0))

    summary["scores"] = normalized_scores
    summary["ppl_metrics"] = ppl_metrics
    summary["aux_metrics"] = aux_metrics
    if normalized_scores:
        summary["mean_score"] = float(sum(normalized_scores.values()) / len(normalized_scores))
    else:
        summary["mean_score"] = 0.0

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("Pretraining Benchmark Suite")
    print("=" * 72)
    for bench in benchmarks:
        if bench in normalized_scores:
            print(f"{bench:<16} {normalized_scores[bench]:.4f}")
        elif bench in ppl_metrics:
            corpus = aux_metrics.get("triviaqa_corpus_ppl_top_alias")
            if bench == "triviaqa" and corpus is not None:
                print(f"{bench:<16} ppl={ppl_metrics[bench]:.4f} (corpus={corpus:.4f})")
            else:
                print(f"{bench:<16} ppl={ppl_metrics[bench]:.4f}")
        else:
            print(f"{bench:<16} FAILED")
    print("-" * 72)
    print(f"Mean score: {summary['mean_score']:.4f}")
    print(f"Summary saved to {summary_path}")
    if failures:
        print("Failures:")
        for bench, err in failures.items():
            print(f"- {bench}: {err}")
    print("=" * 72)


if __name__ == "__main__":
    main()
