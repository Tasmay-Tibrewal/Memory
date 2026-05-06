#!/usr/bin/env python3
"""Evaluate ARC-Challenge or ARC-Easy accuracy."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval_mcq_benchmark import main as eval_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--variant",
        type=str,
        default="challenge",
        choices=["challenge", "easy"],
        help="ARC variant",
    )
    known, remaining = parser.parse_known_args()
    benchmark = "arc_challenge" if known.variant == "challenge" else "arc_easy"
    eval_main(argv=remaining, default_benchmark=benchmark)


if __name__ == "__main__":
    main()
