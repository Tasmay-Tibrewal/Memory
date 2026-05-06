#!/usr/bin/env python3
"""
Training script for Memory-Augmented Transformer.

Usage:
    python scripts/train.py --config configs/base_small.yaml
    
    # Multi-GPU with Accelerate:
    accelerate launch scripts/train.py --config configs/adapter_qwen2.5_1.5b.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory_transformer.config import load_config
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Memory-Augmented Transformer")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help=(
            "Path to checkpoint to initialize model weights only (fresh run: "
            "optimizer/scheduler/global_step are reset)."
        ),
    )
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume and --init_from are mutually exclusive.")
    
    # Load config
    config = load_config(args.config)
    
    # Override resume if specified
    if args.resume:
        config.training.resume_from_checkpoint = args.resume
    if args.init_from:
        config.training.init_from_checkpoint = args.init_from

    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
