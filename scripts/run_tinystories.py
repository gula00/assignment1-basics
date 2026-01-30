#!/usr/bin/env python
"""
One-click training script for TinyStories dataset.

Usage:
    python scripts/run_tinystories.py                    # Basic training
    python scripts/run_tinystories.py --wandb            # With wandb logging
    python scripts/run_tinystories.py --fast             # Quick test run
    python scripts/run_tinystories.py --lr_sweep         # Learning rate sweep
"""

import subprocess
import sys
import argparse
from pathlib import Path


def get_base_cmd():
    """Get base training command with default TinyStories config."""
    return [
        sys.executable, "scripts/train_lm.py",
        # Data paths
        "--train_data", "out/tinystories-train-tokens.npy",
        "--val_data", "out/tinystories-valid-tokens.npy",
        "--vocab_path", "out/tinystories-10k-vocab.txt",
        "--merges_path", "out/tinystories-10k-merges.txt",
        # Model architecture (17M params)
        "--vocab_size", "10000",
        "--context_length", "256",
        "--d_model", "512",
        "--num_layers", "4",
        "--num_heads", "16",
        "--d_ff", "1344",
        # Training defaults
        "--batch_size", "128",
        "--max_steps", "10000",
        "--max_lr", "6e-3",
        "--min_lr", "0",
        "--warmup_ratio", "0.01",
        "--weight_decay", "0.01",
        "--grad_clip", "1.0",
        # Logging
        "--log_interval", "100",
        "--eval_interval", "1000",
        "--eval_steps", "50",
        "--checkpoint_interval", "5000",
        "--generate_interval", "2000",
    ]


def main():
    parser = argparse.ArgumentParser(description="Train on TinyStories")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-tinystories", help="Wandb project")
    parser.add_argument("--fast", action="store_true", help="Quick test run (500 steps)")
    parser.add_argument("--lr_sweep", action="store_true", help="Run learning rate sweep")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name")
    parser.add_argument("--max_lr", type=float, default=None, help="Override max learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    args = parser.parse_args()

    # Check data exists
    data_files = [
        "out/tinystories-train-tokens.npy",
        "out/tinystories-valid-tokens.npy",
        "out/tinystories-10k-vocab.txt",
        "out/tinystories-10k-merges.txt",
    ]
    for f in data_files:
        if not Path(f).exists():
            print(f"Error: Data file not found: {f}")
            print("Please run tokenization first.")
            sys.exit(1)

    if args.lr_sweep:
        # Run learning rate sweep
        learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2]
        for lr in learning_rates:
            print(f"\n{'='*60}")
            print(f"Running with learning rate: {lr}")
            print(f"{'='*60}\n")

            cmd = get_base_cmd()
            cmd.extend(["--max_lr", str(lr)])
            cmd.extend(["--run_name", f"lr_{lr}"])
            cmd.extend(["--max_steps", "5000"])  # Shorter for sweep

            if args.wandb:
                cmd.extend(["--wandb", "--wandb_project", args.wandb_project])
                cmd.extend(["--wandb_tags", "lr_sweep"])

            if args.compile:
                cmd.append("--compile")
            if args.mixed_precision:
                cmd.append("--mixed_precision")

            subprocess.run(cmd, check=True)
        return

    # Single training run
    cmd = get_base_cmd()

    if args.fast:
        # Quick test configuration
        cmd.extend(["--max_steps", "500"])
        cmd.extend(["--eval_interval", "100"])
        cmd.extend(["--log_interval", "10"])
        cmd.extend(["--checkpoint_interval", "500"])
        cmd.extend(["--generate_interval", "250"])
        cmd.extend(["--run_name", args.run_name or "fast_test"])
    else:
        cmd.extend(["--run_name", args.run_name or "tinystories"])

    if args.wandb:
        cmd.extend(["--wandb", "--wandb_project", args.wandb_project])
        cmd.extend(["--wandb_tags", "tinystories"])

    if args.compile:
        cmd.append("--compile")

    if args.mixed_precision:
        cmd.append("--mixed_precision")

    # Override parameters if specified
    if args.max_lr:
        # Find and replace max_lr
        idx = cmd.index("--max_lr")
        cmd[idx + 1] = str(args.max_lr)

    if args.batch_size:
        idx = cmd.index("--batch_size")
        cmd[idx + 1] = str(args.batch_size)

    if args.max_steps:
        idx = cmd.index("--max_steps")
        cmd[idx + 1] = str(args.max_steps)

    print("Running command:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
