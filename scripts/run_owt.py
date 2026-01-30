#!/usr/bin/env python
"""
One-click training script for OpenWebText dataset.

Usage:
    python scripts/run_owt.py                    # Basic training
    python scripts/run_owt.py --wandb            # With wandb logging
    python scripts/run_owt.py --fast             # Quick test run
"""

import subprocess
import sys
import argparse
from pathlib import Path


def get_base_cmd():
    """Get base training command with default OpenWebText config."""
    return [
        sys.executable, "scripts/train_lm.py",
        # Data paths
        "--train_data", "out/owt-train-tokens.npy",
        "--val_data", "out/owt-valid-tokens.npy",
        "--vocab_path", "out/owt-32k-vocab.txt",
        "--merges_path", "out/owt-32k-merges.txt",
        # Model architecture (same as TinyStories but larger vocab)
        "--vocab_size", "32000",
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
    parser = argparse.ArgumentParser(description="Train on OpenWebText")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-owt", help="Wandb project")
    parser.add_argument("--fast", action="store_true", help="Quick test run (500 steps)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name")
    parser.add_argument("--max_lr", type=float, default=None, help="Override max learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    args = parser.parse_args()

    # Check data exists
    data_files = [
        "out/owt-train-tokens.npy",
        "out/owt-valid-tokens.npy",
        "out/owt-32k-vocab.txt",
        "out/owt-32k-merges.txt",
    ]
    for f in data_files:
        if not Path(f).exists():
            print(f"Error: Data file not found: {f}")
            print("Please run tokenization first.")
            sys.exit(1)

    # Build command
    cmd = get_base_cmd()

    if args.fast:
        cmd.extend(["--max_steps", "500"])
        cmd.extend(["--eval_interval", "100"])
        cmd.extend(["--log_interval", "10"])
        cmd.extend(["--checkpoint_interval", "500"])
        cmd.extend(["--generate_interval", "250"])
        cmd.extend(["--run_name", args.run_name or "owt_fast"])
    else:
        cmd.extend(["--run_name", args.run_name or "owt"])

    if args.wandb:
        cmd.extend(["--wandb", "--wandb_project", args.wandb_project])
        cmd.extend(["--wandb_tags", "owt"])

    if args.compile:
        cmd.append("--compile")

    if args.mixed_precision:
        cmd.append("--mixed_precision")

    # Override parameters
    if args.max_lr:
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
