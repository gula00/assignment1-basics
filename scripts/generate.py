#!/usr/bin/env python
"""
Generate text from a trained model.

Usage:
    python scripts/generate.py --checkpoint runs/tinystories/checkpoints/best.pt --prompt "Once upon a time"
    python scripts/generate.py --checkpoint runs/owt/checkpoints/best.pt --prompt "The meaning of life"
"""

import argparse
import json
import torch
from pathlib import Path

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.generation import generate


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Optional: override config paths
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--merges_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    # Find config file
    if args.config:
        config_path = Path(args.config)
    else:
        # Try to find config.json in parent directories
        config_path = checkpoint_path.parent.parent / "config.json"
        if not config_path.exists():
            config_path = checkpoint_path.parent / "config.json"

    if not config_path.exists():
        print(f"Error: Cannot find config.json at {config_path}")
        print("Please specify --config path or --vocab_path and --merges_path")
        return

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print(f"Loaded config from: {config_path}")
    print(f"Model: d_model={config['d_model']}, layers={config['num_layers']}, heads={config['num_heads']}")

    # Load tokenizer
    vocab_path = args.vocab_path or config.get("vocab_path")
    merges_path = args.merges_path or config.get("merges_path")

    if not vocab_path or not merges_path:
        print("Error: vocab_path and merges_path not found in config")
        print("Please specify --vocab_path and --merges_path")
        return

    # Try .pkl version if .txt doesn't work
    vocab_path_pkl = vocab_path.replace(".txt", ".pkl") if vocab_path.endswith(".txt") else vocab_path
    merges_path_pkl = merges_path.replace(".txt", ".pkl") if merges_path.endswith(".txt") else merges_path

    # Check which files exist
    if Path(vocab_path_pkl).exists() and Path(merges_path_pkl).exists():
        vocab_path = vocab_path_pkl
        merges_path = merges_path_pkl

    print(f"Loading tokenizer from: {vocab_path}")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    # Create model
    print("Creating model...")
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config.get("rope_theta", 10000.0),
        weight_tying=config.get("weight_tying", False),
    )

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # Handle compiled model state dict
    state_dict = checkpoint["model_state_dict"]
    # Remove "_orig_mod." prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(args.device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Device: {args.device}")
    print()

    # Interactive generation loop
    print("="*60)
    print("Text Generation (Ctrl+C to exit)")
    print("="*60)

    prompt = args.prompt
    while True:
        print(f"\nPrompt: {prompt}")
        print("-"*40)

        with torch.no_grad():
            output = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )

        print(output)
        print("-"*40)

        # Ask for next prompt
        try:
            prompt = input("\nEnter new prompt (or press Enter to use same, Ctrl+C to exit): ").strip()
            if not prompt:
                prompt = args.prompt
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
