#!/usr/bin/env python
"""Training script for Transformer language models with wandb support."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.generation import generate
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (numpy file)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data (numpy file)")
    parser.add_argument("--vocab_path", type=str, default=None, help="Path to vocabulary file (for generation)")
    parser.add_argument("--merges_path", type=str, default=None, help="Path to merges file (for generation)")

    # Model architecture
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--weight_tying", action="store_true", help="Tie input/output embeddings")

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_lr", type=float, default=6e-3)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Warmup as ratio of max_steps")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Override warmup_ratio with fixed steps")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--checkpoint_interval", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default=None, help="Run name (auto-generated if not provided)")
    parser.add_argument("--log_file", type=str, default=None, help="CSV log file path")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-lm", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[], help="Wandb tags")

    # Device and precision
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")

    # Generation
    parser.add_argument("--generate_interval", type=int, default=2000, help="Generate samples every N steps")
    parser.add_argument("--generate_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    return parser.parse_args()


def load_data(path: str, dtype=np.uint16):
    """Load tokenized data using memory mapping."""
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    else:
        return np.memmap(path, dtype=dtype, mode="r")


def get_peak_memory(device):
    """Get peak GPU memory usage in MB."""
    if device != "cuda":
        return 0
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()
    return peak_memory


@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device, num_steps, use_amp=False):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    dtype = torch.bfloat16 if use_amp and device == "cuda" else torch.float32

    for _ in range(num_steps):
        x, y = get_batch(val_data, batch_size, context_length, device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()

    model.train()
    return total_loss / num_steps


def train(args):
    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = f"run_{int(time.time())}"

    # Setup output directory
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Initialize wandb
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed, disabling wandb logging")
            print("Install with: pip install wandb")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config=vars(args),
                tags=args.wandb_tags,
                dir=str(output_dir),
            )
            print(f"Wandb run: {wandb_run.url}")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to: {config_path}")

    # Setup CSV logging
    log_file = None
    log_path = output_dir / "metrics.csv"
    log_file = open(log_path, "w")
    log_file.write("step,train_loss,val_loss,lr,perplexity,wallclock_time,tokens_processed,tokens_per_sec\n")

    # Load data
    print(f"Loading training data from {args.train_data}")
    train_data = load_data(args.train_data)
    print(f"Training data shape: {train_data.shape}, dtype: {train_data.dtype}")

    print(f"Loading validation data from {args.val_data}")
    val_data = load_data(args.val_data)
    print(f"Validation data shape: {val_data.shape}, dtype: {val_data.dtype}")

    # Determine warmup steps
    warmup_steps = args.warmup_steps if args.warmup_steps else int(args.warmup_ratio * args.max_steps)

    # Determine dtype
    device = args.device
    use_amp = args.mixed_precision and device == "cuda"
    dtype = torch.bfloat16 if use_amp else torch.float32

    # Create model
    print("Creating model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        weight_tying=args.weight_tying,
    )
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    if args.weight_tying:
        print("Weight tying enabled")

    # Log model info to wandb
    if wandb_run:
        wandb_run.config.update({"num_parameters": num_params})

    # Compile model if requested
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        torch.set_float32_matmul_precision("high")

    # Create optimizer with weight decay grouping
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"Decayed params: {len(decay_params)} tensors, {num_decay_params:,} parameters")
    print(f"Non-decayed params: {len(nodecay_params)} tensors, {num_nodecay_params:,} parameters")

    optimizer = AdamW(
        optim_groups,
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from step {start_step}")

    # Load tokenizer for generation
    tokenizer = None
    if args.vocab_path and args.merges_path:
        try:
            from cs336_basics.tokenizer import Tokenizer
            tokenizer = Tokenizer.from_files(
                args.vocab_path,
                args.merges_path,
                special_tokens=["<|endoftext|>"]
            )
            print("Tokenizer loaded for generation")
        except Exception as e:
            print(f"Could not load tokenizer: {e}")

    # Training info
    tokens_per_step = args.batch_size * args.context_length * args.grad_accum_steps
    total_tokens = args.max_steps * tokens_per_step

    print(f"\n{'='*60}")
    print(f"Run name: {args.run_name}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Total steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum_steps} = {args.batch_size * args.grad_accum_steps}")
    print(f"Context length: {args.context_length}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Max LR: {args.max_lr}")
    print(f"Mixed precision: {use_amp}")
    print(f"Wandb: {'enabled' if wandb_run else 'disabled'}")
    print(f"{'='*60}\n")

    model.train()
    start_time = time.time()
    tokens_processed = start_step * tokens_per_step
    best_val_loss = float("inf")

    for step in range(start_step, args.max_steps):
        t0 = time.time()

        # Get learning rate
        lr = get_lr_cosine_schedule(step, args.max_lr, args.min_lr, warmup_steps, args.max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation loop
        loss_accum = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
                logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / args.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed += tokens_per_step
        train_loss = loss_accum.item()
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed

        # Logging
        if step % args.log_interval == 0:
            ppl = math.exp(min(train_loss, 20))
            mem = get_peak_memory(device)
            print(
                f"Step {step:5d}/{args.max_steps} | "
                f"Loss: {train_loss:.4f} | PPL: {ppl:.2f} | "
                f"LR: {lr:.2e} | "
                f"dt: {dt*1000:.0f}ms | "
                f"Tok/s: {tokens_per_sec:,.0f}"
                + (f" | Mem: {mem:.0f}MB" if mem > 0 else "")
            )

            if wandb_run:
                wandb_run.log({
                    "train/loss": train_loss,
                    "train/perplexity": ppl,
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step_time_ms": dt * 1000,
                    "train/tokens_processed": tokens_processed,
                    "train/peak_memory_mb": mem,
                    "step": step,
                })

        # Evaluation
        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device, args.eval_steps, use_amp)
            val_ppl = math.exp(min(val_loss, 20))
            print(f">>> Step {step:5d} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

            # Log to CSV
            log_file.write(f"{step},{train_loss:.6f},{val_loss:.6f},{lr:.8f},{val_ppl:.4f},{elapsed:.2f},{tokens_processed},{tokens_per_sec:.0f}\n")
            log_file.flush()

            # Log to wandb
            if wandb_run:
                wandb_run.log({
                    "eval/loss": val_loss,
                    "eval/perplexity": val_ppl,
                    "step": step,
                })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "best.pt"
                save_checkpoint(model, optimizer, step, best_path)
                print(f">>> New best model saved to {best_path}")

        # Generation
        if tokenizer and step > 0 and step % args.generate_interval == 0:
            print("\n--- Generated Sample ---")
            prompt = "Once upon a time"
            generated = generate(model, tokenizer, prompt, max_new_tokens=args.generate_tokens,
                               temperature=args.temperature, top_p=args.top_p, device=device)
            print(generated)
            print("--- End Sample ---\n")

            if wandb_run:
                wandb_run.log({"generation/sample": wandb.Html(f"<pre>{generated}</pre>"), "step": step})

        # Checkpointing
        if step > 0 and step % args.checkpoint_interval == 0:
            ckpt_path = checkpoint_dir / f"step_{step}.pt"
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # Final evaluation
    print("\n" + "="*60)
    print("Final evaluation...")
    val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device, args.eval_steps * 3, use_amp)
    val_ppl = math.exp(min(val_loss, 20))
    elapsed = time.time() - start_time

    print(f"Final Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
    print(f"Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total tokens processed: {tokens_processed:,}")
    print(f"Average tokens/sec: {tokens_processed/elapsed:,.0f}")

    # Save final checkpoint
    final_path = checkpoint_dir / "final.pt"
    save_checkpoint(model, optimizer, args.max_steps, final_path)
    print(f"Final checkpoint saved to {final_path}")

    # Log final metrics
    log_file.write(f"{args.max_steps},{train_loss:.6f},{val_loss:.6f},{lr:.8f},{val_ppl:.4f},{elapsed:.2f},{tokens_processed},{tokens_processed/elapsed:.0f}\n")
    log_file.close()

    if wandb_run:
        wandb_run.log({
            "final/val_loss": val_loss,
            "final/val_perplexity": val_ppl,
            "final/total_time_sec": elapsed,
            "final/total_tokens": tokens_processed,
        })
        wandb_run.finish()

    # Final generation
    if tokenizer:
        print("\n--- Final Generated Sample ---")
        prompt = "Once upon a time"
        generated = generate(model, tokenizer, prompt, max_new_tokens=256,
                           temperature=args.temperature, top_p=args.top_p, device=device)
        print(generated)
        print("--- End Sample ---\n")

    print(f"\nRun complete! Results saved to: {output_dir}")
    return val_loss


if __name__ == "__main__":
    args = get_args()
    train(args)
