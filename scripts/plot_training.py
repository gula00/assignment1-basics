#!/usr/bin/env python
"""
Plot training curves from CSV logs or wandb.

Usage:
    # From local CSV file
    python scripts/plot_training.py --csv runs/tinystories/metrics.csv --output plots/

    # From multiple runs (comparison)
    python scripts/plot_training.py --csv runs/lr_1e-3/metrics.csv runs/lr_6e-3/metrics.csv --labels "lr=1e-3" "lr=6e-3"

    # From wandb
    python scripts/plot_training.py --wandb --project cs336-lm --runs run1 run2
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_from_csv(csv_files, labels=None, output_dir="plots", title_prefix=""):
    """Plot training curves from CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    if labels is None:
        labels = [Path(f).parent.name for f in csv_files]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)

        # Plot train loss
        axes[0, 0].plot(df['step'], df['train_loss'], label=label, alpha=0.8)

        # Plot val loss
        if 'val_loss' in df.columns:
            axes[0, 1].plot(df['step'], df['val_loss'], label=label, marker='o', markersize=3)

        # Plot perplexity
        if 'perplexity' in df.columns:
            axes[1, 0].plot(df['step'], df['perplexity'], label=label, marker='o', markersize=3)
        elif 'val_loss' in df.columns:
            ppl = np.exp(np.clip(df['val_loss'], 0, 20))
            axes[1, 0].plot(df['step'], ppl, label=label, marker='o', markersize=3)

        # Plot learning rate
        if 'lr' in df.columns:
            axes[1, 1].plot(df['step'], df['lr'], label=label, alpha=0.8)

    # Formatting
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Train Loss')
    axes[0, 0].set_title(f'{title_prefix}Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Val Loss')
    axes[0, 1].set_title(f'{title_prefix}Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Perplexity')
    axes[1, 0].set_title(f'{title_prefix}Validation Perplexity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title(f'{title_prefix}Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{title_prefix.replace(" ", "_")}training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_loss_comparison(csv_files, labels=None, output_dir="plots", title="Loss Comparison"):
    """Plot only loss comparison for multiple runs."""
    os.makedirs(output_dir, exist_ok=True)

    if labels is None:
        labels = [Path(f).parent.name for f in csv_files]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))

    for csv_file, label, color in zip(csv_files, labels, colors):
        df = pd.read_csv(csv_file)

        ax1.plot(df['step'], df['train_loss'], label=label, color=color, alpha=0.8)

        if 'val_loss' in df.columns:
            ax2.plot(df['step'], df['val_loss'], label=label, color=color, marker='o', markersize=4)

    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Val Loss', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_")
    output_path = os.path.join(output_dir, f'{safe_title}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_from_wandb(project, run_names=None, entity=None, output_dir="plots"):
    """Plot training curves from wandb."""
    try:
        import wandb
    except ImportError:
        print("Error: wandb not installed. Install with: pip install wandb")
        return

    os.makedirs(output_dir, exist_ok=True)

    api = wandb.Api()

    if run_names:
        runs = [api.run(f"{entity}/{project}/{name}" if entity else f"{project}/{name}")
                for name in run_names]
    else:
        runs = api.runs(f"{entity}/{project}" if entity else project)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for run in runs:
        history = run.history()
        label = run.name

        if 'train/loss' in history.columns:
            axes[0, 0].plot(history['_step'], history['train/loss'], label=label, alpha=0.8)

        if 'eval/loss' in history.columns:
            eval_data = history[history['eval/loss'].notna()]
            axes[0, 1].plot(eval_data['_step'], eval_data['eval/loss'], label=label, marker='o')

        if 'eval/perplexity' in history.columns:
            eval_data = history[history['eval/perplexity'].notna()]
            axes[1, 0].plot(eval_data['_step'], eval_data['eval/perplexity'], label=label, marker='o')

        if 'train/lr' in history.columns:
            axes[1, 1].plot(history['_step'], history['train/lr'], label=label, alpha=0.8)

    # Formatting (same as above)
    titles = ['Training Loss', 'Validation Loss', 'Validation Perplexity', 'Learning Rate']
    ylabels = ['Train Loss', 'Val Loss', 'Perplexity', 'Learning Rate']

    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{project}_training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--csv", type=str, nargs="+", help="CSV file(s) to plot")
    parser.add_argument("--labels", type=str, nargs="+", help="Labels for each CSV file")
    parser.add_argument("--output", type=str, default="plots", help="Output directory")
    parser.add_argument("--title", type=str, default="", help="Title prefix for plots")
    parser.add_argument("--compare", action="store_true", help="Plot loss comparison only")

    # Wandb options
    parser.add_argument("--wandb", action="store_true", help="Fetch data from wandb")
    parser.add_argument("--project", type=str, help="Wandb project name")
    parser.add_argument("--entity", type=str, help="Wandb entity")
    parser.add_argument("--runs", type=str, nargs="+", help="Wandb run names")

    args = parser.parse_args()

    if args.wandb:
        if not args.project:
            print("Error: --project required for wandb")
            return
        plot_from_wandb(args.project, args.runs, args.entity, args.output)
    elif args.csv:
        if args.compare:
            plot_loss_comparison(args.csv, args.labels, args.output, args.title or "Loss Comparison")
        else:
            plot_from_csv(args.csv, args.labels, args.output, args.title)
    else:
        print("Error: Specify --csv or --wandb")
        parser.print_help()


if __name__ == "__main__":
    main()
