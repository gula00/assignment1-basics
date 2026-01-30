"""Model serialization utilities."""

import os
from typing import IO, BinaryIO

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Save model, optimizer state, and iteration to a checkpoint file.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        iteration: Current training iteration
        out: Path or file-like object to save to
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """Load model and optimizer state from a checkpoint file.

    Args:
        src: Path or file-like object to load from
        model: Model to restore state to
        optimizer: Optimizer to restore state to

    Returns:
        The iteration number from the checkpoint
    """
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
