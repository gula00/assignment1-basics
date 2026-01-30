"""Data loading utilities."""

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of input sequences and labels from dataset.

    Args:
        dataset: 1D numpy array of integer token IDs
        batch_size: Number of sequences to sample
        context_length: Length of each sequence
        device: PyTorch device string

    Returns:
        (x, y) where:
            x: (batch_size, context_length) input sequences
            y: (batch_size, context_length) target sequences (shifted by 1)
    """
    max_start_idx = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    x = np.stack([dataset[i : i + context_length] for i in start_indices])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_indices])

    return (
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    )
