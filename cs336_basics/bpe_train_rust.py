"""BPE Training using Rust extension (bpe_rust)."""

from __future__ import annotations

import os

try:
    from bpe_rust import train_bpe as _train_bpe_rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    merges_outpath: str | os.PathLike | None = None,
    vocab_outpath: str | os.PathLike | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer using Rust extension.

    Args:
        input_path: Path to the training corpus.
        vocab_size: Target vocabulary size (including special tokens).
        special_tokens: List of special tokens to add to vocabulary.
        merges_outpath: Optional path to save merges file.
        vocab_outpath: Optional path to save vocab file.

    Returns:
        vocab: Mapping from token ID to token bytes.
        merges: List of merge operations as (token1_bytes, token2_bytes).
    """
    if not HAS_RUST:
        raise ImportError(
            "bpe_rust extension not found. "
            "Build it with: cd bpe_rust && maturin develop --release"
        )

    vocab_dict, merges_list = _train_bpe_rust(
        str(input_path),
        vocab_size,
        special_tokens,
        str(merges_outpath) if merges_outpath else None,
        str(vocab_outpath) if vocab_outpath else None,
    )

    # Convert to expected types
    vocab = {int(k): bytes(v) for k, v in vocab_dict.items()}
    merges = [(bytes(a), bytes(b)) for a, b in merges_list]

    return vocab, merges
