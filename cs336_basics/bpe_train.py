"""BPE Training implementation."""

from __future__ import annotations

import os
from collections import defaultdict

import regex as re

# Fallback pattern for standard re module
GPT2_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")


def pretokenize(text: str) -> list[str]:
    """Pre-tokenize text using GPT-2 pattern."""
    return GPT2_PATTERN.findall(text)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    merges_outpath: str | os.PathLike | None = None,
    vocab_outpath: str | os.PathLike | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given corpus.

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
    # Read and pre-tokenize the corpus
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Pre-tokenize and count word frequencies
    # Words are stored as tuples of token IDs (initially byte values offset by num special tokens)
    num_special = len(special_tokens)
    word_freqs: dict[tuple[int, ...], int] = defaultdict(int)

    # Split text by special tokens and pre-tokenize each segment separately
    # This ensures special tokens don't contribute to BPE merges
    if special_tokens:
        special_pattern = re.compile("|".join(re.escape(t) for t in special_tokens))
        # Split by special tokens (this removes them and keeps segments)
        segments = special_pattern.split(text)
    else:
        segments = [text]

    for segment in segments:
        for word in pretokenize(segment):
            # Convert word to tuple of token IDs (byte value + num_special)
            word_bytes = word.encode("utf-8")
            word_ids = tuple(b + num_special for b in word_bytes)
            word_freqs[word_ids] += 1

    # Initialize vocabulary with special tokens and byte values
    vocab: dict[int, bytes] = {}
    idx = 0

    # Add special tokens first
    for token in special_tokens:
        vocab[idx] = token.encode("utf-8")
        idx += 1

    # Add all 256 byte values
    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1

    # Mapping from token ID to its byte representation
    id_to_bytes: dict[int, bytes] = dict(vocab)

    # Calculate number of merges needed
    num_merges = vocab_size - len(vocab)
    merges: list[tuple[bytes, bytes]] = []

    if num_merges <= 0:
        return vocab, merges

    # Build initial pair counts
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq

    # Perform merges
    for _ in range(num_merges):
        if not pair_counts:
            break

        # Find most frequent pair
        # Break ties by (first_token_bytes, second_token_bytes) lexicographically, larger first
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], (id_to_bytes[p[0]], id_to_bytes[p[1]])))
        best_count = pair_counts[best_pair]

        if best_count == 0:
            break

        # Get bytes for the pair
        a, b = best_pair
        a_bytes = id_to_bytes[a]
        b_bytes = id_to_bytes[b]

        # Create new token
        new_bytes = a_bytes + b_bytes
        new_id = idx
        vocab[new_id] = new_bytes
        id_to_bytes[new_id] = new_bytes
        merges.append((a_bytes, b_bytes))
        idx += 1

        # Update words and pair counts
        new_word_freqs: dict[tuple[int, ...], int] = {}

        for word, freq in word_freqs.items():
            if len(word) < 2:
                new_word_freqs[word] = new_word_freqs.get(word, 0) + freq
                continue

            # Check if this word contains the pair to merge
            has_pair = False
            for i in range(len(word) - 1):
                if word[i] == a and word[i + 1] == b:
                    has_pair = True
                    break

            if not has_pair:
                new_word_freqs[word] = new_word_freqs.get(word, 0) + freq
                continue

            # Decrement old pair counts for this word
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] -= freq

            # Merge the pair in the word
            new_word: list[int] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_tuple = tuple(new_word)
            new_word_freqs[new_word_tuple] = new_word_freqs.get(new_word_tuple, 0) + freq

            # Increment new pair counts for the merged word
            for i in range(len(new_word_tuple) - 1):
                pair = (new_word_tuple[i], new_word_tuple[i + 1])
                pair_counts[pair] += freq

        word_freqs = new_word_freqs

        # Remove the merged pair from counts
        del pair_counts[best_pair]

    # Save merges to file if path provided
    if merges_outpath is not None:
        with open(merges_outpath, "w", encoding="utf-8") as f:
            for a_bytes, b_bytes in merges:
                f.write(f"{a_bytes!r} {b_bytes!r}\n")

    # Save vocab to file if path provided
    if vocab_outpath is not None:
        with open(vocab_outpath, "w", encoding="utf-8") as f:
            for idx, token_bytes in vocab.items():
                f.write(f"{idx}\t{token_bytes!r}\n")

    return vocab, merges
