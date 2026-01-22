"""BPE Training implementation."""

from __future__ import annotations

import os
import time
from collections import defaultdict

import regex as re

# GPT-2 pretokenization pattern (using Unicode-aware character classes)
# \p{L} matches any Unicode letter, \p{N} matches any Unicode number
GPT2_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def pretokenize(text: str) -> list[str]:
    """Pre-tokenize text using GPT-2 pattern."""
    return GPT2_PATTERN.findall(text)


def compute_word_freqs(
    text: str,
    special_tokens: list[str],
    num_special: int,
) -> dict[tuple[int, ...], int]:
    """
    Pre-tokenize text and compute word frequencies.

    Args:
        text: Input text to process.
        special_tokens: List of special tokens to exclude from tokenization.
        num_special: Number of special tokens (used for ID offset).

    Returns:
        Dictionary mapping word (as tuple of token IDs) to frequency.
    """
    word_freqs: dict[tuple[int, ...], int] = defaultdict(int)

    # Split text by special tokens and pre-tokenize each segment separately
    # This ensures special tokens don't contribute to BPE merges
    if special_tokens:
        special_pattern = re.compile("|".join(re.escape(t) for t in special_tokens))
        segments = special_pattern.split(text)
    else:
        segments = [text]

    for segment in segments:
        for word in pretokenize(segment):
            # Convert word to tuple of token IDs (byte value + num_special)
            word_bytes = word.encode("utf-8")
            word_ids = tuple(b + num_special for b in word_bytes)
            word_freqs[word_ids] += 1

    return word_freqs


def init_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    """
    Initialize vocabulary with special tokens and byte values.

    Args:
        special_tokens: List of special tokens to add first.

    Returns:
        Tuple of (vocab dict, next available index).
    """
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

    return vocab, idx


def build_pair_counts(
    word_freqs: dict[tuple[int, ...], int],
) -> dict[tuple[int, int], int]:
    """
    Build initial pair counts from word frequencies.

    Args:
        word_freqs: Dictionary mapping words to their frequencies.

    Returns:
        Dictionary mapping adjacent token pairs to their total frequency.
    """
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq

    return pair_counts


def find_best_pair(
    pair_counts: dict[tuple[int, int], int],
    id_to_bytes: dict[int, bytes],
) -> tuple[int, int] | None:
    """
    Find the most frequent pair, breaking ties lexicographically.

    Args:
        pair_counts: Dictionary of pair frequencies.
        id_to_bytes: Mapping from token ID to bytes.

    Returns:
        Best pair to merge, or None if no valid pairs.
    """
    if not pair_counts:
        return None

    best_pair = max(
        pair_counts,
        key=lambda p: (pair_counts[p], (id_to_bytes[p[0]], id_to_bytes[p[1]])),
    )

    if pair_counts[best_pair] == 0:
        return None

    return best_pair


def apply_merge(
    word_freqs: dict[tuple[int, ...], int],
    pair_counts: dict[tuple[int, int], int],
    a: int,
    b: int,
    new_id: int,
) -> dict[tuple[int, ...], int]:
    """
    Apply a single merge operation to update word frequencies and pair counts.

    Args:
        word_freqs: Current word frequencies.
        pair_counts: Current pair counts (modified in place).
        a: First token ID of the pair to merge.
        b: Second token ID of the pair to merge.
        new_id: New token ID for the merged pair.

    Returns:
        Updated word frequencies dictionary.
    """
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

    return new_word_freqs


def perform_merges(
    word_freqs: dict[tuple[int, ...], int],
    vocab: dict[int, bytes],
    num_merges: int,
    start_idx: int,
) -> list[tuple[bytes, bytes]]:
    """
    Perform BPE merge operations.

    Args:
        word_freqs: Initial word frequencies.
        vocab: Vocabulary dictionary (modified in place).
        num_merges: Number of merges to perform.
        start_idx: Starting index for new token IDs.

    Returns:
        List of merge operations as (token1_bytes, token2_bytes).
    """
    print("Build pair counts: start")
    start_time = time.time()
    id_to_bytes: dict[int, bytes] = dict(vocab)
    pair_counts = build_pair_counts(word_freqs)
    print(f"Build pair counts: finished in {time.time() - start_time:.2f}s")

    merges: list[tuple[bytes, bytes]] = []
    idx = start_idx

    print("Merge: start")
    merge_start_time = time.time()
    for _ in range(num_merges):
        best_pair = find_best_pair(pair_counts, id_to_bytes)
        if best_pair is None:
            break

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

        # Apply merge
        word_freqs = apply_merge(word_freqs, pair_counts, a, b, new_id)

        # Remove the merged pair from counts
        del pair_counts[best_pair]

    print(f"Merge: finished in {time.time() - merge_start_time:.2f}s")
    return merges


def save_merges(merges: list[tuple[bytes, bytes]], path: str | os.PathLike) -> None:
    """Save merges to file."""
    with open(path, "w", encoding="utf-8") as f:
        for a_bytes, b_bytes in merges:
            f.write(f"{a_bytes!r} {b_bytes!r}\n")


def save_vocab(vocab: dict[int, bytes], path: str | os.PathLike) -> None:
    """Save vocabulary to file."""
    with open(path, "w", encoding="utf-8") as f:
        for idx, token_bytes in vocab.items():
            f.write(f"{idx}\t{token_bytes!r}\n")


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
    train_start_time = time.time()

    # Read corpus
    print("Read corpus: start")
    start_time = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Read corpus: finished in {time.time() - start_time:.2f}s")

    # Pre-tokenize and count word frequencies
    num_special = len(special_tokens)
    print("Pre-tokenize: start")
    start_time = time.time()
    word_freqs = compute_word_freqs(text, special_tokens, num_special)
    print(f"Pre-tokenize: finished in {time.time() - start_time:.2f}s")

    # Initialize vocabulary
    print("Init vocab: start")
    start_time = time.time()
    vocab, idx = init_vocab(special_tokens)
    print(f"Init vocab: finished in {time.time() - start_time:.2f}s")

    # Calculate number of merges needed
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # Perform merges
    merges = perform_merges(word_freqs, vocab, num_merges, idx)

    # Save results
    if merges_outpath is not None:
        print("Save merges: start")
        start_time = time.time()
        save_merges(merges, merges_outpath)
        print(f"Save merges: finished in {time.time() - start_time:.2f}s")

    if vocab_outpath is not None:
        print("Save vocab: start")
        start_time = time.time()
        save_vocab(vocab, vocab_outpath)
        print(f"Save vocab: finished in {time.time() - start_time:.2f}s")

    print(f"Training completed in {time.time() - train_start_time:.2f}s")

    return vocab, merges
