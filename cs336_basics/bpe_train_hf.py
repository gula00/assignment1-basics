"""Fast BPE Training implementation using HuggingFace tokenizers for pre-tokenization."""

from __future__ import annotations

import os
from collections import defaultdict

import regex as re
from tokenizers import pre_tokenizers


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Returns a mapping between every possible byte to a printable unicode string."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def gpt2_unicode_to_bytes() -> dict[str, int]:
    """Returns the inverse mapping of gpt2_bytes_to_unicode."""
    return {v: k for k, v in gpt2_bytes_to_unicode().items()}


# Precompute the decoder
_GPT2_BYTE_DECODER = gpt2_unicode_to_bytes()


def compute_word_freqs_fast(
    text: str,
    special_tokens: list[str],
    num_special: int,
) -> dict[tuple[int, ...], int]:
    """
    Fast pre-tokenization using HuggingFace's ByteLevel pre-tokenizer.
    """
    word_freqs: dict[tuple[int, ...], int] = defaultdict(int)

    # Split text by special tokens first
    if special_tokens:
        special_pattern = re.compile("|".join(re.escape(t) for t in special_tokens))
        segments = special_pattern.split(text)
    else:
        segments = [text]

    # Create a ByteLevel pre-tokenizer
    pre_tok = pre_tokenizers.ByteLevel(add_prefix_space=False)

    for segment in segments:
        if not segment:
            continue
        # Use HuggingFace pre-tokenizer
        tokens = pre_tok.pre_tokenize_str(segment)
        for token_str, _ in tokens:
            # Convert GPT-2 style token to bytes
            word_bytes = bytes([_GPT2_BYTE_DECODER[c] for c in token_str])
            word_ids = tuple(b + num_special for b in word_bytes)
            word_freqs[word_ids] += 1

    return word_freqs


def init_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    """Initialize vocabulary with special tokens and byte values."""
    vocab: dict[int, bytes] = {}
    idx = 0

    for token in special_tokens:
        vocab[idx] = token.encode("utf-8")
        idx += 1

    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1

    return vocab, idx


def perform_merges_optimized(
    word_freqs: dict[tuple[int, ...], int],
    vocab: dict[int, bytes],
    num_merges: int,
    start_idx: int,
) -> list[tuple[bytes, bytes]]:
    """
    Perform BPE merge operations with inverted index optimization.
    Only processes words that contain the pair being merged.
    """
    id_to_bytes: dict[int, bytes] = dict(vocab)

    # Build pair counts and inverted index
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    merges: list[tuple[bytes, bytes]] = []
    idx = start_idx

    for _ in range(num_merges):
        if not pair_counts:
            break

        # Find best pair (max frequency, then lexicographic tie-break)
        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], (id_to_bytes[p[0]], id_to_bytes[p[1]])),
        )

        if pair_counts[best_pair] == 0:
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

        # Get words containing this pair and remove from index
        affected_words = pair_to_words.pop(best_pair, set())
        del pair_counts[best_pair]

        # Process only affected words
        for word in affected_words:
            freq = word_freqs.pop(word, 0)
            if freq == 0:
                continue

            # Remove old pairs from counts and index
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                if p in pair_counts:
                    pair_counts[p] -= freq
                    pair_to_words[p].discard(word)

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
            word_freqs[new_word_tuple] = word_freqs.get(new_word_tuple, 0) + freq

            # Add new pairs to counts and index
            for i in range(len(new_word_tuple) - 1):
                p = (new_word_tuple[i], new_word_tuple[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(new_word_tuple)

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
    Train a BPE tokenizer using fast HuggingFace pre-tokenization
    and optimized merge operations.

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
    # Read corpus
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Pre-tokenize and count word frequencies (fast path)
    num_special = len(special_tokens)
    word_freqs = compute_word_freqs_fast(text, special_tokens, num_special)

    # Initialize vocabulary
    vocab, idx = init_vocab(special_tokens)

    # Calculate number of merges needed
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # Perform merges with optimization
    merges = perform_merges_optimized(word_freqs, vocab, num_merges, idx)

    # Save results
    if merges_outpath is not None:
        save_merges(merges, merges_outpath)

    if vocab_outpath is not None:
        save_vocab(vocab, vocab_outpath)

    return vocab, merges
