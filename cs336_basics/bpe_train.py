"""BPE Training implementation."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(file: BinaryIO, num_chunks: int, split_token: bytes) -> list[int]:
    """Find chunk boundaries aligned to split_token."""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // num_chunks
    boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    boundaries[-1] = file_size

    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        file.seek(pos)
        while True:
            chunk = file.read(4096)
            if not chunk:
                boundaries[bi] = file_size
                break
            found = chunk.find(split_token)
            if found != -1:
                boundaries[bi] = pos + found + len(split_token)
                break
            pos += 4096

    return sorted(set(boundaries))


def _process_chunk(args: tuple) -> dict[tuple[int, ...], int]:
    """Worker: pre-tokenize a file chunk and count word frequencies."""
    filepath, start, end, special_tokens, num_special = args
    freqs: dict[tuple[int, ...], int] = defaultdict(int)

    special_pat = re.compile("|".join(re.escape(t) for t in special_tokens)) if special_tokens else None

    with open(filepath, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    segments = special_pat.split(text) if special_pat else [text]
    for seg in segments:
        for word in PAT.findall(seg):
            word_ids = tuple(b + num_special for b in word.encode("utf-8"))
            freqs[word_ids] += 1

    return dict(freqs)


def pre_tokenize(input_path: str | os.PathLike, special_tokens: list[str], num_special: int) -> dict[tuple[int, ...], int]:
    """Pre-tokenize file in parallel and return word frequencies."""
    num_workers = cpu_count()
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, split_token)

    chunk_args = [(str(input_path), s, e, special_tokens, num_special) for s, e in zip(boundaries[:-1], boundaries[1:])]

    with Pool(num_workers) as pool:
        results = pool.map(_process_chunk, chunk_args)

    # Merge results
    merged: dict[tuple[int, ...], int] = defaultdict(int)
    for d in results:
        for k, v in d.items():
            merged[k] += v
    return merged


def get_pair_freqs(freqs: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    """Build pair frequency table from word frequencies."""
    pair_freqs: dict[tuple[int, int], int] = defaultdict(int)
    for word, freq in freqs.items():
        for i in range(len(word) - 1):
            pair_freqs[word[i], word[i + 1]] += freq
    return pair_freqs


def merge(
    freqs: dict[tuple[int, ...], int],
    pair_freqs: dict[tuple[int, int], int],
    a: int,
    b: int,
    new_id: int,
) -> dict[tuple[int, ...], int]:
    """Merge pair (a, b) -> new_id in freqs, update pair_freqs in place."""
    new_freqs: dict[tuple[int, ...], int] = {}

    for word, freq in freqs.items():
        # Check if word contains the pair
        if len(word) < 2 or (a, b) not in zip(word, word[1:]):
            new_freqs[word] = new_freqs.get(word, 0) + freq
            continue

        # Decrement old pairs
        for i in range(len(word) - 1):
            pair_freqs[word[i], word[i + 1]] -= freq

        # Build merged word
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_word = tuple(new_word)
        new_freqs[new_word] = new_freqs.get(new_word, 0) + freq

        # Increment new pairs
        for i in range(len(new_word) - 1):
            pair_freqs[new_word[i], new_word[i + 1]] += freq

    return new_freqs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE tokenizer on corpus."""
    start_time = time.time()
    num_special = len(special_tokens)

    # Initialize vocab: special tokens + 256 bytes
    vocab = {i: t.encode("utf-8") for i, t in enumerate(special_tokens)}
    for i in range(256):
        vocab[num_special + i] = bytes([i])

    # Pre-tokenize
    print("Pre-tokenize: start")
    t0 = time.time()
    freqs = pre_tokenize(input_path, special_tokens, num_special)
    print(f"Pre-tokenize: {time.time() - t0:.2f}s")

    # Build pair frequencies
    print("Get pair freqs: start")
    t0 = time.time()
    id_to_bytes = dict(vocab)
    pair_freqs = get_pair_freqs(freqs)
    print(f"Get pair freqs: {time.time() - t0:.2f}s")

    # Perform merges
    print("Merge: start")
    t0 = time.time()
    merges = []
    n_merges = vocab_size - len(vocab)
    idx = len(vocab)

    for _ in range(n_merges):
        if not pair_freqs:
            break

        # Find best pair (max freq, then max lex order for ties)
        best = max(pair_freqs, key=lambda p: (pair_freqs[p], id_to_bytes[p[0]], id_to_bytes[p[1]]))
        if pair_freqs[best] <= 0:
            break

        a, b = best
        new_bytes = id_to_bytes[a] + id_to_bytes[b]
        vocab[idx] = new_bytes
        id_to_bytes[idx] = new_bytes
        merges.append((id_to_bytes[a], id_to_bytes[b]))

        freqs = merge(freqs, pair_freqs, a, b, idx)
        del pair_freqs[best]
        idx += 1

    print(f"Merge: {time.time() - t0:.2f}s ({len(merges)} merges)")
    print(f"Total: {time.time() - start_time:.2f}s")

    return vocab, merges
