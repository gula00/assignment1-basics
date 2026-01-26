"""BPE Training with Heap + Inverted Index optimization."""

from __future__ import annotations

import heapq
import os
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(file: BinaryIO, num_chunks: int, split_token: bytes) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    boundaries = [i * (file_size // num_chunks) for i in range(num_chunks + 1)]
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
    filepath, start, end, special_tokens, num_special = args
    freqs: dict[tuple[int, ...], int] = defaultdict(int)
    special_pat = re.compile("|".join(re.escape(t) for t in special_tokens)) if special_tokens else None

    with open(filepath, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    for seg in (special_pat.split(text) if special_pat else [text]):
        for word in PAT.findall(seg):
            freqs[tuple(b + num_special for b in word.encode("utf-8"))] += 1
    return dict(freqs)


def pre_tokenize(input_path: str | os.PathLike, special_tokens: list[str], num_special: int) -> dict[tuple[int, ...], int]:
    num_workers = cpu_count()
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, split_token)

    with Pool(num_workers) as pool:
        results = pool.map(_process_chunk, [
            (str(input_path), s, e, special_tokens, num_special)
            for s, e in zip(boundaries[:-1], boundaries[1:])
        ])

    merged: dict[tuple[int, ...], int] = defaultdict(int)
    for d in results:
        for k, v in d.items():
            merged[k] += v
    return merged


class _RB:  # Reversed Bytes for heap tie-breaking
    __slots__ = ('b',)
    def __init__(self, b: bytes): self.b = b
    def __lt__(self, other): return self.b > other.b
    def __eq__(self, other): return self.b == other.b


class BPETrainer:
    """Heap + Inverted Index optimized BPE trainer."""

    def __init__(self, freqs: dict[tuple[int, ...], int], id_to_bytes: dict[int, bytes]):
        self.freqs = dict(freqs)
        self.id_to_bytes = id_to_bytes
        self.pair_freqs: dict[tuple[int, int], int] = defaultdict(int)
        self.pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
        self.heap: list = []

        # Build index
        for word, freq in self.freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self.pair_freqs[pair] += freq
                self.pair_to_words[pair].add(word)
        for pair, freq in self.pair_freqs.items():
            self._push(pair, freq)

    def _push(self, pair: tuple[int, int], freq: int):
        a, b = pair
        heapq.heappush(self.heap, (-freq, _RB(self.id_to_bytes[a]), _RB(self.id_to_bytes[b]), pair))

    def get_best_pair(self) -> tuple[int, int] | None:
        while self.heap:
            neg_freq, _, _, pair = self.heap[0]
            actual = self.pair_freqs.get(pair, 0)
            if actual > 0 and actual == -neg_freq:
                heapq.heappop(self.heap)
                return pair
            heapq.heappop(self.heap)
        return None

    def merge(self, a: int, b: int, new_id: int):
        target = (a, b)
        affected: set[tuple[int, int]] = set()
        words = self.pair_to_words.pop(target, set()).copy()
        self.pair_freqs[target] = 0

        for word in words:
            if word not in self.freqs:
                continue
            freq = self.freqs.pop(word)

            # Remove old pairs
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                self.pair_to_words.get(p, set()).discard(word)
                self.pair_freqs[p] -= freq
                affected.add(p)

            # Build merged word
            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            self.freqs[new_word] = self.freqs.get(new_word, 0) + freq

            # Add new pairs
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                self.pair_to_words[p].add(new_word)
                self.pair_freqs[p] += freq
                affected.add(p)

        for p in affected:
            if self.pair_freqs.get(p, 0) > 0:
                self._push(p, self.pair_freqs[p])


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start_time = time.time()
    num_special = len(special_tokens)

    vocab = {i: t.encode("utf-8") for i, t in enumerate(special_tokens)}
    vocab.update({num_special + i: bytes([i]) for i in range(256)})

    print("Pre-tokenize: start")
    t0 = time.time()
    freqs = pre_tokenize(input_path, special_tokens, num_special)
    print(f"Pre-tokenize: {time.time() - t0:.2f}s")

    print("Build index: start")
    t0 = time.time()
    id_to_bytes = dict(vocab)
    trainer = BPETrainer(freqs, id_to_bytes)
    print(f"Build index: {time.time() - t0:.2f}s")

    print("Merge: start")
    t0 = time.time()
    merges, idx = [], len(vocab)

    for _ in range(vocab_size - len(vocab)):
        best = trainer.get_best_pair()
        if not best:
            break
        a, b = best
        new_bytes = id_to_bytes[a] + id_to_bytes[b]
        vocab[idx] = id_to_bytes[idx] = new_bytes
        trainer.id_to_bytes = id_to_bytes
        merges.append((id_to_bytes[a], id_to_bytes[b]))
        trainer.merge(a, b, idx)
        idx += 1

    print(f"Merge: {time.time() - t0:.2f}s ({len(merges)} merges)")
    print(f"Total: {time.time() - start_time:.2f}s")
    return vocab, merges
