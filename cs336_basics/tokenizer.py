"""BPE Tokenizer implementation."""

from collections.abc import Iterable, Iterator
import regex as re
import pickle

from cs336_basics import bpe_train


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}
        self.encode_cache = {}

        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"

            next_id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                token_bytes = token.encode("UTF-8")
                if token_bytes not in self.vocab_inv:
                    self.vocab[next_id] = token_bytes
                    self.vocab_inv[token_bytes] = next_id
                    next_id += 1
        else:
            self.special_tokens = None
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_chunk(text)

        ids = []
        for part in re.split(self.special_pattern, text):
            if part in self.special_tokens:
                ids.append(self.vocab_inv[part.encode("UTF-8")])
            else:
                ids.extend(self._encode_chunk(part))
        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        ids = []
        for p in bpe_train.PAT.findall(text):
            if p in self.encode_cache:
                ids.extend(self.encode_cache[p])
            else:
                rep = [bytes([b]) for b in p.encode("UTF-8")]
                merged = self._merge_subword(rep)
                token_ids = [self.vocab_inv[subword] for subword in merged]
                self.encode_cache[p] = token_ids
                ids.extend(token_ids)
        return ids

    def _merge_subword(self, rep: list[bytes]) -> list[bytes]:
        while True:
            best_rank = float("inf")
            best_idx = None

            for i in range(len(rep) - 1):
                rank = self.merges_dict.get((rep[i], rep[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx is None:
                return rep

            rep = rep[:best_idx] + [rep[best_idx] + rep[best_idx + 1]] + rep[best_idx + 2:]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id] for id in ids).decode("UTF-8", errors="replace")
