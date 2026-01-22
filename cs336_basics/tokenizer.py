"""BPE Tokenizer implementation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import IO

import regex as re

# GPT-2 pretokenization pattern (using Unicode-aware character classes)
# \p{L} matches any Unicode letter, \p{N} matches any Unicode number
GPT2_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    """BPE Tokenizer that uses a provided vocabulary and merge list."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            vocab: Mapping from token ID to token bytes.
            merges: List of merge operations as (token1_bytes, token2_bytes).
            special_tokens: Optional list of special tokens that should not be split.
        """
        self.vocab = vocab  # id -> bytes
        self.bytes_to_id = {v: k for k, v in vocab.items()}  # bytes -> id
        self.merges = merges
        # Build merge ranking: (bytes1, bytes2) -> merge priority (lower = earlier)
        self.merge_ranking: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # Handle special tokens
        self.special_tokens = special_tokens or []
        # Sort by length descending for longest match first
        self.special_tokens_sorted = sorted(
            self.special_tokens, key=len, reverse=True
        )
        # Build special token pattern for splitting
        if self.special_tokens_sorted:
            escaped = [re.escape(t) for t in self.special_tokens_sorted]
            self.special_pattern = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_pattern = None

    def _pretokenize(self, text: str) -> list[str]:
        """Pre-tokenize text using GPT-2 pattern."""
        return GPT2_PATTERN.findall(text)

    def _bpe_encode_word(self, word_bytes: bytes) -> list[int]:
        """
        Apply BPE to a single word (sequence of bytes).

        Args:
            word_bytes: The word as bytes.

        Returns:
            List of token IDs after applying BPE merges.
        """
        if len(word_bytes) == 0:
            return []

        # Start with individual bytes as tokens
        tokens = [bytes([b]) for b in word_bytes]

        while len(tokens) >= 2:
            # Find the pair with lowest merge ranking (earliest merge)
            best_pair = None
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranking:
                    rank = self.merge_ranking[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                        best_idx = i

            if best_pair is None:
                # No more merges possible
                break

            # Apply the merge
            new_token = best_pair[0] + best_pair[1]
            tokens = tokens[:best_idx] + [new_token] + tokens[best_idx + 2:]

        # Convert bytes tokens to IDs
        return [self.bytes_to_id[t] for t in tokens]

    def encode(self, text: str) -> list[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: The text to encode.

        Returns:
            List of token IDs.
        """
        if not text:
            return []

        result: list[int] = []

        # Split by special tokens if any
        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            # Check if this part is a special token
            if part in self.special_tokens:
                # Get the token ID for this special token
                special_bytes = part.encode("utf-8")
                if special_bytes in self.bytes_to_id:
                    result.append(self.bytes_to_id[special_bytes])
                continue

            # Pre-tokenize and encode each word
            words = self._pretokenize(part)
            for word in words:
                word_bytes = word.encode("utf-8")
                token_ids = self._bpe_encode_word(word_bytes)
                result.extend(token_ids)

        return result

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back to text.

        Args:
            token_ids: List of token IDs to decode.

        Returns:
            Decoded text string.
        """
        if not token_ids:
            return ""

        # Concatenate all token bytes
        byte_sequence = b"".join(self.vocab[token_id] for token_id in token_ids)

        # Decode as UTF-8, using surrogatepass to handle partial sequences
        # that might occur when decoding individual tokens
        return byte_sequence.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str] | IO[str]) -> Iterator[int]:
        """
        Encode an iterable of strings (like a file object) into token IDs.

        This is a memory-efficient streaming encoder that yields token IDs
        one at a time.

        Args:
            iterable: An iterable of strings (e.g., file object opened in text mode).

        Yields:
            Token IDs one at a time.
        """
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id
