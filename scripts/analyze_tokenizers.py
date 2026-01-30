"""Comprehensive tokenizer analysis for assignment questions."""

import os
import pickle
import sys
import time
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.tokenizer import Tokenizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tokenizer(prefix, special_tokens=None):
    vocab_path = os.path.join(PROJECT_ROOT, "out", f"{prefix}-vocab.pkl")
    merges_path = os.path.join(PROJECT_ROOT, "out", f"{prefix}-merges.pkl")
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def sample_documents(filepath, n=10, sep="<|endoftext|>"):
    """Sample n documents from a file split by separator."""
    random.seed(42)
    docs = []
    current = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if sep in line:
                parts = line.split(sep)
                current.append(parts[0])
                doc = "".join(current).strip()
                if doc:
                    docs.append(doc)
                current = [parts[-1]] if len(parts) > 1 else []
            else:
                current.append(line)

    # Add last document if exists
    if current:
        doc = "".join(current).strip()
        if doc:
            docs.append(doc)

    if len(docs) < n:
        return docs
    return random.sample(docs, n)


def compression_ratio(tokenizer, documents):
    """Calculate bytes/token compression ratio."""
    total_bytes = 0
    total_tokens = 0
    for doc in documents:
        text_bytes = len(doc.encode("utf-8"))
        tokens = tokenizer.encode(doc)
        total_bytes += text_bytes
        total_tokens += len(tokens)
    return total_bytes / total_tokens if total_tokens > 0 else 0


def main():
    print("=" * 70)
    print("TOKENIZER ANALYSIS")
    print("=" * 70)

    # Load tokenizers
    print("\nLoading tokenizers...")
    ts_tok = load_tokenizer("tinystories-10k", ["<|endoftext|>"])
    print(f"  TinyStories tokenizer: vocab={len(ts_tok.vocab)}")

    owt_vocab_path = os.path.join(PROJECT_ROOT, "out", "owt-32k-vocab.pkl")
    owt_tok = None
    if os.path.exists(owt_vocab_path):
        owt_tok = load_tokenizer("owt-32k", ["<|endoftext|>"])
        print(f"  OpenWebText tokenizer: vocab={len(owt_tok.vocab)}")
    else:
        print("  OpenWebText tokenizer: NOT YET TRAINED (skipping OWT-specific analysis)")

    # ================================================================
    # Q3a: Compression ratio on 10 sampled documents
    # ================================================================
    print("\n" + "=" * 70)
    print("Q3a: Compression ratios (10 sampled documents)")
    print("=" * 70)

    ts_train_path = os.path.join(PROJECT_ROOT, "data", "TinyStoriesV2-GPT4-train.txt")
    owt_train_path = os.path.join(PROJECT_ROOT, "data", "owt_train", "owt_train.txt")

    print("\nSampling 10 documents from TinyStories...")
    ts_docs = sample_documents(ts_train_path, 10)
    print(f"  Sampled {len(ts_docs)} documents, total {sum(len(d.encode('utf-8')) for d in ts_docs)} bytes")

    ts_on_ts = compression_ratio(ts_tok, ts_docs)
    print(f"  TinyStories tokenizer on TinyStories: {ts_on_ts:.2f} bytes/token")

    if owt_tok:
        print("\nSampling 10 documents from OpenWebText...")
        owt_docs = sample_documents(owt_train_path, 10)
        print(f"  Sampled {len(owt_docs)} documents, total {sum(len(d.encode('utf-8')) for d in owt_docs)} bytes")

        owt_on_owt = compression_ratio(owt_tok, owt_docs)
        print(f"  OpenWebText tokenizer on OpenWebText: {owt_on_owt:.2f} bytes/token")

    # ================================================================
    # Q3b: Cross-tokenization (TinyStories tokenizer on OpenWebText)
    # ================================================================
    print("\n" + "=" * 70)
    print("Q3b: Cross-tokenization (TS tokenizer on OWT sample)")
    print("=" * 70)

    if owt_tok:
        ts_on_owt = compression_ratio(ts_tok, owt_docs)
        print(f"  TinyStories tokenizer on OpenWebText: {ts_on_owt:.2f} bytes/token")
        print(f"  OpenWebText tokenizer on OpenWebText: {owt_on_owt:.2f} bytes/token")
        print(f"  Ratio difference: TS tok is {owt_on_owt/ts_on_owt:.2f}x the compression of OWT tok on OWT data")

        # Qualitative example
        print("\n  Qualitative example (first 200 chars of first OWT doc):")
        sample = owt_docs[0][:200]
        ts_encoded = ts_tok.encode(sample)
        owt_encoded = owt_tok.encode(sample)
        print(f"    Text: {sample[:100]}...")
        print(f"    TS tokenizer: {len(ts_encoded)} tokens")
        print(f"    OWT tokenizer: {len(owt_encoded)} tokens")

        # Show some tokens from TS tokenizer on OWT
        print(f"\n    TS tokens (first 20): {[ts_tok.decode([t]) for t in ts_encoded[:20]]}")
        print(f"    OWT tokens (first 20): {[owt_tok.decode([t]) for t in owt_encoded[:20]]}")

    # ================================================================
    # Q3c: Throughput estimation
    # ================================================================
    print("\n" + "=" * 70)
    print("Q3c: Throughput estimation")
    print("=" * 70)

    # Use a decent chunk of text for throughput measurement
    test_text = "\n".join(ts_docs)
    test_bytes = len(test_text.encode("utf-8"))

    # Warm up
    ts_tok.encode(test_text[:1000])

    # Measure
    start = time.time()
    n_reps = 3
    for _ in range(n_reps):
        ts_tok.encode(test_text)
    elapsed = time.time() - start
    bytes_per_sec = (test_bytes * n_reps) / elapsed

    print(f"  Test text size: {test_bytes} bytes")
    print(f"  Time for {n_reps} repetitions: {elapsed:.3f}s")
    print(f"  Throughput: {bytes_per_sec:.0f} bytes/second ({bytes_per_sec/1e6:.2f} MB/s)")

    pile_size_bytes = 825 * 1e9  # 825 GB
    pile_time_seconds = pile_size_bytes / bytes_per_sec
    pile_time_hours = pile_time_seconds / 3600
    pile_time_days = pile_time_hours / 24
    print(f"  Estimated time for Pile (825GB): {pile_time_hours:.1f} hours ({pile_time_days:.1f} days)")

    if owt_tok:
        # Also measure OWT tokenizer throughput
        owt_test = "\n".join(owt_docs)
        owt_test_bytes = len(owt_test.encode("utf-8"))
        owt_tok.encode(owt_test[:1000])  # warm up
        start = time.time()
        for _ in range(n_reps):
            owt_tok.encode(owt_test)
        elapsed = time.time() - start
        owt_bps = (owt_test_bytes * n_reps) / elapsed
        print(f"  OWT tokenizer throughput: {owt_bps:.0f} bytes/second ({owt_bps/1e6:.2f} MB/s)")

    # ================================================================
    # Q3d: Encode datasets to uint16
    # ================================================================
    print("\n" + "=" * 70)
    print("Q3d: Encoding datasets to uint16 numpy arrays")
    print("=" * 70)

    print(f"  TinyStories vocab size: {len(ts_tok.vocab)} (max id: {max(ts_tok.vocab.keys())})")
    if owt_tok:
        print(f"  OpenWebText vocab size: {len(owt_tok.vocab)} (max id: {max(owt_tok.vocab.keys())})")
    print(f"  uint16 max value: {np.iinfo(np.uint16).max}")
    print(f"  uint16 is appropriate because both vocab sizes (10k, 32k) < 65535 = 2^16 - 1")

    # Encode TinyStories train
    print("\n  Encoding TinyStories train...")
    ts_out = os.path.join(PROJECT_ROOT, "out", "tinystories-train-tokens.npy")
    if not os.path.exists(ts_out):
        encode_file_to_npy(ts_tok, ts_train_path, ts_out)
    else:
        arr = np.load(ts_out)
        print(f"    Already exists: {ts_out} ({len(arr)} tokens, {arr.nbytes/1e6:.1f} MB)")

    # Encode TinyStories valid
    ts_valid_path = os.path.join(PROJECT_ROOT, "data", "TinyStoriesV2-GPT4-valid.txt")
    ts_valid_out = os.path.join(PROJECT_ROOT, "out", "tinystories-valid-tokens.npy")
    if not os.path.exists(ts_valid_out):
        encode_file_to_npy(ts_tok, ts_valid_path, ts_valid_out)
    else:
        arr = np.load(ts_valid_out)
        print(f"    Already exists: {ts_valid_out} ({len(arr)} tokens, {arr.nbytes/1e6:.1f} MB)")

    if owt_tok:
        # Encode OWT train
        owt_out = os.path.join(PROJECT_ROOT, "out", "owt-train-tokens.npy")
        if not os.path.exists(owt_out):
            encode_file_to_npy(owt_tok, owt_train_path, owt_out)
        else:
            arr = np.load(owt_out)
            print(f"    Already exists: {owt_out} ({len(arr)} tokens, {arr.nbytes/1e6:.1f} MB)")

        # Encode OWT valid
        owt_valid_path = os.path.join(PROJECT_ROOT, "data", "owt_valid.txt")
        owt_valid_out = os.path.join(PROJECT_ROOT, "out", "owt-valid-tokens.npy")
        if not os.path.exists(owt_valid_out):
            encode_file_to_npy(owt_tok, owt_valid_path, owt_valid_out)
        else:
            arr = np.load(owt_valid_out)
            print(f"    Already exists: {owt_valid_out} ({len(arr)} tokens, {arr.nbytes/1e6:.1f} MB)")

    # ================================================================
    # Compare tokenizers
    # ================================================================
    if owt_tok:
        print("\n" + "=" * 70)
        print("TOKENIZER COMPARISON (TinyStories vs OpenWebText)")
        print("=" * 70)

        # Longest tokens
        ts_longest = max(ts_tok.vocab.items(), key=lambda x: len(x[1]))
        owt_longest = max(owt_tok.vocab.items(), key=lambda x: len(x[1]))
        print(f"  TS longest token: {ts_longest[1]!r} ({len(ts_longest[1])} bytes)")
        print(f"  OWT longest token: {owt_longest[1]!r} ({len(owt_longest[1])} bytes)")

        # Show some tokens unique to each
        ts_token_set = set(ts_tok.vocab.values())
        owt_token_set = set(owt_tok.vocab.values())

        ts_only = ts_token_set - owt_token_set
        owt_only = owt_token_set - ts_token_set

        print(f"\n  Tokens only in TS: {len(ts_only)}")
        ts_only_decoded = []
        for t in sorted(ts_only, key=len, reverse=True)[:10]:
            try:
                ts_only_decoded.append(t.decode("utf-8"))
            except:
                ts_only_decoded.append(repr(t))
        print(f"    Examples (longest): {ts_only_decoded}")

        print(f"  Tokens only in OWT: {len(owt_only)}")
        owt_only_decoded = []
        for t in sorted(owt_only, key=len, reverse=True)[:10]:
            try:
                owt_only_decoded.append(t.decode("utf-8"))
            except:
                owt_only_decoded.append(repr(t))
        print(f"    Examples (longest): {owt_only_decoded}")


def encode_file_to_npy(tokenizer, input_path, output_path):
    """Encode a text file to a numpy array of uint16 token IDs."""
    print(f"    Encoding {os.path.basename(input_path)}...")
    start = time.time()

    all_ids = []
    chunk_size = 10 * 1024 * 1024  # 10MB chunks

    with open(input_path, "r", encoding="utf-8") as f:
        while True:
            text = f.read(chunk_size)
            if not text:
                break
            ids = tokenizer.encode(text)
            all_ids.extend(ids)

    arr = np.array(all_ids, dtype=np.uint16)
    np.save(output_path, arr)

    elapsed = time.time() - start
    print(f"    Saved {output_path}: {len(arr)} tokens, {arr.nbytes/1e6:.1f} MB, took {elapsed:.1f}s")


if __name__ == "__main__":
    main()
