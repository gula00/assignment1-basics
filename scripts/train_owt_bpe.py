"""Train BPE tokenizer on OpenWebText dataset."""

import os
import pickle
import sys
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.bpe_train import train_bpe


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def save_outputs(vocab, merges, out_dir, prefix):
    """Save vocab and merges as both pickle (for Tokenizer) and text (for inspection)."""
    # Pickle format for Tokenizer.from_files
    with open(os.path.join(out_dir, f"{prefix}-vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(out_dir, f"{prefix}-merges.pkl"), "wb") as f:
        pickle.dump(merges, f)

    # Text format for human inspection
    with open(os.path.join(out_dir, f"{prefix}-vocab.txt"), "w", encoding="utf-8") as f:
        for idx in sorted(vocab.keys()):
            f.write(f"{idx}\t{vocab[idx]!r}\n")
    with open(os.path.join(out_dir, f"{prefix}-merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a!r} {b!r}\n")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, "data", "owt_train", "owt_train.txt")
    out_dir = os.path.join(project_root, "out")

    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    os.makedirs(out_dir, exist_ok=True)

    start_memory = get_memory_usage_mb()
    start_time = time.time()

    print(f"Starting BPE training on {input_path}")
    print(f"Target vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print(f"Initial memory: {start_memory:.2f} MB")
    print("-" * 50)

    num_workers = min(os.cpu_count() or 4, 6)  # Limit workers for 12GB file
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        parallel=num_workers,
        num_workers=num_workers,
        max_chunk_bytes=64 * 1024 * 1024,  # 64MB chunks to limit memory
    )

    end_time = time.time()
    end_memory = get_memory_usage_mb()

    elapsed_seconds = end_time - start_time
    elapsed_hours = elapsed_seconds / 3600

    print("-" * 50)
    print(f"Training completed!")
    print(f"Time elapsed: {elapsed_seconds:.2f} seconds ({elapsed_hours:.4f} hours)")
    print(f"Final memory: {end_memory:.2f} MB")
    print(f"Memory increase: {end_memory - start_memory:.2f} MB")
    print(f"Vocab size: {len(vocab)}")
    print(f"Merges performed: {len(merges)}")

    # Save outputs
    save_outputs(vocab, merges, out_dir, "owt-32k")
    print(f"Files saved to {out_dir}/owt-32k-*")

    # Find the longest token
    longest_token_id = None
    longest_token_bytes = b""

    for token_id, token_bytes in vocab.items():
        if len(token_bytes) > len(longest_token_bytes):
            longest_token_bytes = token_bytes
            longest_token_id = token_id

    print("-" * 50)
    print(f"Longest token analysis:")
    print(f"  Token ID: {longest_token_id}")
    print(f"  Bytes: {longest_token_bytes!r}")
    print(f"  Length: {len(longest_token_bytes)} bytes")

    try:
        decoded = longest_token_bytes.decode("utf-8")
        print(f"  Decoded: '{decoded}'")
    except UnicodeDecodeError:
        print(f"  Cannot decode as UTF-8")


if __name__ == "__main__":
    main()
