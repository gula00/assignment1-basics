"""Train BPE tokenizer on TinyStories dataset."""

import os
import sys
import time

import psutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.bpe_train import train_bpe


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def main():
    # Paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, "data/TinyStoriesV2-GPT4-train.txt")
    out_dir = os.path.join(project_root, "out")

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    merges_outpath = os.path.join(out_dir, "tinystories-10k-merges.txt")
    vocab_outpath = os.path.join(out_dir, "tinystories-10k-vocab.txt")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Track memory and time
    start_memory = get_memory_usage_mb()
    start_time = time.time()

    print(f"Starting BPE training on {input_path}")
    print(f"Target vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print(f"Initial memory: {start_memory:.2f} MB")
    print("-" * 50)

    # Train BPE (use parallel processing for large files)
    # Set parallel=N to use N workers, or parallel=None/0 to use streaming
    num_workers = os.cpu_count() or 4
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        merges_outpath=merges_outpath,
        vocab_outpath=vocab_outpath,
        parallel=num_workers,  # Use parallel mode with all CPU cores
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

    # Try to decode as UTF-8
    try:
        decoded = longest_token_bytes.decode("utf-8")
        print(f"  Decoded: '{decoded}'")
    except UnicodeDecodeError:
        print(f"  Cannot decode as UTF-8")

    print("-" * 50)
    print(f"Files saved:")
    print(f"  Vocab: {vocab_outpath}")
    print(f"  Merges: {merges_outpath}")


if __name__ == "__main__":
    main()
