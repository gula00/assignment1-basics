from cs336_basics.bpe_train import train_bpe

# Train on tiny.txt
(vocab_tiny, merges_tiny) = train_bpe(
    input_path="../data/tiny.txt",
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
    merges_outpath="../out/tiny-merges.txt",
    vocab_outpath="../out/tiny-vocab.txt",
)
print(f"tiny.txt: vocab_size={len(vocab_tiny)}, merges={len(merges_tiny)}")

# # Train on TinyStoriesV2
# (vocab, merges) = train_bpe(
#     input_path="../data/TinyStoriesV2-GPT4-train.txt",
#     vocab_size=10000,
#     special_tokens=["<|endoftext|>"],
#     merges_outpath="../out/ts-train-merges.txt",
#     vocab_outpath="../out/ts-train-vocab.txt",
# )

if __name__ == "__main__":
    print("Hello, World!")

# python -m cs336_basics.test
