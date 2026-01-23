"""Analyze vocabulary file to find longest tokens."""

import os

# Paths relative to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vocab_path = os.path.join(project_root, "out/tinystories-10k-vocab.txt")

vocab = {}
with open(vocab_path, 'r') as f:
    for line in f:
        idx, token_repr = line.strip().split('\t', 1)
        token_bytes = eval(token_repr)
        vocab[int(idx)] = token_bytes

# Sort by length and show top 20
sorted_tokens = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)
print('Top 20 longest tokens:')
print('-' * 60)
for idx, token_bytes in sorted_tokens[:20]:
    try:
        decoded = token_bytes.decode('utf-8')
        print(f'{idx:5d} | {len(token_bytes):2d} bytes | {repr(decoded)}')
    except:
        print(f'{idx:5d} | {len(token_bytes):2d} bytes | {repr(token_bytes)}')
