
● Training Results Summary

  Training Statistics
  ```
  ┌──────────────────┬─────────────────────────────────────────────┐
  │      Metric      │                    Value                    │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ Time             │ 939.54 seconds (0.26 hours / ~15.7 minutes) │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ Memory (start)   │ 24.95 MB                                    │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ Memory (end)     │ 47.52 MB                                    │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ Memory increase  │ 22.58 MB                                    │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ Final vocab size │ 10,000                                      │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ Merges performed │ 9,743                                       │
  └──────────────────┴─────────────────────────────────────────────┘
  ```
  Output Files

  - Vocab: out/tinystories-10k-vocab.txt
  - Merges: out/tinystories-10k-merges.txt

  Longest Token Analysis

  The longest token is  accomplishment (15 bytes, with leading space).

  There are actually 3 tokens tied at 15 bytes:
  ```
  ┌──────────┬─────────────────┬──────────┐
  │ Token ID │      Token      │  Length  │
  ├──────────┼─────────────────┼──────────┤
  │ 7157     │  accomplishment │ 15 bytes │
  ├──────────┼─────────────────┼──────────┤
  │ 9140     │  disappointment │ 15 bytes │
  ├──────────┼─────────────────┼──────────┤
  │ 9376     │  responsibility │ 15 bytes │
  └──────────┴─────────────────┴──────────┘
  ```
  Does the longest token make sense?

  Yes, it makes sense. TinyStories is a dataset of simple children's stories that often contain moral lessons. Words like "accomplishment", "disappointment", and "responsibility" are common.
  
  The leading space is expected in GPT-2 style tokenization—words mid-sentence start with a space

  Other long tokens (14 bytes) include thematically relevant words: understanding, encouragement, unfortunately, compassionate, determination—all common in children's moral stories.
```
Target vocab size: 10000
    Special tokens: ['<|endoftext|>']
    Initial memory: 25.32 MB
    --------------------------------------------------
    Pre-tokenize (parallel): start
      Split into 20 chunks using 20 workers
    Pre-tokenize (parallel): finished in 54.06s
    Init vocab: start
    Init vocab: finished in 0.00s
    Build pair counts: start
    Build pair counts: finished in 0.05s
    Merge: start
    Merge: finished in 300.49s
    Save merges: start
    Save merges: finished in 0.00s
    Save vocab: start
    Save vocab: finished in 0.01s
    Training completed in 354.63s
    --------------------------------------------------
    Training completed!
    Time elapsed: 354.63 seconds (0.0985 hours)
    Final memory: 72.99 MB
    Memory increase: 47.67 MB
    Vocab size: 10000
    Merges performed: 9743
    --------------------------------------------------
    Longest token analysis:
      Token ID: 7160
      Bytes: b' accomplishment'
      Length: 15 bytes
      Decoded: ' accomplishment'
    --------------------------------------------------
    
      ┌─────────┬──────────────────┬───────────────────┬────────┐
      │  阶段   │ 串行 (streaming) │ 并行 (20 workers) │ 加速比 │
      ├─────────┼──────────────────┼───────────────────┼────────┤
      │ 预分词  │ 615.82s          │ 54.06s            │ 11.4x  │
      ├─────────┼──────────────────┼───────────────────┼────────┤
      │ BPE合并 │ 323.64s          │ 300.49s           │ 1.08x  │
      ├─────────┼──────────────────┼───────────────────┼────────┤
      │ 总计    │ 939.54s          │ 354.63s           │ 2.65x  │
      └─────────┴──────────────────┴───────────────────┴────────┘
 ```
