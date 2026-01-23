
● Training Results Summary

  Training Statistics
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
  Output Files

  - Vocab: out/tinystories-10k-vocab.txt
  - Merges: out/tinystories-10k-merges.txt

  Longest Token Analysis

  The longest token is  accomplishment (15 bytes, with leading space).

  There are actually 3 tokens tied at 15 bytes:
  ┌──────────┬─────────────────┬──────────┐
  │ Token ID │      Token      │  Length  │
  ├──────────┼─────────────────┼──────────┤
  │ 7157     │  accomplishment │ 15 bytes │
  ├──────────┼─────────────────┼──────────┤
  │ 9140     │  disappointment │ 15 bytes │
  ├──────────┼─────────────────┼──────────┤
  │ 9376     │  responsibility │ 15 bytes │
  └──────────┴─────────────────┴──────────┘
  Does the longest token make sense?

  Yes, it makes sense. TinyStories is a dataset of simple children's stories that often contain moral lessons. Words like "accomplishment", "disappointment", and "responsibility" are common.
  
  The leading space is expected in GPT-2 style tokenization—words mid-sentence start with a space

  Other long tokens (14 bytes) include thematically relevant words: understanding, encouragement, unfortunately, compassionate, determination—all common in children's moral stories.
