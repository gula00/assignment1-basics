# CS336 Spring 2025 Assignment 1: Basics

## My answers

### 2.1 The Unicode Standard
a. `chr(0)` >>> `\x00`: `chr` 将 Integer 转换为 unicode, 反之用 `ord`

b. `__repr__()` 和 print 的区别: 前者是 debug 用的, 后者优先 `__str__()`

c. 直接 `chr(0)` 用的 `__repr__`, 可见, print 则不可见

### 2.2 Unicode Encodings
a. Prefer training our tokenizer on UTF-8 encoded bytes: 

- shorter sequences, simpler byte-level tokenization, and more efficient learning
- 对于英文字母来说, unicode 和 ASCII 是一样的, emoji 和汉字等则用多个 byte 的 unicode 表示

b. 下面代码的问题
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
```
UTF-8 是 1-4 bytes 动态长度

c. unicode 的规则

- 0x00–0x7F: 单字节字符

- 0xC2–0xDF: 双字节起始

- 0x80–0xBF: 双字节延续

验证举例: `b'\xc3\xa9'.decode('utf-8')`

### 2.3-2.5 BPE Tokenizer Training
See [cs336_basics/bpe_train.py](./cs336_basics/bpe_train.py)

HuggingFace 有 Rust 的 [tokenizers](https://github.com/huggingface/tokenizers) 库, 快多了

Karpathy 的 bpe 实现:
- minbpe: https://github.com/karpathy/minbpe
- rustbpe: https://github.com/karpathy/rustbpe

merge 的优化:
- Inverted Index: 直接查找受影响的词, 不用遍历
- heapq: pop lex_order / freq 最大的

### 2.6-2.8 Tokenizer Training & Analysis
See [tokenizer_answers.md](./tokenizer_answers.md)

### 3. Transformer Language Model Architecture
资源计算 (Resource Accounting):
- GPT-2 XL: 2.05B params, 8.19 GB (float32), 4.5T FLOPs/seq
- FFN 占比最大 (67%), 因为 d_ff = 4 × d_model
- Context 16384: FLOPs 增加 33×, attention 变成主要瓶颈 (56%)

### 4. Training a Transformer LM
训练要点:
- Cross-entropy: log-sum-exp trick 避免数值溢出
- Learning rate: 最佳 lr 在稳定边界附近, 过大会发散
- AdamW 内存: 16N (params + grads + optimizer state) + activations
- GPT-2 XL 在 A100 (FP32, 50% MFU) 训练 400K steps 需要 ~15 年, 所以需要混合精度和多卡

没上 slurm, 那个机子有点忙, 在宿舍的 4070tis 上训的, loss 缩到 2.67, 还有很大的提升空间

训练过程保存在 wandb 上, 有时间做一做不同参数对比
![overview](images/wandb.png )

### Inference

```bash
uv run scripts/generate.py --checkpoint runs/tinystories/checkpoints/best.pt --prompt "Once upon a time" --max_tokens 256 --temperature 0.7
```

## Schedule

  1. train_bpe
  2. tokenizer (依赖 train_bpe)
  3. nn_utils (softmax, cross_entropy, gradient_clipping)
  4. model 基础组件 (linear, embedding, rmsnorm, rope, silu)
  5. model 注意力 (attention, multihead_attention)
  6. model 完整 (swiglu, transformer_block, transformer_lm)
  7. optimizer (adamw, lr_schedule)
  8. data (get_batch)
  9. serialization (checkpointing)
