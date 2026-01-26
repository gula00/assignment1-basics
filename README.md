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

HuggingFace 有 Rust 实现的 [tokenizers](https://github.com/huggingface/tokenizers) 库, 快多了

merge 的优化: heapq, inverse

But at large scale, it doesn't really matter. The major bottleneck is the pre-tokenization step, which can be optimized by parallelizing the tokenization process.
