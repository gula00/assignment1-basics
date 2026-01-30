"""Transformer model components."""

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce
from torch import Tensor


def silu(x: Tensor) -> Tensor:
    """SiLU / Swish activation."""
    return x * torch.sigmoid(x)


class Linear(nn.Module):
    # def __init__(self, d_in: int, d_out: int):
    #     super().__init__()
    #     self.weight = nn.Parameter(torch.empty(d_out, d_in))
    #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        mean = 0
        std = math.sqrt(2 / (out_features + in_features))
        lower = -3 * std
        upper = 3 * std

        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=lower, b=upper)

        self.weight = torch.nn.Parameter(w)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...i, oi -> ...o", x, self.weight)


class Embedding(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        mean = 0
        std = math.sqrt(2 / (out_features + in_features))
        lower = -3 * std
        upper = 3 * std

        w = torch.empty((in_features, out_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=lower, b=upper)

        self.weight = torch.nn.Parameter(w)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Cast to float32 for numerical stability, then cast back
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)


class RoPE(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        self.theta = theta

        # Precompute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        positions = torch.arange(max_seq_len).float()
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (max_seq_len, d_k/2)

        self.register_buffer("cos", torch.cos(angles))  # (max_seq_len, d_k/2)
        self.register_buffer("sin", torch.sin(angles))  # (max_seq_len, d_k/2)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Apply RoPE to x.

        Args:
            x: (batch, num_heads, seq_len, d_k) or (..., seq_len, d_k)
            positions: (batch, seq_len) or (seq_len,) position indices

        Returns:
            Same shape as x with rotary embeddings applied
        """
        cos = self.cos[positions]  # (..., seq_len, d_k/2)
        sin = self.sin[positions]  # (..., seq_len, d_k/2)

        # Handle 4D input from attention (batch, num_heads, seq_len, d_k)
        if x.dim() == 4 and cos.dim() == 3:
            cos = cos.unsqueeze(1)  # (batch, 1, seq_len, d_k/2)
            sin = sin.unsqueeze(1)  # (batch, 1, seq_len, d_k/2)

        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """
    Q: (..., queries, d_k)
    K: (..., keys, d_k)
    V: (..., keys, d_v)
    mask: (..., queries, keys)
    """
    d_k = Q.size(-1)
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("...qk,...kd->...qd", weights, V)

    return output


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: Tensor, rope: RoPE | None = None, positions: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            rope: Optional RoPE module
            positions: Optional position indices (batch, seq_len)

        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        Q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.num_heads)

        if rope is not None:
            if positions is None:
                positions = torch.arange(seq_len, device=x.device)
            # rotate Q K, V stays still
            Q = rope(Q, positions)
            K = rope(K, positions)

        # Causal mask: True = attend, False = mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        y = scaled_dot_product_attention(Q, K, V, mask)
        y = rearrange(y, "b h s d -> b s (h d)")

        return self.output_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.rope = RoPE(d_model // num_heads, max_seq_len, theta)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), self.rope, positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weight_tying: bool = False,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        # Weight tying: share embedding weights between input and output
        if weight_tying:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: (batch, seq_len)

        Returns:
            (batch, seq_len, vocab_size) logits
        """
        batch, seq_len = token_ids.shape
        # [S] -> [1, S] -> [B, S]
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch, seq_len)

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, positions)
        x = self.ln_final(x)
        x = self.lm_head(x)

        return x
