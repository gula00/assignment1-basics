"""Text generation utilities for language models."""

import torch
from torch import Tensor


def softmax_with_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Apply temperature-scaled softmax.

    Args:
        logits: Unnormalized logits of shape (..., vocab_size)
        temperature: Temperature for scaling (lower = more deterministic)

    Returns:
        Probability distribution of same shape as logits
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    scaled = logits / temperature
    scaled_max = scaled.max(dim=-1, keepdim=True).values
    exp_scaled = torch.exp(scaled - scaled_max)
    return exp_scaled / exp_scaled.sum(dim=-1, keepdim=True)


def top_p_filtering(probs: Tensor, top_p: float) -> Tensor:
    """Apply nucleus (top-p) filtering to probability distribution.

    Args:
        probs: Probability distribution of shape (..., vocab_size)
        top_p: Cumulative probability threshold (0 < top_p <= 1)

    Returns:
        Filtered and renormalized probability distribution
    """
    if top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be in (0, 1]")

    if top_p == 1.0:
        return probs

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cutoff: keep tokens until cumsum exceeds top_p
    # Shift cumsum right by 1 so we include the token that crosses the threshold
    cumsum_mask = cumsum_probs - sorted_probs > top_p

    # Zero out probabilities for tokens outside the nucleus
    sorted_probs[cumsum_mask] = 0.0

    # Scatter back to original order
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(-1, sorted_indices, sorted_probs)

    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    return filtered_probs


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str = "cpu",
) -> str:
    """Generate text from a language model.

    Args:
        model: TransformerLM model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_p: Nucleus sampling threshold
        eos_token_id: Token ID to stop generation (if None, uses tokenizer's special token)
        device: Device to run generation on

    Returns:
        Generated text including the prompt
    """
    model.eval()

    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Get EOS token ID if not provided
    if eos_token_id is None:
        # Try to get from tokenizer's special tokens
        eos_token = "<|endoftext|>"
        try:
            eos_token_id = tokenizer.encode(eos_token)[0]
        except:
            eos_token_id = None

    # Generate tokens one at a time
    generated_ids = list(input_ids)

    for _ in range(max_new_tokens):
        # Get current sequence (possibly truncated to context length)
        context_length = getattr(model, 'context_length', None)
        if context_length is None:
            # Try to infer from model
            for layer in model.layers:
                if hasattr(layer, 'rope') and hasattr(layer.rope, 'cos'):
                    context_length = layer.rope.cos.shape[0]
                    break

        if context_length is not None and len(generated_ids) > context_length:
            # Truncate to fit context window
            curr_ids = generated_ids[-context_length:]
        else:
            curr_ids = generated_ids

        curr_tensor = torch.tensor([curr_ids], dtype=torch.long, device=device)

        # Forward pass
        logits = model(curr_tensor)

        # Get logits for the last position
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply temperature
        probs = softmax_with_temperature(next_logits, temperature)

        # Apply top-p filtering
        probs = top_p_filtering(probs, top_p)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Append to generated sequence
        generated_ids.append(next_token)

        # Check for EOS
        if eos_token_id is not None and next_token == eos_token_id:
            break

    # Decode and return
    return tokenizer.decode(generated_ids)


@torch.no_grad()
def generate_batch(
    model,
    input_ids: Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
) -> Tensor:
    """Generate text from a language model with batched inputs.

    Args:
        model: TransformerLM model
        input_ids: Input token IDs of shape (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        eos_token_id: Token ID to stop generation
        pad_token_id: Token ID for padding

    Returns:
        Generated token IDs of shape (batch_size, seq_len + num_generated)
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.shape[0]

    # Track which sequences have finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Forward pass
        logits = model(generated)

        # Get logits for the last position
        next_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # Apply temperature
        probs = softmax_with_temperature(next_logits, temperature)

        # Apply top-p filtering
        probs = top_p_filtering(probs, top_p)

        # Sample next tokens
        next_tokens = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Replace with pad for finished sequences
        next_tokens[finished] = pad_token_id

        # Append to generated sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Update finished status
        if eos_token_id is not None:
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            if finished.all():
                break

    return generated
