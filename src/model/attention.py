"""
Single-head self-attention, built from scratch, for the GPT model.

Implements scaled dot-product attention with causal masking:

    Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + causal_mask ) @ V

Q, K, V are learned linear projections of the input embeddings. The
causal mask ensures token i can only attend to tokens 0..i, never
future tokens — required for next-token prediction to be meaningful.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EMBEDDING_DIM = 128
HEAD_DIM = 128  # for single-head attention, head_dim == embedding_dim
MAX_SEQ_LEN = 128


class SingleHeadAttention(nn.Module):
    """
    Single-head scaled dot-product self-attention with causal masking.
    Input: [batch_size, seq_len, embedding_dim]
    Output: [batch_size, seq_len, head_dim]  (same shape as input when
             head_dim == embedding_dim, which lets attention layers
             be stacked / slotted into a larger architecture later)
    """
    def __init__(self, embedding_dim, head_dim, max_seq_len):
        super().__init__()
        self.query_proj = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key_proj = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embedding_dim, head_dim, bias=False)
        self.head_dim = head_dim

        # Precompute the causal mask once; register as a buffer so it
        # moves to GPU automatically with the rest of the model, but
        # isn't treated as a learnable parameter.
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ V
        return output


if __name__ == "__main__":
    BATCH_SIZE = 4
    SEQ_LEN = 8
    fake_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)

    single_head = SingleHeadAttention(EMBEDDING_DIM, HEAD_DIM, max_seq_len=MAX_SEQ_LEN)
    output = single_head(fake_input)

    print(f"Input shape: {fake_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == fake_input.shape
    print("Shape check OK.")

    # Sanity check that causal masking is actually working: the first
    # token can only attend to itself, so its attention weights should
    # be a one-hot vector (all weight on position 0).
    Q = single_head.query_proj(fake_input)
    K = single_head.key_proj(fake_input)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(HEAD_DIM)
    mask = single_head.causal_mask[:SEQ_LEN, :SEQ_LEN]
    scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    print(f"\nFirst token's attention weights (should be one-hot on position 0):\n{weights[0][0]}")