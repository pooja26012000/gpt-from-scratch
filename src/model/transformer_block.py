"""
The full transformer block, built from scratch, for the GPT model.

Builds up in stages:
    SingleHeadAttention  -> scaled dot-product attention with causal masking
    MultiHeadAttention   -> several SingleHeadAttention instances in parallel,
                             concatenated and projected back to embedding_dim
    FeedForward           -> per-token MLP (expand -> GELU -> contract)
    TransformerBlock      -> attention + feed-forward, each wrapped in a
                             Pre-LN residual connection:
                                 x = x + MultiHeadAttention(LayerNorm(x))
                                 x = x + FeedForward(LayerNorm(x))

Every piece preserves the input shape [batch_size, seq_len, embedding_dim],
which is what lets TransformerBlocks be stacked on top of each other.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EMBEDDING_DIM = 128
HEAD_DIM = 128  # for single-head attention, head_dim == embedding_dim
NUM_HEADS = 4
MAX_SEQ_LEN = 128


class SingleHeadAttention(nn.Module):
    """
    Single-head scaled dot-product self-attention with causal masking.
    Input: [batch_size, seq_len, embedding_dim]
    Output: [batch_size, seq_len, head_dim]
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


class MultiHeadAttention(nn.Module):
    """
    Runs multiple SingleHeadAttention instances in parallel, each
    operating on a smaller slice of the embedding dimension, then
    concatenates their outputs and passes through an output projection.

    Input: [batch_size, seq_len, embedding_dim]
    Output: [batch_size, seq_len, embedding_dim]
    """
    def __init__(self, embedding_dim, num_heads, max_seq_len):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            SingleHeadAttention(embedding_dim, head_dim, max_seq_len)
            for _ in range(num_heads)
        ])
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.output_proj(concatenated)


class FeedForward(nn.Module):
    """
    A small per-token MLP applied identically at every position:
    Linear -> GELU -> Linear, expanding to a larger hidden dimension
    in between (standard GPT convention: 4x embedding_dim).

    Input: [batch_size, seq_len, embedding_dim]
    Output: [batch_size, seq_len, embedding_dim]
    """
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One full transformer block: multi-head attention and feed-forward,
    each wrapped in a Pre-LN residual connection:

        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Pre-LN (normalize before each sub-layer, add to the raw residual)
    keeps the residual path unobstructed, which is what lets gradients
    flow cleanly through many stacked blocks during training.

    Input: [batch_size, seq_len, embedding_dim]
    Output: [batch_size, seq_len, embedding_dim]
    """
    def __init__(self, embedding_dim, num_heads, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, hidden_dim=4 * embedding_dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


if __name__ == "__main__":
    BATCH_SIZE = 4
    SEQ_LEN = 8
    fake_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)

    # --- Single-head attention ---
    single_head = SingleHeadAttention(EMBEDDING_DIM, HEAD_DIM, max_seq_len=MAX_SEQ_LEN)
    sh_output = single_head(fake_input)
    print(f"SingleHeadAttention output shape: {sh_output.shape}")
    assert sh_output.shape == fake_input.shape
    print("Single-head shape check OK.")

    # Sanity check causal masking: the first token can only attend to
    # itself, so its attention weights should be one-hot on position 0.
    Q = single_head.query_proj(fake_input)
    K = single_head.key_proj(fake_input)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(HEAD_DIM)
    mask = single_head.causal_mask[:SEQ_LEN, :SEQ_LEN]
    scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    print(f"First token's attention weights (should be one-hot on position 0):\n{weights[0][0]}")

    # --- Multi-head attention ---
    multi_head = MultiHeadAttention(EMBEDDING_DIM, NUM_HEADS, max_seq_len=MAX_SEQ_LEN)
    mh_output = multi_head(fake_input)
    print(f"\nMultiHeadAttention output shape: {mh_output.shape}")
    assert mh_output.shape == fake_input.shape
    print("Multi-head shape check OK.")

    # --- Feed-forward ---
    ffn = FeedForward(EMBEDDING_DIM, hidden_dim=4 * EMBEDDING_DIM)
    ffn_output = ffn(mh_output)
    print(f"\nFeedForward output shape: {ffn_output.shape}")
    assert ffn_output.shape == fake_input.shape
    print("FeedForward shape check OK.")

    # --- Full transformer block ---
    block = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, max_seq_len=MAX_SEQ_LEN)
    block_output = block(fake_input)
    print(f"\nTransformerBlock output shape: {block_output.shape}")
    assert block_output.shape == fake_input.shape
    print("TransformerBlock shape check OK.")