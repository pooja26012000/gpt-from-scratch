"""
Token + positional embeddings for the from-scratch GPT model.

Token embeddings map each token ID to a learnable vector ("what is
this token"). Positional embeddings map each sequence position to a
learnable vector ("where in the sequence is this token"). The two are
added together (broadcasting the position embeddings across the batch
dimension) so the model has both pieces of information at once.
"""
import torch
import torch.nn as nn


VOCAB_SIZE = 1534
EMBEDDING_DIM = 128
MAX_SEQ_LEN = 128


class GPTEmbedding(nn.Module):
    """
    Combines token embeddings and learned positional embeddings.
    Input: token IDs, shape [batch_size, seq_len]
    Output: combined embeddings, shape [batch_size, seq_len, embedding_dim]
    """
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        token_embeds = self.token_embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_embeds = self.position_embedding(positions)
        return token_embeds + pos_embeds


if __name__ == "__main__":
    BATCH_SIZE = 4
    SEQ_LEN = 8

    fake_batch = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    embedding_layer = GPTEmbedding(VOCAB_SIZE, EMBEDDING_DIM, MAX_SEQ_LEN)
    output = embedding_layer(fake_batch)

    print(f"Input shape: {fake_batch.shape}")
    print(f"GPTEmbedding output shape: {output.shape}")
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
    print("Shape check OK.")