# src/model/gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.transformer_block import TransformerBlock


VOCAB_SIZE = 1534
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
MAX_SEQ_LEN = 128


class GPT(nn.Module):
    """
    The full GPT model: embeddings -> stack of transformer blocks ->
    final layer norm -> output projection to vocabulary logits.

    Input: token IDs, shape [batch_size, seq_len]
    Output: logits, shape [batch_size, seq_len, vocab_size]
    """
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, max_seq_len)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids, targets=None):
        batch_size, seq_len = token_ids.shape
        token_embeds = self.token_embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.output_proj(x)

        loss = None
        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
    
if __name__ == "__main__":
    BATCH_SIZE = 4
    SEQ_LEN = 8
    fake_token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    model = GPT(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN)
    # logits = model(fake_token_ids)

    logits, loss = model(fake_token_ids)
    print(f"Input shape: {fake_token_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    print("Shape check OK.")
    print(f"Loss (no targets provided): {loss}")  # should print None

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")