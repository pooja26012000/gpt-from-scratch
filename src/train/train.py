import json

import torch
import torch.optim as optim

import math

from src.data.tokenizer import extract_movetext, encode, build_full_vocab, build_stoi_itos
from src.model.gpt import GPT

def load_trained_tokenizer(path):
    """Load the base vocab and merges saved by tokenizer.py."""
    with open(path) as f:
        data = json.load(f)
    base_vocab = data["base_vocab"]
    merges = [(tuple(pair), new_token) for pair, new_token in data["merges"]]
    return base_vocab, merges

def encode_corpus_efficiently(text, merges, stoi):
    """
    Encode a large corpus efficiently by encoding each unique word
    only once (cached), then reusing that result for every occurrence
    — avoids redundantly re-running merges on repeated words.
    """
    words = text.split()
    cache = {}
    all_ids = []
    for word in words:
        if word not in cache:
            word_tokens = encode(word, merges)[0]  # encode() returns a list of one word's tokens
            cache[word] = [stoi[tok] for tok in word_tokens]
        all_ids.extend(cache[word])
    return all_ids

def get_batch(data, batch_size, seq_len, device):
    """
    Sample a random batch of (input, target) pairs from the tokenized
    data, where target is input shifted one position to the right.
    """
    max_start = len(data) - seq_len - 1
    start_indices = torch.randint(0, max_start, (batch_size,))

    inputs = torch.stack([data[i : i + seq_len] for i in start_indices])
    targets = torch.stack([data[i + 1 : i + seq_len + 1] for i in start_indices])

    return inputs.to(device), targets.to(device)

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """
    Linear warmup for `warmup_steps`, then cosine decay down to
    `min_lr` by `max_steps`.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")

def train(model, data, batch_size, seq_len, num_steps, max_lr, min_lr, warmup_steps, device, checkpoint_every=500):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr)

    for step in range(num_steps):
        lr = get_lr(step, warmup_steps, num_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        xb, yb = get_batch(data, batch_size, seq_len, device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}/{num_steps} — loss: {loss.item():.4f} — lr: {lr:.6f}")

        if step > 0 and step % checkpoint_every == 0:
            save_checkpoint(model, f"checkpoints/gpt_step_{step}.pt")

    save_checkpoint(model, "checkpoints/gpt_final.pt")
    return model

if __name__ == "__main__":
    base_vocab, merges = load_trained_tokenizer("data/processed/tokenizer.json")
    full_vocab = build_full_vocab(base_vocab, merges)
    stoi, itos = build_stoi_itos(full_vocab)
    print(f"Loaded vocabulary size: {len(full_vocab)}")

    movetext_corpus = extract_movetext("data/raw/games.pgn")
    all_ids = encode_corpus_efficiently(movetext_corpus, merges, stoi)
    data = torch.tensor(all_ids, dtype=torch.long)
    print(f"Total tokens in encoded corpus: {len(data)}")

    BATCH_SIZE = 4
    SEQ_LEN = 8
    xb, yb = get_batch(data, BATCH_SIZE, SEQ_LEN, device="cpu")
    print(f"\nInput batch shape: {xb.shape}")
    print(f"Target batch shape: {yb.shape}")
    print(f"First input sequence (token IDs): {xb[0]}")
    print(f"First target sequence (token IDs): {yb[0]}")
    print(f"Decoded input:  {[itos[i.item()] for i in xb[0]]}")
    print(f"Decoded target: {[itos[i.item()] for i in yb[0]]}")

    model = GPT(vocab_size=len(full_vocab), embedding_dim=128, num_heads=4, num_layers=4, max_seq_len=128)
    logits, loss = model(xb, yb)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    print("\nStarting training run...")
    trained_model = train(
        model, data,
        batch_size=32, seq_len=128, num_steps=5000,
        max_lr=3e-4, min_lr=3e-5, warmup_steps=100,
        device=device,
        checkpoint_every=500
    )
    print("\nTraining loop finished.")

     # Quick before/after check: run one more forward pass post-training
    xb_check, yb_check = get_batch(data, batch_size=32, seq_len=64, device="cpu")
    _, final_loss = trained_model(xb_check, yb_check)
    print(f"Loss on a fresh batch after training: {final_loss.item():.4f}")