"""
Text generation for the from-scratch GPT model.

Four decoding strategies, each trading off safety vs. variety:
    generate_greedy           - always pick the single most likely token
    generate_with_temperature - sample from the full distribution,
                                 temperature controls how random
    generate_with_top_k       - sample only among the k most likely
                                 tokens at each step
    generate_with_top_p       - sample from the smallest set of tokens
                                 whose cumulative probability exceeds p
                                 (adapts pool size to model confidence)

All four repeatedly: run the model on the current sequence, look at
the prediction for the last position, pick a next token, append it,
and repeat.
"""
import torch

from src.model.gpt import GPT
from src.data.tokenizer import build_full_vocab, build_stoi_itos, encode
from src.train.train import load_trained_tokenizer


def generate_greedy(model, prompt_ids, max_new_tokens, device):
    """Always pick the single highest-probability next token."""
    model.eval()
    ids = prompt_ids.to(device)

    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)

    return ids


def generate_with_temperature(model, prompt_ids, max_new_tokens, temperature, device):
    """Sample from the full probability distribution; temperature
    controls how sharp (low T) or flat (high T) that distribution is."""
    model.eval()
    ids = prompt_ids.to(device)

    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_token], dim=1)

    return ids


def generate_with_top_k(model, prompt_ids, max_new_tokens, k, temperature, device):
    """Sample only among the top-k most likely candidates at each step."""
    model.eval()
    ids = prompt_ids.to(device)

    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        next_token_logits = logits[:, -1, :] / temperature

        top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)
        probs = torch.softmax(top_k_logits, dim=-1)
        sampled_index = torch.multinomial(probs, num_samples=1)
        next_token = torch.gather(top_k_indices, dim=-1, index=sampled_index)

        ids = torch.cat([ids, next_token], dim=1)

    return ids


def generate_with_top_p(model, prompt_ids, max_new_tokens, p, temperature, device):
    """Sample from the smallest set of candidates whose cumulative
    probability exceeds p (the 'nucleus') — pool size adapts to how
    confident the model is at each step."""
    model.eval()
    ids = prompt_ids.to(device)

    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        cutoff_mask = cumulative_probs - sorted_probs > p
        sorted_probs[cutoff_mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled_index = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_indices, dim=-1, index=sampled_index)

        ids = torch.cat([ids, next_token], dim=1)

    return ids


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    """Load the trained tokenizer and a trained GPT checkpoint together."""
    base_vocab, merges = load_trained_tokenizer(tokenizer_path)
    full_vocab = build_full_vocab(base_vocab, merges)
    stoi, itos = build_stoi_itos(full_vocab)

    model = GPT(vocab_size=len(full_vocab), embedding_dim=128, num_heads=4, num_layers=4, max_seq_len=128)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    return model, merges, stoi, itos


def prompt_to_ids(prompt, merges, stoi, device):
    """Encode a text prompt into a batch-of-1 tensor of token IDs."""
    word_token_lists = encode(prompt, merges)
    ids = [stoi[tok] for word in word_token_lists for tok in word]
    return torch.tensor([ids], dtype=torch.long).to(device)


def decode_ids(ids, itos):
    return ' '.join(itos[i] for i in ids)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, merges, stoi, itos = load_model_and_tokenizer(
        "checkpoints/gpt_final.pt", "data/processed/tokenizer.json", device
    )

    prompt = "1. e4"
    prompt_ids = prompt_to_ids(prompt, merges, stoi, device)
    print(f"Prompt: '{prompt}'")

    greedy_ids = generate_greedy(model, prompt_ids, max_new_tokens=20, device=device)
    print(f"\nGreedy:\n{decode_ids(greedy_ids[0].tolist(), itos)}")

    temp_ids = generate_with_temperature(model, prompt_ids, max_new_tokens=20, temperature=1.0, device=device)
    print(f"\nTemperature (T=1.0):\n{decode_ids(temp_ids[0].tolist(), itos)}")

    top_k_ids = generate_with_top_k(model, prompt_ids, max_new_tokens=20, k=10, temperature=1.0, device=device)
    print(f"\nTop-k (k=10):\n{decode_ids(top_k_ids[0].tolist(), itos)}")

    top_p_ids = generate_with_top_p(model, prompt_ids, max_new_tokens=20, p=0.9, temperature=1.0, device=device)
    print(f"\nTop-p (p=0.9):\n{decode_ids(top_p_ids[0].tolist(), itos)}")