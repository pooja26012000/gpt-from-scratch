import torch

from src.model.gpt import GPT
from src.data.tokenizer import build_full_vocab, build_stoi_itos
from src.train.train import load_trained_tokenizer


def generate_greedy(model, prompt_ids, max_new_tokens, device):
    """
    Generate tokens by always picking the single highest-probability
    next token at each step ("greedy decoding").
    """
    model.eval()
    ids = prompt_ids.to(device)

    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)

    return ids

if __name__ == "__main__":
    base_vocab, merges = load_trained_tokenizer("data/processed/tokenizer.json")
    full_vocab = build_full_vocab(base_vocab, merges)
    stoi, itos = build_stoi_itos(full_vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(vocab_size=len(full_vocab), embedding_dim=128, num_heads=4, num_layers=4, max_seq_len=128)
    model.load_state_dict(torch.load("checkpoints/gpt_final.pt", map_location=device))
    model.to(device)

    prompt = "1. e4"
    from src.data.tokenizer import encode
    prompt_tokens = encode(prompt, merges)
    prompt_ids = torch.tensor([[stoi[tok] for word in prompt_tokens for tok in word]], dtype=torch.long)

    print(f"Prompt: '{prompt}'")
    print(f"Prompt IDs shape: {prompt_ids.shape}")

    generated_ids = generate_greedy(model, prompt_ids, max_new_tokens=20, device=device)
    print(f"Generated IDs shape: {generated_ids.shape}")

    generated_tokens = [itos[i] for i in generated_ids[0].tolist()]
    print(f"Generated text: {' '.join(generated_tokens)}")