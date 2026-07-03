import chess
import torch

from src.model.generate import (
    load_model_and_tokenizer, prompt_to_ids, decode_ids,
    generate_greedy, generate_with_temperature, generate_with_top_k, generate_with_top_p
)


def extract_moves(generated_text):
    """
    Strip move-number tokens (e.g. '1.', '23.') from generated text,
    leaving just the actual move tokens in order.
    """
    tokens = generated_text.split()
    moves = [tok for tok in tokens if not tok.endswith('.')]
    return moves


def check_legality(moves, verbose=False):
    board = chess.Board()
    legal_count = 0
    for i, move in enumerate(moves):
        try:
            board.push_san(move)
            legal_count += 1
        except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError) as e:
            if verbose:
                print(f"Broke at move {i}: '{move}' — {type(e).__name__}: {e}")
            break
    return legal_count, len(moves)

def evaluate_strategy(model, prompt_ids, generate_fn, generate_kwargs, num_samples, itos):
    """
    Generate `num_samples` games using the given generation function,
    check legality on each, and return aggregate statistics.
    """
    results = []
    for _ in range(num_samples):
        generated_ids = generate_fn(model, prompt_ids, **generate_kwargs)
        generated_text = decode_ids(generated_ids[0].tolist(), itos)
        moves = extract_moves(generated_text)
        legal, total = check_legality(moves)
        results.append((legal, total))

    legal_fractions = [legal / total for legal, total in results if total > 0]
    avg_legal_fraction = sum(legal_fractions) / len(legal_fractions)
    avg_moves_before_break = sum(legal for legal, _ in results) / len(results)

    return {
        "avg_legal_fraction": avg_legal_fraction,
        "avg_moves_before_break": avg_moves_before_break,
        "raw_results": results,
    }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, merges, stoi, itos = load_model_and_tokenizer(
        "checkpoints/gpt_final.pt", "data/processed/tokenizer.json", device
    )
    prompt_ids = prompt_to_ids("1. e4", merges, stoi, device)

    NUM_SAMPLES = 30
    MAX_NEW_TOKENS = 30

    strategies = {
        "Greedy": (generate_greedy, {"max_new_tokens": MAX_NEW_TOKENS, "device": device}),
        "Temperature (T=1.0)": (generate_with_temperature, {"max_new_tokens": MAX_NEW_TOKENS, "temperature": 1.0, "device": device}),
        "Top-k (k=10)": (generate_with_top_k, {"max_new_tokens": MAX_NEW_TOKENS, "k": 10, "temperature": 1.0, "device": device}),
        "Top-p (p=0.9)": (generate_with_top_p, {"max_new_tokens": MAX_NEW_TOKENS, "p": 0.9, "temperature": 1.0, "device": device}),
    }

    for name, (fn, kwargs) in strategies.items():
        stats = evaluate_strategy(model, prompt_ids, fn, kwargs, NUM_SAMPLES, itos)
        print(f"{name}: avg legal fraction = {stats['avg_legal_fraction']:.2%}, "
              f"avg moves before break = {stats['avg_moves_before_break']:.1f}")