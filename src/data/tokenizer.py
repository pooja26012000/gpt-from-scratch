"""
A from-scratch Byte-Pair Encoding (BPE) tokenizer, trained on chess
PGN move text.

Pipeline: strip PGN headers -> build character vocab -> count word
frequencies -> train BPE merges (weighted by word frequency for
speed) -> encode/decode -> save trained tokenizer to disk.
"""
import os
import json
import time
from collections import Counter

import chess.pgn


# ---------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------

def extract_movetext(path):
    """
    Parse the PGN file game by game and extract just the move text
    (with trailing result), discarding all header metadata.
    """
    movetexts = []
    with open(path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
            movetext = game.accept(exporter).strip()
            movetexts.append(movetext)
    return "\n".join(movetexts)


# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------

def get_base_vocab(corpus):
    """Get the set of unique characters in the corpus, sorted for determinism."""
    return sorted(set(corpus))


def build_full_vocab(base_vocab, merges):
    """
    Build the complete vocabulary: base characters plus every merged
    token, in the order they were created (so IDs are stable/deterministic).
    """
    return list(base_vocab) + [new_token for (pair, new_token) in merges]


def build_stoi_itos(vocab):
    """Build token-to-id and id-to-token mappings."""
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for i, tok in enumerate(vocab)}
    return stoi, itos


# ---------------------------------------------------------------------
# BPE core: pair counting and merging
# ---------------------------------------------------------------------

def merge_pair(tokens, pair, new_token):
    """
    Replace every occurrence of `pair` (adjacent tokens) in `tokens`
    with a single `new_token`. Returns the new, shorter token list.
    """
    merged = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged.append(new_token)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def get_word_freqs(corpus):
    """
    Split the corpus into whitespace-separated words and count how
    often each unique word occurs. Returns dict: word -> count.
    """
    return Counter(corpus.split())


def word_to_symbols(word):
    """Represent a word as a tuple of its individual characters."""
    return tuple(word)


def get_weighted_pair_counts(word_freqs, word_symbols):
    """
    Count pairs across all unique words, weighted by each word's
    frequency in the corpus (avoids rescanning the full raw corpus
    on every merge iteration).

    word_freqs: dict, word (str) -> frequency count
    word_symbols: dict, word (str) -> tuple of current symbols/tokens

    Returns dict: (symbol_a, symbol_b) -> total weighted count
    """
    pair_counts = {}
    for word, freq in word_freqs.items():
        symbols = word_symbols[word]
        for a, b in zip(symbols, symbols[1:]):
            pair = (a, b)
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
    return pair_counts


def train_bpe(word_freqs, num_merges):
    """
    Train BPE merges on word-frequency data.

    Returns:
        word_symbols: dict, word -> final tuple of learned tokens
        merges: ordered list of (pair, new_token) applied, in the
                exact order they were learned (order matters for encode)
    """
    word_symbols = {word: word_to_symbols(word) for word in word_freqs}
    merges = []

    for i in range(num_merges):
        pair_counts = get_weighted_pair_counts(word_freqs, word_symbols)
        if not pair_counts:
            print("No more pairs to merge.")
            break

        best_pair = max(pair_counts, key=pair_counts.get)
        new_token = best_pair[0] + best_pair[1]

        for word in word_symbols:
            word_symbols[word] = tuple(
                merge_pair(list(word_symbols[word]), best_pair, new_token)
            )

        merges.append((best_pair, new_token))

        if (i + 1) % 100 == 0:
            print(f"Merge {i+1}/{num_merges}: {best_pair} -> '{new_token}' (count: {pair_counts[best_pair]})")

    return word_symbols, merges


# ---------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------

def encode(text, merges):
    """
    Tokenize new text using the trained BPE merges, applied in the
    exact order they were learned during training. Returns a list of
    token lists, one per whitespace-separated word.
    """
    words = text.split()
    encoded_words = []
    for word in words:
        symbols = list(word)
        for pair, new_token in merges:
            symbols = merge_pair(symbols, pair, new_token)
        encoded_words.append(symbols)
    return encoded_words


def encode_to_ids(text, merges, stoi):
    """Tokenize text with trained BPE merges and flatten to integer IDs."""
    ids = []
    for word_tokens in encode(text, merges):
        for token in word_tokens:
            ids.append(stoi[token])
    return ids


def decode(ids, itos):
    """Convert a list of token IDs back into text."""
    return " ".join(itos[i] for i in ids)


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def save_tokenizer(base_vocab, merges, path):
    """
    Save the trained tokenizer (base vocab + merges) to a JSON file.
    Merges are saved as a list of [[a, b], new_token] since JSON
    doesn't support tuples directly.
    """
    data = {
        "base_vocab": base_vocab,
        "merges": [[list(pair), new_token] for pair, new_token in merges],
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Tokenizer saved to {path}")


# ---------------------------------------------------------------------
# Train and save
# ---------------------------------------------------------------------

if __name__ == "__main__":
    NUM_MERGES = 1500

    movetext_corpus = extract_movetext("data/raw/games.pgn")
    print(f"Move-only corpus: {len(movetext_corpus)} characters")

    base_vocab = get_base_vocab(movetext_corpus)
    print(f"Base (character) vocabulary size: {len(base_vocab)}")

    word_freqs = get_word_freqs(movetext_corpus)
    print(f"Unique words: {len(word_freqs)} (from {sum(word_freqs.values())} total)")

    start = time.time()
    word_symbols, merges = train_bpe(word_freqs, num_merges=NUM_MERGES)
    print(f"Training took {time.time() - start:.1f}s")

    total_chars = sum(len(word) * freq for word, freq in word_freqs.items())
    total_tokens = sum(len(word_symbols[word]) * freq for word, freq in word_freqs.items())
    print(f"Compression ratio: {total_chars / total_tokens:.2f}x "
          f"({total_chars} chars -> {total_tokens} tokens)")

    full_vocab = build_full_vocab(base_vocab, merges)
    stoi, itos = build_stoi_itos(full_vocab)
    print(f"Final vocabulary size: {len(full_vocab)}")

    # Sanity check: encode -> decode should round-trip exactly
    sample_text = "1. e4 e5 2. Nf3 Nc6"
    ids = encode_to_ids(sample_text, merges, stoi)
    decoded_text = decode(ids, itos)
    print(f"\nRound-trip check: '{sample_text}' -> {ids} -> '{decoded_text}'")
    assert decoded_text == sample_text, "Encode/decode round-trip failed!"
    print("Round-trip OK.")

    os.makedirs("data/processed", exist_ok=True)
    save_tokenizer(base_vocab, merges, "data/processed/tokenizer.json")
