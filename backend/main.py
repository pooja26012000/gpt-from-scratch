"""
FastAPI backend serving the trained GPT-Chess model. Wraps the
existing generation functions from src/model/generate.py behind a
simple HTTP API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch

from src.model.generate import (
    load_model_and_tokenizer, prompt_to_ids, decode_ids,
    generate_greedy, generate_with_temperature, generate_with_top_k, generate_with_top_p
)

app = FastAPI(title="GPT-Chess-From-Scratch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, merges, stoi, itos = load_model_and_tokenizer(
    "checkpoints/gpt_final.pt", "data/processed/tokenizer.json", device
)


class GenerateRequest(BaseModel):
    prompt: str = "1. e4"
    strategy: str = "top_k"
    max_new_tokens: int = 40
    temperature: float = 1.0
    k: int = 10
    p: float = 0.9


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/generate")
def generate(request: GenerateRequest):
    prompt_ids = prompt_to_ids(request.prompt, merges, stoi, device)

    if request.strategy == "greedy":
        ids = generate_greedy(model, prompt_ids, request.max_new_tokens, device)
    elif request.strategy == "temperature":
        ids = generate_with_temperature(model, prompt_ids, request.max_new_tokens, request.temperature, device)
    elif request.strategy == "top_k":
        ids = generate_with_top_k(model, prompt_ids, request.max_new_tokens, request.k, request.temperature, device)
    elif request.strategy == "top_p":
        ids = generate_with_top_p(model, prompt_ids, request.max_new_tokens, request.p, request.temperature, device)
    else:
        return {"error": f"Unknown strategy: {request.strategy}"}

    generated_text = decode_ids(ids[0].tolist(), itos)
    return {"prompt": request.prompt, "strategy": request.strategy, "generated_text": generated_text}
