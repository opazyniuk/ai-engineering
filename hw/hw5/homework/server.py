"""
Крок 1.4 (частина 2): FastAPI сервер для inference.

Завантажує checkpoint.pt при старті, обслуговує POST /generate.
"""
import time
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

import nano_gpt as ng


CKPT_PATH = "checkpoint.pt"


# ----- Pydantic схеми -----
class GenerateRequest(BaseModel):
    prompt: str = Field(default="", description="Початковий текст. Пустий = старт з нуля.")
    max_tokens: int = Field(default=200, ge=1, le=2000, description="Скільки символів згенерувати.")
    temperature: float = Field(default=1.0, gt=0.0, le=3.0, description="0.5=обережно, 1.5=креативно.")
    top_k: Optional[int] = Field(default=None, ge=1, description="Обрізати до top_k символів. None=всі.")


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str
    tokens_generated: int
    gen_time_s: float
    tokens_per_s: float


# ----- Завантаження моделі -----
print(f"loading checkpoint from {CKPT_PATH}...")
ckpt = torch.load(CKPT_PATH, map_location=ng.device, weights_only=False)
stoi = ckpt["vocab"]["stoi"]
itos = ckpt["vocab"]["itos"]

model = ng.NanoGPT().to(ng.device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"model loaded: {sum(p.numel() for p in model.parameters()):,} params on {ng.device}")
print(f"vocab_size={len(stoi)} block_size={ng.BLOCK_SIZE}")


def encode_local(s: str) -> list[int]:
    return [stoi[c] for c in s if c in stoi]


def decode_local(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


# ----- FastAPI -----
app = FastAPI(
    title="nano-GPT inference",
    description="Char-level GPT на українському корпусі. /generate генерує текст.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "model": "nano-GPT",
        "device": str(ng.device),
        "params": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(stoi),
        "block_size": ng.BLOCK_SIZE,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    prompt_ids = encode_local(req.prompt) if req.prompt else [0]
    ctx = torch.tensor([prompt_ids], dtype=torch.long, device=ng.device)

    t0 = time.perf_counter()
    out = model.generate(
        ctx,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    gen_time = time.perf_counter() - t0

    out_ids = out[0].tolist()
    full = decode_local(out_ids)
    completion = full[len(req.prompt):] if req.prompt else full

    return GenerateResponse(
        prompt=req.prompt,
        completion=completion,
        full_text=full,
        tokens_generated=req.max_tokens,
        gen_time_s=gen_time,
        tokens_per_s=req.max_tokens / gen_time,
    )
