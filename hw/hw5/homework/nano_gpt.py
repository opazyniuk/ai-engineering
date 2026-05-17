"""
nano-GPT: мінімальна реалізація GPT на рівні символів.

Структура коду відповідає крокам у візуалізаторі (gpt_visualizer.html):
  1. Токенізація      — рядки 20-45
  2. Embeddings        — NanoGPT.__init__ (tok_emb + pos_emb)
  3. Self-Attention    — CausalSelfAttention
  4. Multi-Head        — розбивка на N_HEAD голів у CausalSelfAttention.forward
  5. Causal Mask       — register_buffer("mask", ...) + masked_fill
  6. Transformer Block — Block (LayerNorm → Attention → +Residual → LayerNorm → MLP → +Residual)
  7. Генерація         — NanoGPT.generate
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# ==============================================================
# Конфігурація моделі
# ==============================================================
BLOCK_SIZE = 128    # максимальна довжина контексту (скільки токенів модель бачить)
BATCH_SIZE = 64     # кількість прикладів в одному батчі
N_EMBED    = 192    # розмірність embedding-вектора кожного токена
N_HEAD     = 6      # кількість attention-голів (кожна працює з N_EMBED/N_HEAD = 32 dim)
N_LAYER    = 4      # кількість transformer-блоків (кожен = attention + MLP)
DROPOUT    = 0.2    # ймовірність dropout (регуляризація при тренуванні)
LR         = 3e-4   # learning rate для оптимізатора
MAX_ITERS  = 5000   # скільки кроків тренувати
EVAL_INTERVAL = 250 # кожні N кроків — оцінити val loss
EVAL_ITERS    = 40  # скільки батчів для оцінки loss
SEED = 1337

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

# ==============================================================
# Крок 1: ТОКЕНІЗАЦІЯ
# Візуалізатор: таб "1. Токенізація"
#
# Перетворюємо текст на числа. Кожен унікальний символ отримує
# свій індекс. Це найпростіший токенізатор — char-level.
# У реальних LLM використовують BPE (subword), але принцип той самий.
# ==============================================================
DATA_PATH = Path(__file__).parent / "training_text.txt"
TEXT = DATA_PATH.read_text(encoding="utf-8")

chars = sorted(set(TEXT))           # усі унікальні символи, відсортовані
vocab_size = len(chars)             # розмір словника (скільки різних "токенів" існує)
stoi = {c: i for i, c in enumerate(chars)}  # символ → число ("A" → 0, "B" → 1, ...)
itos = {i: c for i, c in enumerate(chars)}  # число → символ (0 → "A", 1 → "B", ...)
encode = lambda s: [stoi[c] for c in s]     # "hello" → [7, 4, 11, 11, 14]
decode = lambda ids: "".join(itos[i] for i in ids)  # [7, 4, 11, 11, 14] → "hello"

# Перетворюємо весь текст у тензор чисел і ділимо на train/val
data = torch.tensor(encode(TEXT), dtype=torch.long)  # shape: (len(TEXT),)
n = int(0.9 * len(data))                             # 90% на тренування
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    """Вибираємо випадкові шматки тексту для тренування/валідації."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))  # випадкові стартові позиції
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])         # вхід: символи 0..127
    y = torch.stack([d[i + 1 : i + 1 + BLOCK_SIZE] for i in ix]) # цілі: символи 1..128 (зсув на 1)
    return x.to(device), y.to(device)


# ==============================================================
# Кроки 3-5: SELF-ATTENTION (з Multi-Head і Causal Mask)
# ==============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Одна матриця генерує Q, K, V одночасно (ефективніше ніж 3 окремі)
        self.qkv = nn.Linear(N_EMBED, 3 * N_EMBED)   # (192) → (576) = 3 × 192
        # Фінальна проєкція після конкатенації голів
        self.proj = nn.Linear(N_EMBED, N_EMBED)        # (192) → (192)
        self.drop = nn.Dropout(DROPOUT)

        # ---- Крок 5: Causal Mask ----
        # Нижня трикутна матриця: токен i може бачити тільки позиції 0..i
        # register_buffer = зберігається з моделлю, але не тренується
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))  # трикутна матриця
            .view(1, 1, BLOCK_SIZE, BLOCK_SIZE)              # (1, 1, T, T) для broadcasting
        )

    def forward(self, x):
        B, T, C = x.shape  # B=batch, T=seq_len, C=N_EMBED(192)

        # ---- Крок 3: генеруємо Q, K, V ----
        # Кожен токен отримує три вектори:
        #   Q (Query)  = "що я шукаю?"
        #   K (Key)    = "що я пропоную?"
        #   V (Value)  = "яку інформацію я несу?"
        q, k, v = self.qkv(x).split(N_EMBED, dim=2)  # кожен: (B, T, 192)

        # ---- Крок 4: Multi-Head — ділимо на 6 голів по 32 dim ----
        head_dim = C // N_HEAD  # 192 / 6 = 32
        q = q.view(B, T, N_HEAD, head_dim).transpose(1, 2)  # (B, 6, T, 32)
        k = k.view(B, T, N_HEAD, head_dim).transpose(1, 2)  # (B, 6, T, 32)
        v = v.view(B, T, N_HEAD, head_dim).transpose(1, 2)  # (B, 6, T, 32)

        # ---- Крок 3: Q · K^T → attention scores ----
        # Скалярний добуток Q і K: "наскільки токен i цікавиться токеном j?"
        att = (q @ k.transpose(-2, -1))   # (B, 6, T, T) — матриця "хто на кого дивиться"
        att = att / math.sqrt(head_dim)    # ділимо на √32 щоб softmax не насичувався

        # ---- Крок 5: Causal Mask — блокуємо майбутнє ----
        # Ставимо -∞ туди, де маска = 0 (верхній трикутник)
        # Після softmax -∞ перетворюється на 0 (нульова увага)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # ---- Крок 3: softmax → ймовірності ----
        att = F.softmax(att, dim=-1)  # рядки тепер сумуються до 1.0
        self.last_attn = att.detach()  # зберігаємо для візуалізації (крок 2.2 heatmap)
        att = self.drop(att)           # dropout деяких зв'язків (тільки при тренуванні)

        # ---- Крок 3: зважена сума V ----
        # Кожен токен стає зваженою комбінацією Value-векторів інших токенів
        y = att @ v  # (B, 6, T, 32) — новий embedding для кожного токена

        # ---- Крок 4: збираємо голови назад ----
        # Конкатенуємо 6 голів по 32 dim → 192 dim
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, 192)

        # Фінальна проєкція — змішує інформацію від різних голів
        return self.proj(y)  # (B, T, 192)


# ==============================================================
# Крок 6: TRANSFORMER BLOCK
#
# Один блок = Attention + MLP, кожен з LayerNorm і Residual.
# У нашій моделі 4 такі блоки послідовно.
# ==============================================================
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(N_EMBED)         # нормалізація перед attention
        self.attn = CausalSelfAttention()          # multi-head self-attention
        self.ln2  = nn.LayerNorm(N_EMBED)          # нормалізація перед MLP
        self.mlp  = nn.Sequential(                 # feed-forward мережа
            nn.Linear(N_EMBED, 4 * N_EMBED),       #   розширення: 192 → 768 ("простір для думання")
            nn.GELU(),                              #   активація (нелінійність)
            nn.Linear(4 * N_EMBED, N_EMBED),        #   стиснення: 768 → 192
            nn.Dropout(DROPOUT),                     #   регуляризація
        )

    def forward(self, x):
        # Attention + Residual: x залишається як "шосе", attention додає контекст
        x = x + self.attn(self.ln1(x))  # LayerNorm → Attention → + оригінал

        # MLP + Residual: MLP "думає" над кожним токеном окремо
        x = x + self.mlp(self.ln2(x))   # LayerNorm → MLP → + попередній

        return x  # shape не змінився: (B, T, 192) → (B, T, 192)


# ==============================================================
# ПОВНА МОДЕЛЬ: збираємо все разом
# Візуалізатор: усі 7 кроків послідовно
# ==============================================================
class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # ---- Крок 2: Embeddings ----
        self.tok_emb = nn.Embedding(vocab_size, N_EMBED)  # таблиця: token ID → вектор 192d
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBED)  # таблиця: позиція → вектор 192d

        # ---- Крок 6: Transformer Blocks ×4 ----
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])

        # ---- Вихід ----
        self.ln_f = nn.LayerNorm(N_EMBED)                         # фінальна нормалізація
        self.head = nn.Linear(N_EMBED, vocab_size, bias=False)    # 192 → vocab_size (logits)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B=batch, T=seq_len; idx містить token IDs

        # Крок 2: Embedding = token + position
        pos = torch.arange(T, device=idx.device)           # [0, 1, 2, ..., T-1]
        x = self.tok_emb(idx) + self.pos_emb(pos)          # (B, T, 192) + (T, 192) → (B, T, 192)

        # Крок 6: пропускаємо через 4 transformer-блоки
        x = self.blocks(x)   # (B, T, 192) → (B, T, 192) — shape не змінюється!

        # Вихід: вектор → logits (ненормалізовані ймовірності для кожного символу)
        x = self.ln_f(x)     # фінальна нормалізація
        logits = self.head(x) # (B, T, 192) → (B, T, vocab_size)

        # Рахуємо loss якщо є правильні відповіді (при тренуванні)
        loss = None
        if targets is not None:
            # cross_entropy порівнює передбачення з реальним наступним символом
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    # ==============================================================
    # Крок 7: ГЕНЕРАЦІЯ (autoregressive)
    # Візуалізатор: таб "7. Генерація"
    #
    # Цикл: prompt → model → softmax → sample → append → repeat
    # Кожен крок генерує ОДИН новий символ.
    # ==============================================================
    @torch.no_grad()  # не будуємо граф для backward — економимо пам'ять
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Обрізаємо контекст до BLOCK_SIZE (модель не бачить довше)
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Forward pass: отримуємо logits для КОЖНОЇ позиції
            logits, _ = self(idx_cond)

            # Беремо logits тільки останньої позиції — це передбачення наступного токена
            logits = logits[:, -1, :]       # (B, vocab_size)

            # Temperature: контролює "впевненість" вибору
            #   < 1.0 — загострює розподіл (безпечніше)
            #   > 1.0 — розмазує розподіл (креативніше)
            logits = logits / temperature

            # top_k фільтрація: залишаємо тільки k найімовірніших токенів,
            # решту обнуляємо (через -inf після softmax → 0).
            # Корисно при високій temperature — обрізає "хвіст" дурних варіантів.
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)            # v[..., -1] = k-те найбільше
                logits[logits < v[:, [-1]]] = float("-inf")  # все нижче порога → -inf

            # Softmax: перетворюємо logits на ймовірності (сума = 1)
            probs = F.softmax(logits, dim=-1)

            # Семплінг: обираємо один токен випадково за ймовірностями
            nxt = torch.multinomial(probs, num_samples=1)

            # Додаємо новий токен до послідовності і повторюємо
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ==============================================================
# ТРЕНУВАННЯ
# ==============================================================
@torch.no_grad()
def estimate_loss(model):
    """Оцінюємо loss на train і val без оновлення ваг."""
    out = {}
    model.eval()  # вимикаємо dropout
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # вмикаємо dropout назад
    return out


def main():
    print(f"device={device} vocab_size={vocab_size} params=", end="")
    model = NanoGPT().to(device)
    print(sum(p.numel() for p in model.parameters()))

    # AdamW — стандартний оптимізатор для трансформерів
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # Цикл тренування: forward → loss → backward → update
    for it in range(MAX_ITERS + 1):
        if it % EVAL_INTERVAL == 0:
            l = estimate_loss(model)
            print(f"iter {it:4d} | train {l['train']:.3f} | val {l['val']:.3f}")

        xb, yb = get_batch("train")    # випадковий батч
        _, loss = model(xb, yb)         # forward pass → loss
        opt.zero_grad(set_to_none=True) # обнулити градієнти (інакше акумулюються!)
        loss.backward()                 # backward pass → обчислити градієнти
        opt.step()                      # оновити ваги

    # Генерація: починаємо з порожнього контексту
    print("\n--- sample ---")
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=300)[0].tolist()
    print(decode(out))


if __name__ == "__main__":
    main()
