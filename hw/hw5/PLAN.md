# HW5: Дослідження та розширення nano-GPT — План

## 1. Суть задачі: що таке nano-GPT і чому ми його розширюємо

**nano-GPT** — це ~250 рядків PyTorch коду, які реалізують ту саму архітектуру, що й GPT-4, тільки в мініатюрі. Її дають саме тому, що вона:

- **Маленька** — 1.8M параметрів (GPT-3 = 175B, тобто в ~100 000 разів менше). Тренується на ноутбуці за хвилини.
- **Без абстракцій** — нема `transformers` бібліотеки, нема Hugging Face. Кожен крок видно "голим оком".
- **Це справжній transformer** — той самий self-attention, multi-head, causal mask, residual connections, що в "великих" LLM.

**Що вона робить?** Це **language model на рівні символів** (char-level). Тобто:
- Bхід: послідовність символів `"hello worl"`
- Вихід: ймовірності для кожного наступного символу — `'d'` буде з ймовірністю 0.7, `'k'` з 0.05, тощо
- Генерація: семплимо символ → додаємо до контексту → повторюємо

Це **autoregressive generation** — те саме, що ChatGPT робить, тільки там токенайзер BPE (subword) замість символів і модель в мільярдах разів більша.

**Аналогія Ruby:** уяви Markov chain, але замість `Hash[prev_word][next_word_count]` — нейромережа з 1.8M параметрів, яка вчить розподіл `P(next_char | context)` через градієнтний спуск.

**Що нас просять зробити:** взяти цей готовий код і:
1. Натренувати на власному датасеті (а не лекційному)
2. Подивитись як loss падає (overfitting аналіз)
3. Покрутити sampling параметри (temperature, top_k) — це **inference-time** контроль "креативності"
4. Загорнути в HTTP API — як **продуктивно деплоїться будь-яка ML модель**

---

## 2. Як LLM вчиться — на пальцях

Уяви, що ти даєш ШІ книгу і кажеш: "вгадай наступну літеру". Він вгадує. Ти кажеш правильну відповідь. Він коригує свої "ваги". Повторюємо мільйон разів — і він починає вгадувати непогано.

Технічно:

```
TEXT = "котики дуже милі"

batch (один приклад):
  x = "котики дуж"  ← 10 символів
  y = "отики дуже"  ← ті ж 10 символів зсунуті на 1 — це "правильна відповідь"

forward pass:
  model(x) → logits (B, T, vocab_size)  ← для кожної позиції — розподіл по словнику

loss = cross_entropy(logits, y)  ← наскільки сильно модель помиляється

backward pass:
  loss.backward()  ← обчислює градієнти (∂loss/∂кожен_параметр)
  optimizer.step()  ← коригує параметри в напрямку зменшення loss
```

Ruby-аналогія: cross_entropy це як "штраф за впевнену помилку". Якщо модель сказала "наступна буква 'А' з ймовірністю 0.99", а правильна 'Б' — loss величезний. Якщо сказала 'А' з 0.3 — loss маленький.

---

## 3. Архітектура nano-GPT: 7 кроків з візуалізатора

```
text → [1.tokenize] → [2.embed] → [3-5.attention×N_LAYER] → [head] → logits → [7.sample] → text
```

**Крок 1. Токенізація** (рядки 49-54 у `nano_gpt.py`)
- Вхід: рядок `"hello"`
- Вихід: список цілих `[7, 4, 11, 11, 14]`
- У нас **char-level**: кожен унікальний символ — окремий токен. Vocab_size зазвичай 60-100 для англомовного тексту.
- У real LLM — **BPE** (subword): "hello" може бути одним токеном, "running" може бути ["run", "ning"]. Vocab_size ~50k-200k.
- Ruby-аналогія: `text.chars.map { |c| stoi[c] }`.

**Крок 2. Embeddings** (рядки 171-172, 186)
- Кожен токен ID перетворюється на вектор фіксованої розмірності (`N_EMBED=192`).
- `tok_emb` — таблиця "ID → вектор", вчиться. Тобто схожі за змістом токени отримують схожі вектори.
- `pos_emb` — додатковий вектор для позиції (бо attention сам по собі **позиційно-інваріантний** — без цього "abc" і "cba" виглядали б однаково).
- Сума: `x = tok_emb(id) + pos_emb(position)`.

**Кроки 3-5. Self-Attention** — серце transformer'а. Це в `CausalSelfAttention.forward()`.

Кожен токен задає три питання:
- **Q (Query)** — "що мені треба знайти в контексті?"
- **K (Key)** — "що я можу запропонувати іншим?"
- **V (Value)** — "яку інформацію я несу?"

Алгоритм:
```python
attention_scores = Q @ K.T / sqrt(d)        # (T, T) матриця "хто на кого дивиться"
attention_scores = mask(attention_scores)    # causal mask — не можна дивитись у майбутнє
weights = softmax(attention_scores)          # рядки сумуються до 1.0
output = weights @ V                          # зважена сума value-векторів
```

**Multi-Head** — паралельно запускаємо 6 таких "attention механізмів" з різними Q/K/V проєкціями. Ідея: різні голови вчаться різних патернів (одна — синтаксис, інша — далекий контекст, тощо). Саме тому в 2.2 нас просять подивитись heatmaps по головах — побачити **що саме** кожна голова вивчила.

**Causal mask** — ставимо `-∞` у верхньому трикутнику `attention_scores`, щоб після softmax там був 0. Тобто токен на позиції `i` може дивитись тільки на 0..i, не в майбутнє. Це робить модель autoregressive (інакше вона б "списувала" відповідь з майбутнього).

**Крок 6. Transformer Block** — Attention + MLP, обидва з residual + LayerNorm:
```python
x = x + attn(layer_norm(x))   # attention "змішує" токени між собою
x = x + mlp(layer_norm(x))     # MLP "думає" над кожним токеном окремо
```
Residual (`x + ...`) — щоб градієнти могли пройти через всі шари без затухання. Без цього глибокі мережі не вчаться.

**Крок 7. Генерація** — autoregressive цикл:
```python
for _ in range(max_new_tokens):
    logits = model(idx[:, -BLOCK_SIZE:])[:, -1, :]   # передбачення наступного
    probs = softmax(logits / temperature)
    next_token = sample(probs)
    idx = concat(idx, next_token)
```

**Temperature** контролює "сміливість":
- `T < 1` — загострює розподіл (модель частіше обирає найімовірніше → consistent, нудно)
- `T > 1` — згладжує (більше випадковості → креативно, але часто маячня)

**top_k** обмежує семплінг тільки `k` найімовірнішими токенами (решту обнуляємо). Це відсікає хвіст випадкового шуму.

---

## 4. Інструменти і для чого

| Інструмент | Для чого в HW5 |
|---|---|
| **PyTorch** (`torch`) | Тензори + autograd. Сам нейронний фреймворк. `torch.nn` — шари, `torch.optim` — оптимізатори, `torch.save/load` — серіалізація моделі. |
| **MPS / CUDA / CPU** | Backend для обчислень. На Apple Silicon → MPS (GPU на M1/M2). У 3-5x швидше за CPU. |
| **matplotlib** | Для 1.2 — графік train/val loss. Для 2.2 — heatmaps attention weights. Стандартний tool для plotting у Python. |
| **FastAPI** | Для 1.4 — HTTP server. Сучасний async-фреймворк, конкурент Flask. Pydantic валідація request/response, авто Swagger UI. Аналог в Ruby — Sinatra/Grape. |
| **uvicorn** | ASGI-сервер для FastAPI (Puma-аналог). |
| **tiktoken** (тільки для 2.1 BPE) | OpenAI-шна BPE токенізація. Замість char-level. |
| **curl / Swagger UI** | Тестування API endpoint'а. |
| **time.perf_counter** | Latency benchmark — заміряти скільки мс генерується 50/100/200 токенів. |
| **yt-dlp** (опціонально) | Витягнути субтитри YouTube як датасет: `yt-dlp --write-auto-sub --skip-download --sub-lang uk <url>`. |

---

## 5. План виконання (детально)

### Підготовка
- [ ] Зробити venv з Python 3.12 (PyTorch ще не має wheels для 3.14)
- [ ] `pip install torch matplotlib fastapi uvicorn`
- [ ] Завантажити обраний датасет, скласти `training_text.txt` (≥500KB)

### 1.1 Свій датасет (3 бали)
Замінюємо `training_text.txt`. Запускаємо `python nano_gpt.py`. Чекаємо ~3-10 хв (залежить від розміру). Зберігаємо 3 семпли + рахуємо vocab_size.

### 1.2 Графік loss (3 бали)
**Модифікація коду:** в `main()` накопичуємо `train_losses[]` і `val_losses[]` в момент `estimate_loss`. Після тренування:
```python
import matplotlib.pyplot as plt
plt.plot(iters, train_losses, label='train')
plt.plot(iters, val_losses, label='val')
plt.axvline(iters[np.argmin(val_losses)], color='red', linestyle='--', label='min val')
plt.savefig('plots/loss.png')
```
**Аналіз overfitting:** якщо train loss ↓ а val loss ↑ або плато — модель завчила тренувальні дані (memorization), а не generalize. Маленький датасет → швидко overfit.

### 1.3 Temperature × top_k (4 бали)
**Модифікація `generate()`:**
```python
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        ...
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        ...
```
9 семплів — для кожної комбінації (3 temp × 3 top_k) генеруємо текст і кладемо в табличку у звіті.

### 1.4 FastAPI server (4 бали)
**Чекпоінт:** наприкінці тренування —
```python
torch.save({
    'model_state': model.state_dict(),
    'config': {'BLOCK_SIZE':..., 'N_EMBED':..., ...},
    'vocab': {'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size},
}, 'checkpoint.pt')
```
**`server.py`:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

# load checkpoint, rebuild model, model.eval()

class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0

@app.post("/generate")
def generate(req: GenRequest):
    t0 = time.perf_counter()
    ids = encode(req.prompt)
    out = model.generate(torch.tensor([ids]), req.max_tokens, req.temperature)
    text = decode(out[0].tolist())
    return {"text": text, "latency_ms": (time.perf_counter()-t0)*1000}
```
**Запуск:** `uvicorn server:app --reload` → відкрити http://localhost:8000/docs (Swagger) → тестити.

**Latency benchmark:** скрипт що тричі викликає `/generate` для max_tokens=50/100/200 і пише середнє.

**"Чому в проді не використовують такий сервер":** у відповіді буде про **vLLM/TGI** — вони мають PagedAttention, continuous batching, KV-cache, оптимізовані CUDA kernels. Наш сервер обробляє запити по одному → GPU простоює більшість часу.

### Level 2 (на вибір, +3 бали)
- **2.2 Attention heatmap** (рекомендую — швидше, наочніше): зберегти `att` після softmax як `self.last_att = att` у `CausalSelfAttention.forward()`. Прогнати одне речення, дістати по головах, зробити 6 subplots з matplotlib `imshow()`. Шукати голову яка дивиться на `i-1` (діагональ під головною).
- **2.1 BPE**: замінити char токенізатор на `tiktoken.get_encoding("cl100k_base")`. Vocab ~100k → треба зменшити N_EMBED або використати tied weights щоб параметри не вибухнули. Більше роботи, але реалістичніше.

### Level 3 bonus (+3 бали): KV-cache
**Проблема:** на кожному кроці generate() ми перераховуємо attention для **всього контексту** довжиною T, але реально нам потрібен Q тільки для останнього токена, а K/V попередніх токенів **не змінюються**.

**Розв'язання:** зберігаємо K, V в кеші. На кроці t:
- обчислюємо `q, k_new, v_new` тільки для нового токена (один)
- `K = cat([K_cache, k_new])`, `V = cat([V_cache, v_new])`
- attention: `q @ K.T` — це (1, t) замість (t, t) → **O(t)** замість **O(t²)**

Speedup росте з T бо без кешу компʼют квадратичний по контексту.

### REPORT.md
Зведений звіт усіх чотирьох/шести завдань: текст, графіки, таблиці, висновки. Це власне те, що оцінюється.

---

## 6. Структура папки

```
hw/hw5/
  PLAN.md                       # цей файл
  gpt_visualizer.html           # інтерактивна візуалізація архітектури
  homework/
    nano_gpt.py                 # модифікована модель
    training_text.txt           # власний датасет (≥500KB)
    server.py                   # FastAPI endpoint
    checkpoint.pt               # збережені ваги після тренування
    REPORT.md                   # фінальний звіт
    plots/
      loss.png                  # train/val loss (1.2)
      attention_heatmap.png     # 6 голів (2.2, якщо обрано)
```

---

## 7. Оцінювання

| Завдання | Бали |
|---|---|
| 1.1 Свій датасет + 3 семпли + порівняння | 3 |
| 1.2 Графік loss + аналіз overfitting | 3 |
| 1.3 top_k + таблиця 3×3 + пояснення | 4 |
| 1.4 FastAPI сервер + latency benchmark | 4 |
| **Сума Рівень 1** | **14** |
| Рівень 2 (одне завдання з двох) | +3 |
| Рівень 3 (KV-cache) | +3 |
| **Максимум** | **20** |

- Прохідний: 12/16 (Рівень 1 повністю)
- На максимум: Рівень 1 + Рівень 2
- Бонус: Рівень 3

---

**Бюджет часу:** ~6-8 годин на все, з тренуваннями на MPS.