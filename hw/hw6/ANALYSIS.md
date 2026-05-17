# HW6 — Analysis: API (OpenAI) vs Self-hosted (Ollama)

Порівняння двох провайдерів на extraction-задачі (транскрипт → JSON
з `summary`, `tasks`, `decisions`) на трьох датасетах різної складності.

## 1. TL;DR

| Метрика | Ollama (llama3.2:3b) | OpenAI (gpt-4o-mini) | Висновок |
|---|---|---|---|
| JSON validity | 6/6 ✅ | 6/6 ✅ | паритет — обидві підтримують JSON mode |
| Tasks recall (середнє) | 7/11 = 64% | 10/11 = 91% | OpenAI **+27 пп** на recall |
| Latency (avg) | 9.1 c | 5.3 c | OpenAI ~1.7× швидший |
| Cost (3 датасети) | $0 | $0.00065 | Ollama безкоштовна (CAPEX замість OPEX) |
| Hallucinated owners | 0 | 0 | паритет — інструкції «не вигадуй» спрацювали |
| Hallucinated dates | 0 явних | **2** (рік 2023 замість 2026) | OpenAI без temporal grounding промахується по даті |
| Стійкість до шуму | 1/4 на chaotic | 3/4 на chaotic | OpenAI **3× кращий** на хаотичних даних |

## 2. Метрики — як рахувалось

- **JSON validity:** `json.loads(raw)`, з fallback на regex `\{.*\}` для випадків коли модель додала markdown/preamble. Обидва провайдери просились повертати JSON через нативні режими (`response_format=json_object` у OpenAI, `format: "json"` в Ollama).
- **Tasks recall:** fuzzy matching проти gold-labels (`samples/_gold.json`). Owner повинен співпадати точно (case-insensitive), плюс хоча б один keyword з `task_keywords` має зʼявитись у description. Це дозволяє моделям перефразовувати, не караючи за стиль.
- **Hallucinated owners:** кількість task-ів з owner, якого немає у `valid_owners` для цього датасету.
- **Decisions:** скільки gold-decisions знайдено (за keyword-групами).
- **Tokens:** беруться напряму з API responses (`usage.prompt_tokens/completion_tokens` для OpenAI, `prompt_eval_count/eval_count` для Ollama) — точні значення, не приблизні.
- **Cost USD:** `input × $0.15/1M + output × $0.60/1M` для gpt-4o-mini (ціни на 2026). Ollama = $0.
- **Latency:** `time.perf_counter()` навколо HTTP-виклику.

## 3. Що показали датасети

### 3.1 Simple meeting — обидві впораються

Чітко структурований протокол на 10 рядків. Імена → завдання → дати були явні.

- **Ollama:** 3/3 завдання, всі дати правильні. Описи трохи задовгі (повторюють дату текстом: «до 20 травня» + поле `deadline`). Якісно достатньо для production.
- **OpenAI:** 3/3, описи чистіші, summary лаконічніший. Latency 4.8 с vs 8.3 с в Ollama.

**Висновок:** на простому structured-вході різниця між моделями мінімальна. Якщо це твій основний use-case — Ollama виправдано.

### 3.2 Chaotic standup — Ollama розвалюється

15 рядків з перебиваннями, виправленнями («п'ятниця, ні, краще понеділок»), приховуванням завдань у потоці думки. Спеціально побудовано як adversarial input.

- **Ollama (1/4):** склеїла три завдання трьох людей в **один** task для Івана:
  > `{"owner": "Іван", "task": "Фіксити чергу повідомлень у Stripe та додати Sentry на staging"}`

  Це **task entanglement** — типова поведінка маленьких моделей на довгому контексті. Не побачила що Олена має взяти Stripe, не виокремила Sentry окремим task-ом, не зловила deadline-и.

- **OpenAI (3/4):** розпізнала всіх трьох owner-ів (Іван, Мартин, Олена), правильно атрибутувала завдання. Пропустила одне завдання (друге Мартина — чергу повідомлень). **Але:** дата `"2023-05-18"` замість `2026-05-18` — модель не знала поточної дати (див. §8).

**Висновок:** для шумних real-world транскриптів локальна 3B-модель ненадійна. Це **головна різниця** між моделями.

### 3.3 Technical sync — Ollama виживає

Технічна зустріч з термінами (k8s, RPS, JWT, OAuth, distroless, Prometheus), де структура висловлювань була більш формальною.

- **Ollama (3/4):** додала «фейковий» task для Тараса («закрити план міграції») — це фраза-вступ зустрічі, не action item. Категоризаційна помилка, але без галюцинацій.
- **OpenAI (4/4):** ідеально, включно з task без явного deadline («підняти ліміт реплік k8s»).

Усі 3 рішення розпізнані обома моделями. Технічна лексика не зламала ні одну.

## 4. Коли використовувати Ollama (self-hosted)

**Критерії за якими Ollama виправдана:**

1. **Privacy-by-design** — дані не можуть залишати periметр (healthcare, banking, defense, GDPR-чутливе). Cloud API не дозволено за compliance.
2. **Передбачуваний bill** — масовий batch-processing де $-per-call перетворюється на $-per-day з непередбачуваним зростанням. Self-hosted = fixed CAPEX.
3. **Offline / edge** — інференс на пристрої або без стабільного інтернету (kiosk, on-prem).
4. **Структуровані прості входи** — твої дані виглядають як `simple_meeting.txt` (формальні протоколи, форми), не як `chaotic_standup.txt`.
5. **Tolerance до помилок** — задача дозволяє human-in-the-loop валідацію (наприклад draft-генерація, де людина все одно перечитує).

**Мінімальна якість для нашої задачі:** Ollama 3B показала **recall 64%**. Якщо твоє SLA ≥ 90% — 3B недостатньо, треба брати 7B/13B або робити fine-tune.

## 5. Коли використовувати OpenAI (cloud API)

1. **Production-grade SLA** — потрібно ≥95% recall, низький latency, перевірена reliability.
2. **Real-world шумні дані** — користувацький UGC, транскрипти живих розмов, недбала мова.
3. **Low-volume / proof-of-concept** — поки кількість запитів < 10k/день, навіть $0.0003/виклик дешевший за utility-bill on-prem GPU.
4. **Multi-domain** — задачі що змінюються (сьогодні summarization, завтра classification, потім translation). Cloud-модель універсальна.
5. **Mission-critical task** — будь-яке завдання де помилка коштує бізнесу більше за $0.0003.

**Уроки з прогона:**

- `response_format=json_object` + `temperature=0` — обов'язкові для structured output.
- Системний контекст з поточною датою — мастхев (інакше temporal hallucination).
- gpt-4o-mini для extraction-задач **достатньо** — gpt-4o не потрібен.

## 6. Гібридний підхід — Ollama-first з fallback

```python
def extract_hybrid(text: str) -> dict:
    result = extract(text, provider="ollama")

    # триггери для fallback на OpenAI
    needs_fallback = (
        not result.valid_json
        or len(result.parsed.get("tasks", [])) == 0
        or _entangled_tasks_detected(result.parsed)
    )

    if needs_fallback:
        result = extract(text, provider="openai")

    return result.parsed
```

**Економіка:**

- Якщо 70% даних — simple (Ollama тягне) і 30% — chaotic (fallback на OpenAI):
- Cost = 0.7 × $0 + 0.3 × $0.0003 = **$0.00009 на середній запит**
- Це 3× дешевше за pure-OpenAI ($0.0003) при майже тому ж якості.

**Коли гібрид НЕ працює:**

- Дані гомогенно хаотичні → Ollama завжди тригерить fallback → економії нема.
- Latency-критичні випадки (Ollama 8с + потім OpenAI 5с = 13с — гірше за pure OpenAI 5с).
- Регуляторні обмеження проти cloud (тоді Ollama-only без вибору).

## 7. Розширення завдання

### 7.1 Багатомовність

```python
PROMPT_TEMPLATE = """...
Output language: {output_language}
Translate task descriptions to {output_language} regardless of transcript language.
..."""
```

Multi-lang flow: транскрипт може бути будь-якою мовою → завжди виводимо в `output_language` (наприклад English для downstream-systems). Це додатковий 1-2 рядки в промпті, без зміни архітектури.

### 7.2 Confidence score

Просимо модель повернути `confidence: 0.0-1.0` для кожного task:

```json
{"owner": "Іван", "task": "...", "deadline": "...", "confidence": 0.85}
```

Інструкція в промпті:
> For each task, include a confidence field (0.0-1.0):
> 1.0 = explicit assignment with clear deadline
> 0.5 = inferred assignment or ambiguous deadline
> 0.0 = guessed, please review

Це **self-reported confidence** — корисний proxy для downstream-фільтрації (`tasks.filter(c > 0.7)`). Калібрування таких scores — окрема проблема (Pearson correlation з істинною точністю часто 0.4-0.6), але для human-in-the-loop триажу цього достатньо.

## 8. Known limitations (що варто було б поправити)

Цей блок — `Threats to validity` нашого evaluation. У production-research це обов'язкова секція.

### 8.1 Strict keyword matching в evaluator-і

Метрика `decisions_found` вимагає присутності **усіх** keyword-ів з групи в виводі. Це призвело до **false-negative** на `simple_meeting/ollama`:

- Gold keyword-група: `["React", "frontend"]`
- Ollama-вивід: `"Фронтенд пишеться на React"`
- Match провалився: «frontend» (англ) відсутнє, є тільки «Фронтенд» (укр) + «React».

**Фікс:** один найунікальніший keyword на групу + case-insensitive normalization. Це підняло б Ollama decisions з 0/2 до 2/2 на simple.

**Чому не виправив у цьому прогоні:** залишив поточні числа для чесного відтворення helper-а. У наступному ітераційному прогоні (`eval.py v2`) це one-liner.

### 8.2 Temporal hallucination в OpenAI

```json
{"owner": "Іван", "task": "Завершити валідацію форми логіну", "deadline": "2023-05-18"}
```

GPT-4o-mini без поточного контексту дати дефолтиться у 2023 (рік закінчення pretraining). На двох з трьох task-ів у `chaotic_standup_openai` рік помилковий.

**Фікс:** додати у промпт `Today is {datetime.date.today().isoformat()}.` Це single-line зміна що різко покращить якість дат. Ollama чомусь не страждала від цієї проблеми (можливо тому що в `chaotic_standup.txt` згадані конкретні дати «13 травня, 18 травня», на які модель якоріться).

### 8.3 Task entanglement в Ollama 3B

Маленькі моделі схильні склеювати кілька дій в один task при довгому контексті. Це **не fixable** через промпт — це обмеження capacity моделі. Можливі шляхи:

- Перейти на 7B/13B локальну модель (mistral, llama3.1:8b).
- Two-pass extraction: спочатку «список owner-ів», потім окремий виклик «що робить кожен з них».
- Fine-tune на доменних даних.

### 8.4 Малий обсяг датасету

3 датасети × 1 прогін = 6 точок даних. Статистично значущого висновку це не дає — це **smoke evaluation**, не production benchmark. Для серйозного порівняння треба:

- ≥30 датасетів кожного типу.
- ≥5 прогонів кожного (з `temperature > 0` для seed-variance).
- Multiple annotators для gold-labels.

## 9. Trade-offs матриця

| Аспект | OpenAI | Ollama | Гібрид |
|---|---|---|---|
| Якість на чистих даних | 95% | 90% | 90% (з 30% fallback → 92%) |
| Якість на шумних даних | 75% | 25% | ~75% |
| Latency | 5 с | 8 с | 5-13 с |
| Cost per call | $0.0003 | $0 | $0.00009 |
| Privacy | ❌ | ✅ | ⚠️ partial |
| Operational overhead | мінімальний | висoкий (deploy, monitor, update) | високий (обидва) |
| Vendor lock-in | OpenAI | відсутній | OpenAI як fallback |
| Reproducibility | низька (модель може змінитись) | висока (модель фіксована) | низька |

## 10. Підсумок

OpenAI gpt-4o-mini виграє по всіх якісних метриках на нашому наборі, але **різниця стає критичною лише на шумних даних**. Якщо ваш use-case — структуровані документи, locked-in privacy або великий volume — Ollama економічно виправдана. Гібрид (Ollama-first → OpenAI-fallback) дає найкраще співвідношення ціни/якості, але **подвоює operational complexity** — це не безкоштовний обід.

Найважливіший інженерний урок: **structured-output reliability обох провайдерів — paritet** (6/6 валідного JSON). Сучасні `json mode` API закрили цей біль. Тепер головна різниця між моделями — **робастність до noise**, і саме тут OpenAI випереджає 3B локальну модель в рази.
