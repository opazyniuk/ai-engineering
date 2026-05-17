# HW6 — LLM Engineering: API vs Self-hosted

Extraction-агент який витягує з транскрипту зустрічі структурований JSON
(`summary`, `tasks`, `decisions`) через **два провайдери**:

- **OpenAI** (`gpt-4o-mini`) — cloud API
- **Ollama** (`llama3.2:3b`) — локальна self-hosted модель

Мета — порівняти **якість / швидкість / вартість / приватність** на 3 датасетах
різної складності (простий, хаотичний, технічний).

## Структура

```
hw6/
├── extraction_agent.py    # CLI-точка: запускає одну модель на одному файлі
├── eval.py                # прогін усіх комбінацій + метрики → eval_results.csv
├── samples/               # вхідні транскрипти
│   ├── simple_meeting.txt
│   ├── chaotic_standup.txt
│   └── technical_sync.txt
├── results/               # JSON-відповіді {dataset}_{provider}.json
├── screenshots/           # докази запуску
├── eval_results.csv       # метрики (генерується eval.py)
├── ANALYSIS.md            # висновки
├── pyproject.toml
└── .env.example
```

## Запуск

```bash
# 1. Підняти Ollama локально
ollama serve                          # окремий термінал
ollama pull llama3.2:3b               # одноразово

# 2. Налаштувати env
cp .env.example .env
# відредагувати .env: вставити OPENAI_API_KEY

# 3. Створити venv та встановити залежності
uv venv && source .venv/bin/activate
uv pip install -e .

# 4. Один запит до однієї моделі
python extraction_agent.py samples/simple_meeting.txt --provider openai
python extraction_agent.py samples/simple_meeting.txt --provider ollama

# 5. Повний прогін усіх 6 комбінацій + метрики
python eval.py
```

## Дедлайн
2026-05-04