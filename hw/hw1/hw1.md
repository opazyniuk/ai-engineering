# HW1: Опис курсового проєкту

## Назва проєкту

**Settle AI Assistant — інтелектуальний помічник для користувачів платіжної платформи Settle**

## Опис системи

Settle AI Assistant — це система, що поєднує RAG (Retrieval-Augmented Generation) та AI Agent підходи для двох ключових сценаріїв взаємодії з користувачем:

### 1. Help Center Assistant (Knowledge Base Q&A)

Система індексує весь контент Help Center платформи Settle (https://www.settle.com/) та надає точні, контекстуальні відповіді на запитання користувачів щодо функціоналу платформи. Замість того, щоб шукати відповідь серед десятків статей, користувач отримує конкретну відповідь з посиланням на джерело.

**Приклади запитань:**
- "Як налаштувати автоматичну оплату інвойсів?"
- "Які інтеграції підтримує Settle?"
- "Як додати нового користувача до акаунту?"

### 2. Business Data Assistant (API-driven Q&A)

Система інтерпретує бізнес-запитання користувача природною мовою, формує відповідні API-запити до Settle API, отримує дані та повертає агреговану відповідь.

**Приклади запитань:**
- "Скільки грошей мені треба буде заплатити до кінця тижня?"
- "Які інвойси прострочені?"
- "Покажи топ-5 постачальників за сумою оплат за останній місяць"

## Архітектура (високий рівень)

```
               User Query
                    │
                    ▼
            ┌──────────────┐
            │   FastAPI     │ ← async, rate limiting, Redis cache
            │   API Layer   │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │  LLM Router  │ ← визначає тип запиту (knowledge vs data)
            └──────┬───────┘
                   │
             ┌─────┴─────┐
             ▼           ▼
        ┌─────────┐ ┌─────────┐
        │   RAG   │ │   AI    │
        │  Chain  │ │  Agent  │ ← ReAct + Tool Calling (LangGraph)
        └────┬────┘ └────┬────┘
             │           │
             ▼           ▼
        Vector DB   Settle API        ┌──────────────┐
        (Qdrant)    (tool calling)──► │  Guardrails  │
             │           │            │  & Eval      │
             └─────┬─────┘            └──────────────┘
                   ▼
            ┌──────────────┐
            │  LLM Response│ ← hallucination check, PII masking
            │  Generation  │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │  Monitoring  │ ← Prometheus, Grafana, drift detection
            └──────────────┘
```

## Основні компоненти

### Data Ingestion & Processing (заняття 2-3)
| Компонент | Призначення |
|-----------|------------|
| **Web Scraper / Crawler** | Збір та оновлення контенту Help Center (HTML parsing) |
| **Data Pipeline** | Cleaning, normalization, deduplication зібраних даних |
| **Chunking Engine** | Розбиття документів на chunks з урахуванням типу контенту (semantic chunking) |
| **Metadata Pipeline** | Збереження контексту та походження кожного chunk (джерело, дата, розділ) |

### Embeddings & Vector Search (заняття 7-8)
| Компонент | Призначення |
|-----------|------------|
| **Embedding Pipeline** | Векторизація документів Help Center |
| **Vector Store (Qdrant)** | Зберігання та hybrid search (BM25 + vector) з filtering |

### RAG System (заняття 9)
| Компонент | Призначення |
|-----------|------------|
| **Retriever** | Пошук релевантних chunks з оптимізацією context window |
| **RAG Chain** | Генерація відповідей на основі знайдених документів з посиланнями на джерела |

### AI Agent & Tool Calling (заняття 11)
| Компонент | Призначення |
|-----------|------------|
| **LLM Router** | Класифікація запиту (knowledge base чи business data) |
| **API Agent (ReAct + LangGraph)** | Reasoning → вибір потрібних tools → виконання API-запитів → агрегація результатів |
| **Tool Definitions** | Набір tools для Settle API (get_bills, get_payments, get_vendors тощо), які LLM викликає через tool calling |

### API Layer & Infrastructure (заняття 10, 13)
| Компонент | Призначення |
|-----------|------------|
| **FastAPI Service** | Async API з background jobs для довгих запитів |
| **Redis Cache** | Кешування відповідей та rate limiting |
| **Docker** | Контейнеризація всіх сервісів |

### LLM Engineering (заняття 6)
| Компонент | Призначення |
|-----------|------------|
| **Model Selection** | Вибір LLM з урахуванням tokenomics, latency та якості (OpenAI / Claude API) |

### Evaluation, Safety & Guardrails (заняття 19)
| Компонент | Призначення |
|-----------|------------|
| **Hallucination Detection** | Перевірка faithfulness та groundedness відповідей |
| **Prompt Injection Protection** | Захист від зловмисних інструкцій |
| **PII Masking** | Детекція та маскування персональних даних (фінансова інформація) |
| **Eval Pipeline** | Регулярна перевірка якості відповідей |

### Monitoring & Observability (заняття 15)
| Компонент | Призначення |
|-----------|------------|
| **Document Drift Detection** | Моніторинг змін у Help Center та тригер переіндексації |
| **Observability Stack** | Prometheus + Grafana для метрик (latency, token usage, error rate) |
| **Alerting** | Автоматичні сповіщення при деградації якості або аномаліях |

## Очікувана цінність

- Зменшення навантаження на підтримку Settle за рахунок автоматичних відповідей з Help Center
- Миттєвий доступ до бізнес-даних без необхідності навігації по UI платформи
- Природна мова як інтерфейс до складних API-запитів та аналітики
- Production-ready система з моніторингом, безпекою та контролем якості