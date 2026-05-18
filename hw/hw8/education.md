# pgvector — деталізована довідка по всіх значеннях

Документ описує **кожен параметр, оператор, SQL-конструкцію і функцію Python**, які використано у `src/benchmarks/pgvector_db.py`. Без винятків.

Архітектурно pgvector — це **розширення PostgreSQL**. Все, що нижче — це SQL/PG механіки, які pgvector доточує до звичайного Postgres.

---

## 1. Розширення та тип `vector`

### `CREATE EXTENSION IF NOT EXISTS vector`

Postgres має механізм **extensions** — додаткові модулі, які реєструють нові типи даних, функції, оператори, індекси без перекомпіляції самого Postgres. `pgvector` — одне з таких розширень. Поставляється бінарними файлами в образі `pgvector/pgvector:pg16`.

- `IF NOT EXISTS` — ідемпотентно, не падає якщо вже встановлено.
- Реєструється на рівні **бази даних** (не кластера). Створив у `bench` → доступне тільки в `bench`.

### Тип `vector(N)`

```sql
embedding vector(1536)
```

Це **колонковий тип**, який зберігає `N` чисел `float4` (4 байти) суцільним масивом + кілька байт заголовка. Розмір на диску ≈ `4 * N + 8` байт = для нас ~6 KB на рядок.

- Розмірність `(N)` **обов'язково фіксована** при створенні колонки. Не можна додати вектор іншої довжини.
- Підтримує до **16,000 вимірів** для `vector` (для `halfvec` — до 4,000 з 2 байтами на число; для `bit` — до 64K).
- pgvector також має типи `halfvec(N)` (float16), `sparsevec(N)` (sparse), `bit(N)` — для нас тільки `vector(N)` потрібен.

---

## 2. Структура таблиці

```sql
CREATE TABLE bench (
    id INTEGER PRIMARY KEY,
    embedding vector(1536)
);
```

- `id INTEGER PRIMARY KEY` — наш position-based int id (0..N-1). PRIMARY KEY автоматично створює унікальний btree-індекс.
- **TOAST**: будь-яке значення вектора більше ~2 KB Postgres автоматично виносить у окрему TOAST-таблицю (`pg_toast_<oid>`), у головній лишається посилання. Наш vector(1536) = 6 KB → завжди TOAST'иться. Це **прозоро** для нас, але впливає на disk_size.

--- 

## 3. COPY BINARY — як ми заливаємо 523K векторів

### Чому не INSERT

| Метод | Швидкість на 523K | Чому |
|---|---|---|
| `INSERT` по 1 рядку | ~10 годин | парсинг SQL + транзакція на кожен рядок |
| Multi-row `INSERT INTO ... VALUES (...), (...)` | ~30 хв | менше overhead'у |
| `COPY FROM STDIN WITH (FORMAT BINARY)` | **~40 сек** | бінарний протокол + zero parsing |

COPY — це окремий **wire-protocol** Postgres'у. Клієнт відкриває потік, заливає сирі байти у спеціальному форматі, сервер пише їх безпосередньо в heap. Жодного SQL parsing'а, жодного planner'у.

### Бінарний формат COPY у psycopg3

```python
with cur.copy(f"COPY {TABLE} (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
    copy.set_types(["int4", "vector"])
    for i in range(n):
        copy.write_row((i, vectors[i]))
```

- `cur.copy(...)` повертає **context manager**, який тримає відкритий COPY-потік.
- `copy.set_types(["int4", "vector"])` — каже psycopg, як серіалізувати кожну колонку. **Без цього** psycopg би передавав значення як `TEXT` (повільніше і потребує парсингу на сервері).
  - `"int4"` — 32-bit integer (4 байти).
  - `"vector"` — pgvector тип. Серіалізатор реєструється через `register_vector(conn)`.
- `copy.write_row((id, np_vector))` — pgvector psycopg-helper напряму бере numpy array і пакує його у бінарний формат pgvector'а.

### Чому COPY ТАК ШВИДКЕ

1. **Bypassing SQL layer**: жодного парсера, planner'а, statement cache. Сирі байти → heap.
2. **Bulk WAL**: один запис у WAL на чанк, не на рядок.
3. **No row-by-row trigger overhead** (триггерів у нас немає, але якби були — теж пропускалися б на FREEZE).
4. **TCP throughput**: великі пакети, не дрібні `INSERT` команди.

---

## 4. Чому CREATE INDEX **ПІСЛЯ** COPY, а не до

Якщо HNSW індекс існує **на момент COPY**, кожна вставка має оновити граф. Це:
- Lock на граф → серіалізує вставки.
- O(log N) на кожну вставку → ще + 10-30 мікросекунд на рядок.
- Загалом ~10× повільніше.

Стратегія `COPY → CREATE INDEX`:
- COPY заливає всі дані без жодних індексів (тільки PRIMARY KEY btree оновлюється).
- `CREATE INDEX ... USING hnsw` будує граф **одним проходом по готовому heap** — це швидше, ніж 523K інкрементальних вставок.

Це **стандартний паттерн bulk-load у Postgres** для будь-якого типу індексу.

---

## 5. CREATE INDEX — деталізація

```sql
CREATE INDEX ON bench
USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 200);
```

### 5.1 `USING hnsw`

Тип індексу. У pgvector є **два варіанти**:

| Тип | Що це | Коли обирати |
|---|---|---|
| `hnsw` | Hierarchical Navigable Small World — граф | Default for production. Кращий recall, лінійна побудова, не потребує training |
| `ivfflat` | Inverted file with flat lists (k-means clusters) | Менший диск, швидша побудова, але потребує training-set і гірший recall на edge cases |

Ми обираємо `hnsw` — це SOTA для ANN на текстових embedding'ах.

### 5.2 `vector_cosine_ops` — operator class

**Operator class** — це механізм Postgres, який каже індексу: «який оператор вважати дистанцією». pgvector реєструє три:

| Operator class | Оператор | Метрика | Коли |
|---|---|---|---|
| `vector_l2_ops` | `<->` | Euclidean | для не-нормалізованих векторів |
| `vector_ip_ops` | `<#>` | Negative inner product | для нормалізованих, але хочеш швидше (без 1 SQRT) |
| `vector_cosine_ops` | `<=>` | Cosine distance | універсально для тексту |

Ми обираємо **`vector_cosine_ops`**, бо:
- Наші embedding'и нормалізовані → cosine дає правильні результати.
- pgvector сам нормалізує вектори в індексі ще раз → байдуже, чи вхід нормалізований.
- Це найзвичніший вибір — твій майбутній код буде швидше зрозумілий іншим.

**Важливо**: `vector_cosine_ops` працює **тільки з `<=>`**. Якщо в `ORDER BY` використати `<->`, цей індекс **не буде задіяний** — query пройде full scan.

### 5.3 `WITH (m = 32, ef_construction = 200)` — параметри HNSW

#### `m` (M у термінах HNSW paper)

Кількість зв'язків (сусідів) кожного вузла на рівні 0 графа. На верхніх рівнях — `m/2`.

| Значення `m` | Recall | Memory overhead | Build time | Search time |
|---|---|---|---|---|
| 8 | 0.85-0.90 | мало | швидко | дуже швидко |
| **16** (default pgvector) | 0.90-0.95 | помірно | помірно | швидко |
| **32** (наш вибір) | 0.95-0.99 | +1.6 MB на 100K векторів | +50% | +20% |
| 48-64 | 0.98-0.999 | багато | повільно | помірно |

**Чому `m=32`**: SOTA recommendation для high-quality semantic search. Default pgvector (`m=16`) дає помітно гірший recall на embedding'ах від OpenAI.

**Не можна змінити** `m` після `CREATE INDEX`. Тільки `DROP INDEX` + `CREATE INDEX` заново.

#### `ef_construction`

Розмір **dynamic candidate list** під час побудови графа. Більше = краща структура графа = вищий recall назавжди (не тільки для конкретного запиту).

| Значення | Build time | Recall improvement |
|---|---|---|
| 64 (default pgvector) | швидко | baseline |
| 128 | +30% | +1-2% |
| **200** (наш вибір) | +60% | +2-3% |
| 400 | +130% | +3-4% |
| 800 | +250% | +0.5% додатково |

**Чому `200`**: sweet spot. Більше — diminishing returns.

**Не можна змінити** після `CREATE INDEX`.

#### Чого тут **немає**: `ef_search` / `hnsw.ef_search`

`ef_search` (бо це query-time параметр) задається **окремо** через `SET`, бо може змінюватися per-query без rebuild'у. Розглянуто нижче.

---

## 6. SET hnsw.ef_search — головна ручка для Pareto frontier

```sql
SET hnsw.ef_search = 64;
```

- `SET` — це **session-local** GUC (Grand Unified Configuration) — змінна, яка живе протягом одного коннекту.
- Альтернативи:
  - `SET LOCAL hnsw.ef_search = X` — діє тільки до кінця транзакції.
  - `ALTER DATABASE bench SET hnsw.ef_search = X` — назавжди як дефолт для нової сесії.
  - `SHOW hnsw.ef_search` — переглянути поточне значення.

### Як `ef_search` впливає на пошук

```
Beam search на рівні 0 HNSW:
  cur_candidates = {entry_point}
  while is_better_neighbor_available:
      expand top candidate, add neighbors to cur_candidates
      keep only top-N candidates, where N = ef_search
  return top-K from cur_candidates
```

- **Менше `ef_search`** = вужча черга кандидатів = швидше, але вища ймовірність пропустити справжніх сусідів (нижчий recall).
- **Більше `ef_search`** = ширша черга = повільніше, але вищий recall.
- `ef_search ≥ top_k` — обов'язково (інакше неможливо повернути K результатів).
- На практиці: `ef_search = 2 × top_k` — мінімальна розумна точка.

### Чому це Pareto-ручка

Один індекс — багато точок на Pareto frontier. Без rebuild'у. У runner ми робимо sweep `[16, 32, 64, 128, 256]` за лічені секунди.

---

## 7. Оператор `<=>` (cosine distance)

```sql
SELECT id, embedding <=> $1 AS d FROM bench ORDER BY d LIMIT 10;
```

### Що повертає

```
<=>  →  cosine_distance = 1 - cosine_similarity
```

Для **нормалізованих** векторів (наш кейс):
- `<=>` ∈ [0, 2]
- 0 = ідентичні вектори
- 1 = перпендикулярні (cos=0)
- 2 = протилежні (cos=-1)

Для **ненормалізованих**:
- pgvector сам нормалізує перед обчисленням → значення також у [0, 2].
- Тому **завжди можна** використовувати `<=>` без власної нормалізації.

### Зворотна конверсія до similarity

```python
similarity = 1.0 - distance  # ∈ [-1, +1]
```

У нашому коді: `[(self.int_to_id[_id], 1.0 - float(d)) for _id, d in rows]`. Так віддаємо звичну шкалу «більше = краще», як FAISS/Qdrant.

### Чому `ORDER BY d` + `LIMIT 10` — оптимально

- Postgres planner бачить `ORDER BY embedding <=> $1` + наявний `hnsw` index на цій колонці з `vector_cosine_ops` → **bound the operator class**.
- Planner використовує індекс як **index scan**, який повертає рядки **уже відсортовані** за `<=>`.
- `LIMIT 10` зупиняє після 10 рядків → reads тільки 10-50 listings з графа.

**Без правильного operator class** (наприклад, якщо створив `vector_l2_ops`, а в query `<=>`) — planner ігнорує індекс, робить full sequential scan + sort. Це повільно на ~3 порядки.

---

## 8. ANALYZE bench

```sql
ANALYZE bench;
```

- Збирає статистику для query planner'а: розподіл значень, NDV (number of distinct values), null fraction.
- Без ANALYZE planner може хибно обрати sequential scan, ігноруючи наш HNSW.
- Після bulk-load — **обов'язково**. Autovacuum зробить це з затримкою — для бенчмарка треба негайно.

---

## 9. pg_total_relation_size — як ми міряємо disk

```sql
SELECT pg_total_relation_size('bench');
```

Повертає **bytes**, які займає таблиця **разом з усіма допоміжними структурами**:

| Що включено | Розмір приблизно |
|---|---|
| Heap (рядки таблиці) | ~6 KB × N = ~3 GB |
| TOAST таблиця (вектори > 2 KB) | основна частина |
| PRIMARY KEY btree (id) | ~10 MB на 523K |
| **HNSW індекс** | ~1.5-2 GB |
| FSM (free space map) | трохи |
| VM (visibility map) | трохи |

Тому `pg_total_relation_size('bench')` ≈ 4.5-5 GB на повному корпусі.

Альтернативи:
- `pg_relation_size('bench')` — тільки heap (без індексів і TOAST).
- `pg_indexes_size('bench')` — тільки індекси.
- `pg_table_size('bench')` — heap + TOAST, але без індексів.

Ми обираємо `pg_total_relation_size` бо в реальному prod-середовищі весь дисковий footprint — це heap + TOAST + всі індекси. Це чесне порівняння з Qdrant/Chroma.

---

## 10. autocommit = True

```python
conn = psycopg.connect(PG_DSN, autocommit=True)
```

### Чому autocommit

- **`CREATE INDEX` у транзакції блокує `UPDATE/DELETE`** — для нас не критично, але хороша звичка.
- **`CREATE INDEX CONCURRENTLY`** (для production hot rebuild без блокування) **взагалі не можна в транзакції**.
- `COPY` чудово працює в autocommit.

### Альтернатива

Без autocommit psycopg відкриває транзакцію на першу команду, треба явно `conn.commit()` / `conn.rollback()`. Для аналітичного скрипта типу нашого — автоматизація через autocommit зручніша.

---

## 11. register_vector — pgvector integration з psycopg3

```python
from pgvector.psycopg import register_vector
register_vector(conn)
```

Без цього:
- `cur.execute("SELECT %s::vector", (np.array([1,2,3]),))` — fail. psycopg не знає, як серіалізувати numpy array у `vector`.
- Треба вручну: `'[1,2,3]'::vector` — рядок, повільно, requiрed escaping.

З `register_vector`:
- numpy `np.ndarray` (1D float32) **прозоро** конвертується в bin format pgvector'а.
- Результат `<=>`, `<->` приходить як `float` — це не vector, це distance.
- COPY BINARY теж використовує цей серіалізатор (тому ми пишемо `copy.set_types(["int4", "vector"])`).

**Важливо**: `register_vector` треба викликати **на конкретному з'єднанні** (`conn`), не глобально. Кожен новий коннект — новий `register_vector(conn)`.

---

## 12. Connection DSN

```python
PG_DSN = (
    f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
    f"port={os.environ.get('POSTGRES_PORT', '5434')} "
    f"user={os.environ.get('POSTGRES_USER', 'bench')} "
    f"password={os.environ.get('POSTGRES_PASSWORD', 'bench')} "
    f"dbname={os.environ.get('POSTGRES_DB', 'bench')}"
)
```

DSN (Data Source Name) — стандартний libpq формат. Альтернативи: URI (`postgresql://bench:bench@localhost:5434/bench`).

- `host=localhost` — TCP. Швидше було б через Unix socket (`host=/var/run/postgresql`), але контейнер виставляє тільки TCP.
- `port=5434` — наш ремап у docker-compose (бо `5432` зайнятий хост-постгресом).
- `user/password=bench` — задано в `docker-compose.yml` через `POSTGRES_USER/PASSWORD`.
- `dbname=bench` — задано через `POSTGRES_DB`.

Усе через ENV → можна змінити в `.env` без правок коду.

---

## 13. psycopg vs psycopg2 — чому psycopg (3.x)

`psycopg` (без цифри) — це **psycopg 3**, переписаний з нуля наступник psycopg2.

| Аспект | psycopg2 | psycopg 3 (наш) |
|---|---|---|
| Бінарний протокол | Підтримує | За замовч., швидше |
| Async (asyncio) | Окрема бібліотека | Вбудовано |
| COPY API | `cur.copy_expert()` (clumsy) | `cur.copy(...)` context manager |
| pgvector integration | через `pgvector.psycopg2` | через `pgvector.psycopg` |
| Activity | Підтримується, але legacy | Активна, всі нові features |

Для нового коду — обов'язково psycopg 3.

---

## 14. Чого ми **НЕ** використали (і чому)

### 14.1 `vector_l2_ops` / `vector_ip_ops`

Eq cosine, але не потрібні для нашого кейсу (нормалізовані вектори → cosine оптимальний).

### 14.2 `ivfflat` індекс

Альтернатива HNSW з меншим диском, але:
- Потребує **training set** (`CREATE INDEX ... WITH (lists = N)` — треба обрати кількість кластерів).
- Гірший recall на «корбатих» розподілах embedding'ів.
- На сучасному pgvector (0.5+) HNSW майже завжди кращий.

### 14.3 `halfvec(N)` (float16)

Зменшив би диск удвічі. Але:
- Втрата precision на ~3-й-4-й знак similarity.
- На текстових embedding'ах це може дати -1-2% recall'у.
- Не варто для benchmark'а — лишимо чистий `vector(N)`.

### 14.4 Filtering за payload

pgvector підтримує `WHERE category = 'X' ORDER BY embedding <=> $1 LIMIT 10` — але вимагає окремого тюнинга (`SET hnsw.iterative_scan = strict_order`). У нашому Quora датасеті ніяких categories немає → не використовуємо.

### 14.5 `max_parallel_maintenance_workers`

```sql
SET max_parallel_maintenance_workers = 4;
```

Дозволяє pgvector будувати HNSW в N паралельних потоків — у ~3-4× швидше. Залишили дефолт (1 поток) для **чесного порівняння** з Qdrant/Chroma, які теж однопоточні в нашому benchmark'у. Якщо ставити в prod — обов'язково увімкнути.

---

## 15. Що варто запам'ятати назавжди

1. **`vector_cosine_ops` для текстових embedding'ів** — стандарт.
2. **`<=>` повертає distance, не similarity** → конвертувати `1 - distance` на клієнті.
3. **COPY BINARY у 100× швидше за INSERT** — без винятків при bulk load > 10K рядків.
4. **CREATE INDEX **після** COPY** — інакше build у 10× повільніше.
5. **`m=32, ef_construction=200`** — sane defaults для high-quality semantic search; pgvector defaults (m=16, ef_c=64) занадто скромні.
6. **`SET hnsw.ef_search = X`** — єдиний параметр, який тюнимо без rebuild'у. Sweep для Pareto.
7. **`register_vector(conn)`** на кожен коннект — інакше numpy↔vector не працює.
8. **`autocommit=True`** для bulk-скриптів — спрощує COPY/CREATE INDEX.
9. **`pg_total_relation_size`** — найчесніший вимір дискового footprint'у.
10. **`ANALYZE` після bulk load** — обов'язково, інакше planner може ігнорувати HNSW.
