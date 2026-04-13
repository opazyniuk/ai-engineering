# HW2: Data Engineering for AI

## Три завдання

| # | Завдання                                | Тип            | Що робимо                       |
|---|-----------------------------------------|----------------|---------------------------------|
| 1 | Ipynb — Локальна робота з документами   | Код (notebook) | 6 функцій обробки документів    |
| 2 | AWS Pipeline — PDF Ingestion            | Інфраструктура | PDF → S3 → SQS → Lambda → S3   |
| 3 | Самостійне вивчення                     | Теорія         | S3, SQS, Lambda, IAM            |

---

## Завдання 1: Notebook — 6 функцій

Файл: `homework/homework.ipynb`. Перевірка: `python evaluate.py` (>= 90%).

Порядок реалізації: **1 → 2 → 3 → 5 → 6 → 4** (safe_parse залежить від detect_file_type).

---

### 1. `detect_and_read()` — визначення кодування файлу

**Навіщо:** Legacy-системи віддають файли в різних кодуваннях (Windows-1251, Latin-1, UTF-8 з BOM) без вказання charset. Відкриєш з неправильним кодуванням — отримаєш кракозябри. `charset_normalizer` аналізує розподіл байтів і статистично вгадує правильне кодування.

**BOM** (Byte Order Mark) — 3 байти `\xef\xbb\xbf` на початку файлу, артефакт Windows. Не є частиною тексту — треба вирізати.

**Реалізація:**
```python
raw = file_path.read_bytes()

had_bom = raw.startswith(b"\xef\xbb\xbf")
if had_bom:
    raw = raw[3:]

result = from_bytes(raw).best()
if result is not None:
    encoding = result.encoding
    text = str(result)
else:
    encoding = "utf-8"
    text = raw.decode("utf-8", errors="replace")
```

---

### 2. `detect_file_type()` — magic bytes

**Навіщо:** Розширення файлу може брехати — хтось зберіг HTML як `.pdf`. Перші байти файлу (magic bytes) визначають реальний формат: PDF = `%PDF`, ZIP/XLSX = `PK`. Бібліотека `filetype` перевіряє ці сигнатури. HTML не має magic bytes — перевіряється вручну.

**Реалізація:**
```python
# Порожній файл
if file_path.stat().st_size == 0:
    return {..., "detected_type": None, "is_mismatch": True, "issue": "empty file"}

# Magic bytes через filetype
guess = filetype_lib.guess(str(file_path))
if guess is not None:
    detected_type = guess.extension
else:
    # filetype не визначив — перевіряємо HTML вручну
    head = file_path.read_bytes()[:512].lower()
    if b"<html" in head or b"<!doctype" in head:
        detected_type = "html"
    else:
        detected_type = None

is_mismatch = detected_type is not None and detected_type != declared_type
issue = f"declared .{declared_type} but detected {detected_type}" if is_mismatch else None
```

---

### 3. `extract_clean_text()` — чистка HTML

**Навіщо:** Enterprise HTML (Word-експорт, CMS) — 95% шуму: навігація, скрипти, стилі. BeautifulSoup парсить HTML в дерево і дозволяє видалити "шумні" гілки, а потім витягти тільки текст.

`tag.decompose()` — повністю видаляє тег з дерева разом з вмістом (не просто ховає, а знищує). `get_text(separator="\n", strip=True)` — збирає текст з усіх залишених вузлів.

**Реалізація:**
```python
soup = BeautifulSoup(raw_html, "html.parser")

for tag_name in ["script", "style", "nav", "header", "footer", "aside"]:
    for tag in soup.find_all(tag_name):
        tag.decompose()

text = soup.get_text(separator="\n", strip=True)
```

---

### 4. `safe_parse()` — безпечний парсинг

**Навіщо:** `unstructured.partition` — universal парсер для будь-яких документів (PDF, DOCX, HTML, TXT). Але він падає на corrupted/empty/binary файлах. В production потрібна обгортка: перевірити файл ДО парсингу, зловити exception ЯКЩО впаде, класифікувати помилку.

**Реалізація:**
```python
# 1. Порожній?
if file_path.stat().st_size == 0:
    return {"file": file_path.name, "status": "error",
            "error_type": "empty", "error_message": "file is empty"}

# 2. Тип файлу збігається з розширенням?
file_info = detect_file_type(str(file_path))
if file_info["is_mismatch"]:
    return {"file": file_path.name, "status": "error",
            "error_type": "type_mismatch",
            "error_message": file_info.get("issue", "type mismatch")}

# 3. Парсинг з try/except
try:
    elements = partition(filename=str(file_path))
    text = "\n".join(str(el) for el in elements)
    return {"file": file_path.name, "status": "ok",
            "text": text, "char_count": len(text)}
except Exception as e:
    return {"file": file_path.name, "status": "error",
            "error_type": "corrupted", "error_message": str(e)}
```

---

### 5. `extract_tables_from_pdf()` — таблиці з PDF

**Навіщо:** `unstructured` витягує текст з PDF лінійно — рядки і колонки таблиць перемішуються. `pdfplumber` аналізує лінії і відстані на сторінці, розпізнає табличну структуру і повертає 2D масиви. Перший рядок таблиці — заголовки, решта — дані. `dict(zip(headers, row))` конвертує кожен рядок у словник.

**Реалізація:**
```python
with pdfplumber.open(file_path) as pdf:
    for page in pdf.pages:
        for raw_table in page.extract_tables():
            headers = raw_table[0]
            table = [dict(zip(headers, row)) for row in raw_table[1:]]
            all_tables.append(table)
return all_tables
```

Очікуваний результат: 2 таблиці — revenue by region (5 рядків), revenue by product (4 рядки).

---

### 6. `chunk_text()` — chunking для RAG

**Навіщо:** Для RAG текст зберігається в vector DB як embeddings. Кожен embedding = один chunk. `RecursiveCharacterTextSplitter` ріже текст по ієрархії: спочатку по `\n\n` (абзаци), потім `\n`, потім пробілах — зберігаючи семантичні блоки цілими. `chunk_overlap` дублює кінець попереднього чанка на початку наступного, щоб контекст не губився на стиках.

**Реалізація:**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
return splitter.split_text(text)
```

---

## Завдання 2: AWS Pipeline — PDF Ingestion

**Що будуємо:** Завантажуєш PDF в S3 → автоматично тригериться SQS → Lambda читає PDF, витягує текст, зберігає .txt назад в S3.

**Навіщо така архітектура:** S3 — зберігання файлів. SQS — буфер між S3 і Lambda (якщо Lambda тимчасово перевантажена, повідомлення чекають в черзі, нічого не втрачається). Lambda — serverless обробка (платиш тільки за час виконання, не тримаєш сервер 24/7). IAM — кожен сервіс має мінімально необхідні дозволи.

### Покрокова реалізація

**Крок 1: S3 Bucket**
```
AWS Console → S3 → Create bucket
Name: pdf-ingestion-<your-name>
Region: eu-central-1 (або ваш)
```
Завантажити тестовий PDF вручну в prefix `uploads/`.

**Крок 2: SQS Queue**
```
AWS Console → SQS → Create queue
Type: Standard
Name: pdf-processing-queue
Visibility timeout: 60 seconds (>= Lambda timeout)
```

**Крок 3: S3 Event → SQS**
```
S3 bucket → Properties → Event notifications → Create
Event types: s3:ObjectCreated:*
Prefix: uploads/
Suffix: .pdf
Destination: SQS queue (pdf-processing-queue)
```

SQS потребує policy що дозволяє S3 слати повідомлення — AWS Console зазвичай пропонує додати автоматично.

**Крок 4: Lambda**

IAM Role для Lambda (створюється автоматично, але перевір permissions):
- `s3:GetObject` на `arn:aws:s3:::pdf-ingestion-*/uploads/*`
- `s3:PutObject` на `arn:aws:s3:::pdf-ingestion-*/processed/*`
- `sqs:ReceiveMessage`, `sqs:DeleteMessage`, `sqs:GetQueueAttributes`
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`

```
AWS Console → Lambda → Create function
Name: pdf-text-extractor
Runtime: Python 3.12
Trigger: SQS → pdf-processing-queue
Timeout: 30 seconds
```

pypdf не входить в Lambda runtime — потрібен Layer або ZIP deployment:
```bash
# Локально
mkdir layer && cd layer
pip install pypdf -t python/
zip -r pypdf-layer.zip python/
# Upload як Lambda Layer
```

Код Lambda:
```python
import json
import boto3
from pypdf import PdfReader
from io import BytesIO

s3 = boto3.client("s3")

def lambda_handler(event, context):
    for record in event["Records"]:
        body = json.loads(record["body"])
        s3_event = body["Records"][0]
        bucket = s3_event["s3"]["bucket"]["name"]
        key = s3_event["s3"]["object"]["key"]

        response = s3.get_object(Bucket=bucket, Key=key)
        pdf_bytes = response["Body"].read()

        reader = PdfReader(BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

        output_key = key.replace("uploads/", "processed/").replace(".pdf", ".txt")
        s3.put_object(Bucket=bucket, Key=output_key, Body=text.encode("utf-8"))

    return {"statusCode": 200}
```

**Крок 5: Тест**
```bash
aws s3 cp test.pdf s3://pdf-ingestion-<name>/uploads/test.pdf
# Зачекати ~5-10 сек
aws s3 ls s3://pdf-ingestion-<name>/processed/
# Має з'явитись test.txt
aws s3 cp s3://pdf-ingestion-<name>/processed/test.txt - | head
```

Якщо не працює — CloudWatch Logs для Lambda покаже помилку.

**Крок 6: Cleanup**
```bash
aws s3 rm s3://pdf-ingestion-<name> --recursive
# AWS Console → Billing → Budgets → Create Budget ($5 alert)
```

---

## Завдання 3: Самостійне вивчення

Що почитати по кожному сервісу — ключові речі для наступних занять:

**S3:** Як працюють buckets/objects/keys. Event notifications (тригерять SQS/Lambda). Storage classes (Standard vs Glacier — ціна vs доступність). Versioning.

**SQS:** Standard vs FIFO (порядок і дедуплікація). Visibility timeout (скільки часу consumer має на обробку перед тим як повідомлення стане видимим для інших). Dead Letter Queue (куди йдуть повідомлення що не вдалось обробити N разів).

**Lambda:** Cold start (перший виклик повільніший — завантажується runtime). Layers (спосіб додати залежності). Concurrency (скільки копій Lambda працюють паралельно). Limits (15 хв, 10GB RAM).

**IAM:** Principle of least privilege — кожен сервіс отримує мінімально необхідні дозволи. Policy document (JSON з Effect/Action/Resource). Role vs User (role — для сервісів, user — для людей).

---

## Що здавати

1. Скріншот `evaluate.py` з >= 90% для notebook
2. Скріншоти AWS pipeline: S3 bucket, SQS queue, Lambda, результат обробки PDF
3. Підтвердження cleanup (budget alert)