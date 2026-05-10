"""
Урок 4: ML in Production — Model Registry
Сховище моделей з версіями, стадіями і релізами.

Це як Git для моделей. Кожна навчена модель — це коміт.
Можна бачити всі версії, перемикатись між ними,
позначити яка зараз у продакшені — і відкотитись якщо
нова версія зламала щось у реальності.

Стадії моделі:
  v1  →  archived     (перша версія, замінена)
  v2  →  archived     (була в продакшені)
  v3  →  Production   (поточна робоча модель)
  v4  →  Staging      (тестується перед релізом)

Аналог у світі LLM:
  - HuggingFace Hub — теж registry
  - версії fine-tuned моделей
  - той самий принцип: staging → prod
  - відкат якщо нова версія гірша
"""

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ── ДАНІ ───────────────────────────────────────────────────────

data = pd.DataFrame({
    "вік":          [35, 22, 45, 28, 52, 19, 40, 31, 55, 24, 38, 27, 60, 33, 21, 47, 29, 50, 36, 23],
    "дохід":        [4500, 1200, 7000, 2100, 5500, 900, 6200, 3100, 8000, 1500,
                     5000, None, 7500, 3800, 1100, 6800, 2400, None, 4800, 1300],
    "прострочення": [0, 3, 0, 2, 1, 4, 0, 1, 0, 3, 0, 2, 0, 1, 4, 0, 2, 1, 0, 3],
    "освіта":       ["вища", "середня", "вища", "середня", "вища", "середня",
                     "вища", "вища", "вища", "середня", "вища", "середня",
                     "вища", "вища", "середня", "вища", "середня", "вища", "вища", "середня"],
    "рішення":      [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
})

X = data.drop(columns=["рішення"])
y = data["рішення"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ── ПРЕПРОЦЕСИНГ ──────────────────────────────────────────────

num_features = ["вік", "дохід", "прострочення"]
cat_features = ["освіта"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])

# ── ІМЕНА ──────────────────────────────────────────────────────

MODEL_NAME = "fraud-detector"

# ── КРОК 1: ТРЕНУЄМО КІЛЬКА ВЕРСІЙ ────────────────────────────
# Кожен run = нова версія моделі з різними параметрами

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("credit-approval-registry")

versions_config = [
    {"name": "v1 — baseline",      "model": LogisticRegression(random_state=42)},
    {"name": "v2 — RF shallow",    "model": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)},
    {"name": "v3 — RF medium",     "model": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)},
    {"name": "v4 — RF deep",       "model": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)},
]

print("=" * 65)
print("КРОК 1 — ТРЕНУЄМО 4 ВЕРСІЇ МОДЕЛІ")
print("=" * 65)
print()

run_ids = []

for cfg in versions_config:
    with mlflow.start_run(run_name=cfg["name"]) as run:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model",        cfg["model"]),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("f1", round(f1, 2))
        mlflow.log_metric("accuracy", round(acc, 2))
        mlflow.sklearn.log_model(pipe, "model")

        run_ids.append(run.info.run_id)
        print(f"  {cfg['name']:<25} f1={f1:.2f}  accuracy={acc:.2f}  run_id={run.info.run_id[:8]}...")

# ── КРОК 2: РЕЄСТРУЄМО МОДЕЛІ В REGISTRY ─────────────────────
# mlflow.register_model() — кожен виклик створює нову версію

print()
print("=" * 65)
print("КРОК 2 — РЕЄСТРАЦІЯ В MODEL REGISTRY")
print("=" * 65)
print(f"  Модель: '{MODEL_NAME}'")
print()

client = MlflowClient()

model_versions = []
for i, run_id in enumerate(run_ids):
    # register_model — бере модель з конкретного run і реєструє під спільним
    # іменем у Registry. Кожен виклик з тим самим іменем = нова версія.
    # Аналог git tag v1, v2, v3...
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    model_versions.append(mv)
    print(f"  Зареєстровано: {MODEL_NAME} версія {mv.version}  (run_id={run_id[:8]}...)")

# ── КРОК 3: ПРИЗНАЧАЄМО СТАДІЇ (ALIASES) ─────────────────────
# Alias — це мітка, яка вказує на конкретну версію
# "Production"  = версія в продакшені
# "Staging"     = версія на тестуванні
# "champion"    = найкраща модель

print()
print("=" * 65)
print("КРОК 3 — ПРИЗНАЧАЄМО СТАДІЇ (aliases)")
print("=" * 65)
print()

# Alias = вказівник на версію (як симлінк або DNS-запис).
# Деплой = переключити alias на нову версію. Відкат = повернути на стару.
# Сервіс завжди завантажує "@Production", не конкретну версію v3 —
# тому деплой не вимагає змін у коді сервісу.
client.set_registered_model_alias(MODEL_NAME, "Production", model_versions[2].version)
print(f"  версія {model_versions[2].version} → Production   (поточна робоча модель)")

# v4 → Staging (тестуємо перед релізом)
client.set_registered_model_alias(MODEL_NAME, "Staging", model_versions[3].version)
print(f"  версія {model_versions[3].version} → Staging      (тестується перед релізом)")

# v1, v2 → просто старі версії (без alias = archived)
print(f"  версія {model_versions[0].version} → (archived)   (перша версія, замінена)")
print(f"  версія {model_versions[1].version} → (archived)   (була в продакшені)")

# ── КРОК 4: ЗАВАНТАЖУЄМО PRODUCTION МОДЕЛЬ ────────────────────
# В коді сервісу завжди завантажуємо по alias — не по версії

print()
print("=" * 65)
print("КРОК 4 — ЗАВАНТАЖЕННЯ PRODUCTION МОДЕЛІ")
print("=" * 65)
print()

# Завантажити поточну production модель
prod_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@Production")

y_pred_prod = prod_model.predict(X_test)
f1_prod = f1_score(y_test, y_pred_prod, zero_division=0)

print(f"  model = mlflow.sklearn.load_model('models:/{MODEL_NAME}@Production')")
print(f"  Prediction: {list(y_pred_prod)}")
print(f"  F1 score:   {f1_prod:.2f}")

# ── КРОК 5: ПЕРЕВОДИМО STAGING → PRODUCTION ──────────────────
# Якщо staging модель краща — робимо її production

print()
print("=" * 65)
print("КРОК 5 — ПЕРЕВОДИМО STAGING → PRODUCTION")
print("=" * 65)
print()

staging_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@Staging")
y_pred_staging = staging_model.predict(X_test)
f1_staging = f1_score(y_test, y_pred_staging, zero_division=0)

print(f"  Staging  (v{model_versions[3].version}): f1={f1_staging:.2f}")
print(f"  Production (v{model_versions[2].version}): f1={f1_prod:.2f}")

if f1_staging >= f1_prod:
    client.set_registered_model_alias(MODEL_NAME, "Production", model_versions[3].version)
    print(f"\n  → Staging модель не гірша — переводимо v{model_versions[3].version} в Production!")
else:
    print(f"\n  → Staging модель гірша — залишаємо v{model_versions[2].version} в Production")

# ── КРОК 6: СПИСОК ВСІХ ВЕРСІЙ ───────────────────────────────

print()
print("=" * 65)
print("КРОК 6 — ВСІ ВЕРСІЇ МОДЕЛІ")
print("=" * 65)
print()

# Отримати всі версії
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

# Отримати aliases
registered_model = client.get_registered_model(MODEL_NAME)
aliases = registered_model.aliases  # dict: alias → version

# Побудувати зворотну мапу: version → list of aliases
version_aliases = {}
for alias, ver in aliases.items():
    version_aliases.setdefault(ver, []).append(alias)

print(f"  {'версія':<10} {'aliases':<20} {'f1':<8} {'accuracy':<10}")
print(f"  {'─'*10} {'─'*20} {'─'*8} {'─'*10}")

for mv in sorted(all_versions, key=lambda x: int(x.version)):
    run = client.get_run(mv.run_id)
    f1_val = run.data.metrics.get("f1", "—")
    acc_val = run.data.metrics.get("accuracy", "—")
    als = version_aliases.get(mv.version, ["—"])
    print(f"  v{mv.version:<9} {', '.join(als):<20} {f1_val:<8} {acc_val:<10}")

# ── ПІДСУМОК ──────────────────────────────────────────────────

print()
print("=" * 65)
print("ПІДСУМОК")
print("=" * 65)
print("""
  Model Registry — це Git для моделей:

  1. register_model()       — зареєструвати нову версію
  2. set_alias("Production") — позначити яка в проді
  3. load_model("@Production") — завантажити по alias
  4. set_alias("Production", new_version) — переключити прод

  В коді сервісу ЗАВЖДИ:
    model = mlflow.sklearn.load_model("models:/fraud-detector@Production")

  Деплой = змінити alias. Відкат = повернути alias на стару версію.
""")
