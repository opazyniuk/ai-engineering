"""
Урок 4: ML in Production — Демо: Схвалення кредиту
Простий приклад ML-пайплайну від даних до оцінки моделі.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier

# ── КРОК 1 — ДАНІ ──────────────────────────────────────────
# Датасет: схвалення кредиту
# рядок = людина · мітка = рішення банку

data = pd.DataFrame({
    "вік":           [35, 22, 45, 28, 52, 19],
    "дохід":         [4500, 1200, 7000, 2100, 5500, 900],
    "прострочення":  [0, 3, 0, 2, 1, 4],
    "рішення":       [1, 0, 1, 0, 1, 0],  # 1 = схвалено, 0 = відмовлено
})

print("КРОК 1 — ДАНІ")
print("Датасет: схвалення кредиту")
print(data.to_string(index=False))
print()

# ── КРОК 2 — FEATURE ENGINEERING ────────────────────────────
# Які ознаки важливі: прострочення, дохід, вік

X = data[["вік", "дохід", "прострочення"]]
y = data["рішення"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("КРОК 2 — FEATURE ENGINEERING")
print(f"X = [вік, дохід, прострочення]")
print(f"y = {list(y.values)}")
print(f"train {int(0.8*100)}% · test {int(0.2*100)}%")
print()

# ── КРОК 3 — НАВЧАННЯ ──────────────────────────────────────
# model.fit(X_train, y_train)

# GradientBoosting — 100 дерев тренуються ПОСЛІДОВНО,
# кожне наступне виправляє помилки попередніх (на відміну від
# RandomForest де дерева тренуються паралельно і голосують).
model = GradientBoostingClassifier(
    # n_estimators — скільки дерев побудувати.
    n_estimators=100,
    # max_depth — глибина одного дерева (скільки вкладених if/else).
    # Більше = складніші правила, але ризик overfitting.
    max_depth=3,
    # learning_rate — наскільки сильно кожне нове дерево впливає на результат.
    # Малий learning_rate = повільніше, але стабільніше; потрібно більше дерев.
    learning_rate=0.1,
    random_state=42,
)
model.fit(X_train, y_train)

print("КРОК 3 — НАВЧАННЯ")
print("model.fit(X_train, y_train)")
print(f"  GradientBoostingClassifier | n_estimators=100 | max_depth=3 | learning_rate=0.1")
print(f"  модель побудувала 100 дерев, кожне виправляє помилки попереднього")
print()

# ── КРОК 4 — ОЦІНКА ────────────────────────────────────────
# Результати на тестових даних
# (на 6 рядках метрики будуть нестабільні — це демо)

y_pred = model.predict(X_test)

# accuracy  — % правильних відповідей загалом
# precision — зі всіх кому сказали "схвалено", скільки реально мали бути схвалені
# recall    — зі всіх хто реально мав бути схвалений, скільки модель знайшла
# precision і recall конфліктують: жорстка модель → високий precision, низький recall.
# F1 (не тут, але див. mlflow_tracking_demo.py) — компроміс між ними.
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

print("КРОК 4 — ОЦІНКА")
print("Результати на тестових даних:")
print(f"  Accuracy:  {acc:.0%}")
print(f"  Precision: {prec:.0%}")
print(f"  Recall:    {rec:.0%}")
print()

# ── Feature importance ──────────────────────────────────────
# Дерева вміють показати яка фіча найбільше впливала на рішення.
# Числа додаються до 1.0 — це частка "ваги" фічі в моделі.
# argsort повертає індекси відсортовані від найменшого, [::-1] реверсує.
importances = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

print("Важливість ознак:")
for i in sorted_idx:
    bar = "█" * int(importances[i] * 40)
    print(f"  {features[i]:<15} {bar} {importances[i]:.0%}")
