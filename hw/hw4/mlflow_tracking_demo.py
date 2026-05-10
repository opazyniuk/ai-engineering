"""
Урок 4: ML in Production — MLflow: трекінг експериментів
Щоб не забути що пробував і який запуск був найкращим.

Уяви що тренуєш модель 20 разів з різними параметрами.
Без MLflow — через тиждень не пам'ятаєш що дало F1 = 0.94.
MLflow — це щоденник кожного запуску: параметри, метрики, модель.

Що зберігає MLflow:
  - параметри кожного запуску
  - метрики (f1, accuracy, precision, recall)
  - саму модель — файл
  - датасет і версію коду

Нащо AI-інженеру:
  - порівняти fine-tuning запуски
  - відтворити найкращий результат
  - задеплоїти модель з registry
  - показати клієнту що змінилось
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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

# ── ЕКСПЕРИМЕНТИ З MLFLOW ─────────────────────────────────────
# Порівнюємо 3 запуски з різними параметрами RandomForest

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("credit-approval")

experiments = [
    {"n_estimators": 50,  "max_depth": 3},
    {"n_estimators": 100, "max_depth": 4},
    {"n_estimators": 200, "max_depth": 6},
]

print("=" * 65)
print("MLflow — трекінг експериментів")
print("=" * 65)
print()

results = []

for i, params in enumerate(experiments, 1):
    # start_run — як транзакція в БД. Все що логується всередині `with`
    # прив'язується до цього run і автоматично закривається в кінці.
    with mlflow.start_run(run_name=f"run_{i}"):

        # PARAM vs METRIC:
        # param  — конфігурація, ЗАДАНА перед тренуванням (n_estimators=50)
        # metric — результат, ПОРАХОВАНИЙ після тренування (f1=0.95)
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        # 2. Тренуємо модель
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
            )),
        ])
        pipe.fit(X_train, y_train)

        # 3. Рахуємо метрики
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # 4. Логуємо метрики
        mlflow.log_metric("accuracy", round(acc, 2))
        mlflow.log_metric("f1", round(f1, 2))
        mlflow.log_metric("precision", round(prec, 2))
        mlflow.log_metric("recall", round(rec, 2))

        # log_model — серіалізує pipeline (preprocessor + model) у файл.
        # Потім можна завантажити mlflow.sklearn.load_model() в іншому процесі
        # і робити predict без доступу до тренувального коду.
        mlflow.sklearn.log_model(pipe, "model")

        results.append({
            "запуск": f"run_{i}",
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "f1": round(f1, 2),
            "accuracy": round(acc, 2),
        })

        print(f"  run_{i}: n_estimators={params['n_estimators']:<4} max_depth={params['max_depth']}"
              f"  →  f1={f1:.2f}  accuracy={acc:.2f}")

# ── ПОРІВНЯННЯ ЗАПУСКІВ ────────────────────────────────────────

print()
print("=" * 65)
print("ПОРІВНЯННЯ ЗАПУСКІВ")
print("=" * 65)

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

best = df_results.loc[df_results["f1"].idxmax()]
print(f"\n  → найкращий: {best['запуск']}  (f1={best['f1']}, accuracy={best['accuracy']})")

# ── UI ─────────────────────────────────────────────────────────

print()
print("=" * 65)
print("MLFLOW UI")
print("=" * 65)
print()
print("  Запусти в терміналі:")
print("  $ mlflow ui")
print()
print("  Відкрий http://127.0.0.1:5000")
print("  Побачиш всі запуски, метрики, параметри, моделі")
print("  Можна порівнювати, фільтрувати, сортувати")
