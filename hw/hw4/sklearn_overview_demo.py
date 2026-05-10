"""
Урок 4: ML in Production — Scikit-learn: огляд бібліотеки
Один приклад, який показує всі блоки зі слайду:
  - Алгоритми (LogisticRegression, RandomForest, XGBClassifier)
  - Препроцесинг (StandardScaler, OneHotEncoder, SimpleImputer, ColumnTransformer)
  - Утиліти (train_test_split, cross_val_score, accuracy_score, classification_report)
  - Пайплайн (Pipeline, make_pipeline)

Головна ідея: будь-яка модель працює однаково —
  model.fit(X_train, y_train)
  model.predict(X_test)
  model.score(X_test, y_test)

Хочеш замінити модель — міняєш один рядок, решта не змінюється.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ── ДАНІ ───────────────────────────────────────────────────────
# Датасет: схвалення кредиту (більший, з пропусками і категоріями)

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

print("=" * 60)
print("ДАНІ")
print("=" * 60)
print(data.to_string(index=False))
print(f"\nРядків: {len(data)} · Пропуски в 'дохід': {data['дохід'].isna().sum()}")
print()

# ── РОЗБИВКА ───────────────────────────────────────────────────

X = data.drop(columns=["рішення"])
y = data["рішення"]

# train_test_split — розрізає дані на дві частини:
# train — на цьому модель ВЧИТЬСЯ (бачить і фічі, і відповіді).
# test  — на цьому ЧЕСНО перевіряємо (модель ніколи не бачила).
# Без цього оцінка завжди оптимістична — як здати тест по списаних відповідях.
# В production-проєкті часто ділять на train/val/test (3 частини): val для вибору
# моделі/параметрів, test для фінальної перевірки. Див. credit-scoring-demo.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("=" * 60)
print("РОЗБИВКА")
print("=" * 60)
print(f"train: {len(X_train)} рядків · test: {len(X_test)} рядків")
print()

# ── ПРЕПРОЦЕСИНГ (ColumnTransformer) ──────────────────────────
# ColumnTransformer = "для різних колонок роби різні речі".
# Числові: SimpleImputer (медіана для пропусків — стійка до викидів)
#          → StandardScaler (mean=0, std=1, інакше модель "думатиме" що дохід
#          важливіший за вік просто бо числа більші)
# Категоріальні: OneHotEncoder (модель не розуміє текст, лише числа)

num_features = ["вік", "дохід", "прострочення"]
cat_features = ["освіта"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), num_features),
    # handle_unknown="ignore" — якщо в test з'явиться нова категорія
    # (напр. "PhD"), не падає з помилкою, ставить [0, 0].
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])

print("=" * 60)
print("ПРЕПРОЦЕСИНГ (ColumnTransformer)")
print("=" * 60)
print(f"  Числові {num_features}:")
print(f"    → SimpleImputer(strategy='median')  — заповнює пропуски медіаною")
print(f"    → StandardScaler()                  — нормалізує (mean=0, std=1)")
print(f"  Категоріальні {cat_features}:")
print(f"    → OneHotEncoder()                   — 'вища' → [1,0], 'середня' → [0,1]")
print()

# ── ПАЙПЛАЙН ──────────────────────────────────────────────────
# Pipeline = препроцесинг + модель в одному об'єкті.
# Гарантує що test дані пройдуть ТУ САМУ обробку що і train.
# Без Pipeline легко забути нормалізувати test або зробити це по-іншому —
# класичне джерело багів в ML (data leakage).

# ⬇ ЗАМІНИТИ МОДЕЛЬ — ЗМІНИТИ ЛИШЕ ЦЕЙ РЯДОК ⬇
model_step = LogisticRegression(random_state=42)
# model_step = RandomForestClassifier(n_estimators=100, random_state=42)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model",        model_step),
])

print("=" * 60)
print("ПАЙПЛАЙН")
print("=" * 60)
print("  Pipeline([")
print("      ('preprocessor', ColumnTransformer(...)),")
print(f"      ('model',        {type(model_step).__name__}),")
print("  ])")
print()

# ── FIT / PREDICT / SCORE ─────────────────────────────────────
# Будь-яка модель працює однаково:

pipe.fit(X_train, y_train)           # навчити
y_pred = pipe.predict(X_test)        # передбачити
score  = pipe.score(X_test, y_test)  # оцінити

print("=" * 60)
print("FIT / PREDICT / SCORE")
print("=" * 60)
print(f"  pipe.fit(X_train, y_train)      → навчили")
print(f"  pipe.predict(X_test)            → {list(y_pred)}")
print(f"  pipe.score(X_test, y_test)      → accuracy = {score:.0%}")
print()

# ── МЕТРИКИ ────────────────────────────────────────────────────

print("=" * 60)
print("МЕТРИКИ (classification_report)")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["відмова", "схвалено"], zero_division=0))

# ── CROSS-VALIDATION ───────────────────────────────────────────
# Один train/test split може випадково "пощастити". CV вирішує це:
# дані діляться на cv=3 частини. Тренує 3 рази, кожен раз інша частина — тестова.
# Результат — 3 оцінки, беремо середнє → надійніша оцінка якості моделі.

cv_scores = cross_val_score(pipe, X, y, cv=3, scoring="accuracy")

print("=" * 60)
print("CROSS-VALIDATION (cv=3)")
print("=" * 60)
print(f"  Кожен фолд: {[f'{s:.0%}' for s in cv_scores]}")
print(f"  Середнє:    {cv_scores.mean():.0%} ± {cv_scores.std():.0%}")
print()

# ── ЗАМІНА МОДЕЛІ ─────────────────────────────────────────────
# Головна ідея: міняєш один рядок — решта коду не змінюється

print("=" * 60)
print("ЗАМІНА МОДЕЛІ — один рядок")
print("=" * 60)

models = {
    "LogisticRegression": LogisticRegression(random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, m in models.items():
    p = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        m),
    ])
    cv = cross_val_score(p, X, y, cv=3, scoring="accuracy")
    print(f"  {name:<25} accuracy = {cv.mean():.0%} ± {cv.std():.0%}")

print()
print("Міняєш модель — міняєш один рядок. Решта не змінюється. ✓")
