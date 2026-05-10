"""
Credit Scoring Demo — повний ML пайплайн від сирих даних до рішення.

Запуск:
    python run_pipeline.py

Після завершення:
    mlflow ui --backend-store-uri sqlite:///mlflow.db   (порт 5000)
    uvicorn app.main:app --reload                       (порт 8000)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data import generate_dataset
from src.features import build_preprocessor, split_data
from src.train import train_all_models
from src.evaluate import evaluate_models
from src.registry import register_best_model


def main():
    # ── 01 ДАНІ ────────────────────────────────────────────────
    print("=" * 65)
    print("01 — ДАНІ: генерація синтетичного датасету")
    print("=" * 65)

    df = generate_dataset(n_samples=2000)
    print(f"  Рядків: {len(df)}")
    print(f"  Схвалено: {df['approved'].sum()} ({df['approved'].mean():.0%})")
    print(f"  Відмовлено: {(1 - df['approved']).sum():.0f} ({1 - df['approved'].mean():.0%})")
    print()

    # ── 02 FEATURES ────────────────────────────────────────────
    print("=" * 65)
    print("02 — FEATURES: препроцесинг + train/val/test split")
    print("=" * 65)

    # 3 набори даних:
    #   train (70%) — модель ВЧИТЬСЯ (підбирає ваги)
    #   val   (15%) — ОБИРАЄМО найкращу модель/параметри (можна використовувати багато разів)
    #   test  (15%) — ФІНАЛЬНА чесна оцінка (бачимо лише раз)
    # Чому val окремо: якщо підбирати моделі по test — ти "підглядаєш" у фінальну
    # перевірку і ризикуєш обрати модель яка випадково попала на ці рядки.
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    preprocessor = build_preprocessor()

    print(f"  train: {len(X_train)} рядків")
    print(f"  val:   {len(X_val)} рядків")
    print(f"  test:  {len(X_test)} рядків")
    print()

    # ── 03 НАВЧАННЯ ────────────────────────────────────────────
    print("=" * 65)
    print("03 — НАВЧАННЯ: 3 моделі → MLflow")
    print("=" * 65)
    print()

    results = train_all_models(preprocessor, X_train, X_val, y_train, y_val)
    print()

    # ── 04 ОЦІНКА ─────────────────────────────────────────────
    print("=" * 65)
    print("04 — ОЦІНКА: метрики на test set")
    print("=" * 65)

    best_run_id = evaluate_models(results, X_test, y_test)
    print()

    # ── 05+06 REGISTRY ─────────────────────────────────────────
    print("=" * 65)
    print("05+06 — MODEL REGISTRY: найкраща → Production")
    print("=" * 65)
    print()

    register_best_model(best_run_id)
    print()

    # ── 07 PREDICT ─────────────────────────────────────────────
    print("=" * 65)
    print("07 — PREDICT: тестуємо production модель")
    print("=" * 65)
    print()

    from src.predict import predict

    test_cases = [
        {"age": 35, "monthly_income": 45000, "num_delinquencies": 0,
         "credit_term_months": 24, "credit_amount": 100000, "debt_to_income": 0.09},
        {"age": 22, "monthly_income": 12000, "num_delinquencies": 3,
         "credit_term_months": 60, "credit_amount": 200000, "debt_to_income": 0.28},
    ]

    for tc in test_cases:
        result = predict(tc)
        print(f"  Клієнт: вік={tc['age']}, дохід={tc['monthly_income']}, "
              f"прострочень={tc['num_delinquencies']}")
        print(f"  → {result['label']}"
              + (f" (ймовірність: {result['probability']:.0%})" if result['probability'] else ""))
        print()

    # ── ГОТОВО ─────────────────────────────────────────────────
    print("=" * 65)
    print("ГОТОВО!")
    print("=" * 65)
    print()
    print("  Наступні кроки:")
    print("  1. MLflow UI:   mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("                  http://127.0.0.1:5000")
    print()
    print("  2. FastAPI UI:  uvicorn app.main:app --reload")
    print("                  http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
