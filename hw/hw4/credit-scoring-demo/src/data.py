"""01 — Генерація синтетичного датасету для credit scoring."""

import numpy as np
import pandas as pd
from src.config import FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_STATE


def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """Генерує синтетичний датасет кредитного скорингу."""
    rng = np.random.RandomState(RANDOM_STATE)

    age = rng.randint(20, 66, size=n_samples)
    monthly_income = rng.normal(35000, 15000, size=n_samples).clip(5000, 120000).astype(int)
    num_delinquencies = rng.poisson(0.5, size=n_samples).clip(0, 10)
    credit_term_months = rng.choice([6, 12, 24, 36, 48, 60], size=n_samples)
    credit_amount = (monthly_income * rng.uniform(1, 8, size=n_samples)).astype(int)
    debt_to_income = np.round(credit_amount / (monthly_income * credit_term_months), 3)

    # Правило схвалення (з шумом ~10%)
    score = (
        (monthly_income > 20000).astype(float) * 1.5
        + (num_delinquencies == 0).astype(float) * 2.0
        + (debt_to_income < 0.3).astype(float) * 1.5
        + (age > 25).astype(float) * 0.5
        + (credit_term_months <= 36).astype(float) * 0.5
    )
    threshold = 3.5
    approved = (score >= threshold).astype(int)

    # Додаємо шум
    noise_mask = rng.random(n_samples) < 0.08
    approved[noise_mask] = 1 - approved[noise_mask]

    df = pd.DataFrame({
        "age": age,
        "monthly_income": monthly_income,
        "num_delinquencies": num_delinquencies,
        "credit_term_months": credit_term_months,
        "credit_amount": credit_amount,
        "debt_to_income": debt_to_income,
        "approved": approved,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    print("01 — ДАНІ")
    print(f"  Рядків: {len(df)}")
    print(f"  Схвалено: {df['approved'].sum()} ({df['approved'].mean():.0%})")
    print(f"  Відмовлено: {(1 - df['approved']).sum():.0f} ({1 - df['approved'].mean():.0%})")
    print()
    print(df.head(10).to_string(index=False))
