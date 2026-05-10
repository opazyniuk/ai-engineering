"""Конфігурація проєкту credit scoring."""

import os

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlflow.db')}"
EXPERIMENT_NAME = "credit-scoring"
MODEL_NAME = "credit-scoring-model"

# Ознаки
FEATURE_COLUMNS = [
    "age",
    "monthly_income",
    "num_delinquencies",
    "credit_term_months",
    "credit_amount",
    "debt_to_income",
]

TARGET_COLUMN = "approved"

# Українські назви для UI
FEATURE_LABELS_UA = {
    "age": "Вік",
    "monthly_income": "Місячний дохід (грн)",
    "num_delinquencies": "Кількість прострочень",
    "credit_term_months": "Термін кредиту (місяці)",
    "credit_amount": "Сума кредиту (грн)",
    "debt_to_income": "Співвідношення борг/дохід",
}
