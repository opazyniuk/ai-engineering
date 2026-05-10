"""07 — Завантаження production моделі і передбачення."""

import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME, FEATURE_COLUMNS

_model = None


def load_production_model():
    """Завантажує production модель (singleton)."""
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@Production")
    return _model


def predict(data: dict) -> dict:
    """Передбачення для одного клієнта.

    Args:
        data: dict з ключами з FEATURE_COLUMNS
    Returns:
        {"approved": 0/1, "probability": float, "label": str}
    """
    model = load_production_model()
    df = pd.DataFrame([{col: data[col] for col in FEATURE_COLUMNS}])

    prediction = int(model.predict(df)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = round(float(model.predict_proba(df)[0][1]), 4)

    return {
        "approved": prediction,
        "probability": proba,
        "label": "Схвалено" if prediction == 1 else "Відмовлено",
    }
