"""05+06 — Model Registry: реєстрація найкращої моделі, staging → production."""

import mlflow
from mlflow import MlflowClient

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME


def register_best_model(best_run_id: str):
    """Реєструє найкращу модель і ставить alias Production."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"  Зареєстровано: {MODEL_NAME} v{mv.version}")

    # Ставимо alias Production
    client.set_registered_model_alias(MODEL_NAME, "Production", mv.version)
    print(f"  Alias 'Production' → v{mv.version}")
    print()
    print(f"  Завантажити в коді:")
    print(f"    model = mlflow.sklearn.load_model('models:/{MODEL_NAME}@Production')")

    return mv.version
