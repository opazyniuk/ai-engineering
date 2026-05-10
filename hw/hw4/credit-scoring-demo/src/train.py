"""03 — Тренування моделей + логування в MLflow."""

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier

from src.config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, RANDOM_STATE


# Кожна модель має свої "ручки" що контролюють складність:
#   C            — для LogisticRegression. Малий C = простіша модель (regularization).
#                  C=1.0 default; C=100 — агресивна (може перенавчитись).
#   n_estimators — кількість дерев (для RF паралельно, для XGB послідовно).
#   max_depth    — глибина одного дерева (скільки вкладених if/else).
#   learning_rate — для XGB. Малий = повільніше, стабільніше; потрібно більше дерев.
# Тут кожна модель з ОДНІЄЮ конфігурацією. Реальний тюнінг — через GridSearchCV
# або RandomizedSearchCV: автоматично перебирає комбінації і перевіряє на val.
MODELS = [
    {
        "name": "LogisticRegression — baseline",
        "model": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
    },
    {
        "name": "RandomForest — середній",
        "model": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
    },
    {
        "name": "XGBoost — фінальний",
        "model": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
    },
]


def train_all_models(preprocessor, X_train, X_val, y_train, y_val):
    """Тренує 3 моделі, логує кожну в MLflow. Повертає список результатів."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    for cfg in MODELS:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", cfg["model"]),
            ])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_val)
            metrics = {
                "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
                "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
                "accuracy": round(accuracy_score(y_val, y_pred), 4),
            }

            # Логуємо параметри
            model_obj = cfg["model"]
            params = model_obj.get_params()
            for k in ["n_estimators", "max_depth", "learning_rate", "C", "max_iter"]:
                if k in params and params[k] is not None:
                    mlflow.log_param(k, params[k])

            # Логуємо метрики
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Зберігаємо модель
            mlflow.sklearn.log_model(pipe, "model")

            results.append({
                "name": cfg["name"],
                "run_id": run.info.run_id,
                "metrics": metrics,
                "pipeline": pipe,
            })

            print(f"  {cfg['name']:<35} f1={metrics['f1']:.4f}  "
                  f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}")

    return results
