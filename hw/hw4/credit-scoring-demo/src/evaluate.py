"""04 — Оцінка моделей на test set: метрики + feature importance."""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src.config import FEATURE_COLUMNS


def evaluate_models(results, X_test, y_test):
    """Оцінює всі моделі на тестовому наборі. Повертає run_id найкращої."""
    print()
    rows = []
    best_f1 = -1
    best_run_id = None
    best_name = None
    best_pipe = None

    for r in results:
        pipe = r["pipeline"]
        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        rows.append({
            "модель": r["name"],
            "f1": round(f1, 4),
            "accuracy": round((y_pred == y_test).mean(), 4),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_run_id = r["run_id"]
            best_name = r["name"]
            best_pipe = pipe

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\n  → найкраща: {best_name}  (f1={best_f1:.4f})")

    # Classification report для найкращої
    y_pred_best = best_pipe.predict(X_test)
    print(f"\n  Classification report ({best_name}):")
    print(classification_report(y_test, y_pred_best,
                                target_names=["відмова", "схвалено"], zero_division=0))

    # Feature importance (для tree-based моделей)
    model_step = best_pipe.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("  Feature importance:")
        for i in sorted_idx:
            bar = "█" * int(importances[i] * 40)
            print(f"    {FEATURE_COLUMNS[i]:<25} {bar} {importances[i]:.1%}")

    return best_run_id
