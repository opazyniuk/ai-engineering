"""02 — Feature engineering: препроцесинг + train/val/test split."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, VAL_SIZE


def build_preprocessor():
    """ColumnTransformer: StandardScaler для всіх числових ознак."""
    return ColumnTransformer([
        ("num", SkPipeline([
            ("scaler", StandardScaler()),
        ]), FEATURE_COLUMNS),
    ])


def split_data(df):
    """Розбиває датасет на train (70%) / val (15%) / test (15%)."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Спочатку відділяємо test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # Потім з решти — val
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
