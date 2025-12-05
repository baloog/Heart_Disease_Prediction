# src/train.py

import joblib
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_PATH,
    ensure_directories,
)
from .preprocessing import (
    load_raw_data,
    basic_cleaning,
    get_features_and_target,
    train_test_split_data,
)


def build_preprocessing_transformer() -> ColumnTransformer:
    """Create a ColumnTransformer for numeric + categorical features."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def build_model_pipeline() -> Pipeline:
    """
    Build the full ML pipeline:
    preprocessing (scaling + one-hot encoding) + RandomForest.
    """
    preprocessor = build_preprocessing_transformer()

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return model


def train_and_evaluate() -> None:
    """Train the model, evaluate it, and save it to disk."""
    ensure_directories()

    # 1. Load and clean data
    df_raw = load_raw_data()
    df_clean = basic_cleaning(df_raw)

    # 2. Split into features + target
    X, y = get_features_and_target(df_clean)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 4. Build model
    model = build_model_pipeline()

    # 5. Train
    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    # 6. Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)

    # For ROC-AUC need probabilities of class "1"
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = np.nan

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"Accuracy: {acc:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # 7. Save model
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, DEFAULT_MODEL_PATH)
    print(f"\nModel saved to: {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
