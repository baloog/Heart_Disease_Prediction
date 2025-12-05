# src/train_xgboost.py

import joblib
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

from .config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    XGB_MODEL_PATH,
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


def build_xgb_pipeline() -> Pipeline:
    """
    Build the full ML pipeline:
    preprocessing (scaling + one-hot encoding) + XGBoost classifier.
    """
    preprocessor = build_preprocessing_transformer()

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return model


def train_and_evaluate_xgb() -> None:
    """Train the XGBoost model, evaluate it, and save it to disk."""
    ensure_directories()

    # 1. Load and clean data
    df_raw = load_raw_data()
    df_clean = basic_cleaning(df_raw)

    # 2. Split into features + target
    X, y = get_features_and_target(df_clean)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 4. Build model
    model = build_xgb_pipeline()

    # 5. Train
    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    # 6. Evaluate on test set
    print("\nEvaluating XGBoost on test set...")
    y_pred = model.predict(X_test)

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
    XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, XGB_MODEL_PATH)
    print(f"\nXGBoost model saved to: {XGB_MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate_xgb()
