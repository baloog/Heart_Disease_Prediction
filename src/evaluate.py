# src/evaluate.py

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .config import DEFAULT_MODEL_PATH
from .preprocessing import (
    load_raw_data,
    basic_cleaning,
    get_features_and_target,
    train_test_split_data,
)


def evaluate_saved_model() -> None:
    """Load the saved model and evaluate it on a fresh train/test split."""
    print(f"Loading model from: {DEFAULT_MODEL_PATH}")
    model = joblib.load(DEFAULT_MODEL_PATH)

    # Reload and clean data
    df_raw = load_raw_data()
    df_clean = basic_cleaning(df_raw)
    X, y = get_features_and_target(df_clean)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("Evaluating saved model on test set...")
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
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


if __name__ == "__main__":
    evaluate_saved_model()
