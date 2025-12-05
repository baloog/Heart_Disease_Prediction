# src/inference.py

from typing import Dict, Any

import joblib
import pandas as pd

from .config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    DEFAULT_MODEL_PATH,
)


def load_model(path=DEFAULT_MODEL_PATH):
    """Load the trained model pipeline from disk."""
    model = joblib.load(path)
    return model


def predict_single_patient(
    model,
    patient_data: Dict[str, Any],
) -> None:
    """
    Run inference for a single patient.

    patient_data must contain all feature keys:
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
    """

    # Ensure all required features are present
    missing_keys = [col for col in ALL_FEATURES if col not in patient_data]
    if missing_keys:
        raise ValueError(f"Missing keys in patient_data: {missing_keys}")

    # Create DataFrame with a single row
    df_patient = pd.DataFrame([patient_data], columns=ALL_FEATURES)

    # Run model
    pred = model.predict(df_patient)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_patient)[0, 1]  # prob of class '1'
    else:
        proba = None

    label = "Heart Disease" if pred == 1 else "No Heart Disease"

    print("=== Inference Result ===")
    print(df_patient)
    print()
    print(f"Prediction: {label} (class = {pred})")
    if proba is not None:
        print(f"Confidence (probability of heart disease): {proba:.4f}")


if __name__ == "__main__":
    # Example usage with a sample patient
    model = load_model()

    example_patient = {
        "age": 67,
        "sex": 1,
        "cp": 4,
        "trestbps": 160,
        "chol": 286,
        "fbs": 0,
        "restecg": 2,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 3,
        "thal": 3
    }

    predict_single_patient(model, example_patient)
