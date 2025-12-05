# web/app.py

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

# ------------------------
# Paths and model loading
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.joblib"

# Must match feature order used during training
ALL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca",
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]

app = Flask(__name__)

print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")


def build_explanation_and_advice(patient: dict, pred: int, prob: float):
    """
    Very simple rule-based explanation using key risk factors
    (not true SHAP, but gives a human-friendly summary).
    """
    reasons = []

    age = patient["age"]
    trestbps = patient["trestbps"]
    chol = patient["chol"]
    thalach = patient["thalach"]
    oldpeak = patient["oldpeak"]
    ca = patient["ca"]
    exang = patient["exang"]
    cp = patient["cp"]

    # If model predicts heart disease
    if pred == 1:
        if ca >= 2:
            reasons.append("Multiple major vessels (ca) indicate reduced blood flow to the heart.")
        if oldpeak >= 2.0:
            reasons.append("High ST depression (oldpeak) during exercise suggests possible ischemia.")
        if thalach <= 130:
            reasons.append("Lower maximum heart rate (thalach) can indicate reduced exercise tolerance.")
        if exang == 1:
            reasons.append("Exercise-induced angina (exang = 1) is strongly associated with heart disease.")
        if chol >= 240:
            reasons.append("Elevated cholesterol (chol) increases cardiovascular risk.")
        if trestbps >= 140:
            reasons.append("Higher resting blood pressure (trestbps) is a risk factor.")
        if cp == 3:
            reasons.append("Asymptomatic chest pain type (cp = 3) is often seen in higher-risk patients.")
        if not reasons:
            reasons.append("The combination of your test results suggests increased heart disease risk.")

        advice = (
            "Your profile shows signs that may be associated with heart disease. "
            "Please consult a cardiologist or healthcare professional for a full evaluation. "
            "In general, focus on: regular check-ups, managing blood pressure and cholesterol, "
            "stopping smoking if you smoke, staying physically active, eating a heart-healthy diet, "
            "and managing blood sugar and stress."
        )
        status_image = "unhealthy.png"

    else:
        # Predicted as no heart disease
        if ca == 0:
            reasons.append("No major vessels (ca = 0) appear affected, which is a good sign.")
        if oldpeak < 1.0:
            reasons.append("Low ST depression (oldpeak) suggests good exercise tolerance.")
        if thalach >= 140:
            reasons.append("Higher maximum heart rate (thalach) indicates good exercise capacity.")
        if exang == 0:
            reasons.append("No exercise-induced angina (exang = 0) reported.")
        if chol < 240:
            reasons.append("Cholesterol level is in a reasonable range.")
        if not reasons:
            reasons.append("Your overall pattern looks similar to non-heart disease cases in the dataset.")

        advice = (
            "Your profile currently looks closer to non-heart disease cases in this model. "
            "This does NOT replace medical advice. Keep up healthy habits: regular physical activity, "
            "balanced diet, avoiding smoking and excessive alcohol, managing stress, and periodic health check-ups."
        )
        status_image = "healthy.png"

    # Risk category based on probability of heart disease
    if prob is not None:
        if prob < 0.3:
            risk_level = "Low estimated risk"
        elif prob < 0.6:
            risk_level = "Moderate estimated risk"
        else:
            risk_level = "High estimated risk"
    else:
        risk_level = None

    return reasons, advice, status_image, risk_level


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    reasons = None
    advice = None
    status_image = None
    risk_level = None

    if request.method == "POST":
        try:
            # Read form data
            age = float(request.form["age"])
            sex = int(request.form["sex"])
            cp = int(request.form["cp"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = int(request.form["fbs"])
            restecg = int(request.form["restecg"])
            thalach = float(request.form["thalach"])
            exang = int(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            slope = int(request.form["slope"])
            ca = float(request.form["ca"])
            thal = int(request.form["thal"])

            # Build a single-row DataFrame
            patient_data = {
                "age": age,
                "trestbps": trestbps,
                "chol": chol,
                "thalach": thalach,
                "oldpeak": oldpeak,
                "ca": ca,
                "sex": sex,
                "cp": cp,
                "fbs": fbs,
                "restecg": restecg,
                "exang": exang,
                "slope": slope,
                "thal": thal,
            }

            df_patient = pd.DataFrame([patient_data], columns=ALL_FEATURES)

            # Run prediction
            pred = model.predict(df_patient)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df_patient)[0, 1]  # prob of heart disease
            else:
                proba = None

            label = "Heart Disease" if pred == 1 else "No Heart Disease"

            prediction = label
            probability = proba

            # Explanation + advice + image
            reasons, advice, status_image, risk_level = build_explanation_and_advice(
                patient_data, pred, proba if proba is not None else 0.0
            )

        except Exception as e:
            prediction = f"Error: {e}"
            probability = None
            reasons = None
            advice = None
            status_image = None
            risk_level = None

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        reasons=reasons,
        advice=advice,
        status_image=status_image,
        risk_level=risk_level,
    )


if __name__ == "__main__":
    # Run on localhost
    app.run(debug=True)
