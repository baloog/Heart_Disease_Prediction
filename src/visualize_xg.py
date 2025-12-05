import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    classification_report
)

from .config import XGB_MODEL_PATH, OUTPUTS_DIR
from .preprocessing import (
    load_raw_data,
    basic_cleaning,
    get_features_and_target,
    train_test_split_data
)

OUTPUT_FOLDER = OUTPUTS_DIR / "figures"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def visualize_xgboost():
    print("📦 Loading XGBoost model...")
    model = joblib.load(XGB_MODEL_PATH)

    print("📊 Loading dataset...")
    df = load_raw_data()
    df = basic_cleaning(df)
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("🔮 Running predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # =============== CONFUSION MATRIX =================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(OUTPUT_FOLDER / "confusion_matrix.png")
    plt.close()

    # =============== ROC CURVE ========================
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost ROC Curve")
    plt.legend()
    plt.savefig(OUTPUT_FOLDER / "roc_curve.png")
    plt.close()

    # =============== PRECISION-RECALL CURVE ===========
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("XGBoost Precision-Recall Curve")
    plt.savefig(OUTPUT_FOLDER / "precision_recall_curve.png")
    plt.close()

    # =============== FEATURE IMPORTANCE ===============
    try:
        # For tree-based models inside pipelines:
        booster = model.named_steps["clf"]
        importance = booster.feature_importances_

        # Get one-hot encoded feature names
        feature_names = model.named_steps["preprocess"].get_feature_names_out()

        sorted_idx = np.argsort(importance)[::-1]
        top_features = feature_names[sorted_idx][:15]
        top_importance = importance[sorted_idx][:15]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_importance, y=top_features)
        plt.title("XGBoost Feature Importance (Top 15)")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(OUTPUT_FOLDER / "feature_importance.png")
        plt.close()

    except Exception as e:
        print("⚠️ Could not extract feature importance:", e)

    print("✅ Visualization complete!")
    print(f"📁 Saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    visualize_xgboost()
