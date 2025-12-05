# src/visualize_evaluation.py

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)

from .config import FIGURES_DIR
from .preprocessing import (
    load_raw_data,
    basic_cleaning,
    get_features_and_target,
    train_test_split_data,
)


def _ensure_figures_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_model_and_data():
    """Load saved model + prepare test split."""
    from .config import DEFAULT_MODEL_PATH

    model = joblib.load(DEFAULT_MODEL_PATH)

    df_raw = load_raw_data()
    df_clean = basic_cleaning(df_raw)
    X, y = get_features_and_target(df_clean)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    return model, X_test, y_test


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[0, 1],
        yticklabels=[0, 1],
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path = FIGURES_DIR / "roc_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve to {out_path}")


def plot_precision_recall(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")

    fig.tight_layout()
    out_path = FIGURES_DIR / "precision_recall_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved Precision–Recall curve to {out_path}")


def plot_feature_importance(model):
    """Plot feature importance for tree-based models (RandomForest)."""
    # Unpack pipeline: preprocess + clf
    clf = getattr(model, "named_steps", {}).get("clf", None)
    preprocessor = getattr(model, "named_steps", {}).get("preprocess", None)

    if clf is None or preprocessor is None or not hasattr(clf, "feature_importances_"):
        print("Model does not expose feature_importances_. Skipping plot.")
        return

    # Get feature names from the ColumnTransformer
    numeric_features = preprocessor.transformers_[0][2]
    cat_transformer = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]

    # OneHotEncoder feature names
    if hasattr(cat_transformer, "get_feature_names_out"):
        cat_names = cat_transformer.get_feature_names_out(cat_features)
    else:
        cat_names = cat_features

    feature_names = list(numeric_features) + list(cat_names)
    importances = clf.feature_importances_

    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Plot top 15
    top_n = min(15, len(sorted_names))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), sorted_importances[:top_n][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_names[:top_n][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances (Random Forest)")
    fig.tight_layout()

    out_path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance plot to {out_path}")


def main():
    _ensure_figures_dir()
    model, X_test, y_test = _load_model_and_data()

    # Predictions & scores
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        # fall back to decision_function if no predict_proba
        if hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            print("Model has no probability or decision scores; some plots skipped.")
            y_scores = None

    # 1. Confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # 2. ROC curve
    if y_scores is not None:
        plot_roc_curve(y_test, y_scores)
        # 3. Precision–Recall curve
        plot_precision_recall(y_test, y_scores)

    # 4. Feature importance
    plot_feature_importance(model)


if __name__ == "__main__":
    main()
