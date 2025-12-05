# src/tune_random_forest.py

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    XGB_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    ensure_directories
)
from .preprocessing import (
    load_raw_data,
    basic_cleaning,
    get_features_and_target,
)


def build_preprocessor():
    """Column transformer."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def tune_random_forest():
    ensure_directories()

    # Load data
    df = load_raw_data()
    df = basic_cleaning(df)
    X, y = get_features_and_target(df)

    preprocessor = build_preprocessor()

    # Base estimator
    rf = RandomForestClassifier(random_state=42)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", rf)
    ])

    # Parameter space to search
    param_grid = {
        "clf__n_estimators": [200, 300, 500, 800, 1000],
        "clf__max_depth": [None, 3, 5, 7, 10],
        "clf__min_samples_split": [2, 4, 6, 10],
        "clf__min_samples_leaf": [1, 2, 3, 4],
        "clf__max_features": ["sqrt", "log2"],
        "clf__bootstrap": [True, False],
        "clf__class_weight": ["balanced", None]
    }

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=40,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    print(''
          ' Tuning RandomForest... please wait...')
    search.fit(X, y)

    print("\nBest ROC-AUC:", search.best_score_)
    print("Best params:", search.best_params_)

    # Save tuned model
    joblib.dump(search.best_estimator_, DEFAULT_MODEL_PATH)
    print(f"Tuned RandomForest saved to {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    tune_random_forest()
