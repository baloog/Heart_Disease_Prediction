# src/preprocessing.py

from typing import Tuple

from pandas import Series, read_csv, DataFrame
from sklearn.model_selection import train_test_split

from .config import (
    RAW_DATA_PATH,
    TARGET_COL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    ensure_directories,
)


def load_raw_data(path=RAW_DATA_PATH) -> DataFrame:
    """Load the raw heart dataset from CSV."""
    df = read_csv(path)
    return df


def basic_cleaning(df: DataFrame) -> DataFrame:
    """
    Basic cleaning:
    - drop duplicate rows
    - handle missing values: numeric -> median, categorical -> mode
    """
    df = df.copy()

    # Drop exact duplicates, if any
    df = df.drop_duplicates()

    # Handle missing numeric features
    for col in NUMERIC_FEATURES:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Handle missing categorical features
    for col in CATEGORICAL_FEATURES:
        if df[col].isnull().any():
            mode_val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_val)

    return df


def get_features_and_target(
    df: DataFrame,
) -> Tuple[DataFrame, Series]:
    """Split DataFrame into X (features) and y (target)."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


def train_test_split_data(
    X: DataFrame,
    y: Series,
):
    """
    Perform train/test split with stratification on the target.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Quick manual test
    ensure_directories()
    df_raw = load_raw_data()
    df_clean = basic_cleaning(df_raw)
    X, y = get_features_and_target(df_clean)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("Data loaded and split successfully.")
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
