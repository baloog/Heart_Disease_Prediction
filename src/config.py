# src/config.py

from pathlib import Path

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "heart.csv"

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "heart_disease_model.joblib"
XGB_MODEL_PATH = MODELS_DIR / "heart_disease_xgb_model.joblib"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# -------------------------
# Feature configuration
# -------------------------
TARGET_COL = "target"

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# -------------------------
# Training configuration
# -------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -------------------------
# Utility
# -------------------------
def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
