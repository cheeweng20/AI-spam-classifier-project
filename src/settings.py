"""Shared project paths, schema, and reproducibility settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "loan_approval_dataset.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

ID_COLUMN = "loan_id"
TARGET_COLUMN = "loan_status"
EXPECTED_LABELS = frozenset({"Approved", "Rejected"})

NUMERIC_FEATURES = (
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
)
CATEGORICAL_FEATURES = (
    "education",
    "self_employed",
)
FEATURE_COLUMNS = (
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
)

ALLOWED_CATEGORIES = {
    "education": frozenset({"Graduate", "Not Graduate"}),
    "self_employed": frozenset({"Yes", "No"}),
}
