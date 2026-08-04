"""Shared project paths and configuration values."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "messages.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

# The browser enforces the character limit and Flask independently caps the
# complete request body to prevent unexpectedly large inference requests.
MAX_MESSAGE_CHARS = 10_000
MAX_REQUEST_BYTES = 64 * 1024

EXPECTED_LABELS = frozenset({"ham", "spam"})
