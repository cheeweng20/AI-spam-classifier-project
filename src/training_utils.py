"""Validation helpers shared by model-training scripts."""

import joblib


ARTIFACT_NAMES = (
    "X_train_vec.joblib",
    "X_test_vec.joblib",
    "y_train.joblib",
    "y_test.joblib",
)


def load_training_data(processed_dir):
    """Load processed artifacts and fail early when they are incomplete."""
    missing = [name for name in ARTIFACT_NAMES if not (processed_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed files in {processed_dir}: {', '.join(missing)}. "
            "Run prepare_data.py for this dataset first."
        )

    X_train, X_test, y_train, y_test = (
        joblib.load(processed_dir / name) for name in ARTIFACT_NAMES
    )

    if X_train.shape[0] != len(y_train) or X_test.shape[0] != len(y_test):
        raise ValueError("Feature and label row counts do not match. Reprepare the data.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test feature counts differ. Reprepare the data.")
    if set(y_train) != {"ham", "spam"} or set(y_test) != {"ham", "spam"}:
        raise ValueError("Training and test labels must both contain ham and spam.")

    return X_train, X_test, y_train, y_test
