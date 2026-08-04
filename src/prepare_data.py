"""
Step 1: Prepare the dataset.
Loads data/messages.csv, cleans it, creates a shared train/test split, and
saves the text and labels to data/processed/ so both classifiers use the same
data. Feature extraction is fitted inside cross-validation in the training
scripts to prevent validation-data leakage.

Usage:
    python src/prepare_data.py
"""

import re

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from settings import (
    CV_FOLDS,
    DATA_PATH,
    EXPECTED_LABELS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from text_processing import clean_text


def load_message_data(path):
    """
    Load the message dataset from CSV.

    The expected columns are message and label, but common alternative column
    names are supported to make format errors easier to diagnose.
    """
    try:
        try:
            data = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            data = pd.read_csv(path, encoding="windows-1252")
    except pd.errors.EmptyDataError as exception:
        raise ValueError(f"The dataset is empty: {path}") from exception
    except pd.errors.ParserError as exception:
        raise ValueError(f"The dataset is not a valid CSV file: {path}") from exception

    normalized_columns = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in data.columns
    }
    text_col_candidates = ["message", "text", "body", "content"]
    label_col_candidates = [
        "label",
        "category",
        "spamham",
        "spam",
        "class",
        "target",
        "v1",
    ]

    text_col = next(
        (normalized_columns[c] for c in text_col_candidates if c in normalized_columns),
        None,
    )
    label_col = next(
        (
            normalized_columns[c]
            for c in label_col_candidates
            if c in normalized_columns
        ),
        None,
    )

    if text_col is not None and label_col is not None:
        return pd.DataFrame({
            "label": data[label_col],
            "text": data[text_col],
        })

    raise ValueError(
        f"Could not auto-detect message and label columns.\n"
        f"Actual columns in your file: {list(data.columns)}\n"
        "Expected CSV columns such as message,label."
    )


def validate_and_clean_data(data, dataset):
    """Normalize labels/text and reject unusable or ambiguous training data."""
    required_columns = {"label", "text"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"The {dataset} dataset is missing columns: {sorted(missing_columns)}."
        )

    labels = data["label"].astype("string").str.lower().str.strip()
    labels = labels.replace({
        "1": "spam", "1.0": "spam", "true": "spam", "yes": "spam",
        "0": "ham", "0.0": "ham", "false": "ham", "no": "ham",
    })
    raw_text = data["text"].fillna("").astype("string").str.strip()
    spreadsheet_errors = {
        "#div/0!",
        "#error!",
        "#n/a",
        "#name?",
        "#null!",
        "#num!",
        "#ref!",
        "#value!",
    }
    usable_text = ~raw_text.str.lower().isin(spreadsheet_errors)
    cleaned = pd.DataFrame({
        "label": labels[usable_text],
        "text": raw_text[usable_text].map(clean_text),
    }).dropna(subset=["label"])
    cleaned = cleaned[cleaned["text"].str.len() > 0]

    unknown_labels = sorted(set(cleaned["label"]) - EXPECTED_LABELS)
    if unknown_labels:
        raise ValueError(
            f"Unsupported label values in the {dataset} dataset: "
            f"{unknown_labels[:10]}. Expected ham/spam or 0/1."
        )

    conflicting = cleaned.groupby("text")["label"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        raise ValueError(
            f"The {dataset} dataset contains {len(conflicting)} messages with "
            "conflicting ham/spam labels."
        )

    cleaned = cleaned.drop_duplicates(subset="text").reset_index(drop=True)
    label_counts = cleaned["label"].value_counts()
    if set(label_counts.index) != EXPECTED_LABELS or label_counts.min() < 2:
        raise ValueError(
            f"The {dataset} dataset needs at least two usable ham and spam messages. "
            f"Found: {label_counts.to_dict()}."
        )

    return cleaned


def validate_training_split(X_train, X_test, y_train, y_test):
    """Reject a split that cannot support leakage-safe five-fold validation."""
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Text and label row counts do not match after splitting.")

    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    if set(train_counts.index) != EXPECTED_LABELS:
        raise ValueError("The training split must contain both ham and spam.")
    if set(test_counts.index) != EXPECTED_LABELS:
        raise ValueError("The test split must contain both ham and spam.")
    if train_counts.min() < CV_FOLDS:
        raise ValueError(
            f"Each training class needs at least {CV_FOLDS} messages for "
            f"{CV_FOLDS}-fold cross-validation. Found: {train_counts.to_dict()}."
        )

    overlap = set(X_train).intersection(X_test)
    if overlap:
        raise ValueError(
            f"Detected {len(overlap)} duplicated messages across the training "
            "and test sets. Reprepare and deduplicate the dataset."
        )


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training file not found: {DATA_PATH}\n"
            "Put messages.csv in the data directory."
        )
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading message data from {DATA_PATH} ...")
    raw_data = load_message_data(DATA_PATH)
    data = validate_and_clean_data(raw_data, "message")
    removed_count = len(raw_data) - len(data)
    print(f"Loaded {len(raw_data)} rows; using {len(data)} unique messages "
          f"({removed_count} empty/duplicate rows removed).")
    print(data["label"].value_counts())

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    validate_training_split(X_train, X_test, y_train, y_test)

    print(f"Training set: {len(X_train)} messages")
    print(f"Test set: {len(X_test)} messages")

    joblib.dump(X_train, PROCESSED_DATA_DIR / "X_train.joblib")
    joblib.dump(X_test, PROCESSED_DATA_DIR / "X_test.joblib")
    joblib.dump(y_train, PROCESSED_DATA_DIR / "y_train.joblib")
    joblib.dump(y_test, PROCESSED_DATA_DIR / "y_test.joblib")
    print(f"\nDone. Processed data saved to {PROCESSED_DATA_DIR}/")


if __name__ == "__main__":
    main()
