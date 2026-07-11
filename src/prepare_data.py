"""
Step 1: Prepare the dataset.
Supports multiple raw dataset formats (SMS Spam Collection, Enron-Spam CSV).
Run once per dataset. Cleans, splits, vectorizes, and saves to
data/processed/<dataset>/ so both teammates train on identical data.

Usage:
    python src/prepare_data.py --dataset sms
    python src/prepare_data.py --dataset enron
"""

import re
import argparse
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from text_processing import clean_text

RANDOM_STATE = 42
TEST_SIZE = 0.3

# Resolve files from the project directory, not from whichever directory the
# command happens to be run in.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where each dataset's raw file lives. Adjust the filename if yours differs.
DATASET_PATHS = {
    "sms": PROJECT_ROOT / "data" / "SMSSpamCollection",
    "enron": PROJECT_ROOT / "data" / "enron_spam_data.csv",
}


def load_sms_data(path):
    """UCI SMS Spam Collection: tab-separated, no header, 2 columns (label, text)."""
    data = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="windows-1252",
    )
    return data


def load_enron_data(path):
    """
    Enron-Spam CSV. Column names vary between Kaggle uploads, so we try to
    auto-detect the text/label columns and report the available headers when
    the format is unsupported.
    """
    try:
        data = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Some Enron exports use the Windows Western encoding.
        data = pd.read_csv(path, encoding="windows-1252")

    # Match headers without being sensitive to spaces, punctuation, or case.
    normalized_columns = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in data.columns
    }
    text_col_candidates = ["text", "message", "body", "content", "email"]
    label_col_candidates = ["label", "category", "spamham", "spam", "class", "target"]

    text_col = next(
        (normalized_columns[c] for c in text_col_candidates if c in normalized_columns),
        None,
    )
    label_col = next(
        (normalized_columns[c] for c in label_col_candidates if c in normalized_columns),
        None,
    )

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect columns.\nActual columns in your file: {list(data.columns)}\n"
            "Expected a text/message/body column and a label/category/spam-ham column."
        )

    # The included Enron file has both Subject and Message. Keep a subject-only
    # email instead of dropping it, and include the subject when a body exists.
    text = data[text_col].fillna("").astype(str).str.strip()
    subject_col = normalized_columns.get("subject")
    if subject_col is not None and subject_col != text_col:
        subject = data[subject_col].fillna("").astype(str).str.strip()
        text = (subject + " " + text).str.strip()

    labels = data[label_col].astype("string").str.lower().str.strip()
    labels = labels.replace({
        "1": "spam", "1.0": "spam", "true": "spam", "yes": "spam",
        "0": "ham", "0.0": "ham", "false": "ham", "no": "ham",
    })
    return pd.DataFrame({"label": labels, "text": text})


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
    cleaned = pd.DataFrame({
        "label": labels,
        "text": data["text"].fillna("").map(clean_text),
    }).dropna(subset=["label"])
    cleaned = cleaned[cleaned["text"].str.len() > 0]

    unknown_labels = sorted(set(cleaned["label"]) - {"ham", "spam"})
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
    if set(label_counts.index) != {"ham", "spam"} or label_counts.min() < 2:
        raise ValueError(
            f"The {dataset} dataset needs at least two usable ham and spam messages. "
            f"Found: {label_counts.to_dict()}."
        )

    return cleaned


LOADERS = {
    "sms": load_sms_data,
    "enron": load_enron_data,
}


def main(dataset):
    raw_path = DATASET_PATHS[dataset]
    processed_dir = PROJECT_ROOT / "data" / "processed" / dataset

    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Training file not found: {raw_path}\n"
            f"Put the '{dataset}' training file at that location."
        )
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{dataset}' data from {raw_path} ...")
    raw_data = LOADERS[dataset](raw_path)
    data = validate_and_clean_data(raw_data, dataset)
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

    vectorizer = CountVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Training set: {X_train_vec.shape[0]} messages, {X_train_vec.shape[1]} features")
    print(f"Test set: {X_test_vec.shape[0]} messages")

    joblib.dump(X_train_vec, processed_dir / "X_train_vec.joblib")
    joblib.dump(X_test_vec, processed_dir / "X_test_vec.joblib")
    joblib.dump(y_train, processed_dir / "y_train.joblib")
    joblib.dump(y_test, processed_dir / "y_test.joblib")
    joblib.dump(vectorizer, processed_dir / "vectorizer.joblib")
    joblib.dump(
        {
            "dataset": dataset,
            "raw_rows": len(raw_data),
            "usable_rows": len(data),
            "features": X_train_vec.shape[1],
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "sklearn_version": sklearn.__version__,
        },
        processed_dir / "metadata.joblib",
    )

    print(f"\nDone. Processed data saved to {processed_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sms", "enron"], default="sms")
    args = parser.parse_args()
    main(args.dataset)
