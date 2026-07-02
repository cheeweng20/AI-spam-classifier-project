"""
Step 1: Prepare the dataset.
Supports multiple raw dataset formats (SMS Spam Collection, Enron-Spam CSV).
Run once per dataset. Cleans, splits, vectorizes, and saves to
data/processed/<dataset>/ so both teammates train on identical data.

Usage:
    python src/prepare_data.py --dataset sms
    python src/prepare_data.py --dataset enron
"""

import os
import re
import string
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

RANDOM_STATE = 42
TEST_SIZE = 0.3

# Where each dataset's raw file lives. Adjust the path if your filename differs.
DATASET_PATHS = {
    "sms": "data/SMSSpamCollection",
    "enron": "data\enron_spam_data.csv",
}


def clean_text(text):
    """Remove numbers, punctuation, lowercase everything."""
    text = re.sub(r"\w*\d\w*", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_sms_data(path):
    """UCI SMS Spam Collection: tab-separated, no header, 2 columns (label, text)."""
    data = pd.read_table(path, header=None, names=["label", "text"], encoding="windows-1252")
    return data


def load_enron_data(path):
    """
    Enron-Spam CSV. Column names vary between Kaggle uploads, so we try to
    auto-detect the text/label columns. If detection fails, print the real
    column names and edit the candidate lists below to match.
    """
    data = pd.read_csv(path)

    text_col_candidates = ["text", "Message", "message", "body", "Body", "Subject"]
    label_col_candidates = ["label", "Category", "category", "Spam/Ham", "spam", "class", "target"]

    text_col = next((c for c in text_col_candidates if c in data.columns), None)
    label_col = next((c for c in label_col_candidates if c in data.columns), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect columns.\nActual columns in your file: {list(data.columns)}\n"
            f"Add the real column names to text_col_candidates / label_col_candidates above."
        )

    data = data.rename(columns={text_col: "text", label_col: "label"})[["label", "text"]]
    data = data.dropna(subset=["text", "label"])

    # Normalize label to the strings "spam" / "ham", whatever the original encoding was.
    if data["label"].dtype != object:
        # Numeric labels: CHECK which number means spam in your actual file first
        # (print data['label'].value_counts()) and fix this mapping if needed.
        data["label"] = data["label"].map({1: "spam", 0: "ham"})
    else:
        data["label"] = data["label"].str.lower().str.strip()
        data["label"] = data["label"].replace({"1": "spam", "0": "ham"})

    return data


LOADERS = {
    "sms": load_sms_data,
    "enron": load_enron_data,
}


def main(dataset):
    raw_path = DATASET_PATHS[dataset]
    processed_dir = f"data/processed/{dataset}"
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading '{dataset}' data from {raw_path} ...")
    data = LOADERS[dataset](raw_path)
    print(f"Loaded {len(data)} messages.")
    print(data["label"].value_counts())

    data["text"] = data["text"].map(clean_text)

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    vectorizer = CountVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Training set: {X_train_vec.shape[0]} messages, {X_train_vec.shape[1]} features")
    print(f"Test set: {X_test_vec.shape[0]} messages")

    joblib.dump(X_train_vec, f"{processed_dir}/X_train_vec.joblib")
    joblib.dump(X_test_vec, f"{processed_dir}/X_test_vec.joblib")
    joblib.dump(y_train, f"{processed_dir}/y_train.joblib")
    joblib.dump(y_test, f"{processed_dir}/y_test.joblib")
    joblib.dump(vectorizer, f"{processed_dir}/vectorizer.joblib")

    print(f"\nDone. Processed data saved to {processed_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sms", "enron"], default="sms")
    args = parser.parse_args()
    main(args.dataset)