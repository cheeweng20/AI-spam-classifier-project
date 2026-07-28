"""Shared loading, evaluation, and plotting helpers for model training."""

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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
            "Run prepare_data.py first."
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


def load_test_data(processed_dir):
    """Load and validate the shared test features and labels."""
    feature_path = processed_dir / "X_test_vec.joblib"
    label_path = processed_dir / "y_test.joblib"
    missing = [
        path.name for path in (feature_path, label_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing processed files in {processed_dir}: {', '.join(missing)}. "
            "Run prepare_data.py first."
        )

    X_test = joblib.load(feature_path)
    y_test = joblib.load(label_path)
    if X_test.shape[0] != len(y_test):
        raise ValueError("Test feature and label row counts do not match.")
    if set(y_test) != {"ham", "spam"}:
        raise ValueError("Test labels must contain both ham and spam.")
    return X_test, y_test


def calculate_metrics(y_true, y_pred):
    """Calculate the shared spam-classification evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label="spam", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label="spam", zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, pos_label="spam", zero_division=0),
    }


def print_results(model_name, y_true, y_pred, metrics):
    """Print a consistent metric summary and classification report."""
    print(f"=== {model_name} Results ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("\n", classification_report(y_true, y_pred, zero_division=0))


def save_confusion_matrix(y_true, y_pred, title, color_map, output_path):
    """Save a consistently formatted ham/spam confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    figure, axis = plt.subplots()
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=color_map,
        xticklabels=["pred_ham", "pred_spam"],
        yticklabels=["actual_ham", "actual_spam"],
        ax=axis,
    )
    axis.set_title(title)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
