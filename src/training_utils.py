"""Shared loading, model-selection, evaluation, and plotting helpers."""

import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


ARTIFACT_NAMES = (
    "X_train.joblib",
    "X_test.joblib",
    "y_train.joblib",
    "y_test.joblib",
)
RANDOM_STATE = 42
CV_FOLDS = 5


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

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Text and label row counts do not match. Reprepare the data.")
    if set(y_train) != {"ham", "spam"} or set(y_test) != {"ham", "spam"}:
        raise ValueError("Training and test labels must both contain ham and spam.")

    return X_train, X_test, y_train, y_test


def load_test_data(processed_dir):
    """Load and validate the shared test features and labels."""
    feature_path = processed_dir / "X_test.joblib"
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
    if len(X_test) != len(y_test):
        raise ValueError("Test text and label row counts do not match.")
    if set(y_test) != {"ham", "spam"}:
        raise ValueError("Test labels must contain both ham and spam.")
    return X_test, y_test


def create_grid_search(pipeline, parameter_grid):
    """Create a reproducible five-fold search optimized for spam F1-score."""
    spam_precision = make_scorer(
        precision_score, pos_label="spam", zero_division=0
    )
    spam_recall = make_scorer(recall_score, pos_label="spam", zero_division=0)
    spam_f1 = make_scorer(f1_score, pos_label="spam", zero_division=0)
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring={
            "accuracy": "accuracy",
            "precision": spam_precision,
            "recall": spam_recall,
            "f1": spam_f1,
        },
        refit="f1",
        cv=cross_validation,
        n_jobs=-1,
        return_train_score=False,
        verbose=1,
    )


def save_grid_search_results(search, model_name, artifact_stem, output_dir):
    """Save auditable cross-validation results and the selected configuration."""
    result_columns = [
        "rank_test_f1",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_precision",
        "std_test_precision",
        "mean_test_recall",
        "std_test_recall",
        "mean_test_f1",
        "std_test_f1",
        "params",
    ]
    results = pd.DataFrame(search.cv_results_)
    results = results[result_columns].sort_values("rank_test_f1")
    results.to_csv(output_dir / f"{artifact_stem}_grid_search.csv", index=False)

    summary = {
        "model": model_name,
        "cv_folds": CV_FOLDS,
        "selection_metric": "F1-score (spam is the positive class)",
        "best_cv_f1": float(search.best_score_),
        "best_parameters": search.best_params_,
    }
    with (output_dir / f"{artifact_stem}_training_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"Best 5-fold CV F1: {search.best_score_:.4f}")
    print("Best parameters:")
    for name, value in search.best_params_.items():
        print(f"  {name}: {value}")


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
