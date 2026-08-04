"""Shared loading, model-selection, evaluation, and plotting helpers."""

import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.pipeline import Pipeline
from settings import CV_FOLDS, EXPECTED_LABELS, RANDOM_STATE


ARTIFACT_NAMES = (
    "X_train.joblib",
    "X_test.joblib",
    "y_train.joblib",
    "y_test.joblib",
)


def _load_artifacts(processed_dir, artifact_names):
    """Load a complete set of processed artifacts."""
    missing = [
        name for name in artifact_names if not (processed_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing processed files in {processed_dir}: {', '.join(missing)}. "
            "Run prepare_data.py first."
        )
    return tuple(joblib.load(processed_dir / name) for name in artifact_names)


def load_training_data(processed_dir):
    """Load processed artifacts and fail early when they are incomplete."""
    X_train, X_test, y_train, y_test = _load_artifacts(
        processed_dir, ARTIFACT_NAMES
    )

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Text and label row counts do not match. Reprepare the data.")
    if set(y_train) != EXPECTED_LABELS or set(y_test) != EXPECTED_LABELS:
        raise ValueError("Training and test labels must both contain ham and spam.")
    train_counts = pd.Series(y_train).value_counts()
    if train_counts.min() < CV_FOLDS:
        raise ValueError(
            f"Each training class needs at least {CV_FOLDS} messages for "
            f"{CV_FOLDS}-fold cross-validation. Found: {train_counts.to_dict()}."
        )
    overlap = set(X_train).intersection(X_test)
    if overlap:
        raise ValueError(
            f"Detected {len(overlap)} duplicated messages across the stored "
            "training and test sets."
        )

    return X_train, X_test, y_train, y_test


def load_test_data(processed_dir):
    """Load and validate the shared test features and labels."""
    X_test, y_test = _load_artifacts(
        processed_dir, ("X_test.joblib", "y_test.joblib")
    )
    if len(X_test) != len(y_test):
        raise ValueError("Test text and label row counts do not match.")
    if set(y_test) != EXPECTED_LABELS:
        raise ValueError("Test labels must contain both ham and spam.")
    return X_test, y_test


def create_text_pipeline(classifier):
    """Build the shared leakage-safe text-classification pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
            ),
        ),
        ("classifier", classifier),
    ])


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
        "best_cv_f1_std": float(
            search.cv_results_["std_test_f1"][search.best_index_]
        ),
        "best_parameters": search.best_params_,
        "scikit_learn_version": sklearn.__version__,
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
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def train_and_evaluate_model(
    classifier,
    classifier_parameter_grid,
    model_name,
    artifact_stem,
    confusion_matrix_filename,
    color_map,
    processed_dir,
    output_dir,
):
    """Run the shared training, evaluation, and artifact-saving workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_training_data(processed_dir)

    parameter_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        **classifier_parameter_grid,
    }
    search = create_grid_search(
        create_text_pipeline(classifier), parameter_grid
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    save_grid_search_results(
        search, model_name, artifact_stem, output_dir
    )

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print_results(model_name, y_test, y_pred, metrics)
    save_confusion_matrix(
        y_test,
        y_pred,
        f"{model_name} - Confusion Matrix",
        color_map,
        output_dir / confusion_matrix_filename,
    )

    joblib.dump(model, output_dir / f"{artifact_stem}_model.joblib")
    print(
        f"\nSaved model, grid-search results, and confusion matrix to "
        f"{output_dir}/"
    )
