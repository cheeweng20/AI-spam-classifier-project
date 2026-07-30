"""
Step 3: Train SVM (TEAMMATE's part).
Run AFTER prepare_data.py.

Usage:
    python src/train_svm.py
"""

from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from training_utils import (
    calculate_metrics,
    create_grid_search,
    load_training_data,
    print_results,
    save_confusion_matrix,
    save_grid_search_results,
)


def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_training_data(processed_dir)

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
            ),
        ),
        ("classifier", LinearSVC(random_state=42, max_iter=5000)),
    ])
    parameter_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [0.5, 1.0, 2.0],
        "classifier__class_weight": [None, "balanced"],
    }
    search = create_grid_search(pipeline, parameter_grid)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    save_grid_search_results(
        search,
        "SVM",
        "svm",
        models_dir,
    )

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print_results("SVM", y_test, y_pred, metrics)
    save_confusion_matrix(
        y_test,
        y_pred,
        "SVM - Confusion Matrix",
        "Greens",
        models_dir / "confusion_matrix_svm.png",
    )

    joblib.dump(model, models_dir / "svm_model.joblib")
    print(f"\nSaved model, grid-search results, and confusion matrix to {models_dir}/")


if __name__ == "__main__":
    main()
