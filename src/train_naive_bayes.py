"""
Step 2: Train Naive Bayes (YOUR part).
Run AFTER prepare_data.py.

Usage:
    python src/train_naive_bayes.py
"""

from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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
        ("classifier", MultinomialNB()),
    ])
    parameter_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__alpha": [0.1, 0.5, 1.0],
        "classifier__fit_prior": [True, False],
    }
    search = create_grid_search(pipeline, parameter_grid)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    save_grid_search_results(
        search,
        "Naive Bayes",
        "naive_bayes",
        models_dir,
    )

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print_results("Naive Bayes", y_test, y_pred, metrics)
    save_confusion_matrix(
        y_test,
        y_pred,
        "Naive Bayes - Confusion Matrix",
        "Blues",
        models_dir / "confusion_matrix_nb.png",
    )

    joblib.dump(model, models_dir / "naive_bayes_model.joblib")
    print(f"\nSaved model, grid-search results, and confusion matrix to {models_dir}/")


if __name__ == "__main__":
    main()
