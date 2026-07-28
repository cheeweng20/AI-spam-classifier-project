"""
Step 2: Train Naive Bayes (YOUR part).
Run AFTER prepare_data.py.

Usage:
    python src/train_naive_bayes.py
"""

from pathlib import Path
import joblib
from sklearn.naive_bayes import MultinomialNB
from training_utils import (
    calculate_metrics,
    load_training_data,
    print_results,
    save_confusion_matrix,
)


def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_training_data(processed_dir)

    model = MultinomialNB()
    model.fit(X_train, y_train)
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
    print(f"\nSaved model + confusion matrix to {models_dir}/")


if __name__ == "__main__":
    main()
