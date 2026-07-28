"""
Step 3: Train SVM (TEAMMATE's part).
Run AFTER prepare_data.py.

Usage:
    python src/train_svm.py
"""

from pathlib import Path
import joblib
from sklearn.svm import SVC
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

    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train, y_train)
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
    print(f"\nSaved model + confusion matrix to {models_dir}/")


if __name__ == "__main__":
    main()
