"""
Step 4: Compare both models.
Run AFTER both train_naive_bayes.py and train_svm.py.

Usage:
    python src/compare_models.py
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from settings import MODELS_DIR, PROCESSED_DATA_DIR
from training_utils import calculate_metrics, load_test_data


def main():
    model_paths = {
        "Naive Bayes": MODELS_DIR / "naive_bayes_model.joblib",
        "SVM": MODELS_DIR / "svm_model.joblib",
    }
    missing = [path.name for path in model_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing trained models in {MODELS_DIR}: {', '.join(missing)}. "
            "Train both models first."
        )

    X_test, y_test = load_test_data(PROCESSED_DATA_DIR)
    results = []
    for model_name, model_path in model_paths.items():
        model = joblib.load(model_path)
        metrics = calculate_metrics(y_test, model.predict(X_test))
        results.append({
            "model": model_name,
            **metrics,
        })

    df = pd.DataFrame(results).set_index("model")
    print("=== Model Comparison ===")
    print(df.round(4))

    df.to_csv(MODELS_DIR / "comparison_table.csv")
    print(f"\nSaved table to {MODELS_DIR}/comparison_table.csv")

    test_metrics = df[["accuracy", "precision", "recall", "f1"]].rename(
        columns={
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        }
    )
    ax = test_metrics.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Naive Bayes vs SVM")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric", loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.tight_layout()
    figure = ax.get_figure()
    figure.savefig(MODELS_DIR / "comparison_chart.png", dpi=150)
    plt.close(figure)
    print(f"Saved chart to {MODELS_DIR}/comparison_chart.png")


if __name__ == "__main__":
    main()
