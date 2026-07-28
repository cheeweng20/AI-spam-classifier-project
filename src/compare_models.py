"""
Step 4: Compare both models.
Run AFTER both train_naive_bayes.py and train_svm.py.

Usage:
    python src/compare_models.py
"""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from training_utils import calculate_metrics, load_test_data


def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"

    model_paths = {
        "Naive Bayes": models_dir / "naive_bayes_model.joblib",
        "SVM": models_dir / "svm_model.joblib",
    }
    missing = [path.name for path in model_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing trained models in {models_dir}: {', '.join(missing)}. "
            "Train both models first."
        )

    X_test, y_test = load_test_data(processed_dir)
    results = []
    for model_name, model_path in model_paths.items():
        model = joblib.load(model_path)
        metrics = calculate_metrics(y_test, model.predict(X_test))
        results.append({"model": model_name, **metrics})

    df = pd.DataFrame(results).set_index("model")
    print("=== Model Comparison ===")
    print(df.round(4))

    df.to_csv(models_dir / "comparison_table.csv")
    print(f"\nSaved table to {models_dir}/comparison_table.csv")

    ax = df.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_ylabel("Score")
    ax.set_title("Naive Bayes vs SVM")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    figure = ax.get_figure()
    figure.savefig(models_dir / "comparison_chart.png", dpi=150)
    plt.close(figure)
    print(f"Saved chart to {models_dir}/comparison_chart.png")


if __name__ == "__main__":
    main()
