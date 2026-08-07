"""Compare the trained Decision Tree and Random Forest models."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from settings import MODELS_DIR, PROCESSED_DATA_DIR
from training_utils import calculate_metrics, load_test_data


def main():
    model_paths = {
        "Decision Tree": MODELS_DIR / "decision_tree_model.joblib",
        "Random Forest": MODELS_DIR / "random_forest_model.joblib",
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
        predictions = model.predict(X_test)
        results.append({
            "model": model_name,
            **calculate_metrics(y_test, predictions),
        })

    comparison = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("=== Model Comparison ===")
    print(comparison.set_index("model").round(4))
    comparison.to_csv(MODELS_DIR / "comparison_table.csv", index=False)

    chart_columns = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
    }
    chart_data = comparison.set_index("model")[list(chart_columns)].rename(
        columns=chart_columns
    )
    axis = chart_data.plot(kind="bar", figsize=(9, 5.5), rot=0)
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.set_title("Decision Tree vs Random Forest")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Metric", loc="lower right")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.tight_layout()
    figure = axis.get_figure()
    figure.savefig(MODELS_DIR / "comparison_chart.png", dpi=180)
    plt.close(figure)

    winner = comparison.iloc[0]
    print(f"\nBest F1-score: {winner['model']} ({winner['f1']:.4f})")
    print(f"Saved comparison artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
