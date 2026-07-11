"""
Step 4: Compare both models for one dataset.
Run AFTER both train_naive_bayes.py and train_svm.py have been run for that dataset.

Usage:
    python src/compare_models.py --dataset sms
    python src/compare_models.py --dataset enron
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def main(dataset):
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models" / dataset

    metric_paths = (
        models_dir / "naive_bayes_metrics.joblib",
        models_dir / "svm_metrics.joblib",
    )
    missing = [str(path.name) for path in metric_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing model metrics in {models_dir}: {', '.join(missing)}. "
            "Train both models for this dataset first."
        )

    nb_metrics, svm_metrics = (joblib.load(path) for path in metric_paths)

    df = pd.DataFrame([nb_metrics, svm_metrics]).set_index("model")
    print(f"=== Model Comparison ({dataset}) ===")
    print(df.round(4))

    df.to_csv(models_dir / "comparison_table.csv")
    print(f"\nSaved table to {models_dir}/comparison_table.csv")

    ax = df.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_ylabel("Score")
    ax.set_title(f"Naive Bayes vs SVM ({dataset})")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    figure = ax.get_figure()
    figure.savefig(models_dir / "comparison_chart.png", dpi=150)
    plt.close(figure)
    print(f"Saved chart to {models_dir}/comparison_chart.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sms", "enron"], default="sms")
    args = parser.parse_args()
    main(args.dataset)
