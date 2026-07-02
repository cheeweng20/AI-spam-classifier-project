"""
Step 4: Compare both models for one dataset.
Run AFTER both train_naive_bayes.py and train_svm.py have been run for that dataset.

Usage:
    python src/compare_models.py --dataset sms
    python src/compare_models.py --dataset enron
"""

import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def main(dataset):
    models_dir = f"models/{dataset}"

    nb_metrics = joblib.load(f"{models_dir}/naive_bayes_metrics.joblib")
    svm_metrics = joblib.load(f"{models_dir}/svm_metrics.joblib")

    df = pd.DataFrame([nb_metrics, svm_metrics]).set_index("model")
    print(f"=== Model Comparison ({dataset}) ===")
    print(df.round(4))

    df.to_csv(f"{models_dir}/comparison_table.csv")
    print(f"\nSaved table to {models_dir}/comparison_table.csv")

    ax = df.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_ylabel("Score")
    ax.set_title(f"Naive Bayes vs SVM ({dataset})")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{models_dir}/comparison_chart.png", dpi=150)
    print(f"Saved chart to {models_dir}/comparison_chart.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sms", "enron"], default="sms")
    args = parser.parse_args()
    main(args.dataset)