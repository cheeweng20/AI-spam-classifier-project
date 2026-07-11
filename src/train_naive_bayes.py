"""
Step 2: Train Naive Bayes (YOUR part).
Run AFTER prepare_data.py has been run for the chosen dataset.

Usage:
    python src/train_naive_bayes.py --dataset sms
    python src/train_naive_bayes.py --dataset enron
"""

import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
)


def main(dataset):
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed" / dataset
    models_dir = project_root / "models" / dataset
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train = joblib.load(processed_dir / "X_train_vec.joblib")
    X_test = joblib.load(processed_dir / "X_test_vec.joblib")
    y_train = joblib.load(processed_dir / "y_train.joblib")
    y_test = joblib.load(processed_dir / "y_test.joblib")

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam")
    rec = recall_score(y_test, y_pred, pos_label="spam")
    f1 = f1_score(y_test, y_pred, pos_label="spam")

    print(f"=== Naive Bayes Results ({dataset}) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred_ham", "pred_spam"],
                yticklabels=["actual_ham", "actual_spam"])
    plt.title(f"Naive Bayes - Confusion Matrix ({dataset})")
    plt.savefig(models_dir / "confusion_matrix_nb.png", dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump(model, models_dir / "naive_bayes_model.joblib")
    joblib.dump(
        {"model": "Naive Bayes", "accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        models_dir / "naive_bayes_metrics.joblib"
    )
    print(f"\nSaved model + confusion matrix to {models_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sms", "enron"], default="sms")
    args = parser.parse_args()
    main(args.dataset)
