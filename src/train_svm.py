"""
Step 3: Train SVM (TEAMMATE's part).
Run AFTER prepare_data.py has been run for the chosen dataset.

Usage:
    python src/train_svm.py --dataset sms
    python src/train_svm.py --dataset enron
"""

import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from training_utils import load_training_data
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
)


def main(dataset):
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed" / dataset
    models_dir = project_root / "models" / dataset
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_training_data(processed_dir)

    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)

    print(f"=== SVM Results ({dataset}) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\n", classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    figure, axis = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["pred_ham", "pred_spam"],
        yticklabels=["actual_ham", "actual_spam"],
        ax=axis,
    )
    axis.set_title(f"SVM - Confusion Matrix ({dataset})")
    figure.savefig(models_dir / "confusion_matrix_svm.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    joblib.dump(model, models_dir / "svm_model.joblib")
    joblib.dump(
        {"model": "SVM", "accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        models_dir / "svm_metrics.joblib"
    )
    print(f"\nSaved model + confusion matrix to {models_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sms", "enron"], default="sms")
    args = parser.parse_args()
    main(args.dataset)
