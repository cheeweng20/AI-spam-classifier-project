"""
Step 3: Train SVM (TEAMMATE's part).
Run AFTER prepare_data.py.

Usage:
    python src/train_svm.py
"""

from sklearn.svm import LinearSVC

from settings import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from training_utils import train_and_evaluate_model


def main():
    train_and_evaluate_model(
        classifier=LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
        classifier_parameter_grid={
            "classifier__C": [0.5, 1.0, 2.0],
            "classifier__class_weight": [None, "balanced"],
        },
        model_name="SVM",
        artifact_stem="svm",
        confusion_matrix_filename="confusion_matrix_svm.png",
        color_map="Greens",
        processed_dir=PROCESSED_DATA_DIR,
        output_dir=MODELS_DIR,
    )


if __name__ == "__main__":
    main()
