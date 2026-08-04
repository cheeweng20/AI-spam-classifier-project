"""
Step 2: Train Naive Bayes (YOUR part).
Run AFTER prepare_data.py.

Usage:
    python src/train_naive_bayes.py
"""

from sklearn.naive_bayes import MultinomialNB

from settings import MODELS_DIR, PROCESSED_DATA_DIR
from training_utils import train_and_evaluate_model


def main():
    train_and_evaluate_model(
        classifier=MultinomialNB(),
        classifier_parameter_grid={
            "classifier__alpha": [0.1, 0.5, 1.0],
            "classifier__fit_prior": [True, False],
        },
        model_name="Naive Bayes",
        artifact_stem="naive_bayes",
        confusion_matrix_filename="confusion_matrix_nb.png",
        color_map="Blues",
        processed_dir=PROCESSED_DATA_DIR,
        output_dir=MODELS_DIR,
    )


if __name__ == "__main__":
    main()
