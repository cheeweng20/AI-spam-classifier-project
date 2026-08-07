"""Train and evaluate the Logistic Regression classifier."""

from sklearn.linear_model import LogisticRegression

from settings import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from training_utils import train_and_evaluate_model


def main():
    train_and_evaluate_model(
        classifier=LogisticRegression(
            max_iter=2_000,
            random_state=RANDOM_STATE,
        ),
        classifier_parameter_grid={
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__class_weight": [None, "balanced"],
        },
        scale_numeric=True,
        model_name="Logistic Regression",
        artifact_stem="logistic_regression",
        confusion_matrix_filename="confusion_matrix_logistic_regression.png",
        color_map="Blues",
        processed_dir=PROCESSED_DATA_DIR,
        output_dir=MODELS_DIR,
    )


if __name__ == "__main__":
    main()
