"""Train and evaluate the Random Forest classifier."""

from sklearn.ensemble import RandomForestClassifier

from settings import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from training_utils import train_and_evaluate_model


def main():
    train_and_evaluate_model(
        classifier=RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        classifier_parameter_grid={
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__class_weight": [None, "balanced"],
        },
        scale_numeric=False,
        model_name="Random Forest",
        artifact_stem="random_forest",
        confusion_matrix_filename="confusion_matrix_random_forest.png",
        color_map="Greens",
        processed_dir=PROCESSED_DATA_DIR,
        output_dir=MODELS_DIR,
    )


if __name__ == "__main__":
    main()
