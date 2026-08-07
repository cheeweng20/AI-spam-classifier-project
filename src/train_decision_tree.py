"""Train and evaluate the Decision Tree classifier."""

from sklearn.tree import DecisionTreeClassifier

from settings import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from training_utils import train_and_evaluate_model


def main():
    train_and_evaluate_model(
        classifier=DecisionTreeClassifier(random_state=RANDOM_STATE),
        classifier_parameter_grid={
            "classifier__max_depth": [3, 4, 5, 6, 8, 10, None],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__class_weight": [None, "balanced"],
        },
        scale_numeric=False,
        model_name="Decision Tree",
        artifact_stem="decision_tree",
        confusion_matrix_filename="confusion_matrix_decision_tree.png",
        color_map="Oranges",
        processed_dir=PROCESSED_DATA_DIR,
        output_dir=MODELS_DIR,
    )


if __name__ == "__main__":
    main()
