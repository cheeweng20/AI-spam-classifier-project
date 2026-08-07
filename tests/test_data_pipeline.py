"""Regression tests for loan data, model evaluation, and Streamlit inference."""

import sys
import unittest
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from prepare_data import (  # noqa: E402
    validate_and_clean_data,
    validate_training_split,
)
from settings import FEATURE_COLUMNS  # noqa: E402
from training_utils import (  # noqa: E402
    calculate_metrics,
    create_grid_search,
    create_model_pipeline,
)


def valid_rows():
    return pd.DataFrame({
        "loan_id": [1, 2, 3, 4],
        "no_of_dependents": [0, 1, 2, 3],
        "education": [" Graduate", "Not Graduate", "Graduate", "Not Graduate"],
        "self_employed": [" No", "Yes", "No", "Yes"],
        "income_annum": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
        "loan_amount": [2_000_000, 4_000_000, 5_000_000, 8_000_000],
        "loan_term": [4, 8, 10, 12],
        "cibil_score": [400, 500, 700, 800],
        "residential_assets_value": [-100_000, 1_000_000, 2_000_000, 3_000_000],
        "commercial_assets_value": [0, 500_000, 1_000_000, 1_500_000],
        "luxury_assets_value": [500_000, 1_000_000, 1_500_000, 2_000_000],
        "bank_asset_value": [100_000, 200_000, 300_000, 400_000],
        "loan_status": [" Rejected", "Rejected", "Approved", "Approved"],
    })


class DataValidationTests(unittest.TestCase):
    def test_validation_normalizes_categories_and_preserves_documented_anomaly(self):
        cleaned = validate_and_clean_data(valid_rows())
        self.assertEqual(list(cleaned["education"]), [
            "Graduate", "Not Graduate", "Graduate", "Not Graduate"
        ])
        self.assertEqual(cleaned.loc[0, "residential_assets_value"], -100_000)
        self.assertIn("negative", cleaned.attrs["quality_warnings"][0])

    def test_validation_rejects_missing_column(self):
        data = valid_rows().drop(columns="cibil_score")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_and_clean_data(data)

    def test_validation_rejects_duplicate_loan_id(self):
        data = valid_rows()
        data.loc[1, "loan_id"] = 1
        with self.assertRaisesRegex(ValueError, "Duplicate loan_id"):
            validate_and_clean_data(data)

    def test_validation_rejects_unknown_category(self):
        data = valid_rows()
        data.loc[0, "education"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported or missing"):
            validate_and_clean_data(data)

    def test_split_rejects_application_overlap(self):
        cleaned = validate_and_clean_data(valid_rows())
        X = pd.concat([cleaned.loc[:, FEATURE_COLUMNS]] * 3, ignore_index=True)
        y = pd.concat([cleaned["loan_status"]] * 3, ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicated applications"):
            validate_training_split(X, X.copy(), y, y.copy())


class EvaluationTests(unittest.TestCase):
    def test_metrics_are_limited_to_the_four_required_scores(self):
        metrics = calculate_metrics(
            ["Rejected", "Rejected", "Approved", "Approved"],
            ["Rejected", "Approved", "Approved", "Approved"],
        )
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertEqual(set(metrics), {"accuracy", "precision", "recall", "f1"})

    def test_grid_search_uses_stratified_five_fold_f1(self):
        pipeline = create_model_pipeline(
            DecisionTreeClassifier(random_state=42), scale_numeric=False
        )
        search = create_grid_search(
            pipeline,
            {"classifier__max_depth": [4, 6]},
        )
        self.assertEqual(search.refit, "f1")
        self.assertEqual(search.cv.n_splits, 5)
        self.assertTrue(search.cv.shuffle)
        self.assertEqual(search.cv.random_state, 42)
        self.assertEqual(
            set(search.scoring), {"accuracy", "precision", "recall", "f1"}
        )

    def test_both_model_pipelines_fit_structured_features(self):
        cleaned = validate_and_clean_data(valid_rows())
        X = cleaned.loc[:, FEATURE_COLUMNS]
        y = cleaned["loan_status"]
        for classifier, scale in (
            (DecisionTreeClassifier(max_depth=3, random_state=42), False),
            (RandomForestClassifier(n_estimators=10, random_state=42), False),
        ):
            with self.subTest(classifier=type(classifier).__name__):
                model = create_model_pipeline(classifier, scale)
                model.fit(X, y)
                self.assertEqual(len(model.predict(X)), len(X))


class StreamlitApplicationTests(unittest.TestCase):
    def run_app(self):
        app = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py")
        return app.run(timeout=30)

    def test_initial_page_renders_model_comparison(self):
        app = self.run_app()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "🏦 Loan Approval Prediction")
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(
            list(app.dataframe[0].value.columns),
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
            ],
        )

    def test_valid_application_displays_both_predictions(self):
        app = self.run_app()
        values = [
            2,
            5_000_000,
            10_000_000,
            10,
            750,
            5_000_000,
            2_000_000,
            5_000_000,
            3_000_000,
        ]
        for widget, value in zip(app.number_input, values):
            widget.set_value(value)
        app.button[0].click()
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 2)
        self.assertTrue(
            all(metric.value in {"APPROVED", "REJECTED"} for metric in app.metric)
        )


if __name__ == "__main__":
    unittest.main()
