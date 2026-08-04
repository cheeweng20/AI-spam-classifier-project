"""Focused regression tests for data validation and shared preprocessing."""

import sys
import unittest
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from prepare_data import (  # noqa: E402
    validate_and_clean_data,
    validate_training_split,
)
from text_processing import clean_text  # noqa: E402
from training_utils import calculate_metrics, create_grid_search  # noqa: E402


class TextProcessingTests(unittest.TestCase):
    def test_clean_text_normalizes_case_numbers_and_punctuation(self):
        self.assertEqual(clean_text("WIN prize-123 NOW!!!"), "win prize now")

    def test_validation_normalizes_and_deduplicates_rows(self):
        data = pd.DataFrame({
            "label": ["HAM", "0", "spam", "1", "ham", "ham"],
            "text": [
                "Hello!",
                "hello",
                "WIN now",
                "Different offer",
                "123",
                "A second normal message",
            ],
        })

        cleaned = validate_and_clean_data(data, "test")

        self.assertEqual(len(cleaned), 4)
        self.assertEqual(set(cleaned["label"]), {"ham", "spam"})
        self.assertNotIn("", set(cleaned["text"]))

    def test_validation_rejects_conflicting_duplicate_labels(self):
        data = pd.DataFrame({
            "label": ["ham", "spam", "ham", "spam"],
            "text": ["Same message", "same message!", "Other ham", "Other spam"],
        })

        with self.assertRaisesRegex(ValueError, "conflicting"):
            validate_and_clean_data(data, "test")

    def test_split_rejects_too_few_training_samples_for_five_folds(self):
        X_train = pd.Series([f"train {index}" for index in range(8)])
        y_train = pd.Series(["ham"] * 4 + ["spam"] * 4)
        X_test = pd.Series(["test ham", "test spam"])
        y_test = pd.Series(["ham", "spam"])

        with self.assertRaisesRegex(ValueError, "at least 5"):
            validate_training_split(X_train, X_test, y_train, y_test)

    def test_split_rejects_train_test_message_overlap(self):
        X_train = pd.Series([f"train {index}" for index in range(10)])
        y_train = pd.Series(["ham"] * 5 + ["spam"] * 5)
        X_test = pd.Series(["train 0", "new test"])
        y_test = pd.Series(["ham", "spam"])

        with self.assertRaisesRegex(ValueError, "duplicated messages"):
            validate_training_split(X_train, X_test, y_train, y_test)


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_spam_as_the_positive_class(self):
        metrics = calculate_metrics(
            ["ham", "ham", "spam", "spam"],
            ["ham", "spam", "spam", "spam"],
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 0.8)

    def test_grid_search_uses_stratified_five_fold_cv_and_spam_f1(self):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", MultinomialNB()),
        ])
        search = create_grid_search(
            pipeline,
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "classifier__alpha": [0.5, 1.0],
            },
        )

        self.assertEqual(search.refit, "f1")
        self.assertEqual(search.cv.n_splits, 5)
        self.assertTrue(search.cv.shuffle)
        self.assertEqual(search.cv.random_state, 42)
        self.assertEqual(
            set(search.scoring),
            {"accuracy", "precision", "recall", "f1"},
        )


class StreamlitApplicationTests(unittest.TestCase):
    def run_app(self):
        app = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py")
        return app.run(timeout=30)

    def test_initial_page_renders_without_loading_a_prediction(self):
        app = self.run_app()

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "✉️ Message Spam Classifier")
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.dataframe), 1)

    def test_empty_or_number_only_message_is_rejected(self):
        for message in ("", "123 456 !!!"):
            with self.subTest(message=message):
                app = self.run_app()
                app.text_area[0].input(message)
                app.button[0].click()
                app.run(timeout=30)

                self.assertFalse(app.exception)
                self.assertIn("containing some words", app.warning[0].value)
                self.assertEqual(len(app.metric), 0)

    def test_valid_message_displays_both_predictions_and_agreement(self):
        app = self.run_app()
        app.text_area[0].input(
            "Congratulations! You have won a free cash prize. Click now."
        )
        app.button[0].click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual([metric.value for metric in app.metric], ["SPAM", "SPAM"])
        self.assertIn("Both models agree", app.success[0].value)


if __name__ == "__main__":
    unittest.main()
