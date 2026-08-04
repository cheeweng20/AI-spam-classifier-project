"""Focused regression tests for data validation and shared preprocessing."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import app as app_module  # noqa: E402
from prepare_data import (  # noqa: E402
    validate_and_clean_data,
    validate_training_split,
)
from settings import MAX_MESSAGE_CHARS  # noqa: E402
from text_processing import clean_text  # noqa: E402
from training_utils import calculate_metrics, create_grid_search  # noqa: E402


class StubModel:
    """Small predictable classifier used by the Flask unit tests."""

    def __init__(self, label):
        self.label = label

    def predict(self, messages):
        return [self.label for _ in messages]


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


class FlaskApplicationTests(unittest.TestCase):
    def setUp(self):
        app_module.app.config.update(TESTING=True)
        app_module.load_models.cache_clear()
        self.client = app_module.app.test_client()
        self.comparison_patcher = patch.object(
            app_module, "comparison_table", return_value=None
        )
        self.comparison_patcher.start()
        self.addCleanup(self.comparison_patcher.stop)
        self.addCleanup(app_module.load_models.cache_clear)

    def test_get_page_does_not_load_models(self):
        with patch.object(app_module, "load_models") as load_models:
            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Message Spam Classifier", response.data)
        load_models.assert_not_called()

    def test_empty_or_number_only_message_is_rejected(self):
        for message in ("", "123 456 !!!"):
            with self.subTest(message=message):
                with patch.object(app_module, "load_models") as load_models:
                    response = self.client.post("/", data={"message": message})
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"containing some words", response.data)
                load_models.assert_not_called()

    def test_oversized_message_is_rejected_before_inference(self):
        message = "a" * (MAX_MESSAGE_CHARS + 1)
        with patch.object(app_module, "load_models") as load_models:
            response = self.client.post("/", data={"message": message})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Message is too long", response.data)
        load_models.assert_not_called()

    def test_valid_message_displays_both_predictions_and_agreement(self):
        with patch.object(
            app_module,
            "load_models",
            return_value=(StubModel("spam"), StubModel("spam")),
        ):
            response = self.client.post(
                "/",
                data={"message": "Congratulations, claim your free prize."},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Both models agree", response.data)
        self.assertEqual(response.data.count(b">SPAM<"), 2)

    def test_missing_models_produce_a_user_friendly_error(self):
        with patch.object(
            app_module,
            "load_models",
            side_effect=FileNotFoundError,
        ):
            response = self.client.post(
                "/", data={"message": "A normal readable sentence"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"model files are missing", response.data)

    def test_invalid_comparison_table_produces_a_user_friendly_error(self):
        self.comparison_patcher.stop()
        with patch.object(
            app_module,
            "comparison_table",
            side_effect=ValueError("invalid columns"),
        ):
            response = self.client.get("/")
        self.comparison_patcher.start()

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Could not read the saved model comparison", response.data)

    def test_model_loader_caches_validated_artifacts(self):
        app_module.load_models.cache_clear()
        with patch.object(
            app_module.joblib,
            "load",
            side_effect=(StubModel("ham"), StubModel("spam")),
        ) as joblib_load:
            first = app_module.load_models()
            second = app_module.load_models()

        self.assertIs(first, second)
        self.assertEqual(joblib_load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
