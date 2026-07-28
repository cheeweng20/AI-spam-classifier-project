"""Focused regression tests for data validation and shared preprocessing."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from prepare_data import validate_and_clean_data  # noqa: E402
from text_processing import clean_text  # noqa: E402
from training_utils import calculate_metrics  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
