"""Simple Flask interface for comparing the spam-classifier models.

Run from the project root with:
    python src/app.py
"""

from functools import lru_cache

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file

from settings import MAX_MESSAGE_CHARS, MAX_REQUEST_BYTES, MODELS_DIR
from text_processing import clean_text

COMPARISON_COLUMNS = (
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "cv_f1",
)
COMPARISON_TABLE_PATH = MODELS_DIR / "comparison_table.csv"
COMPARISON_CHART_PATH = MODELS_DIR / "comparison_chart.png"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES


@lru_cache(maxsize=1)
def load_models():
    """Load and validate both fitted pipelines once per application process."""
    nb_model = joblib.load(MODELS_DIR / "naive_bayes_model.joblib")
    svm_model = joblib.load(MODELS_DIR / "svm_model.joblib")
    for model_name, model in (
        ("Naive Bayes", nb_model),
        ("SVM", svm_model),
    ):
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"{model_name} artifact does not provide predict().")
    return nb_model, svm_model


def prediction_result(model, message):
    """Return a display-ready prediction."""
    return str(model.predict([message])[0]).upper()


def comparison_table():
    """Return the validated saved test-set comparison table, if it exists."""
    if not COMPARISON_TABLE_PATH.is_file():
        return None
    table = pd.read_csv(COMPARISON_TABLE_PATH)
    missing_columns = set(COMPARISON_COLUMNS) - set(table.columns)
    if missing_columns:
        raise ValueError(
            "The comparison table is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    table = table[list(COMPARISON_COLUMNS)].rename(columns={
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "cv_f1": "CV F1",
    })
    return table.round(4).to_html(
        classes="comparison-table",
        border=0,
        index=False,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    message = request.form.get("message", "")
    results = None
    errors = []

    if request.method == "POST":
        if len(message) > MAX_MESSAGE_CHARS:
            errors.append(
                f"Message is too long. Enter no more than "
                f"{MAX_MESSAGE_CHARS:,} characters."
            )
        else:
            cleaned_message = clean_text(message)
            if not cleaned_message:
                errors.append(
                    "Enter a message containing some words before classifying."
                )
            else:
                try:
                    nb_model, svm_model = load_models()
                    results = {
                        "naive_bayes": prediction_result(
                            nb_model, cleaned_message
                        ),
                        "svm": prediction_result(svm_model, cleaned_message),
                    }
                    results["agreement"] = (
                        results["naive_bayes"] == results["svm"]
                    )
                except FileNotFoundError:
                    errors.append(
                        "The model files are missing. Run the data-preparation "
                        "and both training scripts."
                    )
                except (
                    OSError,
                    TypeError,
                    ValueError,
                    AttributeError,
                ) as exception:
                    errors.append(
                        f"Could not load compatible model files: {exception}"
                    )

    comparison = None
    try:
        comparison = comparison_table()
    except (
        OSError,
        UnicodeError,
        ValueError,
        pd.errors.ParserError,
    ) as exception:
        errors.append(f"Could not read the saved model comparison: {exception}")

    return render_template(
        "index.html",
        message=message,
        results=results,
        errors=errors,
        comparison=comparison,
        chart_available=COMPARISON_CHART_PATH.is_file(),
        max_message_chars=MAX_MESSAGE_CHARS,
    )


@app.route("/comparison-chart")
def comparison_chart():
    """Serve the saved comparison chart."""
    if not COMPARISON_CHART_PATH.is_file():
        return "Not found", 404
    return send_file(COMPARISON_CHART_PATH)


if __name__ == "__main__":
    app.run()
