"""Simple Flask interface for comparing the spam-classifier models.

Run from the project root with:
    python src/app.py
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file

from text_processing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = Flask(__name__)


def load_models():
    """Load the vectorizer and both trained classification models."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    vectorizer = joblib.load(processed_dir / "vectorizer.joblib")
    nb_model = joblib.load(models_dir / "naive_bayes_model.joblib")
    svm_model = joblib.load(models_dir / "svm_model.joblib")

    feature_count = len(vectorizer.vocabulary_)
    for model_name, model in (("Naive Bayes", nb_model), ("SVM", svm_model)):
        model_features = getattr(model, "n_features_in_", None)
        if model_features != feature_count:
            raise ValueError(
                f"{model_name} expects {model_features} features, but the "
                f"vectorizer creates {feature_count}. Retrain the models."
            )
    return vectorizer, nb_model, svm_model


def prediction_result(model, vector):
    """Return a display-ready prediction."""
    return str(model.predict(vector)[0]).upper()


def comparison_table():
    """Return the saved test-set comparison table, if it exists."""
    table_path = PROJECT_ROOT / "models" / "comparison_table.csv"
    if not table_path.is_file():
        return None
    return pd.read_csv(table_path, index_col=0).round(4).to_html(
        classes="comparison-table", border=0
    )


@app.route("/", methods=["GET", "POST"])
def index():
    message = request.form.get("message", "")
    results = None
    error = None

    try:
        vectorizer, nb_model, svm_model = load_models()
        if request.method == "POST":
            cleaned_message = clean_text(message)
            if not cleaned_message:
                error = "Enter a message containing some words before classifying."
            else:
                vector = vectorizer.transform([cleaned_message])
                results = {
                    "naive_bayes": prediction_result(nb_model, vector),
                    "svm": prediction_result(svm_model, vector),
                }
    except FileNotFoundError:
        error = (
            "The model files are missing. Run prepare_data.py and both "
            "training scripts."
        )
    except (OSError, ValueError, AttributeError) as exception:
        error = f"Could not load compatible model files: {exception}"

    chart_path = PROJECT_ROOT / "models" / "comparison_chart.png"
    return render_template(
        "index.html",
        message=message,
        results=results,
        error=error,
        comparison=comparison_table(),
        chart_available=chart_path.is_file(),
    )


@app.route("/comparison-chart")
def comparison_chart():
    """Serve the saved comparison chart."""
    chart_path = PROJECT_ROOT / "models" / "comparison_chart.png"
    if not chart_path.is_file():
        return "Not found", 404
    return send_file(chart_path)


if __name__ == "__main__":
    app.run(debug=True)
