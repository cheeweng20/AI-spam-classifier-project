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
    """Load both fitted text-classification pipelines."""
    models_dir = PROJECT_ROOT / "models"
    nb_model = joblib.load(models_dir / "naive_bayes_model.joblib")
    svm_model = joblib.load(models_dir / "svm_model.joblib")
    return nb_model, svm_model


def prediction_result(model, message):
    """Return a display-ready prediction."""
    return str(model.predict([message])[0]).upper()


def comparison_table():
    """Return the saved test-set comparison table, if it exists."""
    table_path = PROJECT_ROOT / "models" / "comparison_table.csv"
    if not table_path.is_file():
        return None
    table = pd.read_csv(table_path).rename(columns={
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
    error = None

    try:
        nb_model, svm_model = load_models()
        if request.method == "POST":
            cleaned_message = clean_text(message)
            if not cleaned_message:
                error = "Enter a message containing some words before classifying."
            else:
                results = {
                    "naive_bayes": prediction_result(nb_model, cleaned_message),
                    "svm": prediction_result(svm_model, cleaned_message),
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
