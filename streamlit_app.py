"""Streamlit interface for the trained message spam classifiers.

Run from the project root with:
    streamlit run streamlit_app.py
"""

import joblib
import pandas as pd
import streamlit as st

from src.settings import MAX_MESSAGE_CHARS, MODELS_DIR
from src.text_processing import clean_text


MODEL_PATHS = {
    "Naive Bayes": MODELS_DIR / "naive_bayes_model.joblib",
    "SVM": MODELS_DIR / "svm_model.joblib",
}
COMPARISON_TABLE_PATH = MODELS_DIR / "comparison_table.csv"
COMPARISON_CHART_PATH = MODELS_DIR / "comparison_chart.png"
COMPARISON_COLUMNS = (
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "cv_f1",
)


@st.cache_resource
def load_models():
    """Load and validate the fitted pipelines once per Streamlit process."""
    models = {
        model_name: joblib.load(model_path)
        for model_name, model_path in MODEL_PATHS.items()
    }
    for model_name, model in models.items():
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"{model_name} artifact does not provide predict().")
    return models


@st.cache_data
def load_comparison_table():
    """Load and validate the saved model-comparison results."""
    table = pd.read_csv(COMPARISON_TABLE_PATH)
    missing_columns = set(COMPARISON_COLUMNS) - set(table.columns)
    if missing_columns:
        raise ValueError(
            "The comparison table is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    return table[list(COMPARISON_COLUMNS)].rename(columns={
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "cv_f1": "CV F1",
    })


def predict_message(message):
    """Return each model's display-ready prediction for a message."""
    models = load_models()
    return {
        model_name: str(model.predict([message])[0]).upper()
        for model_name, model in models.items()
    }


st.set_page_config(
    page_title="Message Spam Classifier",
    page_icon="✉️",
    layout="centered",
)

st.title("✉️ Message Spam Classifier")
st.write(
    "Enter a message to compare predictions from Multinomial Naive Bayes "
    "and a Linear Support Vector Machine."
)

with st.form("classification_form"):
    message = st.text_area(
        "Message",
        placeholder="Type or paste a message here...",
        height=180,
        max_chars=MAX_MESSAGE_CHARS,
    )
    submitted = st.form_submit_button(
        "Classify message",
        type="primary",
        width="stretch",
    )

if submitted:
    cleaned_message = clean_text(message)
    if not cleaned_message:
        st.warning("Enter a message containing some words before classifying.")
    else:
        try:
            predictions = predict_message(cleaned_message)
        except FileNotFoundError:
            st.error(
                "The trained model files are missing. Run the data-preparation "
                "and both training scripts first."
            )
        except (OSError, TypeError, ValueError, AttributeError) as exception:
            st.error(f"Could not load compatible model files: {exception}")
        else:
            st.subheader("Prediction")
            first_column, second_column = st.columns(2)
            first_column.metric("Naive Bayes", predictions["Naive Bayes"])
            second_column.metric("SVM", predictions["SVM"])

            if predictions["Naive Bayes"] == predictions["SVM"]:
                st.success("Both models agree on this classification.")
            else:
                st.warning(
                    "The models disagree. Treat this prediction with extra caution."
                )

st.divider()
st.subheader("Model comparison")

if COMPARISON_TABLE_PATH.is_file():
    try:
        comparison = load_comparison_table()
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exception:
        st.warning(f"Could not read the saved model comparison: {exception}")
    else:
        st.dataframe(
            comparison.style.format({
                column: "{:.4f}"
                for column in comparison.columns
                if column != "Model"
            }),
            hide_index=True,
            width="stretch",
        )
else:
    st.info("Run the model-comparison script to generate the results table.")

if COMPARISON_CHART_PATH.is_file():
    st.image(
        str(COMPARISON_CHART_PATH),
        caption="Naive Bayes and SVM test-set performance",
        width="stretch",
    )
