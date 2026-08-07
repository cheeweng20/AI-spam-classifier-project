"""Streamlit interface for the trained loan approval classifiers.

Run from the project root with:
    streamlit run streamlit_app.py
"""

import joblib
import pandas as pd
import streamlit as st

from src.settings import FEATURE_COLUMNS, MODELS_DIR


MODEL_PATHS = {
    "Decision Tree": MODELS_DIR / "decision_tree_model.joblib",
    "Random Forest": MODELS_DIR / "random_forest_model.joblib",
}
COMPARISON_TABLE_PATH = MODELS_DIR / "comparison_table.csv"
COMPARISON_CHART_PATH = MODELS_DIR / "comparison_chart.png"
COMPARISON_COLUMNS = (
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
)


@st.cache_resource
def load_models():
    """Load and validate the two fitted model pipelines once per app process."""
    models = {
        model_name: joblib.load(model_path)
        for model_name, model_path in MODEL_PATHS.items()
    }
    for model_name, model in models.items():
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"{model_name} artifact does not provide predict().")
    return models


def load_comparison_table():
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
        "f1": "F1-score",
    })


def build_application_data(values):
    """Build a one-row DataFrame using the exact training feature schema."""
    application = pd.DataFrame([values])
    missing = set(FEATURE_COLUMNS) - set(application.columns)
    if missing:
        raise ValueError(f"Missing application fields: {sorted(missing)}.")
    return application.loc[:, FEATURE_COLUMNS]


def predict_application(application):
    """Return each model's predicted loan status."""
    return {
        model_name: str(model.predict(application)[0])
        for model_name, model in load_models().items()
    }


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Loan Approval Prediction")
st.write(
    "Enter an applicant's financial and loan information to compare predictions "
    "from Decision Tree and Random Forest."
)
st.info(
    "Educational demonstration only. This prototype predicts patterns in the "
    "provided dataset and must not be used to make real lending decisions."
)

with st.form("loan_application_form"):
    applicant_column, loan_column, assets_column = st.columns(3)
    with applicant_column:
        st.subheader("Applicant")
        no_of_dependents = st.number_input(
            "Number of dependents", min_value=0, max_value=20, value=2, step=1
        )
        education = st.selectbox(
            "Education", options=["Graduate", "Not Graduate"]
        )
        self_employed = st.selectbox(
            "Self-employed", options=["No", "Yes"]
        )
        income_annum = st.number_input(
            "Annual income",
            min_value=100_000,
            max_value=100_000_000,
            value=5_000_000,
            step=100_000,
            help="Use the same monetary units as the training dataset.",
        )

    with loan_column:
        st.subheader("Loan")
        loan_amount = st.number_input(
            "Loan amount",
            min_value=100_000,
            max_value=100_000_000,
            value=15_000_000,
            step=100_000,
        )
        loan_term = st.number_input(
            "Loan term", min_value=1, max_value=40, value=10, step=1
        )
        cibil_score = st.number_input(
            "CIBIL score", min_value=300, max_value=900, value=600, step=1
        )

    with assets_column:
        st.subheader("Assets")
        residential_assets_value = st.number_input(
            "Residential assets value",
            min_value=0,
            max_value=100_000_000,
            value=5_000_000,
            step=100_000,
        )
        commercial_assets_value = st.number_input(
            "Commercial assets value",
            min_value=0,
            max_value=100_000_000,
            value=3_000_000,
            step=100_000,
        )
        luxury_assets_value = st.number_input(
            "Luxury assets value",
            min_value=0,
            max_value=100_000_000,
            value=10_000_000,
            step=100_000,
        )
        bank_asset_value = st.number_input(
            "Bank asset value",
            min_value=0,
            max_value=100_000_000,
            value=4_000_000,
            step=100_000,
        )

    submitted = st.form_submit_button(
        "Predict loan status", type="primary", width="stretch"
    )

if submitted:
    application = build_application_data({
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    })
    try:
        predictions = predict_application(application)
    except FileNotFoundError:
        st.error(
            "The trained model files are missing. Run data preparation and both "
            "training scripts first."
        )
    except (OSError, TypeError, ValueError, AttributeError) as exception:
        st.error(f"Could not load compatible model files: {exception}")
    else:
        st.subheader("Prediction")
        result_columns = st.columns(len(predictions))
        for column, (model_name, predicted_label) in zip(
            result_columns, predictions.items()
        ):
            column.metric(model_name, predicted_label.upper())

        labels = set(predictions.values())
        if len(labels) == 1:
            st.success("Both models agree on this prediction.")
        else:
            st.warning(
                "The models disagree. This illustrates model uncertainty and "
                "should not be interpreted as a lending recommendation."
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
        caption="Decision Tree and Random Forest test-set performance",
        width="stretch",
    )
