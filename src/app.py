"""
Step 5: Streamlit prototype UI.
Run AFTER both models have been trained.

Usage:
    streamlit run src/app.py
"""

from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
from text_processing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_models(dataset):
    processed_dir = PROJECT_ROOT / "data" / "processed" / dataset
    models_dir = PROJECT_ROOT / "models" / dataset
    vectorizer = joblib.load(processed_dir / "vectorizer.joblib")
    nb_model = joblib.load(models_dir / "naive_bayes_model.joblib")
    svm_model = joblib.load(models_dir / "svm_model.joblib")

    feature_count = len(vectorizer.vocabulary_)
    for model_name, model in (("Naive Bayes", nb_model), ("SVM", svm_model)):
        model_features = getattr(model, "n_features_in_", None)
        if model_features != feature_count:
            raise ValueError(
                f"{model_name} expects {model_features} features, but the "
                f"{dataset} vectorizer creates {feature_count}. Retrain the models."
            )
    return vectorizer, nb_model, svm_model


st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Email / SMS Spam Classifier")
st.write("Compare Naive Bayes vs SVM predictions on the same message.")

dataset = st.sidebar.selectbox("Dataset", ["sms", "enron"])
models_dir = PROJECT_ROOT / "models" / dataset

try:
    vectorizer, nb_model, svm_model = load_models(dataset)
except FileNotFoundError:
    st.error(
        f"The {dataset} model files are missing. Run prepare_data.py and both "
        f"training scripts with --dataset {dataset}."
    )
    st.stop()
except (OSError, ValueError) as error:
    st.error(f"Could not load compatible {dataset} model files: {error}")
    st.stop()

message = st.text_area("Enter a message to classify:", height=100,
                        placeholder="e.g. Congratulations! You've won a free prize, click here!")

if st.button("Classify"):
    cleaned = clean_text(message)
    if not cleaned:
        st.warning("Enter a message containing some words before classifying.")
        st.stop()

    vec = vectorizer.transform([cleaned])

    nb_pred = nb_model.predict(vec)[0]
    nb_probabilities = nb_model.predict_proba(vec)[0]
    nb_conf = nb_probabilities[list(nb_model.classes_).index(nb_pred)]

    svm_pred = svm_model.predict(vec)[0]
    svm_probabilities = svm_model.predict_proba(vec)[0]
    svm_conf = svm_probabilities[list(svm_model.classes_).index(svm_pred)]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Naive Bayes")
        st.metric("Prediction", nb_pred.upper(), f"{nb_conf:.1%} confidence")
    with col2:
        st.subheader("SVM")
        st.metric("Prediction", svm_pred.upper(), f"{svm_conf:.1%} confidence")

st.divider()

# Show the comparison table/chart from compare_models.py if available
try:
    comparison_df = pd.read_csv(models_dir / "comparison_table.csv", index_col=0)
    st.subheader("Model Comparison (test set)")
    st.dataframe(comparison_df.round(4))
except FileNotFoundError:
    st.info("Run compare_models.py first to see the comparison table here.")

comparison_chart = models_dir / "comparison_chart.png"
if comparison_chart.is_file():
    st.image(str(comparison_chart))
