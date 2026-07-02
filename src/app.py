"""
Step 5: Streamlit prototype UI.
Run AFTER both models have been trained.

Usage:
    streamlit run src/app.py
"""

import re
import string
import joblib
import streamlit as st
import pandas as pd

MODELS_DIR = "models"


def clean_text(text):
    text = re.sub(r"\w*\d\w*", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_models():
    vectorizer = joblib.load("data/processed/vectorizer.joblib")
    nb_model = joblib.load(f"{MODELS_DIR}/naive_bayes_model.joblib")
    svm_model = joblib.load(f"{MODELS_DIR}/svm_model.joblib")
    return vectorizer, nb_model, svm_model


st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.write("Compare Naive Bayes vs SVM predictions on the same message.")

vectorizer, nb_model, svm_model = load_models()

message = st.text_area("Enter a message to classify:", height=100,
                        placeholder="e.g. Congratulations! You've won a free prize, click here!")

if st.button("Classify") and message.strip():
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])

    nb_pred = nb_model.predict(vec)[0]
    nb_conf = nb_model.predict_proba(vec).max()

    svm_pred = svm_model.predict(vec)[0]
    svm_conf = svm_model.predict_proba(vec).max()

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
    comparison_df = pd.read_csv(f"{MODELS_DIR}/comparison_table.csv", index_col=0)
    st.subheader("Model Comparison (test set)")
    st.dataframe(comparison_df.round(4))
    st.image(f"{MODELS_DIR}/comparison_chart.png")
except FileNotFoundError:
    st.info("Run compare_models.py first to see the comparison table here.")
