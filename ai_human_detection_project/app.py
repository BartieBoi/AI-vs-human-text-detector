import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("AI vs Human Text Detector")

# Load models

svm = joblib.load("models/svm_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

models = {
    "SVM": joblib.load("models/svm_model.pkl"),
    "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
    "AdaBoost": joblib.load("models/adaboost_model.pkl")
}

# User input

text = st.text_area("Paste text here:", height=250)

# Model selector

model_name = st.selectbox("Choose Model", ["SVM", "Decision Tree", "AdaBoost"])

# Text statistics

words = text.split()

st.metric("Word Count", len(words))
st.metric("Unique Words", len(set(words)))

# Analyze button

if st.button("Analyze") and text.strip():
    X = tfidf.transform([text])
    model = models[model_name]
    prediction = model.predict(X)[0]

    # Confidence calculation
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        confidence = (1 / (1 + np.exp(-abs(score))))

    elif hasattr(model, "predict_proba"):

        confidence = np.max(model.predict_proba(X))

    else:

        confidence = 0.0

    # Prediction output
    st.subheader("Prediction")

    if prediction == 1:
        st.success("AI Generated")
    else:
        st.success("Human Written")

    st.metric("Confidence", f"{confidence*100:.2f}%")

    # Model comparison
    results = []

    for name, m in models.items():

        pred = m.predict(X)[0]

        results.append({
            "Model": name,
            "Prediction":
                "AI Generated"
                if pred == 1
                else "Human Written"
        })

    st.subheader("Model Comparison")

    st.dataframe(pd.DataFrame(results))

    # Download report
    report = f'''

        Prediction:
            {"AI Generated" if prediction == 1 else "Human Written"}

        Confidence:
            {confidence*100:.2f}%

        Word Count:
            {len(words)}

        Unique Words:
            {len(set(words))}
    '''

    st.download_button("Download Report", report, file_name="report.txt")
