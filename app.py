import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model.preprocessing import preprocess_data

st.set_page_config(page_title="Bank Marketing ML Classifier", layout="wide")

st.title("📊 Bank Marketing Classification App")
st.markdown("Compare multiple ML models for term deposit prediction.")

# Load available models
MODEL_DIR = "saved_models"

available_models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# Sidebar model selection
selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(available_models.keys())
)

model_path = os.path.join(MODEL_DIR, available_models[selected_model_name])

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file not found. Please train models first.")
    st.stop()

# Upload test dataset
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(test_df.head())

    if "deposit" in test_df.columns:
        X_test = test_df.drop("deposit", axis=1)
        y_test = test_df["deposit"].map({"yes": 1, "no": 0})
    else:
        st.error("Uploaded CSV must contain 'deposit' column for evaluation.")
        st.stop()

    # Predictions
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    st.subheader("📈 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.subheader("🔎 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.success("Evaluation completed successfully!")