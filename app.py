import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide"
)

# --- MODEL LOADING WITH STREAMLIT CACHING ---
@st.cache_resource
def load_model_artifacts():
    model_path = "model.joblib"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

artifacts = load_model_artifacts()

# --- HEADER SECTION ---
st.title("🌸 Iris Flower Species Classifier")
st.markdown("""
This interactive machine learning web application predicts the iris flower species based on floral measurements.
Adjust the sliders in the sidebar or below to see real-time classification probabilities!
""")

if artifacts is None:
    st.error("⚠️ Model file `model.joblib` not found! Please run `python train_model.py` first to train and export the model.")
    st.stop()

model = artifacts["model"]
feature_names = artifacts["feature_names"]
target_names = artifacts["target_names"]
feature_ranges = artifacts["feature_ranges"]

# --- SIDEBAR INPUT CONTROLS ---
st.sidebar.header("🔧 Input Floral Features")
st.sidebar.markdown("Use the sliders to customize flower dimensions:")

user_inputs = {}
for feat in feature_names:
    feat_meta = feature_ranges[feat]
    # Clean up name for display label
    label = feat.replace(" (cm)", "").replace("_", " ").title() + " (cm)"
    user_inputs[feat] = st.sidebar.slider(
        label=label,
        min_value=float(round(feat_meta["min"] * 0.8, 1)),
        max_value=float(round(feat_meta["max"] * 1.2, 1)),
        value=float(round(feat_meta["default"], 1)),
        step=0.1,
        help=f"Select measurement for {label}"
    )

# --- STRICT FEATURE ORDER VALIDATION ---
# Ensure input dataframe matches the exact feature order used during training
input_df = pd.DataFrame([user_inputs])[feature_names]

# --- MAIN DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Selected Input Measurements")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("### 📊 Features Summary")
    for feat in feature_names:
        st.write(f"• **{feat}**: `{user_inputs[feat]} cm`")

with col2:
    st.subheader("🔮 Real-Time Prediction")

    # Run inference
    prediction_idx = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    predicted_species = target_names[prediction_idx].capitalize()
    confidence = probabilities[prediction_idx] * 100

    # Prediction Highlight Box
    st.success(f"**Predicted Species:** ### {predicted_species}")
    st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

    # Probability Distribution Chart
    st.markdown("#### Probability Distribution")
    prob_df = pd.DataFrame({
        "Species": [s.capitalize() for s in target_names],
        "Probability": probabilities
    }).set_index("Species")

    st.bar_chart(prob_df)

# --- EXPANDABLE INFO ACCORDION ---
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    - **Algorithm:** Random Forest Classifier (100 estimators)
    - **Dataset:** Scikit-learn Iris Dataset
    - **Classes:** Setosa, Versicolor, Virginica
    - **Serialization:** Joblib (`model.joblib`)
    - **Caching:** `@st.cache_resource` ensures lightning-fast page interactions without reloading model weights into memory.
    """)
