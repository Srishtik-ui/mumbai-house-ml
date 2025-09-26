# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="House Multi-Predictor", layout="centered")

@st.cache_resource
def load_models(path="model_artifacts/multi_model.joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Upload it to the repo or set correct path.")
    return joblib.load(path)

st.title("🏠 House Multi-Predictor")
st.write("Provide BHK and Type — the app predicts Price, Area, Age, Status, Region, Locality.")

# Load models
try:
    models = load_models()
except Exception as e:
    st.error(str(e))
    st.stop()

# Input widgets
st.sidebar.header("Input features")
bhk = st.sidebar.number_input("BHK (number)", min_value=0, max_value=10, value=2, step=1)
house_type = st.sidebar.selectbox("Type", options=["Apartment", "Independent House", "Villa", "Builder Floor", "Studio", "Other"])

# Prepare input for pipelines (the pipelines expect a DataFrame with exactly the same input columns used in training)
input_df = pd.DataFrame([{"BHK": bhk, "Type": house_type}])

st.subheader("Input")
st.table(input_df)

# Prediction button
if st.button("Predict"):
    with st.spinner("Predicting..."):
        results = {}
        # Price (regression)
        if "price" in models:
            try:
                price_pred = models["price"].predict(input_df)[0]
                results["Price"] = float(price_pred)
            except Exception as e:
                results["Price_error"] = str(e)

        # Area
        if "area" in models:
            try:
                area_pred = models["area"].predict(input_df)[0]
                results["Area"] = float(area_pred)
            except Exception as e:
                results["Area_error"] = str(e)

        # Age (could be numeric or classifier)
        if "age" in models:
            try:
                age_model = models["age"]
                # Check if model predicts numeric or encoded classes
                age_pred = age_model.predict(input_df)
                # If predict returns array of floats, convert to float
                if np.issubdtype(np.array(age_pred).dtype, np.number):
                    results["Age"] = float(age_pred[0])
                else:
                    results["Age"] = age_pred[0]
            except Exception as e:
                results["Age_error"] = str(e)

        # Status (categorical)
        if "status" in models:
            try:
                status_pred = models["status"].predict(input_df)[0]
                results["Status"] = str(status_pred)
            except Exception as e:
                results["Status_error"] = str(e)

        # Region
        if "region" in models:
            try:
                region_pred = models["region"].predict(input_df)[0]
                results["Region"] = str(region_pred)
            except Exception as e:
                results["Region_error"] = str(e)

        # Locality
        if "locality" in models:
            try:
                locality_pred = models["locality"].predict(input_df)[0]
                results["Locality"] = str(locality_pred)
            except Exception as e:
                results["Locality_error"] = str(e)

    st.success("Done — predictions below")
    st.json(results)
