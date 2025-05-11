import streamlit as st
import joblib

# Load the trained logistic regression model and selected features
model = joblib.load("logistic_model.pkl")
selected_features = joblib.load("selected_features.pkl")

st.title("Breast Cancer Predictor")
st.subheader("Enter the following feature values:")

# Collect user inputs for each selected feature
input_data = []
for feature in selected_features:
    value = st.number_input(f"{feature}", step=0.1)
    input_data.append(value)

# Make prediction
if st.button("Predict"):
    prediction = model.predict([input_data])[0]
    result = "Benign" if prediction == 1 else "Malignant"
    st.success(f"Prediction: {result}")
