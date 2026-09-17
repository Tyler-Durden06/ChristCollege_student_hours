import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Get the current folder
BASE_DIR = Path(__file__).resolve().parent

# Model path
model_path = BASE_DIR / "House_price_model.pkl"

# Load model
model = joblib.load(model_path)

# Title
st.title("House Price Predictor")

st.write("Enter the house details to make a prediction.")

# Input: Area
area = st.number_input(
    "Area (sq.ft)",
    min_value=100.0,
    step=100.0
)

# Input: Bedrooms
bedroom = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    step=1
)

# Input: Age
age = st.number_input(
    "Age of House (years)",
    min_value=0,
    step=1
)

# Prediction
if st.button("Predict"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Area": [area],
        "Bedroom": [bedroom],
        "Age": [age]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Result: {prediction}")
