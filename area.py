import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load model
model_path = Path(__file__).resolve().parent / "House_price_model.pkl"
model = joblib.load(model_path)

st.title("House Price Predictor")

st.write("Enter the house details to predict the price.")

# Area
area = st.number_input(
    "Area (sq.ft)",
    min_value=600,
    max_value=6000,
    value=1000,
    step=50
)

# Total Floors
floors = st.number_input(
    "Total Floors",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

# Bedrooms
bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=5,
    value=2,
    step=1
)

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Area_Sq_Ft": [area],
        "Total_Floors": [floors],
        "Bedrooms": [bedrooms]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted House Price: ₹{prediction:.2f} Lakhs"
    )
