import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ------------------------------------------
# Load Model
# ------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "House_price_model.pkl"

model = joblib.load(model_path)


# ------------------------------------------
# Title
# ------------------------------------------

st.title("House Price Predictor")

st.write("Enter the details of the house.")


# ------------------------------------------
# Area
# Maximum = 6000 sq.ft
# ------------------------------------------

area = st.number_input(
    "Area (sq.ft)",
    min_value=600,
    max_value=6000,
    value=1000,
    step=50
)


# ------------------------------------------
# Total Floors
# ------------------------------------------

floors = st.number_input(
    "Total Floors",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)


# ------------------------------------------
# Bedrooms
# Maximum = 5
# ------------------------------------------

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=5,
    value=2,
    step=1
)


# ------------------------------------------
# Prediction
# ------------------------------------------

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Area_Sq_Ft": [area],
        "Total_Floors": [floors],
        "Bedrooms": [bedrooms]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(
        f"Predicted House Price: ₹{prediction:.2f} Lakhs"
    )
