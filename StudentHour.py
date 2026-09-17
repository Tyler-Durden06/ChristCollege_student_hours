import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Get current folder
BASE_DIR = Path(__file__).resolve().parent

# Model path
model_path = BASE_DIR / "student_pass_fail_model.pkl"

# Check if model exists
if not model_path.exists():
    st.error("Model file not found!")
    st.stop()

# Load trained model
model = joblib.load(model_path)

# Title
st.title("Student Pass Predictor")

st.write("Enter the student's study hours and attendance.")

# Input 1: Study Hours
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Input 2: Attendance
attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

# Prediction button
if st.button("Predict"):

    # Create input data
    input_data = pd.DataFrame({
        "Study Hours": [study_hours],
        "Attendance": [attendance]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][int(prediction)]
    else:
        probability = None

    # Display result
    if prediction == 1:

        if probability is not None:
            st.success(
                f"Predicted Result: PASS ({probability:.1%} confidence)"
            )
        else:
            st.success("Predicted Result: PASS")

    else:

        if probability is not None:
            st.error(
                f"Predicted Result: FAIL ({probability:.1%} confidence)"
            )
        else:
            st.error("Predicted Result: FAIL")
