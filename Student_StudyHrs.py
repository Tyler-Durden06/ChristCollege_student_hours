import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Get the folder where this Python file is located
BASE_DIR = Path(__file__).resolve().parent

# Path to the trained model
model_path = BASE_DIR / "student_pass_fail_model.pkl"

# Check whether the model exists
if not model_path.exists():
    st.error("Model file not found!")
    st.write("Expected model location:")
    st.code(str(model_path))
    st.stop()

# Load the model
model = joblib.load(model_path)

# Streamlit UI
st.title("Student Pass Predictor")

st.write("Enter the number of hours studied to predict the result.")

study_hours = st.number_input(
    "Study hours",
    min_value=0.0,
    step=0.5
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "StudyHours": [study_hours]
    })

    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][int(prediction)]
    else:
        probability = None

    if prediction == 1:

        if probability is not None:
            st.success(
                f"Predicted result: Pass ({probability:.1%} confidence)"
            )
        else:
            st.success("Predicted result: Pass")

    else:

        if probability is not None:
            st.error(
                f"Predicted result: Fail ({probability:.1%} confidence)"
            )
        else:
            st.error("Predicted result: Fail")
