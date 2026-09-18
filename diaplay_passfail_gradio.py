import gradio as gr
import joblib
import pandas as pd
import os

# Load model
model = joblib.load("student_pass_fail_model.pkl")


def predict_result(study_hours):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Study_Hours": [study_hours]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probabilities = model.predict_proba(input_data)[0]

    # Assuming:
    # 0 = FAIL
    # 1 = PASS
    pass_probability = probabilities[1] * 100

    if prediction == 1:
        result = "PASS"
    else:
        result = "FAIL"

    return f"Student Result: {result}\nProbability: {pass_probability:.2f}%"


# Gradio interface
demo = gr.Interface(
    fn=predict_result,

    inputs=gr.Number(
        label="Enter Study Hours",
        minimum=0,
        maximum=24
    ),

    outputs=gr.Textbox(
        label="Prediction"
    ),

    title="Student Result Prediction",

    description="Predict Pass or Fail based on Study Hours."
)


# Start Gradio on Render
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
