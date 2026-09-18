import gradio as gr
import joblib
import pandas as pd
import os

model = joblib.load("student_pass_fail_model.pkl")


def predict_result(study_hours):

    input_data = pd.DataFrame({
        "Study Hours": [study_hours]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "PASS"
    else:
        return "FAIL"


demo = gr.Interface(
    fn=predict_result,
    inputs=gr.Number(label="Enter Study Hours"),
    outputs=gr.Textbox(label="Prediction"),
    title="Student Result Prediction",
    description="Predict Pass or Fail based on Study Hours."
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    show_error=True
)
