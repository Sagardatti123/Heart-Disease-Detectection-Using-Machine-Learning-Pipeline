"""
Heart Disease Prediction — Flask Web App
Developed by Sagar Datti

Loads the pre-trained StandardScaler (standerd_model.pkl) and
GaussianNB model (navi_bayes_model.pkl) and serves a simple web
form to collect the 7 features the model was trained on:

    age, sex, cp, thalach, oldpeak, slope, thal

and returns a human-readable prediction: "Normal" or "Heart Disease Detected".
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "standerd_model.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "navi_bayes_model.pkl")

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Order of features the model was trained on (must match training pipeline)
FEATURE_ORDER = ["age", "sex", "cp", "thalach", "oldpeak", "slope", "thal"]

# Human-readable labels for the categorical fields, used to render dropdowns
SEX_OPTIONS = [(1, "Male"), (0, "Female")]

CP_OPTIONS = [
    (0, "Typical Angina"),
    (1, "Atypical Angina"),
    (2, "Non-Anginal Pain"),
    (3, "Asymptomatic"),
]

SLOPE_OPTIONS = [
    (0, "Upsloping"),
    (1, "Flat"),
    (2, "Downsloping"),
]

THAL_OPTIONS = [
    (1, "Normal"),
    (2, "Fixed Defect"),
    (3, "Reversible Defect"),
]

# Numeric field configuration: (label, min, max, step, unit)
NUMERIC_FIELDS = {
    "age": {"label": "Age", "min": 1, "max": 120, "step": 1, "unit": "years"},
    "thalach": {
        "label": "Max Heart Rate Achieved",
        "min": 60,
        "max": 220,
        "step": 1,
        "unit": "bpm",
    },
    "oldpeak": {
        "label": "ST Depression (Oldpeak)",
        "min": 0,
        "max": 6.5,
        "step": 0.1,
        "unit": "",
    },
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    result_class = None
    confidence = None
    form_values = {}

    if request.method == "POST":
        try:
            form_values = {
                "age": request.form.get("age", ""),
                "sex": request.form.get("sex", ""),
                "cp": request.form.get("cp", ""),
                "thalach": request.form.get("thalach", ""),
                "oldpeak": request.form.get("oldpeak", ""),
                "slope": request.form.get("slope", ""),
                "thal": request.form.get("thal", ""),
            }

            features = [
                float(form_values["age"]),
                float(form_values["sex"]),
                float(form_values["cp"]),
                float(form_values["thalach"]),
                float(form_values["oldpeak"]),
                float(form_values["slope"]),
                float(form_values["thal"]),
            ]

            input_array = np.array([features])
            scaled_input = scaler.transform(input_array)

            prediction = model.predict(scaled_input)[0]

            try:
                proba = model.predict_proba(scaled_input)[0]
                confidence = round(max(proba) * 100, 1)
            except Exception:
                confidence = None

            if int(prediction) == 1:
                result = "Heart Disease Detected"
                result_class = "danger"
            else:
                result = "Normal"
                result_class = "safe"

        except Exception as e:
            result = f"Error processing input: {e}"
            result_class = "danger"

    return render_template(
        "index.html",
        sex_options=SEX_OPTIONS,
        cp_options=CP_OPTIONS,
        slope_options=SLOPE_OPTIONS,
        thal_options=THAL_OPTIONS,
        numeric_fields=NUMERIC_FIELDS,
        result=result,
        result_class=result_class,
        confidence=confidence,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)