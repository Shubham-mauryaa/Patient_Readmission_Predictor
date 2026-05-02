"""
app.py
Flask REST API for Clinical Outcome Predictor.
Exposes a /predict endpoint that accepts patient data and returns readmission risk.

Run: python app.py
API: http://localhost:5000
"""

from flask import Flask, request, jsonify
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.predict import predict_single, load_model

app = Flask(__name__)

MODEL_PATH = "models/xgboost_model.pkl"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "Clinical Outcome Predictor",
        "description": "Predicts 30-day hospital readmission risk for diabetic patients",
        "endpoints": {
            "POST /predict": "Returns readmission risk for a single patient",
            "GET /health": "Health check"
        },
        "author": "Shubham Maurya"
    })


@app.route("/health", methods=["GET"])
def health():
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "ok" if model_exists else "model_not_found",
        "model_loaded": model_exists,
        "model_path": MODEL_PATH
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict readmission risk for a single patient.

    Expected JSON body (example):
    {
        "age": 65,
        "time_in_hospital": 5,
        "num_lab_procedures": 42,
        "num_medications": 18,
        "number_inpatient": 2,
        "number_emergency": 0,
        "number_outpatient": 1,
        "total_visits": 3,
        "medication_change_flag": 2,
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7
    }

    Returns:
    {
        "readmission_risk": "High" | "Low",
        "probability": 0.76,
        "model": "XGBoost",
        "note": "..."
    }
    """
    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "error": "Model not found. Please run python src/train.py first."
        }), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Request body must be JSON with patient features."}), 400

    try:
        result = predict_single(data, model_path=MODEL_PATH)
        result["note"] = (
            "Patient flagged as HIGH risk for 30-day readmission. "
            "Consider follow-up care planning."
            if result["readmission_risk"] == "High"
            else "Patient is LOW risk for 30-day readmission."
        )
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch_endpoint():
    """
    Batch prediction from uploaded CSV file.
    Send as multipart/form-data with key 'file'.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send CSV as multipart/form-data with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        import pandas as pd
        import io
        from src.predict import predict_batch

        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))

        # Save temp file
        tmp_path = "/tmp/batch_input.csv"
        df.to_csv(tmp_path, index=False)

        result_df = predict_batch(tmp_path, model_path=MODEL_PATH)
        high_risk_count = (result_df["readmission_risk"] == "High").sum()

        return jsonify({
            "total_patients": len(result_df),
            "high_risk_count": int(high_risk_count),
            "low_risk_count": int(len(result_df) - high_risk_count),
            "predictions": result_df[["probability", "readmission_risk"]].to_dict(orient="records")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Clinical Outcome Predictor API...")
    print("Docs: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
