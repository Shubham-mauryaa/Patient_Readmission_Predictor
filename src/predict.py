"""
predict.py
Inference utilities — load saved model and predict on new patient data.
Can be used standalone or imported by app.py.
"""

import pickle
import pandas as pd
import numpy as np
import os

MODEL_PATH = "models/xgboost_model.pkl"


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_names"]


def align_features(input_df, feature_names):
    """
    Ensures the input DataFrame has the same columns as the training data.
    Missing columns are filled with 0 (handles one-hot encoded columns).
    """
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[feature_names]


def predict_single(patient_dict, model_path=MODEL_PATH):
    """
    Predict readmission risk for a single patient.

    Parameters
    ----------
    patient_dict : dict
        Raw patient features (pre-processed numeric/encoded values).
    model_path : str
        Path to saved model pickle.

    Returns
    -------
    dict with keys: risk_label, probability, model
    """
    model, feature_names = load_model(model_path)
    df = pd.DataFrame([patient_dict])
    df = align_features(df, feature_names)
    prob = model.predict_proba(df)[0][1]
    label = "High" if prob >= 0.5 else "Low"
    return {
        "readmission_risk": label,
        "probability": round(float(prob), 4),
        "model": "XGBoost"
    }


def predict_batch(csv_path, model_path=MODEL_PATH, out_path=None):
    """
    Run predictions on a CSV file of patient records.

    Parameters
    ----------
    csv_path : str
        Path to input CSV (must have same features as training data).
    out_path : str, optional
        If provided, saves predictions CSV to this path.

    Returns
    -------
    DataFrame with appended risk_label and probability columns.
    """
    model, feature_names = load_model(model_path)
    df = pd.read_csv(csv_path)

    # Remove target if accidentally included
    if "readmitted" in df.columns:
        df.drop(columns=["readmitted"], inplace=True)

    df_aligned = align_features(df.copy(), feature_names)
    probs = model.predict_proba(df_aligned)[:, 1]
    df["probability"] = probs.round(4)
    df["readmission_risk"] = ["High" if p >= 0.5 else "Low" for p in probs]

    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Predictions saved → {out_path}")

    return df


if __name__ == "__main__":
    # Quick demo with dummy patient data
    sample_patient = {
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
        "admission_source_id": 7,
    }
    result = predict_single(sample_patient)
    print("\nSample Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
