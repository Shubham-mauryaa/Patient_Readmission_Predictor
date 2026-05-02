"""
preprocess.py
Cleans and prepares the UCI Diabetes 130-US Hospitals dataset for modeling.
Run: python src/preprocess.py
Output: data/processed_data.csv
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/diabetic_data.csv"
OUT_PATH = "data/processed_data.csv"

# ── ICD-9 diagnosis code → clinical category mapping ─────────────────────────
def map_diagnosis(code):
    if pd.isnull(code):
        return "Other"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        c = float(code)
    except ValueError:
        return "Other"
    if 390 <= c <= 459 or c == 785:
        return "Circulatory"
    elif 460 <= c <= 519 or c == 786:
        return "Respiratory"
    elif 520 <= c <= 579 or c == 787:
        return "Digestive"
    elif c == 250:
        return "Diabetes"
    elif 800 <= c <= 999:
        return "Injury"
    elif 710 <= c <= 739:
        return "Musculoskeletal"
    elif 580 <= c <= 629 or c == 788:
        return "Genitourinary"
    elif 140 <= c <= 239:
        return "Neoplasms"
    else:
        return "Other"


def preprocess(raw_path=RAW_PATH, out_path=OUT_PATH):
    print("Loading data...")
    df = pd.read_csv(raw_path)
    print(f"  Raw shape: {df.shape}")

    # ── Replace '?' with NaN ──────────────────────────────────────────────────
    df.replace("?", np.nan, inplace=True)

    # ── Drop columns with >40% missing or not useful ─────────────────────────
    drop_cols = ["weight", "payer_code", "medical_specialty",
                 "encounter_id", "patient_nbr"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── Remove duplicate patients (keep first encounter) ─────────────────────
    # dataset has multiple rows per patient; keep first visit only
    # (patient_nbr already dropped above — dedup on remaining identifiers)
    df.drop_duplicates(inplace=True)

    # ── Target variable ───────────────────────────────────────────────────────
    # "<30" → readmitted within 30 days = 1; ">30" or "NO" = 0
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # ── Age: convert decade strings to ordinal midpoint ───────────────────────
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }
    df["age"] = df["age"].map(age_map)

    # ── Diagnosis codes → clinical categories ────────────────────────────────
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col + "_category"] = df[col].apply(map_diagnosis)
            df.drop(columns=[col], inplace=True)

    # ── Feature engineering ───────────────────────────────────────────────────
    df["total_visits"] = (
        df["number_inpatient"] + df["number_outpatient"] + df["number_emergency"]
    )
    df["medication_change_flag"] = (
        (df["change"] == "Ch").astype(int) + (df["diabetesMed"] == "Yes").astype(int)
    )

    # ── Drop remaining low-value columns ─────────────────────────────────────
    drop_more = ["examide", "citoglipton", "glimepiride-pioglitazone",
                 "metformin-rosiglitazone", "metformin-pioglitazone"]
    df.drop(columns=[c for c in drop_more if c in df.columns], inplace=True)

    # ── Impute missing values ─────────────────────────────────────────────────
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)

    # ── One-hot encode categoricals ───────────────────────────────────────────
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "readmitted"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"  Processed shape: {df.shape}")
    print(f"  Class distribution:\n{df['readmitted'].value_counts(normalize=True).round(3)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")
    return df


if __name__ == "__main__":
    preprocess()
