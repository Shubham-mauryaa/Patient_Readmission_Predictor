# Patient Readmission Predictor
### Predicting 30-Day Hospital Readmission Risk Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green) ![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [API Usage](#api-usage)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)

---

## Overview

This project builds an end-to-end machine learning pipeline to predict whether a diabetic patient will be **readmitted to a hospital within 30 days** of discharge. Early identification of high-risk patients enables healthcare providers to intervene proactively — reducing readmission rates, improving patient outcomes, and lowering operational costs.

The project covers the complete data science workflow: exploratory data analysis (EDA), data preprocessing, feature engineering, model training and comparison, evaluation on imbalanced classes, and deployment via a REST API.

---

## Problem Statement

Hospital readmissions within 30 days are a major quality and cost indicator in healthcare. In the US alone, preventable readmissions cost over **$26 billion annually**. Predictive models that flag high-risk patients at the point of discharge allow clinical teams to prioritize follow-up care and resource allocation.

**Target Variable:** `readmitted` — binary label indicating whether a patient was readmitted within 30 days (`1`) or not (`0`).

**Type of problem:** Binary classification on imbalanced tabular clinical data.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

| Property | Value |
|---|---|
| Rows | ~101,766 patient encounters |
| Features | 50 original columns |
| Target | Readmission within 30 days |
| Class distribution | ~89% not readmitted, ~11% readmitted (imbalanced) |
| Source | 130 US hospitals across 10 years of real EHR data |

---

## Project Structure

```
clinical-outcome-predictor/
│
├── data/
│   ├── diabetic_data.csv            # Raw dataset (download via download_data.py)
│   └── processed_data.csv           # Cleaned dataset after preprocessing
│
├── notebooks/
│   └── eda_and_modeling.ipynb       # Full analysis walkthrough (EDA → results)
│
├── src/
│   ├── preprocess.py                # Data cleaning and feature engineering
│   ├── train.py                     # Model training and evaluation
│   └── predict.py                   # Inference utilities
│
├── models/
│   └── xgboost_model.pkl            # Saved best model (generated after training)
│
├── app.py                           # Flask REST API
├── download_data.py                 # Auto-downloads the UCI dataset
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

### 1. EDA
- Identified class imbalance — only ~11% of patients readmitted within 30 days
- Found patients with 2+ prior inpatient visits show 3x higher readmission rates
- Mapped ICD-9 diagnosis codes into 9 clinical categories

### 2. Preprocessing
- Dropped columns with >40% missing values (`weight`, `payer_code`, `medical_specialty`)
- Mapped age decade strings to ordinal midpoints
- One-hot encoded categorical features

### 3. Feature Engineering
- `total_visits` = inpatient + outpatient + emergency visits
- `medication_change_flag` combining diabetes medication change signals

### 4. Class Imbalance — SMOTE
- Applied SMOTE on training data only (never on test set)
- F1 improvement of ~12% on minority class vs no resampling

### 5. Models Trained

| Model | AUC-ROC | F1 Score |
|---|---|---|
| Logistic Regression (baseline) | 0.71 | 0.48 |
| Random Forest | 0.79 | 0.58 |
| **XGBoost (final)** | **0.84** | **0.63** |

---

## Results

**Best model: XGBoost Classifier — AUC-ROC: 0.84 | F1: 0.63**

Top predictive features:
1. `number_inpatient` — prior inpatient visits
2. `discharge_disposition_id` — discharge destination
3. `time_in_hospital` — length of stay
4. `num_medications` — medications administered
5. `diag_1_category` — primary diagnosis category

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/Shubham-mauryaa/clinical-outcome-predictor.git
cd clinical-outcome-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
python download_data.py

# 4. Preprocess
python src/preprocess.py

# 5. Train models
python src/train.py

# 6. Explore notebook
jupyter notebook notebooks/eda_and_modeling.ipynb

# 7. Start API
python app.py
```

---

## API Usage

**POST /predict**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "time_in_hospital": 5, "num_lab_procedures": 42,
       "num_medications": 18, "number_inpatient": 2, "total_visits": 3}'
```

**Response:**
```json
{
  "readmission_risk": "High",
  "probability": 0.76,
  "model": "XGBoost",
  "note": "Patient flagged as HIGH risk for 30-day readmission."
}
```

---

## Key Learnings

- Accuracy is misleading for imbalanced clinical datasets — AUC-ROC and F1 on the minority class are what matter
- SMOTE must be applied only to training data — not before the train/test split
- Prior inpatient visits dominate over most clinical lab values in predicting readmission
- Grouping ICD-9 codes into clinical categories meaningfully improved model signal

---

---

## Tech Stack

`Python 3.9` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `imbalanced-learn` · `Matplotlib` · `Seaborn` · `Flask` · `Jupyter`

---

**Shubham Maurya** · [LinkedIn](https://www.linkedin.com/in/shubhammauryaa/) · [GitHub](https://github.com/Shubham-mauryaa)
