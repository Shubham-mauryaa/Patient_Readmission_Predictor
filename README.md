# Patient_Readmission_Predictor
This ML model predicts whether a patient is at high risk of being readmitted to a hospital within 30 days of discharge, using their clinical data (age, diagnosis, lab results, prior visits, etc.).

problem : Hospital readmissions cost healthcare systems billions annually. If I can flag high-risk patients early, hospitals can intervene — follow-up calls, extended care, medication checks. This is a real, actively used application in health-tech and pharma analytics.

Project Methodology : 
1. Takes in patient data — a structured tabular dataset where each row is one patient visit, and columns are clinical features like age, number of prior inpatient visits, primary diagnosis code, number of lab procedures, number of medications, etc.
2. Cleans and prepares the data — handles missing values, encodes categorical variables (like diagnosis codes), engineers new features (e.g. total visit count), and addresses the fact that most patients are not readmitted (class imbalance problem).
3. Trains multiple ML models — Logistic Regression (baseline), Random Forest, and XGBoost. Compares them against each other.
4. Evaluates properly — since the dataset is imbalanced (far more non-readmissions than readmissions), accuracy alone is misleading. You use AUC-ROC, Precision-Recall curve, and F1 score.
5. Outputs a prediction — given a new patient's data, the model outputs: "High risk of readmission (78% probability)" or "Low risk (23% probability)".
6. Exposes it via a Flask API — a simple endpoint where you send patient data as JSON and get back a risk prediction. This makes it deployable.
