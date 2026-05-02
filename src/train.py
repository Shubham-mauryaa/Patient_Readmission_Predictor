"""
train.py
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
Handles class imbalance with SMOTE. Saves the best model.
Run: python src/train.py
Output: models/xgboost_model.pkl, models/model_results.csv
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

DATA_PATH = "data/processed_data.csv"
MODEL_DIR = "models"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]
    return X, y


def apply_smote(X_train, y_train, random_state=42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE — class distribution: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not Readmitted", "Readmitted"]))
    return {"model": name, "auc_roc": round(auc, 4), "f1_score": round(f1, 4)}


def plot_confusion_matrix(model, X_test, y_test, name, save_dir):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Not Readmitted", "Readmitted"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {path}")


def plot_feature_importance(model, feature_names, save_dir, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(top_n), importances[indices][::-1], color="#1F3864")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances — XGBoost")
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance_xgboost.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved feature importance plot → {path}")


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading processed data...")
    X, y = load_data()
    print(f"  Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
    print(f"  Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── SMOTE on training data only ───────────────────────────────────────────
    print("\nApplying SMOTE to training data...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # ── Scale for Logistic Regression ─────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ── 1. Logistic Regression (baseline) ────────────────────────────────────
    print("\nTraining Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train_res)
    results.append(evaluate("Logistic Regression", lr, X_test_scaled, y_test))
    plot_confusion_matrix(lr, X_test_scaled, y_test, "Logistic Regression", MODEL_DIR)

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)
    results.append(evaluate("Random Forest", rf, X_test, y_test))
    plot_confusion_matrix(rf, X_test, y_test, "Random Forest", MODEL_DIR)

    # ── 3. XGBoost (final model) ──────────────────────────────────────────────
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train_res, y_train_res,
            eval_set=[(X_test, y_test)],
            verbose=False)
    results.append(evaluate("XGBoost", xgb, X_test, y_test))
    plot_confusion_matrix(xgb, X_test, y_test, "XGBoost", MODEL_DIR)
    plot_feature_importance(xgb, list(X.columns), MODEL_DIR)

    # ── Save results summary ──────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    results_df.to_csv(os.path.join(MODEL_DIR, "model_results.csv"), index=False)
    print(f"\n{'═'*50}")
    print("Model Comparison Summary:")
    print(results_df.to_string(index=False))

    # ── Save best model (XGBoost) ─────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": xgb, "feature_names": list(X.columns)}, f)
    print(f"\nBest model saved → {model_path}")

    # Also save scaler for LR (not needed for tree models but good practice)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return xgb, results_df


if __name__ == "__main__":
    train()
