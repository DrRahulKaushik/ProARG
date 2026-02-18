#!/usr/bin/env python3

"""
Deep Embedding Feature Model Training
Models:
- Logistic Regression
- SVM (RBF)
- Random Forest
- Neural Network (MLP)

Evaluation:
- AUROC
- MCC
- F1

Outputs:
1. validation_performance.tsv
2. cross_validation_10fold.tsv
3. Saved trained models (.joblib)
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score


# ------------------------------------------------
# Threshold Optimization (MCC-based)
# ------------------------------------------------
def optimize_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 200)
    best_thr = 0.5
    best_mcc = -1

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = t

    return best_thr


# ------------------------------------------------
# Evaluation
# ------------------------------------------------
def evaluate(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Threshold": threshold
    }


# ------------------------------------------------
# Main
# ------------------------------------------------
def main(args):

    os.makedirs(args.outdir, exist_ok=True)

    train_df = pd.read_csv(args.train, sep="\t")
    val_df = pd.read_csv(args.val, sep="\t")

    drop_cols = {"seq_id", "label", "set"}

    # -------- Embedding feature selection --------
    feature_cols = [
        c for c in train_df.columns
        if c.startswith("ESM_")
    ]

    if len(feature_cols) == 0:
        raise ValueError("No embedding features detected.")

    print(f"[INFO] Using {len(feature_cols)} embedding features")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values

    # -------------------------
    # Scaling (LR, SVM, MLP)
    # -------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, f"{args.outdir}/feature_scaler.joblib")

    results_val = []
    models = {}

    # ============================================================
    # Logistic Regression
    # ============================================================
    lr = LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=5000,
        n_jobs=args.cpus,
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    prob = lr.predict_proba(X_val_scaled)[:, 1]
    thr = optimize_threshold(y_val, prob)

    results_val.append(
        {"Model": "LogisticRegression", **evaluate(y_val, prob, thr)}
    )
    models["LogisticRegression"] = lr
    joblib.dump(lr, f"{args.outdir}/logistic_regression.joblib")

    # ============================================================
    # SVM (RBF)
    # ============================================================
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    prob = svm.predict_proba(X_val_scaled)[:, 1]
    thr = optimize_threshold(y_val, prob)

    results_val.append(
        {"Model": "SVM", **evaluate(y_val, prob, thr)}
    )
    models["SVM"] = svm
    joblib.dump(svm, f"{args.outdir}/svm.joblib")

    # ============================================================
    # Random Forest
    # ============================================================
    rf = RandomForestClassifier(
        n_estimators=500,
        n_jobs=args.cpus,
        random_state=42
    )
    rf.fit(X_train, y_train)
    prob = rf.predict_proba(X_val)[:, 1]
    thr = optimize_threshold(y_val, prob)

    results_val.append(
        {"Model": "RandomForest", **evaluate(y_val, prob, thr)}
    )
    models["RandomForest"] = rf
    joblib.dump(rf, f"{args.outdir}/random_for_

