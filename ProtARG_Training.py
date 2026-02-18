#!/usr/bin/env python3

"""
Model Training Script
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
import json
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
def evaluate(y_true, y_prob, threshold=0.5):
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

    # -------------------------
    # Load data
    # -------------------------
    train_df = pd.read_csv(args.train, sep="\t")
    val_df = pd.read_csv(args.val, sep="\t")

    drop_cols = {"seq_id", "label"}

    feature_cols = [
        c for c in train_df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

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
    joblib.dump(rf, f"{args.outdir}/random_forest.joblib")

    # ============================================================
    # Neural Network (MLP)
    # ============================================================
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        early_stopping=True,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    prob = mlp.predict_proba(X_val_scaled)[:, 1]
    thr = optimize_threshold(y_val, prob)

    results_val.append(
        {"Model": "MLP", **evaluate(y_val, prob, thr)}
    )
    models["MLP"] = mlp
    joblib.dump(mlp, f"{args.outdir}/mlp.joblib")

    # ------------------------------------------------
    # Save Validation Performance
    # ------------------------------------------------
    df_val = pd.DataFrame(results_val)
    df_val.to_csv(
        f"{args.outdir}/validation_performance.tsv",
        sep="\t",
        index=False
    )

    # ------------------------------------------------
    # 10-fold Cross Validation (Train + Val)
    # ------------------------------------------------
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results_cv = []

    for name, model in models.items():

        if name in ["LogisticRegression", "SVM", "MLP"]:
            X_used = scaler.transform(X_full)
        else:
            X_used = X_full

        prob = cross_val_predict(
            model,
            X_used,
            y_full,
            cv=cv,
            method="predict_proba",
            n_jobs=args.cpus
        )[:, 1]

        thr = optimize_threshold(y_full, prob)

        results_cv.append(
            {"Model": name, **evaluate(y_full, prob, thr)}
        )

    df_cv = pd.DataFrame(results_cv)
    df_cv.to_csv(
        f"{args.outdir}/cross_validation_10fold.tsv",
        sep="\t",
        index=False
    )

    print("\n[COMPLETE] Models trained and evaluated.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--outdir", default="final_models")
    parser.add_argument("--cpus", type=int, default=8)

    args = parser.parse_args()
    main(args)

