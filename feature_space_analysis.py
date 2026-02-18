#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


# =========================================================
# Utility
# =========================================================
def die(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def load_feature_table(path, require_label=True):
    df = pd.read_csv(path, sep="\t")

    if "seq_id" not in df.columns:
        die(f"{path} missing 'seq_id' column")

    if require_label and "label" not in df.columns:
        die(f"{path} missing 'label' column")

    return df

def get_feature_matrix(df):
    # Keep only numeric feature columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Remove label column if present
    if "label" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["label"])

    return numeric_df


# =========================================================
# A. Global feature statistics
# =========================================================
def global_feature_stats(X):
    stats = X.describe().T
    stats["variance"] = X.var()
    return stats


# =========================================================
# B. Class-wise feature distributions
# =========================================================
def classwise_distribution(df):
    df_pos = df[df["label"] == 1]
    df_neg = df[df["label"] == 0]

    features = get_feature_matrix(df).columns

    records = []
    for col in features:
        mean_pos = df_pos[col].mean()
        mean_neg = df_neg[col].mean()
        records.append({
            "feature": col,
            "AMR_mean": mean_pos,
            "NonAMR_mean": mean_neg,
            "abs_mean_diff": abs(mean_pos - mean_neg),
            "AMR_std": df_pos[col].std(),
            "NonAMR_std": df_neg[col].std()
        })

    return pd.DataFrame(records)


# =========================================================
# C. Mann–Whitney + FDR correction
# =========================================================
def mannwhitney_analysis(df):
    df_pos = df[df["label"] == 1]
    df_neg = df[df["label"] == 0]
    features = get_feature_matrix(df).columns

    results = []

    for col in features:
        try:
            stat, p = mannwhitneyu(
                df_pos[col],
                df_neg[col],
                alternative="two-sided"
            )
            results.append((col, p))
        except ValueError:
            continue

    res_df = pd.DataFrame(results, columns=["feature", "p_value"])

    # FDR correction (Benjamini-Hochberg)
    reject, p_adj, _, _ = multipletests(
        res_df["p_value"], method="fdr_bh"
    )

    res_df["adj_p"] = p_adj
    res_df["significant_fdr_0.05"] = reject

    return res_df


# =========================================================
# D. PCA dimensionality analysis
# =========================================================
def pca_summary(X):
    pca = PCA()
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    return {
        "n_features": X.shape[1],
        "50%_variance_components": np.searchsorted(cumvar, 0.50) + 1,
        "80%_variance_components": np.searchsorted(cumvar, 0.80) + 1,
        "95%_variance_components": np.searchsorted(cumvar, 0.95) + 1
    }


# =========================================================
# E. Redundancy (mean |correlation|)
# =========================================================
def mean_absolute_correlation(X):
    corr = np.corrcoef(X.T)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return np.nanmean(np.abs(upper))


# =========================================================
# F. Random-label separability control
# =========================================================
def random_label_control(df):
    X = get_feature_matrix(df).values
    y = df["label"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_perm = np.random.permutation(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_perm, test_size=0.3, stratify=y_perm, random_state=42
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_te, y_prob)

    return auroc


# =========================================================
# Main
# =========================================================
def main(args):

    os.makedirs(args.outdir, exist_ok=True)

    print("[INFO] Loading feature tables...")

    df_classical = load_feature_table(args.classical)
    df_deep = load_feature_table(args.deep)
    df_integrated = load_feature_table(args.integrated)

    # -----------------------------------------------------
    # SECTION 3.2.1 PCA + Dimensionality
    # -----------------------------------------------------
    print("[INFO] Running PCA analyses...")

    pca_results = {
        "Classical": pca_summary(get_feature_matrix(df_classical)),
        "Deep": pca_summary(get_feature_matrix(df_deep)),
        "Integrated": pca_summary(get_feature_matrix(df_integrated))
    }

    pca_df = pd.DataFrame(pca_results).T
    pca_df.to_csv(f"{args.outdir}/feature_pca_variance.tsv", sep="\t")

    # -----------------------------------------------------
    # SECTION 3.2.2 Distribution & Effect Size
    # -----------------------------------------------------
    print("[INFO] Computing class-wise distributions...")

    classwise_df = classwise_distribution(df_integrated)
    classwise_df.to_csv(
        f"{args.outdir}/feature_classwise_stats.tsv",
        sep="\t", index=False
    )

    # Summary for manuscript text
    abs_diff = classwise_df["abs_mean_diff"]

    distribution_summary = pd.DataFrame([{
        "n_features": len(classwise_df),
        "features_higher_in_AMR": int(
            (classwise_df["AMR_mean"] >
             classwise_df["NonAMR_mean"]).sum()
        ),
        "features_higher_in_nonAMR": int(
            (classwise_df["NonAMR_mean"] >
             classwise_df["AMR_mean"]).sum()
        ),
        "mean_abs_mean_diff": abs_diff.mean(),
        "median_abs_mean_diff": abs_diff.median(),
    }])

    distribution_summary.to_csv(
        f"{args.outdir}/feature_distribution_summary.tsv",
        sep="\t", index=False
    )

    # -----------------------------------------------------
    # SECTION 3.2.2 Statistical separability
    # -----------------------------------------------------
    print("[INFO] Running Mann–Whitney tests with FDR correction...")

    mw_df = mannwhitney_analysis(df_integrated)
    mw_df.to_csv(
        f"{args.outdir}/feature_mannwhitney.tsv",
        sep="\t", index=False
    )

    mw_summary = pd.DataFrame([{
        "n_features_tested": len(mw_df),
        "features_FDR_lt_0.05": int((mw_df["adj_p"] < 0.05).sum()),
        "features_FDR_lt_0.01": int((mw_df["adj_p"] < 0.01).sum()),
        "features_FDR_lt_0.001": int((mw_df["adj_p"] < 0.001).sum()),
        "median_p_value": mw_df["p_value"].median()
    }])

    mw_summary.to_csv(
        f"{args.outdir}/feature_mannwhitney_summary.tsv",
        sep="\t", index=False
    )

    # -----------------------------------------------------
    # SECTION 3.2.3 Redundancy
    # -----------------------------------------------------
    print("[INFO] Computing redundancy metrics...")

    redundancy = pd.DataFrame.from_dict({
        "Classical": mean_absolute_correlation(
            get_feature_matrix(df_classical).values
        ),
        "Deep": mean_absolute_correlation(
            get_feature_matrix(df_deep).values
        ),
        "Integrated": mean_absolute_correlation(
            get_feature_matrix(df_integrated).values
        )
    }, orient="index", columns=["mean_absolute_correlation"])

    redundancy.to_csv(
        f"{args.outdir}/feature_redundancy.tsv",
        sep="\t"
    )

    # -----------------------------------------------------
    # SECTION 3.2.4 Random label control
    # -----------------------------------------------------
    print("[INFO] Running random-label separability control...")

    random_auroc = random_label_control(df_integrated)

    random_summary = pd.DataFrame([{
        "random_label_AUROC": random_auroc
    }])

    random_summary.to_csv(
        f"{args.outdir}/random_label_control.tsv",
        sep="\t", index=False
    )

    print("\n[SUCCESS] Feature space analysis complete.")
    print(f"[INFO] Output directory: {args.outdir}")


# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive Feature Space Analysis (Sections 3.2.1–3.2.4)"
    )

    parser.add_argument("--classical", required=True)
    parser.add_argument("--deep", required=True)
    parser.add_argument("--integrated", required=True)
    parser.add_argument("--outdir", default="feature_space_analysis")

    args = parser.parse_args()
    main(args)

