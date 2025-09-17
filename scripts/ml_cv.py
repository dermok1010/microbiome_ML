#!/usr/bin/env python
import argparse
import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def main(args):
    # load
    data = pd.read_csv(args.data, sep=",", header=0)

    # columns
    genera_cols = data.columns[343:501].tolist()
    pheno_cols = ["ANI_ID", "ch4_g_day2_1v3", "SEX", "Age_in_months", "main_breed", "weight"]

    ml_df = data[pheno_cols + genera_cols].dropna()

    # one-hot (use `sparse=False` for compatibility on older sklearns)
    categorical_vars = ["SEX", "main_breed"]
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    encoded = encoder.fit_transform(ml_df[categorical_vars])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_vars), index=ml_df.index)

    X = pd.concat([ml_df[["Age_in_months", "weight"] + genera_cols], encoded_df], axis=1)
    y = ml_df["ch4_g_day2_1v3"]

    # threads from env if available
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", args.threads))

    rf_model = RandomForestRegressor(
        n_estimators=500,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=None,
        max_depth=None,
        n_jobs=threads,
        random_state=42,
    )

    xgb_model = XGBRegressor(
        subsample=1.0,
        reg_lambda=0.1,
        reg_alpha=0,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        gamma=1,
        colsample_bytree=0.6,
        verbosity=0,
        n_jobs=threads,
        random_state=42,
    )

    cv = KFold(n_splits=args.cv, shuffle=True, random_state=42)

    rf_r2 = cross_val_score(rf_model, X, y, cv=cv, scoring="r2")
    rf_mse = -cross_val_score(rf_model, X, y, cv=cv, scoring="neg_mean_squared_error")

    xgb_r2 = cross_val_score(xgb_model, X, y, cv=cv, scoring="r2")
    xgb_mse = -cross_val_score(xgb_model, X, y, cv=cv, scoring="neg_mean_squared_error")

    results = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv": args.cv,
        "rf": {
            "r2_mean": float(rf_r2.mean()),
            "r2_std": float(rf_r2.std()),
            "mse_mean": float(rf_mse.mean()),
            "mse_std": float(rf_mse.std()),
            "r2_all": rf_r2.tolist(),
            "mse_all": rf_mse.tolist(),
        },
        "xgb": {
            "r2_mean": float(xgb_r2.mean()),
            "r2_std": float(xgb_r2.std()),
            "mse_mean": float(xgb_mse.mean()),
            "mse_std": float(xgb_mse.std()),
            "r2_all": xgb_r2.tolist(),
            "mse_all": xgb_mse.tolist(),
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Final dataset shape: {ml_df.shape}")
    print("Random Forest  R²: mean =", results["rf"]["r2_mean"],  "std =", results["rf"]["r2_std"])
    print("Random Forest  MSE: mean =", results["rf"]["mse_mean"], "std =", results["rf"]["mse_std"])
    print("XGBoost       R²: mean =", results["xgb"]["r2_mean"],  "std =", results["xgb"]["r2_std"])
    print("XGBoost       MSE: mean =", results["xgb"]["mse_mean"], "std =", results["xgb"]["mse_std"])
    print(f"Saved results -> {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CLR_micro.csv")
    parser.add_argument("--out", required=True, help="Path to write JSON results")
    parser.add_argument("--cv", type=int, default=10, help="K-Fold splits")
    parser.add_argument("--threads", type=int, default=1, help="Fallback threads if SLURM_CPUS_PER_TASK not set")
    args = parser.parse_args()
    main(args)
