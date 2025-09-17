#!/usr/bin/env python
import argparse
import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

def make_encoder():
    """
    Handle scikit-learn versions:
    - >=1.2: OneHotEncoder(sparse_output=...)
    - older: OneHotEncoder(sparse=...)
    """
    try:
        return OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore")

def cv_metrics_and_importance(model, X, y, genera_cols, cv, n_repeats=30, random_state=42, n_jobs=1):
    """
    Runs manual K-fold CV:
      - fits model on each train fold,
      - evaluates on validation fold (R²/MSE),
      - computes permutation importance on validation fold,
      - aggregates importances across folds.
    Returns:
      metrics dict, genus-only importance DataFrame, full-feature importance DataFrame
    """
    fold_metrics = []
    per_fold_importances = []  # list of Series (index=features, name=fold_k)

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # fresh clone with same params
        m = model.__class__(**model.get_params())
        m.fit(X_tr, y_tr)

        y_hat = m.predict(X_va)
        r2 = r2_score(y_va, y_hat)
        mse = mean_squared_error(y_va, y_hat)

        # permutation importance on the validation fold
        pi = permutation_importance(
            m, X_va, y_va, scoring="r2",
            n_repeats=n_repeats, random_state=random_state + fold_id, n_jobs=n_jobs
        )
        imp_series = pd.Series(pi.importances_mean, index=X.columns, name=f"fold_{fold_id}")
        per_fold_importances.append(imp_series)

        fold_metrics.append({"fold": fold_id, "r2": float(r2), "mse": float(mse)})

    # stack per-fold importances -> rows=features, cols=folds
    imp_df = pd.concat(per_fold_importances, axis=1)
    imp_df["mean_importance"] = imp_df.mean(axis=1)
    imp_df["std_importance"] = imp_df.std(axis=1)

    # genera-only table for priority ranking
    genus_imp = imp_df.loc[genera_cols].copy()
    # rank: higher mean_importance => smaller rank number
    genus_imp["rank"] = (-genus_imp["mean_importance"]).rank(method="min").astype(int)
    genus_imp = genus_imp.sort_values("mean_importance", ascending=False)

    # CV metrics summary
    r2_vals = [m["r2"] for m in fold_metrics]
    mse_vals = [m["mse"] for m in fold_metrics]
    metrics = {
        "r2_mean": float(np.mean(r2_vals)),
        "r2_std": float(np.std(r2_vals, ddof=1)),
        "mse_mean": float(np.mean(mse_vals)),
        "mse_std": float(np.std(mse_vals, ddof=1)),
        "r2_all": r2_vals,
        "mse_all": mse_vals,
    }
    return metrics, genus_imp, imp_df

def main(args):
    # -----------------------------
    # Load data
    # -----------------------------
    data = pd.read_csv(args.data, sep=",", header=0)

    # Column sets
    genera_cols = data.columns[343:501].tolist()  # already CLR
    pheno_cols = ["ANI_ID", "ch4_g_day2_1v3", "SEX", "Age_in_months", "main_breed", "weight"]

    ml_df = data[pheno_cols + genera_cols].dropna().copy()

    # One-hot encode categoricals
    categorical_vars = ["SEX", "main_breed"]
    encoder = make_encoder()
    encoded = encoder.fit_transform(ml_df[categorical_vars])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_vars),
        index=ml_df.index
    )

    # Final X/y (keep covariates + genera + one-hot)
    X = pd.concat([ml_df[["Age_in_months", "weight"] + genera_cols], encoded_df], axis=1)
    y = ml_df["ch4_g_day2_1v3"]

    # Threads
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", args.threads))

    # Models (your fixed configs)
    rf_model = RandomForestRegressor(
        n_estimators=500,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=None,
        max_depth=None,
        n_jobs=threads,
        random_state=args.seed,
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
        random_state=args.seed,
    )

    # CV splitter
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    # -----------------------------
    # RF: CV metrics + importance
    # -----------------------------
    rf_metrics, rf_genus_imp, rf_all_imp = cv_metrics_and_importance(
        rf_model, X, y, genera_cols, cv,
        n_repeats=args.pi_repeats, random_state=args.seed, n_jobs=threads
    )

    # -----------------------------
    # XGB: CV metrics + importance
    # -----------------------------
    xgb_metrics, xgb_genus_imp, xgb_all_imp = cv_metrics_and_importance(
        xgb_model, X, y, genera_cols, cv,
        n_repeats=args.pi_repeats, random_state=args.seed + 1000, n_jobs=threads
    )

    # -----------------------------
    # Results JSON (accuracy only)
    # -----------------------------
    results = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv": args.cv,
        "pi_repeats": args.pi_repeats,
        "rf": rf_metrics,
        "xgb": xgb_metrics,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    base = os.path.splitext(args.out)[0]

    # Write importance tables
    rf_genus_imp.to_csv(f"{base}_rf_genus_importance.csv")
    xgb_genus_imp.to_csv(f"{base}_xgb_genus_importance.csv")
    if args.write_all_features:
        rf_all_imp.to_csv(f"{base}_rf_all_features_importance.csv")
        xgb_all_imp.to_csv(f"{base}_xgb_all_features_importance.csv")

    # -----------------------------
    # Combined RF + XGB ranking (optional but handy)
    # -----------------------------
    comb = pd.DataFrame(index=sorted(set(rf_genus_imp.index) | set(xgb_genus_imp.index)))
    comb["rf_mean_imp"] = rf_genus_imp["mean_importance"]
    comb["xgb_mean_imp"] = xgb_genus_imp["mean_importance"]
    # Normalize per-model (z or min-max). Here: rank-based average (robust to scale)
    comb["rf_rank"]  = (-comb["rf_mean_imp"].fillna(0)).rank(method="min")
    comb["xgb_rank"] = (-comb["xgb_mean_imp"].fillna(0)).rank(method="min")
    comb["mean_rank"] = comb[["rf_rank", "xgb_rank"]].mean(axis=1)
    comb = comb.sort_values("mean_rank")
    comb.to_csv(f"{base}_combined_genus_importance.csv")

    # -----------------------------
    # Console summary
    # -----------------------------
    print(f"Final dataset shape: {ml_df.shape}")
    print("Random Forest  R²: mean =", results["rf"]["r2_mean"],  "std =", results["rf"]["r2_std"])
    print("Random Forest  MSE: mean =", results["rf"]["mse_mean"], "std =", results["rf"]["mse_std"])
    print("XGBoost       R²: mean =", results["xgb"]["r2_mean"],  "std =", results["xgb"]["r2_std"])
    print("XGBoost       MSE: mean =", results["xgb"]["mse_mean"], "std =", results["xgb"]["mse_std"])
    print(f"Saved results JSON -> {args.out}")
    print(f"Wrote: {base}_rf_genus_importance.csv")
    print(f"Wrote: {base}_xgb_genus_importance.csv")
    print(f"Wrote: {base}_combined_genus_importance.csv")
    if args.write_all_features:
        print(f"Wrote: {base}_rf_all_features_importance.csv")
        print(f"Wrote: {base}_xgb_all_features_importance.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CLR_micro.csv")
    parser.add_argument("--out", required=True, help="Path to write JSON results")
    parser.add_argument("--cv", type=int, default=10, help="K-Fold splits")
    parser.add_argument("--threads", type=int, default=1, help="Fallback threads if SLURM_CPUS_PER_TASK not set")
    parser.add_argument("--pi_repeats", type=int, default=30, help="Permutation-importance repeats per fold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--write_all_features", action="store_true", help="Also write covariate+genera importance tables")
    args = parser.parse_args()
    main(args)
