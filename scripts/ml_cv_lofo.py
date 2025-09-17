#!/usr/bin/env python
import argparse, os, json, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ----------------------------
# helpers
# ----------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def pearsonr_safe(y_true, y_pred):
    """Return Pearson r; NaN if fewer than 2 points or zero variance."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return np.nan
    if np.allclose(np.var(y_true, ddof=1), 0.0) or np.allclose(np.var(y_pred, ddof=1), 0.0):
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def build_Xy(df, genera_cols, cat_cols, num_cols, enc=None):
    X_num = df[num_cols + genera_cols].astype(float)
    if enc is None:
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        enc.fit(df[cat_cols])
    X_cat = pd.DataFrame(enc.transform(df[cat_cols]),
                         index=df.index,
                         columns=enc.get_feature_names_out(cat_cols))
    X = pd.concat([X_num, X_cat], axis=1)
    return X, enc

def inner_group_cv_score(model_ctor, param_grid, X, y, groups, k_splits, threads, random_state):
    """
    Simple inner CV tuner with GroupKFold, selects by lowest mean RMSE.
    model_ctor: callable(**params)->model
    param_grid: list of dicts
    Returns: best_params, cv_table(list of dicts)
    """
    gkf = GroupKFold(n_splits=min(k_splits, len(np.unique(groups))))
    results = []
    for params in param_grid:
        rmses = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
            mdl = model_ctor(**params)
            # Set threads if supported
            if hasattr(mdl, "n_jobs"):
                mdl.n_jobs = threads
            if hasattr(mdl, "random_state"):
                mdl.random_state = random_state
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xva)
            rmses.append(np.sqrt(mean_squared_error(yva, pred)))
        results.append({
            "params": params,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "k": int(gkf.n_splits)
        })
    best = min(results, key=lambda d: d["rmse_mean"])
    return best["params"], results

def greedy_params_grid(model_name):
    if model_name == "rf":
        return [
            {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": None},
            {"n_estimators": 600, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": None},
            {"n_estimators": 600, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": None},
        ]
    if model_name == "xgb":
        return [
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.6, "reg_lambda": 0.1, "reg_alpha": 0.0, "gamma": 1.0, "verbosity": 0, "n_jobs": 1},
            {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.7, "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0, "verbosity": 0, "n_jobs": 1},
            {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.1, "gamma": 0.0, "verbosity": 0, "n_jobs": 1},
        ]
    raise ValueError("Unknown model grid")

def make_model_ctor(name, threads, random_state):
    if name == "rf":
        def ctor(**p):
            return RandomForestRegressor(random_state=random_state, n_jobs=threads, **p)
        return ctor
    if name == "xgb":
        def ctor(**p):
            # n_jobs inside params is overwritten by threads
            p = {**p, "n_jobs": threads, "random_state": random_state}
            return XGBRegressor(**p)
        return ctor
    raise ValueError("Unknown model")

# ----------------------------
# main LOBO
# ----------------------------
def main(args):
    rng = np.random.RandomState(args.seed)
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", args.threads))

    # Load
    data = pd.read_csv(args.data, sep=",", header=0)
    if args.breeder_col not in data.columns:
        raise ValueError(f"Breeder column '{args.breeder_col}' not in data.")
    for c in ["ANI_ID", "ch4_g_day2_1v3", "SEX", "Age_in_months", "main_breed", "weight", "hr_off_feed"]:
        if c not in data.columns:
            raise ValueError(f"Required column '{c}' missing.")

    # Feature columns (keep consistent with your current script)
    genera_cols = data.columns[343:501].tolist()   # adjust if needed
    cat_cols    = ["SEX", "main_breed", "hr_off_feed"]
    num_cols    = ["Age_in_months", "weight"]

    use_cols = ["ANI_ID", "ch4_g_day2_1v3", args.breeder_col] + cat_cols + num_cols + genera_cols
    df = data[use_cols].dropna().copy()
    df["hr_off_feed"] = df["hr_off_feed"].astype(str)

    # Split by breeder (outer LOFO)
    breeders = df[args.breeder_col].astype(str)
    unique_breeders = sorted(breeders.unique(), key=lambda x: (len(df.loc[breeders==x]), x))  # optional: sorted by size then id

    # Outputs
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    metrics_rows = []
    diag_rows = []
    preds_rows = []
    grids_rows = []

    for b in unique_breeders:
        test_mask = (breeders == b)
        train_mask = ~test_mask

        df_tr = df.loc[train_mask].copy()
        df_te = df.loc[test_mask].copy()

        # Fit encoder on TRAIN only, apply to train/test
        Xtr, enc = build_Xy(df_tr, genera_cols, cat_cols, num_cols, enc=None)
        Xte, _   = build_Xy(df_te, genera_cols, cat_cols, num_cols, enc=enc)
        ytr = df_tr["ch4_g_day2_1v3"].astype(float)
        yte = df_te["ch4_g_day2_1v3"].astype(float)

        # Diagnostics: shapes & finites
        note = []
        if Xtr.shape[0] != ytr.shape[0]:
            raise RuntimeError(f"Train dim mismatch for breeder {b}: Xtr={Xtr.shape}, ytr={ytr.shape}")
        if Xte.shape[0] != yte.shape[0]:
            raise RuntimeError(f"Test dim mismatch for breeder {b}: Xte={Xte.shape}, yte={yte.shape}")
        if not np.isfinite(Xtr.to_numpy()).all(): note.append("Non-finite in Xtr")
        if not np.isfinite(ytr.to_numpy()).all(): note.append("Non-finite in ytr")
        if not np.isfinite(Xte.to_numpy()).all(): note.append("Non-finite in Xte")
        if not np.isfinite(yte.to_numpy()).all(): note.append("Non-finite in yte")

        # Inner grouped CV (by breeder) on TRAIN
        groups_tr = df_tr[args.breeder_col].astype(str)
        n_groups  = groups_tr.nunique()
        k_inner   = min(args.inner_k, max(2, n_groups))
        # If only 1 group in train (shouldn't happen in LOFO unless only 2 groups total), fallback to plain KFold
        use_group_cv = (n_groups >= 2)

        # Models to run
        model_names = ["rf", "xgb"] if args.models == "both" else [args.models]

        results_this_breeder = {}

        for model_name in model_names:
            model_ctor = make_model_ctor(model_name, threads=threads, random_state=args.seed)
            grid = greedy_params_grid(model_name)

            if use_group_cv:
                best_params, cv_table = inner_group_cv_score(
                    model_ctor, grid, Xtr, ytr, groups_tr, k_splits=k_inner, threads=threads, random_state=args.seed
                )
            else:
                # fallback plain KFold if somehow only one group in train
                kf = KFold(n_splits=min(args.inner_k, max(2, len(ytr)//5)), shuffle=True, random_state=args.seed)
                res = []
                for params in grid:
                    rmses = []
                    for tr_idx, va_idx in kf.split(Xtr, ytr):
                        mdl = model_ctor(**params)
                        mdl.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
                        pred = mdl.predict(Xtr.iloc[va_idx])
                        rmses.append(np.sqrt(mean_squared_error(ytr.iloc[va_idx], pred)))
                    res.append({"params": params, "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)), "k": int(kf.n_splits)})
                best = min(res, key=lambda d: d["rmse_mean"])
                best_params, cv_table = best["params"], res

            # Fit best on full TRAIN, eval on TEST
            best_model = model_ctor(**best_params)
            best_model.fit(Xtr, ytr)
            ypred = best_model.predict(Xte)

            # Metrics
            var_y = float(np.var(yte, ddof=1)) if len(yte) > 1 else np.nan
            r2 = float(r2_score(yte, ypred)) if (len(yte) > 1 and np.isfinite(ypred).all()) else np.nan
            this_rmse = rmse(yte, ypred)
            this_cor = pearsonr_safe(yte, ypred)

            metrics_rows.append({
                "Breeder": b,
                "Model": model_name,
                "R2": r2,
                "RMSE": this_rmse,
                "COR": this_cor,
                "Ntrain": int(Xtr.shape[0]),
                "Ntest": int(Xte.shape[0]),
                "InnerKFolds": int(k_inner),
                "InnerGrouped": bool(use_group_cv),
            })
            diag_rows.append({
                "Breeder": b,
                "Model": model_name,
                "Npred": int(len(ypred)),
                "NA_pred": int(np.isnan(ypred).sum()),
                "NA_y": int(np.isnan(yte).sum()),
                "Var_y": var_y,
                "Note": " | ".join(sorted(set(note))) if note else ""
            })
            preds_rows.append(pd.DataFrame({
                "Breeder": b, "Model": model_name,
                "idx": df_te.index.values,
                "y_obs": yte.values,
                "y_pred": ypred
            }))

            for row in cv_table:
                grids_rows.append({
                    "Breeder": b, "Model": model_name, **row
                })

        # end model loop
    # end breeder loop

    # Write outputs
    metrics_df = pd.DataFrame(metrics_rows)
    diags_df   = pd.DataFrame(diag_rows)
    preds_df   = pd.concat(preds_rows, ignore_index=True) if preds_rows else pd.DataFrame()
    grids_df   = pd.DataFrame(grids_rows)

    base = args.out_prefix
    metrics_df.to_csv(f"{base}_lobo_metrics.csv", index=False)
    diags_df.to_csv(f"{base}_lobo_diagnostics.csv", index=False)
    preds_df.to_csv(f"{base}_lobo_predictions.csv", index=False)
    grids_df.to_csv(f"{base}_lobo_innercv_gridsearch.csv", index=False)

    # quick console summary (now includes COR)
    summ = (metrics_df
            .groupby("Model", as_index=False)
            .agg(mean_R2=("R2", "mean"),
                 sd_R2=("R2", "std"),
                 mean_COR=("COR", "mean"),
                 sd_COR=("COR", "std"),
                 mean_RMSE=("RMSE", "mean"),
                 sd_RMSE=("RMSE", "std")))
    print(summ.to_string(index=False))
    print(f"Saved:\n  {base}_lobo_metrics.csv\n  {base}_lobo_diagnostics.csv\n  {base}_lobo_predictions.csv\n  {base}_lobo_innercv_gridsearch.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CLR_micro.csv")
    ap.add_argument("--out_prefix", required=True, help="Output prefix (no extension)")
    ap.add_argument("--breeder_col", default="breeder", help="Breeder/Group column name")
    ap.add_argument("--inner_k", type=int, default=5, help="Inner GroupKFold splits")
    ap.add_argument("--threads", type=int, default=1, help="Threads if SLURM_CPUS_PER_TASK not set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", choices=["rf","xgb","both"], default="both")
    args = ap.parse_args()
    main(args)
