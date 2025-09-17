#!/usr/bin/env python
import argparse, os, json, numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

# -----------------------------
# Utilities
# -----------------------------
def make_encoder():
    """Handle scikit-learn 1.2+ vs older."""
    try:
        return OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore")

def build_Xy(df, genera_cols, cat_cols, num_cols, enc=None):
    X_num = df[num_cols + genera_cols].astype(float)
    if enc is None:
        enc = make_encoder()
        enc.fit(df[cat_cols])
    X_cat = pd.DataFrame(
        enc.transform(df[cat_cols]),
        index=df.index,
        columns=enc.get_feature_names_out(cat_cols)
    )
    X = pd.concat([X_num, X_cat], axis=1)
    y = df["ch4_g_day2_1v3"].astype(float)
    return X, y, enc

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def default_rf(threads, seed):
    return RandomForestRegressor(
        n_estimators=500,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=None,
        max_depth=None,
        n_jobs=threads,
        random_state=seed,
    )

def default_xgb(threads, seed):
    return XGBRegressor(
        subsample=1.0,
        reg_lambda=0.1,
        reg_alpha=0.0,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        gamma=1.0,
        colsample_bytree=0.6,
        verbosity=0,
        n_jobs=threads,
        random_state=seed,
    )

# -----------------------------
# Main
# -----------------------------
def main(args):
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", args.threads))

    # Load data
    df0 = pd.read_csv(args.data)
    need = ["ANI_ID","ch4_g_day2_1v3","SEX","Age_in_months","main_breed","weight", args.breeder_col]
    missing = [c for c in need if c not in df0.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Feature ranges
    gstart, gend = args.genera_start, args.genera_end
    if gend <= gstart:
        raise ValueError("--genera_end must be > --genera_start")
    genera_cols = df0.columns[gstart:gend].tolist()

    cat_cols = ["SEX","main_breed"]
    num_cols = ["Age_in_months","weight"]

    # Keep only the columns we need
    use_cols = ["ANI_ID","ch4_g_day2_1v3", args.breeder_col] + cat_cols + num_cols + genera_cols
    df = df0[use_cols].dropna().copy()
    df[args.breeder_col] = df[args.breeder_col].astype(str)

    breeders = sorted(df[args.breeder_col].unique(), key=lambda b: (len(df[df[args.breeder_col]==b]), b))
    print(f"Breeders (n={len(breeders)}): {breeders}")

    # Outputs
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Data holders
    metrics_rows = []
    genus_imp_by_model = {"rf": pd.DataFrame(index=genera_cols), "xgb": pd.DataFrame(index=genera_cols)}
    topk_rows = []

    # Loop over breeders (LOFO)
    for i, b in enumerate(breeders, start=1):
        test_mask = (df[args.breeder_col] == b)
        train_mask = ~test_mask

        df_tr = df.loc[train_mask].copy()
        df_te = df.loc[test_mask].copy()

        # Fit encoder on TRAIN only, apply to train/test
        Xtr, ytr, enc = build_Xy(df_tr, genera_cols, cat_cols, num_cols, enc=None)
        Xte, yte, _   = build_Xy(df_te, genera_cols, cat_cols, num_cols, enc=enc)

        # Sanity
        if Xtr.shape[0] != len(ytr) or Xte.shape[0] != len(yte):
            raise RuntimeError(f"Dim mismatch for breeder {b}")

        model_names = ["rf", "xgb"] if args.models == "both" else [args.models]

        for mdl_name in model_names:
            seed = args.seed + i * (17 if mdl_name=="rf" else 29)
            if mdl_name == "rf":
                mdl = default_rf(threads, seed)
            else:
                mdl = default_xgb(threads, seed)

            mdl.fit(Xtr, ytr)
            ypred = mdl.predict(Xte)

            this_r2 = float(r2_score(yte, ypred)) if len(yte) > 1 else np.nan
            this_rmse = rmse(yte, ypred)

            metrics_rows.append({
                "Breeder": b, "Model": mdl_name,
                "R2": this_r2, "RMSE": this_rmse,
                "Ntrain": int(Xtr.shape[0]), "Ntest": int(Xte.shape[0])
            })

            # Permutation importance
            pi = permutation_importance(
                mdl, Xte, yte, scoring="r2",
                n_repeats=args.pi_repeats,
                random_state=seed,
                n_jobs=threads
            )
            imp_series = pd.Series(pi.importances_mean, index=Xte.columns)

            genus_imp = imp_series.reindex(genera_cols).fillna(0.0)
            genus_imp_by_model[mdl_name][b] = genus_imp.values

            top_genus = genus_imp.sort_values(ascending=False).head(args.topk)
            for feat, val in top_genus.items():
                topk_rows.append({"Breeder": b, "Model": mdl_name, "Feature": feat, "Importance": float(val)})

    # Write outputs
    base = args.out_prefix

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(f"{base}_lobo_metrics.csv", index=False)

    genus_imp_by_model["rf"].to_csv(f"{base}_rf_genus_importance_matrix.csv")
    genus_imp_by_model["xgb"].to_csv(f"{base}_xgb_genus_importance_matrix.csv")

    topk_df = pd.DataFrame(topk_rows)
    topk_df = topk_df.sort_values(["Model","Breeder","Importance"], ascending=[True, True, False])
    topk_df.to_csv(f"{base}_top{args.topk}_genus_per_breeder.csv", index=False)

    print(metrics_df.groupby("Model", as_index=False).agg(mean_R2=("R2","mean"),
                                                          sd_R2=("R2","std"),
                                                          mean_RMSE=("RMSE","mean"),
                                                          sd_RMSE=("RMSE","std")).to_string(index=False))
    print("Wrote:")
    print(f"  {base}_lobo_metrics.csv")
    print(f"  {base}_rf_genus_importance_matrix.csv")
    print(f"  {base}_xgb_genus_importance_matrix.csv")
    print(f"  {base}_top{args.topk}_genus_per_breeder.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CLR_micro.csv")
    ap.add_argument("--out_prefix", required=True, help="Output prefix (no extension)")
    ap.add_argument("--breeder_col", default="breeder", help="Breeder/Group column name")
    ap.add_argument("--genera_start", type=int, default=343, help="Start index (0-based, inclusive) of genera cols")
    ap.add_argument("--genera_end",   type=int, default=501, help="End index (0-based, exclusive) of genera cols")
    ap.add_argument("--pi_repeats", type=int, default=30, help="Permutation-importance repeats per breeder")
    ap.add_argument("--topk", type=int, default=20, help="Top-k genera to export per breeder")
    ap.add_argument("--models", choices=["rf","xgb","both"], default="both")
    ap.add_argument("--threads", type=int, default=1, help="Threads if SLURM_CPUS_PER_TASK not set")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    main(args)
