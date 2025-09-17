#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: pcr_lofo_cov3pcs.R <in_csv> <out_base> <seed> <flock_col>", call. = FALSE)
}
infile    <- args[[1]]
out_base  <- args[[2]]
seed      <- as.integer(args[[3]])
flock_col <- args[[4]]
set.seed(seed)

# ---------- load data ----------
dat <- read.csv(infile, header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)

need_cols <- c("ch4_g_day2_1v3", flock_col,
               "weight","Age_in_months","main_breed","SEX")
missing <- setdiff(need_cols, names(dat))
if (length(missing)) stop("Missing required columns: ", paste(missing, collapse = ", "))

if (ncol(dat) < 354) stop("Expected microbiome features in columns 354:n (numeric CLR).")

y     <- as.numeric(dat$ch4_g_day2_1v3)
Xmic  <- as.matrix(dat[, 354:ncol(dat), drop = FALSE])
flock <- as.factor(dat[[flock_col]])

Covars <- data.frame(
  weight        = dat$weight,
  Age_in_months = dat$Age_in_months,
  main_breed    = factor(dat$main_breed),
  sex           = factor(dat$SEX),
  check.names   = FALSE
)

cc <- complete.cases(y, Xmic, flock, Covars)
if (!all(cc)) message(sprintf("Dropped %d incomplete rows.", sum(!cc)))
y      <- y[cc]
Xmic   <- Xmic[cc, , drop = FALSE]
flock  <- droplevels(flock[cc])
Covars <- droplevels(Covars[cc, , drop = FALSE])

# ---------- helpers ----------
add_intercept <- function(M) cbind(`(Intercept)` = 1, M)

safe_project <- function(pca, newdata, k) {
  Z <- predict(pca, newdata = newdata)
  if (is.null(dim(Z))) Z <- matrix(Z, nrow = nrow(newdata), byrow = TRUE)
  Z[, seq_len(min(k, ncol(Z))), drop = FALSE]
}

mk_cov_design <- function(train_df, test_df, flock_name) {
  tr <- train_df
  te <- test_df

  # Ensure factors (coerce characters if any)
  for (nm in names(tr)) {
    if (is.character(tr[[nm]])) tr[[nm]] <- factor(tr[[nm]])
    if (is.character(te[[nm]])) te[[nm]] <- factor(te[[nm]])
  }

  # Drop TRAIN factors with <2 observed levels
  drop_cols <- character(0)
  for (nm in names(tr)) {
    if (is.factor(tr[[nm]])) {
      obs_levels <- levels(droplevels(tr[[nm]]))
      if (length(obs_levels) < 2L) {
        message(sprintf("Flock %s: dropping covariate '%s' (only one level in TRAIN: %s)",
                        flock_name, nm, paste(obs_levels, collapse="|")))
        drop_cols <- c(drop_cols, nm)
      }
    }
  }
  if (length(drop_cols)) {
    tr <- tr[, !(names(tr) %in% drop_cols), drop = FALSE]
    te <- te[, !(names(te) %in% drop_cols), drop = FALSE]
  }

  if (ncol(tr) == 0) {
    return(list(Xtr = matrix(nrow = nrow(tr), ncol = 0),
                Xte = matrix(nrow = nrow(te),  ncol = 0)))
  }

  # Align TEST factor levels to TRAIN; replace unseen TEST levels with a safe reference
  for (nm in names(tr)) {
    if (is.factor(tr[[nm]])) {
      tr_lv <- levels(tr[[nm]])
      te[[nm]] <- factor(te[[nm]], levels = tr_lv)
      if (anyNA(te[[nm]])) {
        ref <- if ("TX" %in% tr_lv) "TX" else tr_lv[1]
        nfix <- sum(is.na(te[[nm]]))
        message(sprintf("Flock %s: replacing %d unseen TEST levels in '%s' with reference '%s'",
                        flock_name, nfix, nm, ref))
        te[[nm]][is.na(te[[nm]])] <- ref
        te[[nm]] <- factor(te[[nm]], levels = tr_lv)  # reassert levels
      }
    }
  }

  # Build model matrices, explicitly remove intercept BEFORE zero-variance check
  trm <- terms(~ ., data = tr)
  Xtr_full <- model.matrix(trm, data = tr)
  Xte_full <- model.matrix(trm, data = te)
  keep_cols <- colnames(Xtr_full) != "(Intercept)"
  Xtr <- Xtr_full[, keep_cols, drop = FALSE]
  Xte <- Xte_full[, keep_cols, drop = FALSE]

  # Drop any zero-variance encoded columns in TRAIN (rare but safe)
  if (ncol(Xtr) > 0) {
    nzv <- apply(Xtr, 2, function(v) var(v) > 0)
    if (any(!nzv)) {
      zv_names <- colnames(Xtr)[!nzv]
      message(sprintf("Flock %s: dropping zero-variance encoded cols: %s",
                      flock_name, paste(zv_names, collapse=",")))
      Xtr <- Xtr[, nzv, drop = FALSE]
      Xte <- Xte[, nzv, drop = FALSE]
    }
  }
  list(Xtr = Xtr, Xte = Xte)
}

# ---------- outputs ----------
metrics <- tibble(Flock = character(), R2 = double(), RMSE = double(),
                  PCs = integer(), Ntrain = integer(), Ntest = integer())
pred_dump <- tibble(Flock = character(), idx = integer(),
                    y_obs = double(), y_pred = double())

# ---------- LOFO ----------
for (fl in levels(flock)) {
  test_idx  <- which(flock == fl)
  train_idx <- which(flock != fl)

  ytr <- y[train_idx]; yte <- y[test_idx]
  Xtr_mic <- Xmic[train_idx, , drop = FALSE]
  Xte_mic <- Xmic[test_idx,  , drop = FALSE]
  Cov_tr  <- droplevels(Covars[train_idx, , drop = FALSE])
  Cov_te  <- droplevels(Covars[test_idx,  , drop = FALSE])

  if (length(ytr) < 3) {
    message(sprintf("Skipping flock %s: too few training samples.", fl))
    next
  }

  # Optional: echo TRAIN factor levels
  for (nm in names(Cov_tr)) {
    if (is.factor(Cov_tr[[nm]])) {
      message(sprintf("Flock %s: TRAIN levels for %s = %s",
                      fl, nm, paste(levels(Cov_tr[[nm]]), collapse=",")))
    }
  }

  # PCA on training microbiome
  pca   <- prcomp(Xtr_mic, center = TRUE, scale. = TRUE)
  k_eff <- min(3L, ncol(pca$x))

  if (k_eff == 0) {
    pred <- rep(mean(ytr), length(yte))
  } else {
    Ztr <- pca$x[, 1:k_eff, drop = FALSE]
    Zte <- safe_project(pca, Xte_mic, k_eff)

    cov_dm   <- mk_cov_design(Cov_tr, Cov_te, fl)
    Xcov_tr  <- cov_dm$Xtr
    Xcov_te  <- cov_dm$Xte

    # Final design: covariates + PCs
    Xtr_final <- cbind(Xcov_tr, Ztr)
    Xte_final <- cbind(Xcov_te, Zte)

    # Sanity: ensure row counts match before fit
    if (nrow(Xtr_final) != length(ytr)) stop(sprintf("Train design row mismatch for flock %s", fl))
    if (nrow(Xte_final) != length(yte)) stop(sprintf("Test design row mismatch for flock %s", fl))

    fit  <- lm.fit(add_intercept(Xtr_final), ytr)
    pred <- as.numeric(add_intercept(Xte_final) %*% fit$coefficients)
  }

  r2   <- if (length(yte) > 1 && var(yte) > 0) cor(yte, pred)^2 else NA_real_
  rmse <- sqrt(mean((yte - pred)^2, na.rm = TRUE))

  metrics <- bind_rows(metrics, tibble(
    Flock = fl, R2 = r2, RMSE = rmse,
    PCs = k_eff, Ntrain = length(ytr), Ntest = length(yte)
  ))
  pred_dump <- bind_rows(pred_dump, tibble(
    Flock = fl, idx = test_idx, y_obs = yte, y_pred = pred
  ))
}

# ---------- write ----------
dir.create(dirname(out_base), recursive = TRUE, showWarnings = FALSE)
write_csv(metrics,     paste0(out_base, "_metrics.csv"))
write_csv(pred_dump,   paste0(out_base, "_predictions.csv"))

cat(sprintf(
  "PCR LOFO (covariates + 3 PCs) complete. Mean R2=%.3f (SD %.3f); Mean RMSE=%.3f (SD %.3f); Flocks=%d\n",
  mean(metrics$R2, na.rm = TRUE), sd(metrics$R2, na.rm = TRUE),
  mean(metrics$RMSE, na.rm = TRUE), sd(metrics$RMSE, na.rm = TRUE),
  nrow(metrics)
))
