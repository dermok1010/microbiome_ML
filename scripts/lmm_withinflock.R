#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(sommer)
  library(dplyr)
  library(readr)
  library(tibble)
  library(purrr)
})

options(sommer.dateWarning = FALSE)

# ------------------ 1) Load + align + clean ------------------
pheno <- read_csv(
  "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/CLR_micro.csv",
  show_col_types = FALSE
)

K <- as.matrix(read.csv(
  "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/MRM_matrix.csv",
  row.names = 1, check.names = FALSE
))
colnames(K) <- sub("^X","",colnames(K))
rownames(K) <- sub("^X","",rownames(K))

pheno$ANI_ID <- as.character(pheno$ANI_ID)
rownames(K)  <- as.character(rownames(K))
colnames(K)  <- as.character(colnames(K))

ids <- intersect(pheno$ANI_ID, rownames(K))
stopifnot(length(ids) > 0)

base <- pheno %>%
  filter(ANI_ID %in% ids) %>%
  mutate(
    id            = factor(ANI_ID),
    breeder       = factor(breeder),
    SEX           = factor(SEX),
    main_breed    = factor(main_breed),
    Age_in_months = suppressWarnings(as.numeric(Age_in_months)),
    weight        = suppressWarnings(as.numeric(weight)),
    hr_off_feed   = factor(hr_off_feed, levels = c("1","2","3"))  # <-- added
  ) %>%
  filter(!is.na(SEX), !is.na(main_breed), !is.na(Age_in_months),
         !is.na(weight), !is.na(breeder), !is.na(hr_off_feed))

K2 <- K[as.character(base$ANI_ID), as.character(base$ANI_ID), drop = FALSE]
K2 <- (K2 + t(K2))/2
K2 <- K2 / mean(diag(K2))

base <- base %>% mutate(id = factor(ANI_ID, levels = rownames(K2)))
stopifnot(identical(levels(base$id), rownames(K2)))

# ------------------ 2) Helpers ------------------
beta_from_Beta <- function(Beta_tbl, X_cols) {
  b <- setNames(Beta_tbl$Estimate, Beta_tbl$Effect)
  miss <- setdiff(X_cols, names(b))
  if (length(miss)) b <- c(b, setNames(rep(0, length(miss)), miss))
  b[X_cols]
}

get_sigma_e_g <- function(fit) {
  sv <- fit$sigmaVector
  nm <- names(sv)
  idx_g <- grep("\\bu:id\\b|u:id\\.", nm, ignore.case = TRUE)[1]
  idx_e <- grep("\\bunits\\b|residual", nm, ignore.case = TRUE)[1]
  list(
    sigma_g2 = as.numeric(sv[idx_g]),
    sigma_e2 = as.numeric(sv[idx_e])
  )
}

predict_kernel_holdout <- function(beta_vec, sigma_g2, sigma_e2, dat, mask, Kfull) {
  # include hr_off_feed in design matrix
  X <- model.matrix(~ SEX + Age_in_months + weight + main_breed + hr_off_feed, dat)
  miss <- setdiff(colnames(X), names(beta_vec))
  if (length(miss)) beta_vec <- c(beta_vec, setNames(rep(0, length(miss)), miss))
  beta_vec <- beta_vec[colnames(X)]

  idx_tr <- which(!mask & !is.na(dat$y))
  idx_te <- which(mask)

  if (!length(idx_te)) return(numeric(0))
  if (!length(idx_tr))  return(rep(NA_real_, length(idx_te)))

  Xtr <- X[idx_tr, , drop = FALSE]
  Xte <- X[idx_te, , drop = FALSE]
  ytr <- dat$y[idx_tr]

  Ktrtr <- Kfull[idx_tr, idx_tr, drop = FALSE]
  Ktetr <- Kfull[idx_te, idx_tr, drop = FALSE]

  delta <- sigma_e2 / sigma_g2
  rhs   <- ytr - drop(Xtr %*% beta_vec)

  cvec <- solve(Ktrtr + diag(delta, nrow(Ktrtr)), rhs)
  drop(Xte %*% beta_vec + Ktetr %*% cvec)
}

# ------------------ 3) Stratified 10-fold CV maker ------------------
make_stratified_folds <- function(groups, k = 10, seed = 123) {
  set.seed(seed)
  folds <- integer(length(groups))
  levs <- levels(groups)
  for (g in levs) {
    idx <- which(groups == g)
    idx <- sample(idx, length(idx))  # random permute within breeder
    cuts <- cut(seq_along(idx), breaks = k, labels = FALSE)
    folds[idx] <- cuts
  }
  folds
}

# Assign 10-folds, stratified by breeder
k <- 10
fold_id <- make_stratified_folds(base$breeder, k = k, seed = 123)
base$cv_fold <- fold_id

# ------------------ 4) One-fold fit/predict ------------------
fit_kfold_once <- function(fold_num) {
  dat  <- base
  mask <- dat$cv_fold == fold_num

  yy <- dat$ch4_g_day2_1v3
  yy[mask] <- NA_real_
  dat$y <- yy

  dat$SEX         <- droplevels(dat$SEX)
  dat$main_breed  <- droplevels(dat$main_breed)
  dat$breeder     <- droplevels(dat$breeder)
  dat$id          <- droplevels(dat$id)
  dat$hr_off_feed <- droplevels(dat$hr_off_feed)  # <-- added

  Ksub <- K2[as.character(dat$id), as.character(dat$id), drop = FALSE]

  fit <- mmer(
    fixed  = y ~ SEX + Age_in_months + weight + main_breed + hr_off_feed,  # <-- added
    random = ~ vsr(id, Gu = Ksub),
    rcov   = ~ units,
    data   = dat,
    verbose = FALSE
  )

  X_cols   <- colnames(model.matrix(~ SEX + Age_in_months + weight + main_breed + hr_off_feed, dat))  # <-- added
  beta_vec <- beta_from_Beta(fit$Beta, X_cols)

  vc <- get_sigma_e_g(fit)
  yhat_te <- predict_kernel_holdout(beta_vec, vc$sigma_g2, vc$sigma_e2, dat, mask, Ksub)

  out <- tibble(
    id      = as.character(dat$id[mask]),
    breeder = as.character(dat$breeder[mask]),
    fold    = fold_num,
    y_obs   = dat$ch4_g_day2_1v3[mask],
    y_hat   = yhat_te
  )

  has_obs <- !is.na(out$y_obs) & !is.na(out$y_hat)

  RMSE <- if (any(has_obs)) sqrt(mean((out$y_hat[has_obs] - out$y_obs[has_obs])^2)) else NA_real_
  COR  <- if (sum(has_obs) >= 2) cor(out$y_hat[has_obs], out$y_obs[has_obs]) else NA_real_
  R2   <- if (any(has_obs)) {
    1 - sum((out$y_hat[has_obs] - out$y_obs[has_obs])^2) /
        sum((out$y_obs[has_obs] - mean(out$y_obs[has_obs]))^2)
  } else NA_real_

  h2 <- if (is.finite(vc$sigma_g2 + vc$sigma_e2) && (vc$sigma_g2 + vc$sigma_e2) > 0) {
    vc$sigma_g2 / (vc$sigma_g2 + vc$sigma_e2)
  } else NA_real_

  tibble(
    pred        = list(out),
    fold        = fold_num,
    n_masked    = nrow(out),
    n_with_obs  = sum(has_obs),
    RMSE = RMSE,
    COR  = COR,
    R2   = R2,
    sigma_g2 = vc$sigma_g2,
    sigma_e2 = vc$sigma_e2,
    h2 = h2
  )
}

# ------------------ 5) Run 10-fold CV & save ------------------
folds <- sort(unique(base$cv_fold))
res_tbl <- lapply(folds, fit_kfold_once) %>% bind_rows()

preds_cv        <- bind_rows(res_tbl$pred)
metrics_by_fold <- res_tbl %>% select(-pred)

overall <- preds_cv %>%
  filter(!is.na(y_obs), !is.na(y_hat)) %>%
  summarize(
    n    = n(),
    RMSE = sqrt(mean((y_hat - y_obs)^2)),
    COR  = if (n() >= 2) cor(y_hat, y_obs) else NA_real_,
    R2   = 1 - sum((y_hat - y_obs)^2) / sum((y_obs - mean(y_obs))^2)
  )

# --- Per-breeder metrics (within-flock view)
metrics_by_breeder <- preds_cv %>%
  filter(!is.na(y_obs), !is.na(y_hat)) %>%
  group_by(breeder) %>%
  summarize(
    n    = n(),
    RMSE = sqrt(mean((y_hat - y_obs)^2)),
    COR  = if (n() >= 2) cor(y_hat, y_obs) else NA_real_,
    R2   = 1 - sum((y_hat - y_obs)^2) / sum((y_obs - mean(y_obs))^2),
    .groups = "drop"
  )

write.csv(preds_cv,             "kfold_predictions.csv",            row.names = FALSE)
write.csv(metrics_by_fold,      "kfold_metrics_by_fold.csv",        row.names = FALSE)
write.csv(metrics_by_breeder,   "kfold_metrics_by_breeder.csv",     row.names = FALSE)
write.csv(overall,              "kfold_overall_metrics.csv",        row.names = FALSE)

vc_table <- metrics_by_fold %>% select(fold, sigma_g2, sigma_e2, h2)
write.csv(vc_table, "kfold_variance_components.csv", row.names = FALSE)

cat("10-fold CV complete.\n",
    "Predictions: kfold_predictions.csv\n",
    "Per-fold metrics: kfold_metrics_by_fold.csv\n",
    "Per-breeder metrics: kfold_metrics_by_breeder.csv\n",
    "Overall metrics: kfold_overall_metrics.csv\n",
    "Variance components: kfold_variance_components.csv\n", sep = "")
