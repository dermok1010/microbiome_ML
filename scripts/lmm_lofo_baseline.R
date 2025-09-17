#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tibble)
})

# 1) Load + align + clean
pheno <- read_csv(
  "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/CLR_micro.csv",
  show_col_types = FALSE
)

base <- pheno %>%
  mutate(
    id            = as.character(ANI_ID),
    breeder       = factor(breeder),
    SEX           = factor(SEX),
    main_breed    = factor(main_breed),
    Age_in_months = suppressWarnings(as.numeric(Age_in_months)),
    weight        = suppressWarnings(as.numeric(weight)),
    hr_off_feed   = factor(hr_off_feed, levels = c("1","2","3")) # explicit levels
  ) %>%
  filter(!is.na(SEX), !is.na(main_breed), !is.na(Age_in_months),
         !is.na(weight), !is.na(breeder), !is.na(hr_off_feed))

# Helper: build safe fixed-effects formula for a given TRAINING slice
fixed_formula_for <- function(dat) {
  terms <- c("Age_in_months", "weight")
  if (nlevels(droplevels(dat$SEX))         >= 2) terms <- c("SEX", terms)
  if (nlevels(droplevels(dat$main_breed))  >= 2) terms <- c(terms, "main_breed")
  if (nlevels(droplevels(dat$hr_off_feed)) >= 2) terms <- c(terms, "hr_off_feed")
  as.formula(paste("y ~", paste(terms, collapse = " + ")))
}

# 2) LOFO (baseline with deployable covariates incl. hr_off_feed)
fit_lofo_baseline_once <- function(holdout_flock) {
  dat  <- base
  mask <- dat$breeder == holdout_flock

  yy <- dat$ch4_g_day2_1v3
  yy[mask] <- NA_real_
  dat$y <- yy

  # Drop unused levels globally
  dat$SEX         <- droplevels(dat$SEX)
  dat$main_breed  <- droplevels(dat$main_breed)
  dat$breeder     <- droplevels(dat$breeder)
  dat$hr_off_feed <- droplevels(dat$hr_off_feed)

  tr <- which(!mask & !is.na(dat$y))
  te <- which(mask)

  if (!length(te)) return(tibble(
    pred=list(tibble(id=character(), breeder=character(), y_obs=numeric(), y_hat=numeric(), heldout=logical())),
    breeder=as.character(holdout_flock), n_masked=0, n_with_obs=0,
    RMSE=NA_real_, COR=NA_real_, R2=NA_real_
  ))

  if (length(tr) < 3) {
    out <- tibble(
      id      = as.character(dat$id[te]),
      breeder = as.character(dat$breeder[te]),
      y_obs   = dat$ch4_g_day2_1v3[te],
      y_hat   = NA_real_,
      heldout = TRUE
    )
    return(tibble(pred=list(out), breeder=as.character(holdout_flock),
                  n_masked=nrow(out), n_with_obs=0,
                  RMSE=NA_real_, COR=NA_real_, R2=NA_real_))
  }

  # Split train/test frames
  dat_tr <- dat[tr, , drop = FALSE]
  dat_te <- dat[te, , drop = FALSE]

  # --- Align TEST factor levels to TRAINING levels (critical) ---
  dat_tr$SEX         <- droplevels(dat_tr$SEX)
  dat_tr$main_breed  <- droplevels(dat_tr$main_breed)
  dat_tr$hr_off_feed <- droplevels(dat_tr$hr_off_feed)

  dat_te$SEX         <- factor(dat_te$SEX,         levels = levels(dat_tr$SEX))
  dat_te$main_breed  <- factor(dat_te$main_breed,  levels = levels(dat_tr$main_breed))
  dat_te$hr_off_feed <- factor(dat_te$hr_off_feed, levels = levels(dat_tr$hr_off_feed))

  # Build formula using TRAIN data only (drops 1-level factors safely)
  fml <- fixed_formula_for(dat_tr)

  fit_lm <- tryCatch(stats::lm(fml, data = dat_tr), error = function(e) NULL)

  y_hat <- if (is.null(fit_lm)) {
    rep(NA_real_, nrow(dat_te))
  } else {
    # allow NAs if test has unseen/empty levels; we aligned levels so this is just a safeguard
    suppressWarnings(predict(fit_lm, newdata = dat_te, na.action = na.pass))
  }

  out <- tibble(
    id      = as.character(dat_te$id),
    breeder = as.character(dat_te$breeder),
    y_obs   = dat_te$ch4_g_day2_1v3,
    y_hat   = y_hat,
    heldout = TRUE
  )

  has <- !is.na(out$y_obs) & !is.na(out$y_hat)
  RMSE <- if (any(has)) sqrt(mean((out$y_hat[has] - out$y_obs[has])^2)) else NA_real_
  COR  <- if (sum(has) >= 2) cor(out$y_hat[has], out$y_obs[has]) else NA_real_
  R2   <- if (any(has)) 1 - sum((out$y_hat[has] - out$y_obs[has])^2) /
                           sum((out$y_obs[has] - mean(out$y_obs[has]))^2) else NA_real_

  tibble(
    pred        = list(out),
    breeder     = as.character(holdout_flock),
    n_masked    = nrow(out),
    n_with_obs  = sum(has),
    RMSE = RMSE, COR = COR, R2 = R2
  )
}

flocks  <- levels(base$breeder)
res_tbl <- lapply(flocks, fit_lofo_baseline_once) %>% bind_rows()

preds_lofo_baseline       <- bind_rows(res_tbl$pred)
metrics_by_flock_baseline <- res_tbl %>% select(-pred)

overall_baseline <- preds_lofo_baseline %>%
  filter(!is.na(y_obs), !is.na(y_hat)) %>%
  summarize(
    n    = n(),
    RMSE = sqrt(mean((y_hat - y_obs)^2)),
    COR  = if (n() >= 2) cor(y_hat, y_obs) else NA_real_,
    R2   = 1 - sum((y_hat - y_obs)^2) / sum((y_obs - mean(y_obs))^2)
  )

write.csv(preds_lofo_baseline,       "lofo_predictions_baseline.csv",        row.names = FALSE)
write.csv(metrics_by_flock_baseline, "lofo_metrics_by_flock_baseline.csv",  row.names = FALSE)
write.csv(overall_baseline,          "lofo_overall_metrics_baseline.csv",   row.names = FALSE)

cat("Baseline LOFO complete.\n",
    "Predictions: lofo_predictions_baseline.csv\n",
    "Per-flock:  lofo_metrics_by_flock_baseline.csv\n",
    "Overall:    lofo_overall_metrics_baseline.csv\n", sep = "")
