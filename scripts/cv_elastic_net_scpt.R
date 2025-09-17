#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(glmnet)
  library(caret)
  library(dplyr)
  library(tidyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: elastic_net_cv.R <in_csv> <out_base> <k_outer> <k_inner> <lambda_choice: lambda.1se|lambda.min> <seed> [alpha_grid_csv]",
       call. = FALSE)
}

infile       <- args[[1]]
out_base     <- args[[2]]                    # e.g. /path/to/results/elastic_net
K_outer      <- as.integer(args[[3]])        # e.g. 10
K_inner      <- as.integer(args[[4]])        # e.g. 5
use_lambda   <- args[[5]]                    # "lambda.1se" or "lambda.min"
seed         <- as.integer(args[[6]])        # e.g. 2025
alpha_grid   <- if (length(args) >= 7) as.numeric(unlist(strsplit(args[[7]], ","))) else c(0, .25, .5, .75, 1)

set.seed(seed)

# ---- Load & prepare ----
dat <- read.csv(infile, header = TRUE)

y <- dat$ch4_g_day2_1v3
# microbiome columns already CLR; adjust index if needed
X_micro <- dat[, 354:ncol(dat), drop = FALSE]

X_all <- data.frame(
  weight        = dat$weight,
  Age_in_months = dat$Age_in_months,
  main_breed    = factor(dat$main_breed),
  sex           = factor(dat$SEX),
  X_micro,
  check.names = FALSE
)

cc <- complete.cases(X_all, y)
X_all <- X_all[cc, , drop=FALSE]
y     <- y[cc]

X <- model.matrix(~ ., data = X_all)[, -1, drop = FALSE]

# ---- Outer CV ----
folds <- caret::createFolds(y, k = K_outer, list = TRUE, returnTrain = FALSE)

metrics <- tibble(Fold = integer(), Alpha = double(), R2 = double(), RMSE = double())
coef_list_min <- list()
coef_list_1se <- list()

for (i in seq_along(folds)) {
  test_idx  <- folds[[i]]
  train_idx <- setdiff(seq_along(y), test_idx)

  Xtr <- X[train_idx, , drop=FALSE]; ytr <- y[train_idx]
  Xte <- X[test_idx,  , drop=FALSE]; yte <- y[test_idx]

  # tune alpha on training set
  alpha_fits <- lapply(alpha_grid, function(a) {
    fit <- cv.glmnet(Xtr, ytr, alpha = a, family = "gaussian",
                     nfolds = K_inner, standardize = TRUE)
    list(alpha = a, fit = fit,
         lambda.min = fit$lambda.min,
         lambda.1se = fit$lambda.1se,
         cvm_min    = min(fit$cvm))
  })
  best <- alpha_fits[[ which.min(vapply(alpha_fits, `[[`, numeric(1), "cvm_min")) ]]
  best_alpha <- best$alpha
  best_fit   <- best$fit

  s_use <- if (use_lambda == "lambda.min") best$lambda.min else best$lambda.1se
  pred  <- as.numeric(predict(best_fit, newx = Xte, s = s_use))
  r2    <- cor(pred, yte)^2
  rmse  <- sqrt(mean((pred - yte)^2))

  metrics <- bind_rows(metrics, tibble(Fold = i, Alpha = best_alpha, R2 = r2, RMSE = rmse))

  tidy_cf <- function(cf_mat) {
    tibble(Variable = rownames(cf_mat), Coef = as.numeric(cf_mat[,1])) |>
      filter(Variable != "(Intercept)") |>
      mutate(Fold = i)
  }
  coef_list_min[[i]] <- tidy_cf(as.matrix(coef(best_fit, s = "lambda.min")))
  coef_list_1se[[i]] <- tidy_cf(as.matrix(coef(best_fit, s = "lambda.1se")))
}

# importance summariser (genera only)
bind_imp <- function(lst, Kfold) {
  df <- bind_rows(lst)

  covar_prefix <- c("sex", "weight", "Age_in_months", "main_breed")
  is_covar <- function(v) any(startsWith(v, covar_prefix))
  df_g <- df |> filter(!vapply(Variable, is_covar, logical(1)))

  genus <- sort(unique(df_g$Variable))
  agg <- lapply(genus, function(g) {
    per_fold <- tibble(Fold = 1:Kfold) |>
      left_join(df_g |> filter(Variable == g) |> select(Fold, Coef), by = "Fold") |>
      mutate(Variable = g, Coef = tidyr::replace_na(Coef, 0))
    tibble(
      Variable = g,
      sel_freq = mean(per_fold$Coef != 0),
      mean_abs = mean(abs(per_fold$Coef)),
      mean_coef = mean(per_fold$Coef)
    )
  })
  bind_rows(agg) |>
    arrange(desc(sel_freq), desc(mean_abs))
}

imp_min <- bind_imp(coef_list_min,  length(folds))
imp_1se <- bind_imp(coef_list_1se, length(folds))

dir.create(dirname(out_base), recursive = TRUE, showWarnings = FALSE)
write_csv(metrics, paste0(out_base, "_cv_metrics_", use_lambda, "_test.csv"))
write_csv(imp_1se, paste0(out_base, "_importance_lambda1se.csv"))
write_csv(imp_min, paste0(out_base, "_importance_lambdamin.csv"))

cat(sprintf("Elastic Net done.\nMean R2=%.3f (SD %.3f); Mean RMSE=%.3f (SD %.3f)\n",
            mean(metrics$R2), sd(metrics$R2), mean(metrics$RMSE), sd(metrics$RMSE)))
