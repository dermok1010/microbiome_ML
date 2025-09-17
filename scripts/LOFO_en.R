#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(glmnet)
  library(dplyr)
  library(tidyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: elastic_net_lofo.R <in_csv> <out_base> <k_inner> <lambda_choice: lambda.1se|lambda.min> <seed> <flock_col> [alpha_grid_csv]",
       call. = FALSE)
}

message("Args: ", paste0(sprintf("[%d]=%s", seq_along(args), args), collapse = " "))

# ---- robust parsers ----
as_int <- function(x, nm) {
  out <- suppressWarnings(as.integer(x))
  if (is.na(out)) stop(sprintf("Argument '%s' is not a valid integer: '%s'", nm, x), call. = FALSE)
  out
}
parse_alphas <- function(x) {
  a <- suppressWarnings(as.numeric(unlist(strsplit(x, "[,; ]+"))))
  if (any(is.na(a))) stop(sprintf("Alpha grid contains non-numeric entries: '%s'", x), call. = FALSE)
  a
}

infile       <- args[[1]]
out_base     <- args[[2]]
K_inner      <- as_int(args[[3]], "k_inner")
use_lambda   <- args[[4]]                      # "lambda.1se" or "lambda.min"
seed         <- as_int(args[[5]], "seed")
flock_col    <- args[[6]]
alpha_grid   <- if (length(args) >= 7) parse_alphas(args[[7]]) else c(0, .25, .5, .75, 1)

set.seed(seed)

# ---- Load & prepare ----
dat <- read.csv(infile, header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)

# Column checks
need_cols <- c("ch4_g_day2_1v3", "weight", "Age_in_months", "main_breed", "SEX", flock_col)
missing <- setdiff(need_cols, names(dat))
if (length(missing) > 0) stop("Missing required columns: ", paste(missing, collapse = ", "))

if (ncol(dat) < 354) stop("Input does not have columns from 354 onward for microbiome features.")
y <- dat$ch4_g_day2_1v3
X_micro <- dat[, 354:ncol(dat), drop = FALSE]

X_all <- data.frame(
  weight        = dat$weight,
  Age_in_months = dat$Age_in_months,
  main_breed    = factor(dat$main_breed),
  sex           = factor(dat$SEX),
  X_micro,
  check.names = FALSE
)

cc <- complete.cases(X_all, y, dat[[flock_col]])
if (!all(cc)) message(sprintf("Dropped %d rows due to missing values.", sum(!cc)))
X_all <- X_all[cc, , drop = FALSE]
y     <- y[cc]
flock <- as.factor(dat[[flock_col]][cc])

X <- model.matrix(~ ., data = X_all)[, -1, drop = FALSE]
storage.mode(X) <- "double"

# ---- helpers ----
# Build non-overlapping, group-balanced fold IDs (1..K) for cv.glmnet
make_grouped_foldid <- function(groups, k_desired) {
  n <- length(groups)
  ug_levels <- levels(as.factor(groups))
  ug <- length(ug_levels)

  k_use <- min(k_desired, ug)
  if (k_use < 2) {
    # fallback to plain CV with at least 3 folds (or sensible cap)
    k_use <- max(3L, min(k_desired, max(3L, floor(n / 5L))))
    return(list(foldid = NULL, k_use = k_use, grouped = FALSE))
  }

  sizes <- vapply(ug_levels, function(g) sum(groups == g), integer(1))
  ord   <- order(sizes, decreasing = TRUE)
  fold_sizes <- integer(k_use)
  grp2fold <- setNames(integer(ug), ug_levels)
  for (idx in ord) {
    j <- which.min(fold_sizes)
    grp2fold[ug_levels[idx]] <- j
    fold_sizes[j] <- fold_sizes[j] + sizes[idx]
  }
  foldid <- as.integer(grp2fold[as.character(groups)])

  # Sanity
  if (any(is.na(foldid))) stop("Internal: NA fold assignments found.")
  u <- sort(unique(foldid))
  if (!identical(u, seq_len(k_use))) {
    map <- setNames(seq_along(u), u)
    foldid <- as.integer(map[as.character(foldid)])
    k_use <- length(u)
  }
  if (any(tabulate(foldid, nbins = k_use) == 0L)) stop("Internal: an inner fold ended up empty.")
  list(foldid = foldid, k_use = k_use, grouped = TRUE)
}

tidy_cf <- function(cf_mat, fold_label) {
  tibble(Variable = rownames(cf_mat), Coef = as.numeric(cf_mat[, 1])) |>
    dplyr::filter(Variable != "(Intercept)") |>
    mutate(Fold = fold_label)
}

bind_imp <- function(lst, fold_levels) {
  df <- bind_rows(lst)
  covar_prefix <- c("sex", "weight", "Age_in_months", "main_breed")
  is_covar <- function(v) any(startsWith(v, covar_prefix))
  df_g <- df |> dplyr::filter(!vapply(Variable, is_covar, logical(1)))
  genus <- sort(unique(df_g$Variable))
  agg <- lapply(genus, function(g) {
    per_fold <- tibble(Fold = fold_levels) |>
      left_join(df_g |> dplyr::filter(Variable == g) |> dplyr::select(Fold, Coef), by = "Fold") |>
      mutate(Variable = g, Coef = tidyr::replace_na(Coef, 0))
    tibble(
      Variable = g,
      sel_freq  = mean(per_fold$Coef != 0),
      mean_abs  = mean(abs(per_fold$Coef)),
      mean_coef = mean(per_fold$Coef)
    )
  })
  bind_rows(agg) |>
    arrange(desc(sel_freq), desc(mean_abs))
}

# ---- outputs ----
dir.create(dirname(out_base), recursive = TRUE, showWarnings = FALSE)
metrics <- tibble(Flock = character(), Alpha = double(), R2 = double(), RMSE = double(),
                  InnerGrouped = logical(), InnerKFolds = integer(),
                  Ntrain = integer(), Ntest = integer())
coef_list_min <- list()
coef_list_1se <- list()
diagnostics <- tibble(Flock=character(), Ntrain=integer(), Ntest=integer(),
                      Npred=integer(), NA_pred=integer(), NA_y=integer(),
                      Var_y=double(), Note=character())
pred_dump <- tibble(Flock=character(), idx=integer(), y_obs=double(), y_pred=double())

# ---- Outer LOFO ----
flock_levels <- levels(flock)
for (fl in flock_levels) {
  test_idx  <- which(flock == fl)
  train_idx <- which(flock != fl)

  Xtr <- X[train_idx, , drop = FALSE]
  Xte <- X[test_idx,  , drop = FALSE]
  ytr <- y[train_idx]
  yte <- y[test_idx]
  grp_tr <- droplevels(flock[train_idx])

  # harden
  Xtr <- as.matrix(Xtr); storage.mode(Xtr) <- "double"
  Xte <- as.matrix(Xte); storage.mode(Xte) <- "double"

  note <- character(0)
  if (nrow(Xtr) != length(ytr)) stop(sprintf("Train dim mismatch for breeder '%s': nrow(Xtr)=%d, length(ytr)=%d", fl, nrow(Xtr), length(ytr)))
  if (nrow(Xte) != length(yte)) stop(sprintf("Test dim mismatch for breeder '%s': nrow(Xte)=%d, length(yte)=%d", fl, nrow(Xte), length(yte)))
  if (any(!is.finite(Xtr))) stop(sprintf("Non-finite values in Xtr for breeder '%s'", fl))
  if (any(!is.finite(ytr))) stop(sprintf("Non-finite values in ytr for breeder '%s'", fl))
  if (any(!is.finite(Xte)))  note <- c(note, "Non-finite in Xte (will become NA preds)")

  # Inner grouped CV by breeder — non-overlapping fold IDs
  inner <- make_grouped_foldid(grp_tr, K_inner)

  # Fit per-alpha with inner CV
  alpha_fits <- lapply(alpha_grid, function(a) {
    if (is.null(inner$foldid)) {
      fit <- cv.glmnet(Xtr, ytr, alpha = a, family = "gaussian",
                       nfolds = inner$k_use, standardize = TRUE)
    } else {
      if (length(inner$foldid) != nrow(Xtr)) {
        stop(sprintf("foldid length (%d) != nrow(Xtr) (%d) for breeder '%s'",
                     length(inner$foldid), nrow(Xtr), fl))
      }
      fit <- cv.glmnet(Xtr, ytr, alpha = a, family = "gaussian",
                       foldid = inner$foldid, standardize = TRUE)
    }
    list(alpha = a, fit = fit,
         lambda.min = fit$lambda.min,
         lambda.1se = fit$lambda.1se,
         cvm_min    = min(fit$cvm))
  })

  best_idx <- which.min(vapply(alpha_fits, `[[`, numeric(1), "cvm_min"))
  best <- alpha_fits[[best_idx]]
  best_alpha <- best$alpha
  best_fit   <- best$fit

  s_use <- if (use_lambda == "lambda.min") best$lambda.min else best$lambda.1se
  pred  <- as.numeric(predict(best_fit, newx = Xte, s = s_use))

  # sanity: predictions align with test
  if (length(pred) != length(yte)) {
    note <- c(note, sprintf("Pred length %d != Ntest %d", length(pred), length(yte)))
  }

  na_pred <- sum(is.na(pred))
  na_y    <- sum(is.na(yte))
  var_y   <- if (length(yte) > 1) var(yte, na.rm = TRUE) else NA_real_

  r2 <- NA_real_
  if (length(yte) > 1 && all(is.finite(pred)) && na_pred == 0 && na_y == 0 && var_y > 0) {
    r2 <- cor(pred, yte)^2
  }
  rmse <- sqrt(mean((pred - yte)^2, na.rm = TRUE))

  # record metrics
  metrics <- bind_rows(metrics, tibble(
    Flock = fl, Alpha = best_alpha, R2 = r2, RMSE = rmse,
    InnerGrouped = !is.null(inner$foldid),
    InnerKFolds  = inner$k_use,
    Ntrain = nrow(Xtr), Ntest = nrow(Xte)
  ))

  # diagnostics + predictions
  diagnostics <- bind_rows(diagnostics, tibble(
    Flock = fl, Ntrain = nrow(Xtr), Ntest = nrow(Xte),
    Npred = length(pred), NA_pred = na_pred, NA_y = na_y,
    Var_y = var_y, Note = paste(unique(note), collapse = " | ")
  ))
  pred_dump <- bind_rows(pred_dump, tibble(
    Flock = fl, idx = test_idx, y_obs = as.numeric(yte), y_pred = pred
  ))

  # store coefs
  coef_list_min[[fl]] <- tidy_cf(as.matrix(coef(best_fit, s = "lambda.min")),  fl)
  coef_list_1se[[fl]] <- tidy_cf(as.matrix(coef(best_fit, s = "lambda.1se")), fl)
}

# ---- Importance aggregation ----
imp_min <- bind_imp(coef_list_min,  flock_levels)
imp_1se <- bind_imp(coef_list_1se, flock_levels)

# ---- Write outputs ----
write_csv(metrics, paste0(out_base, "_lofo_metrics_", use_lambda, "_test.csv"))
write_csv(imp_1se, paste0(out_base, "_lofo_importance_lambda1se.csv"))
write_csv(imp_min,  paste0(out_base, "_lofo_importance_lambdamin.csv"))
write_csv(diagnostics, paste0(out_base, "_lofo_diagnostics.csv"))
write_csv(pred_dump,   paste0(out_base, "_lofo_predictions.csv"))

cat(sprintf(
  "Elastic Net LOFO done.\nMean R2=%.3f (SD %.3f); Mean RMSE=%.3f (SD %.3f); Flocks=%d\n",
  mean(metrics$R2, na.rm = TRUE), sd(metrics$R2, na.rm = TRUE),
  mean(metrics$RMSE, na.rm = TRUE), sd(metrics$RMSE, na.rm = TRUE),
  length(flock_levels)
))
