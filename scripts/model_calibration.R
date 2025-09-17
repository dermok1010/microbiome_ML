
# Inspecting bias

# calibration_benchmark.R
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(broom)
  library(purrr)
})



# --------------------------
# 0) File paths
# --------------------------
lmm_lofo_file  <- "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/scripts/lofo_predictions.csv"                 # cols: id, breeder, y_obs, y_hat, heldout
lmm_kfold_file <- "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/scripts/kfold_predictions.csv"                # cols: id, breeder, fold, y_obs, y_hat
ml_lofo_file   <- "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/results/ml_lofo_625828_lobo_predictions.csv"  # cols: Breeder, Model, idx, y_obs, y_pred

out_dir <- "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/calibration_outputs"
plot_dir <- file.path(out_dir, "plots")
dir.create(out_dir, showWarnings = FALSE)
dir.create(plot_dir, showWarnings = FALSE)

# --------------------------
# 1) Load & harmonise
# --------------------------
read_safe <- function(path) if (file.exists(path)) read_csv(path, show_col_types = FALSE) else NULL

lmm_lofo  <- read_safe(lmm_lofo_file)
lmm_kfold <- read_safe(lmm_kfold_file)
ml_lofo   <- read_safe(ml_lofo_file)

# normalise to: breeder, Model, Dataset, y_obs, y_hat
norm_lmm <- function(df, dataset_label){
  if (is.null(df)) return(NULL)
  stopifnot(all(c("breeder","y_obs","y_hat") %in% names(df)))
  df %>%
    transmute(
      breeder = as.character(breeder),
      Model   = "LMM",
      Dataset = dataset_label,
      y_obs   = as.numeric(y_obs),
      y_hat   = as.numeric(y_hat)
    )
}

norm_ml <- function(df){
  if (is.null(df)) return(NULL)
  stopifnot(all(c("Breeder","Model","y_obs","y_pred") %in% names(df)))
  df %>%
    transmute(
      breeder = as.character(Breeder),
      Model   = as.character(Model),   # rf / xgb
      Dataset = "ML-LOFO",
      y_obs   = as.numeric(y_obs),
      y_hat   = as.numeric(y_pred)
    )
}

dat_all <- bind_rows(
  norm_lmm(lmm_lofo,  "LMM-LOFO"),
  norm_lmm(lmm_kfold, "LMM-10fold"),
  norm_ml(ml_lofo)
) %>%
  filter(is.finite(y_obs), is.finite(y_hat))

if (nrow(dat_all) == 0) stop("No prediction data found; check file paths.")

# --------------------------
# 2) Helper: bias / calibration calculator
# --------------------------
calc_bias_tbl <- function(df, group_cols = c("Dataset","Model","breeder")){
  if (!all(c("y_obs","y_hat") %in% names(df))) stop("Data missing y_obs/y_hat")
  df %>%
    group_by(across(all_of(group_cols))) %>%
    group_modify(function(d, key){
      y  <- d$y_obs
      p  <- d$y_hat
      n  <- length(y)
      if (sum(is.finite(y) & is.finite(p)) < 3) {
        tibble(
          n = n,
          mean_obs = mean(y), mean_pred = mean(p),
          cal_in_large = mean(y) - mean(p),
          mean_error   = mean(p - y),
          rmse = NA_real_, cor = NA_real_, r2 = NA_real_,
          intercept = NA_real_, slope = NA_real_,
          intercept_lo = NA_real_, intercept_hi = NA_real_,
          slope_lo = NA_real_, slope_hi = NA_real_
        )
      } else {
        fit <- lm(y ~ p)
        ci  <- suppressMessages(confint(fit))
        tibble(
          n = n,
          mean_obs = mean(y), mean_pred = mean(p),
          cal_in_large = mean(y) - mean(p),    # obs - pred (ideal 0)
          mean_error   = mean(p - y),          # pred - obs
          rmse = sqrt(mean((p - y)^2)),
          cor  = cor(y, p),
          r2   = 1 - sum((y - p)^2) / sum((y - mean(y))^2),
          intercept = unname(coef(fit)[1]),
          slope     = unname(coef(fit)[2]),
          intercept_lo = if (!is.null(ci)) ci[1,1] else NA_real_,
          intercept_hi = if (!is.null(ci)) ci[1,2] else NA_real_,
          slope_lo     = if (!is.null(ci) && "p" %in% rownames(ci)) ci["p",1] else if (!is.null(ci)) ci[2,1] else NA_real_,
          slope_hi     = if (!is.null(ci) && "p" %in% rownames(ci)) ci["p",2] else if (!is.null(ci)) ci[2,2] else NA_real_
        )
      }
    }) %>%
    ungroup()
}

bias_by_breeder <- calc_bias_tbl(dat_all, c("Dataset","Model","breeder"))
bias_overall    <- calc_bias_tbl(dat_all, c("Dataset","Model"))

write_csv(bias_by_breeder, file.path(out_dir, "bias_by_breeder.csv"))
write_csv(bias_overall,    file.path(out_dir, "bias_overall.csv"))

# --------------------------
# 3) Quick console summary
# --------------------------
cat("\n=== Overall calibration by Dataset × Model ===\n")
print(bias_overall %>%
        select(Dataset, Model, n, rmse, cor, r2, cal_in_large, intercept, slope) %>%
        arrange(Dataset, Model))

# --------------------------
# 4) Visualisations
# --------------------------

# A) Mean bias (calibration-in-the-large) per breeder
p_bias <- bias_by_breeder %>%
  mutate(breeder = factor(breeder, levels = sort(unique(breeder)))) %>%
  ggplot(aes(x = breeder, y = cal_in_large, fill = Model)) +
  geom_hline(yintercept = 0, linetype = 2) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  facet_wrap(~ Dataset, scales = "free_x", ncol = 1) +
  labs(title = "Calibration-in-the-large (obs − pred) by breeder",
       x = "Breeder", y = "Mean bias (obs − pred)") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(plot_dir, "A_bias_bar_by_breeder.png"), p_bias, width = 12, height = 8, dpi = 300)

# B) Calibration slope per breeder (with 95% CI)
p_slope <- bias_by_breeder %>%
  filter(is.finite(slope)) %>%
  mutate(breeder = factor(breeder, levels = sort(unique(breeder)))) %>%
  ggplot(aes(x = breeder, y = slope, color = Model)) +
  geom_hline(yintercept = 1, linetype = 2) +
  geom_pointrange(aes(ymin = slope_lo, ymax = slope_hi),
                  position = position_dodge(width = 0.5)) +
  facet_wrap(~ Dataset, scales = "free_x", ncol = 1) +
  labs(title = "Calibration slope by breeder (ideal = 1)",
       x = "Breeder", y = "Slope (y ~ a + b·ŷ)") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#
#ggsave(file.path(plot_dir, "B_slope_by_breeder.png"), p_slope, width = 12, height = 8, dpi = 300)

# C) Scatter calibration plots (y vs yhat) for a few largest breeders per Dataset×Model
topN_per_panel <- 6

choose_top_breeders <- dat_all %>%
  count(Dataset, Model, breeder, name = "n") %>%
  group_by(Dataset, Model) %>%
  slice_max(n, n = topN_per_panel, with_ties = FALSE) %>%
  ungroup()

dat_top <- dat_all %>%
  semi_join(choose_top_breeders, by = c("Dataset","Model","breeder"))

p_scatter <- dat_top %>%
  ggplot(aes(x = y_hat, y = y_obs)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2) +
  geom_point(alpha = 0.7, size = 1.8) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_grid(Dataset + Model ~ breeder, scales = "free") +
  labs(title = "Calibration scatter (top-size breeders per panel)",
       x = "Predicted (ŷ)", y = "Observed (y)") +
  theme_minimal(base_size = 11)
ggsave(file.path(plot_dir, "C_scatter_top_breeders.png"), p_scatter, width = 14, height = 8, dpi = 300)

# D) Bland–Altman trend: (pred − obs) vs pred
p_ba <- dat_top %>%
  mutate(error = y_hat - y_obs) %>%
  ggplot(aes(x = y_hat, y = error)) +
  geom_hline(yintercept = 0, linetype = 2) +
  geom_point(alpha = 0.7, size = 1.5) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_grid(Dataset + Model ~ breeder, scales = "free") +
  labs(title = "Bland–Altman style: error vs prediction",
       x = "Predicted (ŷ)", y = "Error (ŷ − y)") +
  theme_minimal(base_size = 11)
#ggsave(file.path(plot_dir, "D_bland_altman_top_breeders.png"), p_ba, width = 14, height = 8, dpi = 300)

cat("\nSaved outputs to:\n  - Tables: ", normalizePath(out_dir),
    "\n  - Plots:  ", normalizePath(plot_dir), "\n", sep = "")
