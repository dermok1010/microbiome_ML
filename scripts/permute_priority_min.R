#!/usr/bin/env Rscript
# Usage:
#   Rscript scripts/permute_random50_min.R <N> <K> <seed> <out_rds>
# Example:
#   Rscript scripts/permute_random50_min.R 10000 50 134 results/random_var_explained_N10000_k50_seed134.rds

suppressPackageStartupMessages({
  library(sommer)
  library(dplyr)
  library(tibble)
})

# --- args ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("Usage: Rscript permute_random50_min.R <N> <K> <seed> <out_rds>")
N    <- as.integer(args[[1]])
K    <- as.integer(args[[2]])
SEED <- as.integer(args[[3]])
OUT  <- args[[4]]
set.seed(SEED)

# --- data (absolute path; matches your setup) ---
pheno <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/LR_micro.csv",
                  check.names = FALSE)
pheno$ANI_ID <- as.character(pheno$ANI_ID)

# genera columns (as in your original script)
all_genera <- colnames(pheno)[354:ncol(pheno)]

# --- kernel builder (your original version) ---
make_kernel <- function(df_matrix) {
  clr_centered <- t(apply(df_matrix, 1, function(row) row - mean(row)))
  clr_scaled   <- scale(clr_centered)
  clr_scaled %*% t(clr_scaled) / ncol(clr_scaled)
}

# --- permutations ---
random_var_explained <- rep(NA_real_, N)

for (i in seq_len(N)) {
  # sample random priority/background
  random_priority   <- sample(all_genera, K)
  random_background <- setdiff(all_genera, random_priority)
  
  # build matrices
  priority_mat <- pheno %>%
    select(ANI_ID, dplyr::all_of(random_priority)) %>%
    column_to_rownames("ANI_ID")
  
  background_mat <- pheno %>%
    select(ANI_ID, dplyr::all_of(random_background)) %>%
    column_to_rownames("ANI_ID")
  
  # kernels
  K_priority   <- make_kernel(priority_mat)
  K_background <- make_kernel(background_mat)
  
  rownames(K_priority)   <- colnames(K_priority)   <- rownames(priority_mat)
  rownames(K_background) <- colnames(K_background) <- rownames(background_mat)
  
  # fit LMM, store fraction from priority kernel
  val <- tryCatch({
    model <- mmer(
      ch4_g_day2_1v3 ~ SEX + Age_in_months + weight + main_breed,
      random = ~ vsr(ANI_ID, Gu = K_priority) + vsr(ANI_ID, Gu = K_background),
      rcov   = ~ units,
      data   = pheno
    )
    vc <- unlist(model$sigma)
    vc[1] / sum(vc)
  }, error = function(e) NA_real_)
  
  random_var_explained[i] <- val
}

# --- save just the numeric vector ---
dir.create(dirname(OUT), recursive = TRUE, showWarnings = FALSE)
saveRDS(random_var_explained, OUT)
cat("Saved:", OUT, "\n")
