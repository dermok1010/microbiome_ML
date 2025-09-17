BiocManager::install("Maaslin2")

packageVersion("Maaslin2")

library(Maaslin2)
library(tidyverse)

data <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/CLR_micro.csv")

for (i in seq_along(data)) {
  cat("Column", i, ":", colnames(data)[i], "\n")
}


# 1. Split metadata vs microbiome
meta_df        <- data[, 1:343]             # phenotypes + covariates
genus_clr_mat  <- data[, 344:ncol(data)]    # CLR genus abundances

meta_df <- meta_df %>%
  mutate(
    sex     = factor(SEX),
    breed   = factor(main_breed),
    breeder = factor(breeder),
    ch4_g_day2_1v3 = as.numeric(ch4_g_day2_1v3),
    age     = as.numeric(Age_in_months),
    weight  = as.numeric(weight)
  )


# 3. Run MaAsLin2
fit <- Maaslin2(
  input_data      = as.data.frame(genus_clr_mat),  # microbiome features
  input_metadata  = meta_df,                    # covariates
  output          = "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/results/maaslin2_out",
  fixed_effects   = c("sex","age","weight","breed","ch4_g_day2_1v3"),
  random_effects  = c("breeder"),
  normalization   = "NONE",
  transform       = "NONE",
  analysis_method = "LM",
  correction      = "BH",
  standardize     = FALSE,
  reference = c("breed","TX")
)
