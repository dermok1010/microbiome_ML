
#install.packages("sommer")
library(sommer)

# Load phenotype data
pheno <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/CLR_micro.csv")

K <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/MRM_matrix.csv", row.names = 1)
K <- as.matrix(K)

# Remove 'X' from column and row names
colnames(K) <- gsub("^X", "", colnames(K))
rownames(K) <- gsub("^X", "", rownames(K))


print(head(K))

# Check IDS match with pheno
all(pheno$ANI_ID %in% rownames(K))
# Should return TRUE


pheno$ANI_ID <- as.character(pheno$ANI_ID)
rownames(K) <- as.character(rownames(K))
colnames(K) <- as.character(colnames(K))
# view them

# Reorder K to match pheno$ANI_ID
K <- K[pheno$ANI_ID, pheno$ANI_ID]


# Fit the sommer model
mod <- mmer(
  ch4_g_day2_1v3 ~ SEX + Age_in_months + weight + main_breed,
  random = ~ breeder + vsr(ANI_ID, Gu = K),
  data = pheno
)


summary(mod)

## Feature selection
# Fit the sommer model
mod_exp <- mmer(
  ch4_g_day2_1v3 ~ 1,
  random = ~ vsr(ANI_ID, Gu = K),
  rcov = ~ units,
  data = pheno
)

# Extract random effects (BLUPs) for ANI_ID
blups <- mod_exp$U$`u:ANI_ID`$ch4_g_day2_1v3

# Ensure pheno$ANI_ID is character
pheno$ANI_ID <- as.character(pheno$ANI_ID)

# Create new column matching BLUPs by ANI_ID
pheno$microbiome_blup <- blups[pheno$ANI_ID]

# Subtract to remove microbiome effect
pheno$ch4_no_microbiome <- pheno$ch4_g_day2_1v3 - pheno$microbiome_blup

print(head(pheno$ch4_no_microbiome))



library(glmnet)

# Combine response and predictors
temp <- data.frame(
  ch4_no_microbiome = pheno$ch4_no_microbiome,
  Age_in_months = pheno$Age_in_months,
  weight = pheno$weight,
  SEX = pheno$SEX,
  main_breed = pheno$main_breed,
  rumen = pheno$rumen,
  fat_kg = pheno$fat_kg,
  bone_kg = pheno$bone_kg,
  muscle_kg = pheno$muscle_kg,
  Juicy = pheno$Juicy,
  Tenderness = pheno$Tenderness,
  Tenderness2 = pheno$Tenderness2
)

# Keep complete rows
temp_complete <- temp[complete.cases(temp), ]


# Rebuild X and y
X <- model.matrix(~ Age_in_months + weight + SEX + main_breed +
                    rumen + fat_kg + bone_kg + muscle_kg +
                    Juicy + Tenderness + Tenderness2,
                  data = temp_complete)[, -1]

y <- temp_complete$ch4_no_microbiome


set.seed(123)
cv_lasso <- cv.glmnet(X, y, alpha = 1, nfolds = 10)

summary(cv_lasso)

# Extract coefficients at lambda.min
lasso_coef <- coef(cv_lasso, s = "lambda.min")

#############################

### The rest of this script is probably irreleven and the liklihood of its irrelevence increases the longer it is left untouched

############################



library(rrBLUP)
library(dplyr)
library(caret)  # for createFolds
library(Metrics)  # for rmse


# Create folds
set.seed(123)
folds <- createFolds(pheno$ANI_ID, k = 5)

# Prepare results dataframe
results <- data.frame(
  Fold = integer(),
  Model = character(),
  RMSE = numeric(),
  R2 = numeric(),
  stringsAsFactors = FALSE
)

# Define fixed effects formula
fixed_formula <- as.formula("ch4_g_day2_1v3 ~ SEX + Age_in_months + weight + main_breed")

# Loop through folds
for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  train_data <- pheno[-test_idx, ]
  test_data  <- pheno[test_idx, ]
  
  train_ids <- train_data$ANI_ID
  test_ids  <- test_data$ANI_ID
  
  # ---- SINGLE-KERNEL MODEL ----
  kin_train <- K[train_ids, train_ids]
  
  model_single <- kin.blup(
    data = train_data,
    geno = "ANI_ID",
    pheno = "ch4_g_day2_1v3",
    K = kin_train,
    fixed = fixed_formula
  )
  
  # Match predictions to test set
  preds_single <- model_single$g[test_ids]
  obs_single   <- test_data$ch4_g_day2_1v3
  
  results <- rbind(results, data.frame(
    Fold = i,
    Model = "SingleKernel",
    RMSE = rmse(obs_single, preds_single),
    R2 = cor(obs_single, preds_single)^2
  ))
  
  # ---- MULTI-KERNEL MODEL ----
  K_combined <- K_rf_priority + K_rf_background
  kin_train_mk <- K_combined[train_ids, train_ids]
  
  model_multi <- kin.blup(
    data = train_data,
    geno = "ANI_ID",
    pheno = "ch4_g_day2_1v3",
    K = kin_train_mk,
    fixed = fixed_formula
  )
  
  preds_multi <- model_multi$g[test_ids]
  obs_multi   <- test_data$ch4_g_day2_1v3
  
  results <- rbind(results, data.frame(
    Fold = i,
    Model = "MultiKernel_RF",
    RMSE = rmse(obs_multi, preds_multi),
    R2 = cor(obs_multi, preds_multi)^2
  ))
}

# Summarise results
results_summary <- results %>%
  group_by(Model) %>%
  summarise(
    Mean_RMSE = mean(RMSE),
    Mean_R2 = mean(R2),
    .groups = "drop"
  )

print(results_summary)





############################

### PCA

############################


library(vegan)
X <- as.matrix(pheno[, 354:ncol(pheno)]) # your CLR table
dist <- vegdist(X, method="euclidean")
ord <- cmdscale(dist, k=2)  # PCoA
plot(ord, col=factor(pheno$breeder), pch=19)

legend("topright",
       legend = levels(factor(pheno$breeder)),
       col = 1:length(levels(factor(pheno$breeder))),
       pch = 19,
       cex = 0.8,     # shrink labels a bit
       bty = "n")     # no legend box

adonis2(X ~ breeder, data=pheno, method="euclidean", permutations=999)


##################################

### Random-slope mixed model


##################################

library(lme4)

# e.g. use top microbiome PCs to reduce dimensionality
pcs <- prcomp(pheno[,354:ncol(pheno)], scale.=TRUE)$x[,1:5]
pheno <- cbind(pheno, pcs)

# Random intercept + random slopes for PC1 & PC2 by breeder
m <- lmer(ch4_g_day2_1v3 ~ SEX + Age_in_months + weight + main_breed +
            PC1 + PC2 + (PC1 + PC2 | breeder), data=pheno)


VarCorr(m)







