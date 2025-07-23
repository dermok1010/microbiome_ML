

# Decision tree
install.packages("rpart.plot")
library(rpart.plot)


data <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/CLR_micro.csv")

micro_predictors <- data[, 354:ncol(data)]

View(micro_predictors)

# Add methane response variable
tree_data <- data.frame(ch4_g_day2_1v3 = data$ch4_g_day2_1v3, micro_predictors)

# Fit decision tree
tree_model <- rpart(ch4_g_day2_1v3 ~ ., data = tree_data, method = "anova",
                    control = rpart.control(maxdepth = 6))  # Adjust depth as needed

# Plot the tree
rpart.plot(tree_model, type = 3, fallen.leaves = TRUE, cex = 0.6)

# See which genera were most important
tree_model$variable.importance

############################################################################################################

# Chance an elastic net
install.packages("glmnet")

library(glmnet)

y <- data$ch4_g_day2_1v3

# Microbiome matrix
X_micro <- data[, 354:ncol(data)]

# Combine with covariates
X_all <- cbind(
  weight = data$weight,
  Age_in_months = data$Age_in_months,
  main_breed = data$main_breed,
  sex = as.factor(data$SEX),  # Important: ensure it's a factor
  X_micro
)


complete_rows <- complete.cases(X_all, y)
X_clean <- X_all[complete_rows, ]
y_clean <- y[complete_rows]

X_matrix <- model.matrix(~ ., data = X_clean)[, -1]

# Create model matrix (this automatically one-hot encodes sex)
X <- model.matrix(~ ., data = X_clean)[, -1]  # remove intercept column


set.seed(123)
elastic_model <- cv.glmnet(X_matrix, y_clean, alpha = 0.5, standardize = TRUE, nfolds = 5)

plot(elastic_model)


preds <- predict(elastic_model, newx = X_matrix, s = "lambda.min")
r2 <- cor(preds, y_clean)^2
rmse <- sqrt(mean((preds - y_clean)^2))

cat("Elastic Net R²:", r2, "\nRMSE:", rmse)




coef_min <- coef(elastic_model, s = "lambda.min")
sum(coef_min != 0)  # Includes intercept

selected_vars <- rownames(coef_min)[which(coef_min != 0)]
selected_vars <- selected_vars[selected_vars != "(Intercept)"]
print(selected_vars)


# Convert to data frame
coef_df <- as.data.frame(as.matrix(coef_min))
coef_df$Variable <- rownames(coef_df)
colnames(coef_df)[1] <- "Coefficient"

# Remove intercept and covariates
covariate_vars <- c("sexM", "weight", "Age_in_months", grep("^main_breed", rownames(coef_min), value=TRUE))
genus_coefs <- coef_df[!(coef_df$Variable %in% c("(Intercept)", covariate_vars)), ]

# Top 50 by absolute value
genus_coefs$AbsCoef <- abs(genus_coefs$Coefficient)

# Filter for non-zero coefficients
nonzero_coefs <- genus_coefs[genus_coefs$Coefficient != 0, ]

# Print count of non-zero predictors
cat("Number of non-zero predictors:", nrow(nonzero_coefs), "\n")

# Get top 50 by absolute value
top45_genera <- head(nonzero_coefs[order(-nonzero_coefs$AbsCoef), ], 45)


write.csv(top45_genera, "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/elastic_net_45.csv", row.names = F)

# So evaluate prediction accuracy

library(caret)

set.seed(456)
folds <- createFolds(y_clean, k = 10)

results <- data.frame(
  Fold = integer(),
  R2 = numeric(),
  RMSE = numeric(),
  stringsAsFactors = FALSE
)

for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  train_X <- X_matrix[-test_idx, ]
  train_y <- y_clean[-test_idx]
  test_X  <- X_matrix[test_idx, ]
  test_y  <- y_clean[test_idx]
  
  # Fit Elastic Net on training data
  fold_model <- cv.glmnet(train_X, train_y, alpha = 0.5, standardize = TRUE, nfolds = 5)
  
  # Predict on test data
  preds <- predict(fold_model, newx = test_X, s = "lambda.min")
  
  # Compute metrics
  fold_r2 <- cor(preds, test_y)^2
  fold_rmse <- sqrt(mean((preds - test_y)^2))
  
  results <- rbind(results, data.frame(Fold = i, R2 = fold_r2, RMSE = fold_rmse))
}

# Show results
print(results)
cat("\nAverage R²:", mean(results$R2), "\nAverage RMSE:", mean(results$RMSE), "\n")



##### Look at residuals especially for edge cases

plot(
  y_clean, elastic_residuals,
  main = "Elastic Net Residuals vs True Methane",
  xlab = "True Methane (g/day)", ylab = "Residuals",
  pch = 16, col = "blue"
)
abline(h = 0, col = "red")

















# OK so Methanobrevibacter is showing a negative coefficient with methane so will investigate

# correlation between methane and log-transformed methanobrevibacter
cor(data$ch4_g_day2_1v3, data$Methanobrevibacter, use = "complete.obs")

# partial
summary(lm(ch4_g_day2_1v3 ~ Methanobrevibacter + weight + Age_in_months + SEX + main_breed, data = data))


plot(data$Methanobrevibacter, data$ch4_g_day2_1v3,
     xlab = "Log Methanobrevibacter",
     ylab = "Methane (g/day)",
     main = "Relationship between Methanobrev"
)


# See if correlation holds in raw abundance data
raw_data <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/ct_sensory_micro.csv")

# correlation between methane and log-transformed methanobrevibacter
cor(raw_data$ch4_g_day2_1v3, raw_data$Methanobrevibacter, use = "complete.obs")

summary(raw_data$Methanobrevibacter)
