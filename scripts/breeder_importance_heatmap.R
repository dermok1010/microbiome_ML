
library(pheatmap)



df <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/results/lofo_pi_621273_rf_genus_importance_matrix.csv",
               check.names = FALSE)

rownames(df) <- df[[1]]
mat <- as.matrix(df[, -1, drop = FALSE])

# keep top K genera by max importance (readability)
K <- 50
keep <- order(apply(mat, 1, max, na.rm = TRUE), decreasing = TRUE)[seq_len(min(K, nrow(mat)))]
mat_top <- mat[keep, , drop = FALSE]

# Heatmap with row scaling (z-score per genus across breeders)
pheatmap(mat_top,
         scale = "row",
         main = "RF permutation importance per breeder (row-scaled)",
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         show_rownames = TRUE,
         show_colnames = TRUE,
         na_col = "grey90")
