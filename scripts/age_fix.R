

##### Cleaning Microbiome data  so no missing data

og_micro_ct <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/ct_sensory_micro.csv")

g <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/CLR_micro.csv")

# How many rows
nrow(g)

# how many distinct animals
n_distinct(g$ANI_ID)

# how many methane values
sum(!is.na(g$ch4_g_day2_1v3))

# how many covariates
sum(!is.na(g$weight))
sum(!is.na(g$Age_in_months))
sum(!is.na(g$main_breed))
sum(!is.na(g$SEX))

# So age in months needs to be done again
summary(g$Age_in_months)

library(dplyr)
library(lubridate)

g <- g %>%
  mutate(
    birthdate = case_when(
      grepl("/", birthdate) ~ dmy(birthdate),  # e.g. 01/04/2023
      TRUE ~ ymd(birthdate)                    # e.g. 2023-04-17
    ),
    Age_in_months = interval(birthdate, date) %/% months(1)
  )


write.csv(g, "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/CLR_micro.csv", row.names = F)
