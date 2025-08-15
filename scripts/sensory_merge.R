

# We will try and merge sensory data in
library(haven)

sensory <- read_sas("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/sensory_all.sas7bdat")

# going to remove all ct_ data and merge again
micro <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/ct_full_growings_microbiome.csv")


View(sensory)


# Whitespace in sensory so will have to remove
library(stringr)
library(lubridate)

# Clean up dates in raw_abund
sensory <- sensory %>%
  mutate(
    Eartag_Number = str_replace_all(Eartag_Number, "\\s+", ""),
    sensory_date  = ymd(Date)         # convert and rename
  ) %>%
  select(-Date)     


# Lets merge now, we will have to merge on closest date
micro$EID <- as.character(micro$EID)
# 2. Join and compute date difference
merged <- sensory %>%
  inner_join(micro, by = c("Eartag_Number" = "EID")) %>%
  mutate(
    sensory_date_diff = abs(as.numeric(difftime(sensory_date, date, units = "days")))
  ) %>%
  group_by(Eartag_Number) %>%                # one row per tag
  slice_min(order_by = sensory_date_diff,    # keep the closest date
            n = 1, with_ties = T) %>%    # break ties by first row
  ungroup()

dim(merged)
dim(micro)

View(merged)

# So sensory data was used in a way where multiple people had a vote so have just averaged it.

avg_sensory <- merged %>%        # or whatever object you’re starting from
  select(-Username) %>%                  # 1. drop the unwanted column
  group_by(Eartag_Number) %>%            # 2. one row per animal
  summarise(
    across(
      c(Flavour, Juicy, Tenderness, Tenderness2, OffFlavour, Acceptability),
      ~ mean(.x, na.rm = TRUE)           # 2a. average each score
    ),
    across(
      .cols = -c(Flavour, Juicy, Tenderness, Tenderness2, OffFlavour, Acceptability),
      .fns  = first,                     # 3. keep first value of all other vars
      .names = "{.col}"                  # keep original names
    ),
    .groups = "drop"
  )




avg_sensory <- avg_sensory %>%
  # 1. Drop all .y columns
  select(-matches("\\.y$")) %>%
  
  # 2. Rename all .x columns to remove .x suffix
  rename_with(.fn = ~ str_replace(., "\\.x$", ""), .cols = matches("\\.x$"))

dim(avg_sensory)

avg_sensory <- avg_sensory %>%
  select(-DTS) %>%
  mutate(DTS = abs(as.numeric()))

write.csv(avg_sensory, "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/ct_sensory_micro.csv", row.names = F)

