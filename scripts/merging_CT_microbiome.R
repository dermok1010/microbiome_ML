
###

### Plan here is to merge the full EIDs into the relative abundance, this should allow for easy merge with all other data

###

# Read the first column as *character* and the rest as default
raw_abund <- read.csv(
  "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/genus_relative_abundance.csv",
  colClasses = c("character", rep("numeric", ncol(read.csv(
    "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/genus_relative_abundance.csv", nrows = 1)) - 1))
)

# Rename the first column to 'VID'
colnames(raw_abund)[1] <- "VID"


# Step 1: Detect number of columns
n_cols <- ncol(read.csv(
  "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/Macrogen Sample List Feb 2025.csv",
  nrows = 1))

# Step 2: Set colClasses so that 2nd column is character, rest are default (NA)
classes <- rep(NA, n_cols)
classes[2] <- "character"  # Only the 2nd column (VID) as character

# Step 3: Read in the data
link <- read.csv(
  "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/Macrogen Sample List Feb 2025.csv",
  colClasses = classes,
  stringsAsFactors = FALSE
)

print(head(link))
print(head(raw_abund))


# All looks in order, lets merge based on VID, bring EID to first column in raw_abund

library(dplyr)

raw_abund <- raw_abund %>%
  left_join(select(link, VID, EID, micro_date = Date), by = "VID")

# Ok not all have merged, some are NA for EID, lets investigate

check <- raw_abund %>%
  select(EID, VID) %>%
  filter(is.na(EID))

View(check)


# Thats fixed now I think EID does not match same structure as tag column in growing dataset 

print(head(growings$tag))
str(growings$tag)

print(head(raw_abund$EID))
str(raw_abund$EID)

# Whitespace in raw_abund so will have to remove
library(stringr)
library(lubridate)

# Clean up dates in raw_abund
raw_abund <- raw_abund %>%
  mutate(
    EID = str_replace_all(EID, "\\s+", ""),
    micro_date = dmy(micro_date)
  )

print(head(raw_abund$EID))
print(head(raw_abund$micro_date))
# Lets merge now, we will have to merge on closest date

merged <- raw_abund %>%
  inner_join(growings, by = c("EID" = "tag")) %>%  # all combos
  mutate(micro_date_diff = abs(as.numeric(difftime(micro_date, date, units = "days")))) %>%
  group_by(VID) %>%
  slice_min(order_by = micro_date_diff, n = 1, with_ties = FALSE) %>%  # keep closest only
  ungroup()

dim(merged)
nrow(merged)
nrow(raw_abund)

# Right some did not match, lets see, i think they may just be data that was removed due to filtering in previous analysis
missing <- anti_join(raw_abund, growings, by = c("EID" = "tag"))
View(missing)

# So this is an old growings dataset
old_growings <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_1/Re-run 2024/data/growing_animals_2024_raw.csv")
nrow(old_growings)


found_in_old <- anti_join(raw_abund, old_growings, by = c("EID" = "tag"))


View(found_in_old)
colnames(old_growings)


# Just three mising now, these are actually missing in the OG methane PAC data so should follow up on this, funnily two are in CT data


# So merge the old growings with missing data
unmatched_filled <- missing %>%
  inner_join(old_growings, by = c("EID" = "tag")) %>%  # all combos
  mutate(micro_date_diff = abs(as.numeric(difftime(micro_date, date, units = "days")))) %>%
  group_by(VID) %>%                           # or EID_clean if you prefer
  slice_min(order_by = micro_date_diff, n = 1, with_ties = FALSE) %>%    # keep closest
  ungroup()
           

dim(unmatched_filled)
dim(merged)


# Now merge into dataset, now have 357 animals, 
merged_final <- bind_rows(
  growings_match = merged,
  old_growings_match = unmatched_filled,
  .id = "source"
)

n_distinct(merged_final$ANI_ID)


merged_final$Year <- as.numeric(format(as.Date(merged_final$date), "%Y"))

write.csv(merged_final, "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/full_growings_microbiome.csv", row.names = F)

write.csv(missing, "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/missing.csv", row.names = F)



### 

# Forgot that some will not have CT data, the ones that were in missing, so will try and rectify this now


CT_data <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_1/Re-run 2024/data/CT_data.csv")

# going to remove all ct_ data and merge again
merged_final <- read.csv("/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/full_growings_microbiome.csv")

merged_final <- merged_final %>%
  select(
    -starts_with("ct_"),
    -c(Scan_Date, fat_kg, muscle_kg, bone_kg, gigot, EMA, INF, rumen, date_diff_ct, date_diff_ct_matched, days_rumen_methane,
       days_rumen_slaughter, rumen_matched)
  )

CT_data <- CT_data %>%
  mutate(Year = case_when(
    Year_code == 23 ~ 2023,
    Year_code == 24 ~ 2024,
    TRUE ~ NA_real_  # Default case, if needed
  ))

CT_data <- CT_data %>%
  select(Scan_Date, Flock_prefix, Year, Ear_tag_at_CT, ct_weight, ct_fat_kg, ct_muscle_kg, ct_bone_kg, ct_total_kg, ct_KO,
         ct_M_B, ct_M_F, ct_fat, ct_muscle, ct_bone, ct_gigot_shape, ct_EMA, ct_spine_length, ct_IMF, ct_rumen, fat_kg, muscle_kg, 
         bone_kg, gigot, EMA, INF, rumen, ANI_ID)

CT_data$ANI_ID <- as.character(CT_data$ANI_ID)
merged_final$ANI_ID <- as.character(merged_final$ANI_ID)

merged <- merged_final %>%
  left_join(CT_data, by = c("ANI_ID", "Year"))

# Added in ct_date_diff column

merged <- merged %>%
  mutate(
    date        = ymd(date),         # or as.Date() if already in "YYYY-MM-DD"
    Scan_Date   = dmy(Scan_Date),    # or dmy() if it's "DD/MM/YYYY"
    ct_date_diff = abs(as.numeric(difftime(date, Scan_Date, units = "days")))
  )

print(head(merged$ct_date_diff))



write.csv(merged, "/home/dermot.kelly/Dermot_analysis/Phd/Paper_2/microbiome_ml/data/ct_full_growings_microbiome.csv", row.names = F)
