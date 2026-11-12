# Netflix titles

## Files
| File | Purpose |
|---|---|
| `Person1_Loading_and_Cleaning.ipynb` | Colab/Jupyter notebook with the 4 steps and the checks |
| `netflix_clean.csv` | Cleaned dataset (8,806 rows × 12 columns) |
| `removed_duplicates.csv` | Rows dropped in Step 4 (audit trail) |
| `quarantine_ratings.csv` | Ratings not in the valid certificate list (empty for this file) |

## What was done
1. **Load raw** – all columns as text, 12 expected columns verified, row count matches the file (8,807), checksum logged.
2. **Standardise strings** – leading/trailing spaces stripped (88 `date_added` values, 1 title), empty strings → missing.
3. **Fix shifted durations** – 3 rows with `'74 min'`-style values in `rating` moved to `duration`.
4. **Drop duplicates** – 0 exact duplicates; 1 title+type+release_year duplicate removed ("Consequences", 2014), keep first.

Missing Values and Final Validation

### Files

| File | Purpose |
|---|---|
| `Missing_Values_and_Validation.ipynb` | Colab notebook containing missing-value handling and final validation |
| `netflix_final_cleaned.csv` | Final dataset after missing-value handling and validation (8,803 rows × 12 columns) |

### What was done

1. **Check missing values** – Identified 4,305 missing values across `director`, `cast`, `country`, `date_added`, and `rating`.
2. **Handle missing values** – Replaced missing values in `director`, `cast`, `country`, and `rating` with `"Unknown"`.
3. **Preserve missing dates** – Kept the 10 missing values in `date_added` unchanged because the actual dates were unavailable.
4. **Validate duplicates** – Confirmed 0 exact duplicate rows and 0 duplicate records based on `title`, `type`, and `release_year`.
5. **Final validation** – Verified the expected 12 columns and final dataset shape of 8,803 rows × 12 columns. The only remaining missing values are the 10 in `date_added`.
6. **Save final dataset** – Exported the validated dataset as `netflix_final_cleaned.csv`.
