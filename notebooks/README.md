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
