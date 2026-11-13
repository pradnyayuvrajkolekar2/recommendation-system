# Personalised Recommendation System

A context-aware Movie & TV Show recommendation engine built on the Kaggle Netflix Movies and TV Shows dataset (~8,800 titles, 12 raw columns)[cite: 1, 6].

---

## Stage 1: Data Ingestion & Cleaning Pipeline

Stage 1 transforms the raw Netflix dataset into a validated, ML-ready feature table across a 4-stage sequential pipeline (`P1 -> P2 -> P3 -> P4`)[cite: 1, 4]:

| Order | Owner | Module / Step | Key Transformations & Deliverables |
| :---: | :---: | :--- | :--- |
| **1** | **Apoorva** | Loading & Basic Cleaning | Reads raw CSV as text, validates schema, strips whitespace, fixes shifted duration values (`74 min` in `rating`), and removes exact/title-year duplicates. |
| **2** | **Shrushti** | Missing Values & Validation | Generates before/after missingness reports, imputes missing `director`/`cast`/`country` as `'Unknown'` and `rating` via type-specific mode, drops rows missing `date_added`/`duration`, and runs final schema validation. |
| **3** | **Pradnya** | **Parsing & Multi-Valued Fields** | Parses `date_added` (`year_added`, `month_added`), splits `duration` into numeric `duration_min` (Movies) and `seasons` (TV Shows), maps `rating` to standardized `maturity_group` tiers, converts `listed_in`/`director`/`cast`/`country` into Python lists (capping `cast_list` to top 3), and extracts `primary_country` (grouping countries with `< 20` titles into `'Other'`). |
| **4** | **Jeslida** | Text Cleaning & Feature Engineering | Cleans plot descriptions, builds multi-hot genre labels, computes `content_age` and `is_movie`, constructs the leak-free combined text `'soup'`, selects final features, and exports the clean Parquet table. |

---

## Pradnya -> Output Schema (`parsed_netflix_data.parquet`)

After Step 3, the dataset contains **22 standardized columns** ready for text cleaning and feature engineering[cite: 3, 4]:

* **Preserved Raw Columns:** `show_id`, `type`, `title`, `director`, `cast`, `country`, `listed_in`, `description`, `release_year`[cite: 4]
* **Parsed Temporal & Numeric Fields:** `date_added` (datetime), `year_added` (`Int64`), `month_added` (`Int64`), `duration_min` (float, Movies only), `seasons` (float, TV Shows only)[cite: 3, 4]
* **Standardized Maturity Fields:** `maturity_rating`, `maturity_group` (`Kids`, `Older Kids`, `Teens`, `Adults`, `Unrated`, `Unknown`)[cite: 3, 4]
* **Structured Multi-Valued & Geographic Fields:** `genres` (list), `directors` (list), `cast_list` (top-3 list), `countries` (list), `primary_country` (string, `< 20` threshold mapped to `'Other'`)[cite: 3, 4]

---

## Setup & How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt