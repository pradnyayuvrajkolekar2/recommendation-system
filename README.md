# Personalised Recommendation System

A context-aware Movie & TV Show recommendation engine built on the Kaggle Netflix Movies and TV Shows dataset (~8,800 titles, 12 raw columns)[cite: 1, 6].

---
---

## Setup & How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt


   ---

## Stage 2: Modelling 

### LSA model

**File:** `src/models/lsa.py`

**Method:** The TF-IDF matrix of the catalogue `soup` is compressed with
Truncated SVD into latent topics. Vectors are L2-normalised after SVD, then
titles are compared by cosine similarity. This captures meaning beyond exact
word matches.

**Input:** `parsed_netflix_data_features.parquet` (output of Stage 1).

**Tuning (validation queries only):** ngram range (unigram or bigram) and
number of components (50, 100, 200, 300), chosen by Precision@10.

**Best config:** bigram, 300 components (explained variance 23.55%).

| Set | Precision@10 | NDCG@10 |
| :--- | :---: | :---: |
| Validation (best config) | 0.4560 | 0.7047 |
| Test | 0.4556 | 0.7090 |
| Random baseline (validation) | 0.2359 | - |

**Topics found (examples):** Topic 3 is about school and teen stories
("school, high school, teen, students"). Topic 4 is about documentaries
("documentary, series, stories, follows").

**Run:**

    python src/models/lsa.py

**Outputs (in `results/`):**
- `recs_lsa.parquet` with columns `query_id, rank, rec_id, score`
- `lsa_tuning_table.csv`
- `lsa_explained_variance.png`

**Note:** TF-IDF and SVD are fitted on the whole catalogue (including test
titles). This is unsupervised, so no genre labels are used in fitting.
