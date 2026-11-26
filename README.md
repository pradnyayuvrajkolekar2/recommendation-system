# Netflix Content-Based Recommender
 
A content-based recommender that suggests similar Netflix titles from their descriptions, directors, cast and country. It is built on the Kaggle **Netflix Movies and TV Shows** dataset (8,803 titles, 12 raw columns).
 
The dataset has **no user ratings**, so collaborative filtering is not possible. Instead, for any title the system recommends the 10 most similar titles. Genres are hidden from the models and used only as the answer key for evaluation.
 
> MSc Computing (AI), Dublin City University: Statistics & Machine Learning group project.
 
---
 
## Results (4,402 test queries)
 
| Model | Precision@10 | NDCG@10 | Coverage |
|---|---|---|---|
| **TF-IDF + cosine** | **0.469** | **0.751** | 96.3% |
| LSA (TF-IDF + Truncated SVD) | 0.456 | 0.709 | 95.3% |
| Random baseline | 0.244 | 0.507 | 99.3% |
 
- **Precision@10:** share of the 10 recommendations that share at least one genre with the query title.
- **NDCG@10:** ranking quality within each model's own top-10 list.
- **Coverage:** share of the catalogue recommended at least once.
TF-IDF performs best, nearly doubling the random baseline. The difference from the baseline is statistically significant (Wilcoxon signed-rank, p < 0.001).

 
---
 
## Setup
 
Requires Python 3.10+.
 
```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```
 

 
---
 
## How to run
 
Run the stages in this order from the repository root:
 
1. **Stage 1: cleaning and feature engineering.** Open the notebooks in `notebooks/` and run all cells, in order. This creates `data/processed/parsed_netflix_data_features.parquet`.
2. **Stages 2–3: modelling and evaluation.** Run the pipeline script:
```bash
python modelling_and_evaluation.py
```
 
   It tunes and trains TF-IDF and LSA, saves their recommendations, and writes the comparison table to `results/model_comparison.csv`.

---
## Stage 1: Data cleaning and feature engineering
 
**Where:** `notebooks/`. **Input:** `data/raw/netflix_titles.csv`. **Output:** `data/processed/parsed_netflix_data_features.parquet`.
 
- **Cleaning:** whitespace and empty values standardised, shifted duration values fixed, duplicates removed, missing director/cast/country set to "Unknown".
- **Parsing:** dates parsed, duration split into minutes (films) and seasons (shows), maturity ratings grouped, multi-valued fields split into lists.
- **Text cleaning:** descriptions lower-cased, accents and punctuation removed, English stop words dropped.
- **Text field ("soup"):** cleaned description plus directors, top-three cast and primary country, each name joined into one token (e.g. `kirstenjohnson`).
- **Genre labels:** 42 genres one-hot encoded (`genre_*` columns), used **only** for evaluation and kept out of the soup. A leakage check confirms this.
- **Feature selection:** structured candidates checked for missingness, near-constant values and Spearman correlation.
---
 
## Stage 2: Modelling
 
**Where:** `src/models/TF_IDF.py` and `src/models/lsa.py`, run by `src/models/modelling_and_evaluation.py`.
 
Both models use the same shared query split: 50/50, stratified by type, seed 42, saved to `results/query_split.csv`. Hyper-parameters are tuned on **validation queries only**, and test queries are scored once with the final settings.
 
Both models are unsupervised. The vectoriser and SVD are fitted on the whole catalogue's text, which contains no genre labels, so this does not leak the answer key.
 
### TF-IDF + cosine similarity
 
Each title's soup becomes a TF-IDF vector, and the 10 titles with the highest cosine similarity are recommended (excluding the title itself).
 
- **Tuning:** 36 configurations: n-gram range (unigram, unigram+bigram), `min_df` (1, 2, 3), vocabulary cap (none, 5,000, 20,000), sublinear TF (on/off).
- **Best config:** unigrams, `min_df=1`, no vocabulary cap, sublinear TF.
**Outputs:** `results/recs_tfidf.parquet`, `results/tuning_results.csv`
 
### LSA (Latent Semantic Analysis)
 
The TF-IDF matrix is compressed with Truncated SVD into latent topics. Vectors are L2-normalised after SVD and compared by cosine similarity, which captures meaning beyond exact word matches.
 
- **Tuning:** 8 configurations: n-gram range (unigram, bigram) × components (50, 100, 200, 300).
- **Best config:** bigram, 300 components (explained variance 23.55%).
- **Example topics:** school and teen stories ("school, high school, teen, students"); documentaries ("documentary, series, stories, follows").
| Set | Precision@10 | NDCG@10 |
|---|---|---|
| Validation (best config) | 0.4560 | 0.7047 |
| Test | 0.4556 | 0.7090 |
 
**Outputs:** `results/recs_lsa.parquet`, `results/lsa_tuning_table.csv`, `results/lsa_explained_variance.png`
 
---
 
## Stage 3: Evaluation
 
**Where:** `src/evaluation.py`, run by `src/models/modelling_and_evaluation.py` after the models.
 
Scores every `results/recs_<model>.parquet` file, plus a random baseline, on the same test queries, and writes `results/model_comparison.csv`.
 
**Adding a new model:** save its recommendations as `results/recs_<name>.parquet` with columns `query_id`, `rank` (1–10), `rec_id`, `score`, covering the test queries in `results/query_split.csv`. The evaluation step then includes it in the comparison automatically.
 

---
 
## Limitations
 
- Genre overlap is only a proxy for a good recommendation; with no user data, results are not personalised and evaluation is offline only.
- LSA was tuned over fewer configurations than TF-IDF, so part of the gap may come from settings rather than the SVD step.
- Results come from a single 2021 catalogue snapshot and one train/test split.

