

import pandas as pd
import numpy as np
import os

DATA_PATH = "/content/sample_data/parsed_netflix_data_features (1).parquet"

df = pd.read_parquet(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
display(df.head())

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

SEED = 42

val_ids, test_ids = train_test_split(
    df["show_id"],
    test_size=0.5,
    random_state=SEED,
    stratify=df["type"]
)

os.makedirs("/content/results", exist_ok=True)

split_df = pd.concat([
    pd.DataFrame({
        "show_id": val_ids,
        "split": "val"
    }),
    pd.DataFrame({
        "show_id": test_ids,
        "split": "test"
    })
], ignore_index=True)

split_df.to_csv("/content/results/query_split.csv", index=False)

print("Query split created successfully!")
print(split_df["split"].value_counts())
print(split_df.head())

import sys

sys.path.insert(0, "/content/sample_data")

from evaluation import genre_matrix, evaluate, format_results

g_matrix = genre_matrix(df)

print("Genre matrix shape:", g_matrix.shape)
print("Evaluation functions loaded successfully!")

# Load the shared query split

split_df = pd.read_csv("/content/results/query_split.csv")

row_lookup = pd.Series(
    np.arange(len(df)),
    index=df["show_id"]
)

val_ids = split_df.loc[
    split_df["split"] == "val", "show_id"
]

test_ids = split_df.loc[
    split_df["split"] == "test", "show_id"
]

val_idx = row_lookup.loc[val_ids].to_numpy()
test_idx = row_lookup.loc[test_ids].to_numpy()

print("Validation queries:", len(val_idx))
print("Test queries:", len(test_idx))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

SEED = 42

tfidf_configs = [
    {
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_features": 10000,
        "name": "unigram_min2"
    },
    {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 10000,
        "name": "bigram_min2"
    }
]

print("LSA configurations ready!")
print("Validation queries:", len(val_idx))
print("SVD components:", [50, 100, 200, 300])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

SEED = 42
K = 10

tuning_results = []

best_precision = -1
best_model_info = None

for cfg in tfidf_configs:

    vec = TfidfVectorizer(
        ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"],
        max_features=cfg["max_features"]
    )

    X_tfidf = vec.fit_transform(df["soup"])

    for n_comp in [50, 100, 200, 300]:

        svd = TruncatedSVD(
            n_components=n_comp,
            random_state=SEED
        )

        X_lsa = svd.fit_transform(X_tfidf)
        X_lsa_norm = normalize(X_lsa, norm="l2", axis=1)

        Q_val = X_lsa_norm[val_idx]
        sims_val = np.dot(Q_val, X_lsa_norm.T)

        sims_val[np.arange(len(val_idx)), val_idx] = -np.inf

        top10 = np.argpartition(
            -sims_val, 10, axis=1
        )[:, :10]

        row_idx = np.arange(len(val_idx))[:, None]

        sorted_idx = np.argsort(
            -sims_val[row_idx, top10], axis=1
        )

        final_top10 = np.take_along_axis(
            top10, sorted_idx, axis=1
        )

        # Pradnya's evaluation function
        metrics = evaluate(
            g_matrix,
            val_idx,
            final_top10,
            k=K
        )

        exp_var = svd.explained_variance_ratio_.sum() * 100

        result = {
            "tfidf_config": cfg["name"],
            "ngram_range": str(cfg["ngram_range"]),
            "n_components": n_comp,
            "explained_variance_%": round(exp_var, 2),
            "precision": round(metrics["precision"], 4),
            "ndcg": round(metrics["ndcg"], 4)
        }

        tuning_results.append(result)

        print(
            f"LSA ({cfg['name']}, {n_comp}): "
            f"Precision={metrics['precision']:.4f}, "
            f"NDCG={metrics['ndcg']:.4f}"
        )

        if metrics["precision"] > best_precision:

            best_precision = metrics["precision"]

            best_model_info = {
                "config": cfg,
                "n_components": n_comp,
                "vec": vec,
                "svd": svd,
                "X_lsa_norm": X_lsa_norm
            }

tuning_df = pd.DataFrame(tuning_results)

os.makedirs("/content/results", exist_ok=True)

tuning_df.to_csv(
    "/content/results/lsa_tuning_table.csv",
    index=False
)

print("\nLSA tuning completed!")
print(tuning_df.to_string(index=False))

print("\nBest configuration:")
print(best_model_info["config"])
print("Best components:", best_model_info["n_components"])
print("Best validation precision:", best_precision)

# Generate top-10 LSA recommendations for test queries

X_lsa_norm = best_model_info["X_lsa_norm"]

Q_test = X_lsa_norm[test_idx]
sims_test = np.dot(Q_test, X_lsa_norm.T)

# Exclude recommending the query itself
sims_test[np.arange(len(test_idx)), test_idx] = -np.inf

# Get top 10 recommendations
top10 = np.argpartition(-sims_test, 10, axis=1)[:, :10]

row_idx = np.arange(len(test_idx))[:, None]
sorted_idx = np.argsort(-sims_test[row_idx, top10], axis=1)
final_top10 = np.take_along_axis(top10, sorted_idx, axis=1)

# Save in Pradnya's required format
recommendations = []

for i, query_idx in enumerate(test_idx):
    for rank, rec_idx in enumerate(final_top10[i], start=1):
        recommendations.append({
            "query_id": df.iloc[query_idx]["show_id"],
            "rank": rank,
            "rec_id": df.iloc[rec_idx]["show_id"],
            "score": float(sims_test[i, rec_idx])
        })

recs_lsa = pd.DataFrame(recommendations)
recs_lsa.to_parquet("/content/results/recs_lsa.parquet", index=False)

print("LSA test recommendations saved!")
print("Shape:", recs_lsa.shape)
print(recs_lsa.head(10))

import os
import sys
import shutil
import importlib

# Required folder create
os.makedirs("/content/data/processed", exist_ok=True)

# Dataset correct location मध्ये copy
shutil.copy(
    "/content/sample_data/parsed_netflix_data_features (1).parquet",
    "/content/data/processed/parsed_netflix_data_features.parquet"
)

# Evaluation.py reload
sys.path.insert(0, "/content/sample_data")
os.chdir("/content")

import evaluation
importlib.reload(evaluation)

# Pradnya's evaluation run
evaluation.compare_all()

import os
import matplotlib.pyplot as plt

os.makedirs("/content/reports/figures", exist_ok=True)

# Graph 1: Explained Variance
plt.figure(figsize=(7,4))

for name, g in tuning_df.groupby("tfidf_config"):
    plt.plot(g["n_components"], g["explained_variance_%"],
             marker="o", label=name)

plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")
plt.title("LSA: Explained Variance vs Components")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("/content/reports/figures/lsa_explained_variance.png",
            dpi=150, bbox_inches="tight")
plt.show()


# Graph 2: Validation Precision
plt.figure(figsize=(7,4))

for name, g in tuning_df.groupby("tfidf_config"):
    plt.plot(g["n_components"], g["precision"],
             marker="o", label=name)

plt.xlabel("Number of components")
plt.ylabel("Validation Precision")
plt.title("LSA: Validation Precision vs Components")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("/content/reports/figures/lsa_val_precision_vs_components.png",
            dpi=150, bbox_inches="tight")
plt.show()

print("Both graphs saved successfully!")

import os

print(os.listdir("results"))

from google.colab import files

files.download("results/model_comparison.csv")