import glob
import os
from os import path
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import ndcg_score

# shared setting for evaluation
SEED = 42
K = 10
ID_COL = "show_id"
TYPE_COL = "type"
GENRE_PREFIX = "genre_"

CANDIDATES = [
    f"{d}/{n}"
    for d in ["data/processed", "../data/processed", "."]
    for n in ["parsed_netflix_data_features.parquet", "features_netflix_data.parquet"]
]
DATA_PATH = next((p for p in CANDIDATES if os.path.exists(p)), CANDIDATES[0])
RESULTS_DIR = "results" if os.path.exists("data") else "../results"
SPLIT_PATH = f"{RESULTS_DIR}/query_split.csv"

def genre_matrix(df):
    #lead the genre answer key and test split
    columns = [c for c in df.columns if c.startswith(GENRE_PREFIX)]
    return sparse.csr_matrix(df[columns].values.astype(np.float32))

def load_eval_data(data_path, split_path):
    #loading what is needed for evaluation
    df = pd.read_parquet(path=data_path).reset_index(drop=True)
    g_matrix = genre_matrix(df)
    id_to_index = pd.Series(df.index.values, index=df[ID_COL])
    split_df = pd.read_csv(split_path)
    test_ids = id_to_index.loc[split_df.loc[split_df["split"] == "test", ID_COL]].values
    return df, g_matrix, id_to_index, test_ids

def evaluate(g_matrix, query_ids, recommendations, k=K):
    #evaluate any model
    top_recommendations = recommendations[:, :k]
    g_matrix_arr = g_matrix.toarray()
    shared = (g_matrix_arr[query_ids][:, None, :] * g_matrix_arr[recommendations]).sum(axis=2)
    rel = (shared > 0).astype(float)

    #add all matrices in 4 one-liners
    precision = rel.mean()
    ndcg = ndcg_score(rel, np.tile(np.arange(k, 0, -1), (len(query_ids), 1)), k=k)
    coverage = len(np.unique(top_recommendations)) / g_matrix.shape[0]

    return {
        "precision": float(precision),
        "ndcg": float(ndcg),
        "coverage": float(coverage)
    }

def format_results(metrics):
    #format the results for printing
    return {name: round(value, 4) for name, value in metrics.items()}

def compare_all():
    #compare all the models in the results directory
    df, g_matrix, id_to_index, test_ids = load_eval_data(DATA_PATH, SPLIT_PATH)

    # baseline
    random = np.random.RandomState(SEED)
    random_recommendations = np.array([random.choice(np.arange(len(df)), size=K, replace=False) for _ in range(len(test_ids))])
    rows = [{"model": "random", **evaluate(g_matrix, test_ids, random_recommendations)}]

    # Score every results/recs_<model>.parquet file (tfidf, lsa, etc.)
    for path in sorted(glob.glob(f"{RESULTS_DIR}/recs_*.parquet")):
        name = os.path.basename(path).removeprefix("recs_").removesuffix(".parquet")
        r = pd.read_parquet(path).sort_values(["query_id", "rank"])
        assert (r["query_id"] != r["rec_id"]).all(), f"{name}: recommended itself!"

        q_idx = id_to_index[r["query_id"].drop_duplicates()].values
        recs = id_to_index[r["rec_id"]].values.reshape(len(q_idx), K)
        rows.append({"model": name, **evaluate(g_matrix , q_idx, recs)})

    table = pd.DataFrame(rows).sort_values("precision", ascending=False)
    table.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
    print(table.round(4).to_string(index=False))

if __name__ == "__main__":
    compare_all()