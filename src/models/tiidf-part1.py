import os
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# define teh random state ,k value and teh columns
SEED =42
K=10
ID_COL,TYPE_COL,TITLE_COL="show_id","type","title"

SOUP_COL="soup"
GENRE_COLUMN="listed_in"
GENRE_PREFIX="genre_"

#this function is to find the features file 
def find_features_file():

    folders=["data/processed","../data/processed","."]
    names=["parsed_netflix_data_features.parquet","features_netflix_data.parquet"]
    candidates= [Path(folder) /name for folder in folders for name in names]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])



CLEAN_PATH=find_features_file()
RESULTS_DIR=Path("results") if Path("data").exists() else Path("../results")
SPLIT_PATH= RESULTS_DIR/ "query_split.csv"
RESULTS_DIR.mkdir(parents=True,exist_ok=True)

#load the data
def load_data(path):
    #read teh paraquet file
    df=pd.read_parquet(path).reset_index(drop=True)
# if the ID_COL is not unique raise an Error
    if not df[ID_COL].is_unique:
        raise ValueError("Expected values to be unique")
# if the SOUP_COL is not in df raise an Error

    if SOUP_COL not in df.columns:
        raise KeyError("Missing column")

    df[SOUP_COL]=df[SOUP_COL].fillna("").astype(str)
    #check the empty count
    empty_count= int((df[SOUP_COL].str.strip()=="").sum())
    print(f"Loaded the data : {empty_count} empty")
    return df


#check for leakage in teh dataframe
def leakage(df):
#clean genre column
    genres=df[GENRE_COLUMN].fillna("").str.split(", ").explode().dropna().unique()
    #get multiword genres
    multi_word_genres=[str(g).lower() for g in genres if len(str(g).split())>1]

    soups=df[SOUP_COL].str.lower()
    #cehck for matches
    matches=sum(int(soups.str.contains(genre,regex=False).sum()) for genre in multi_word_genres)
    print(f"multi word genres in soups: {matches}")
    return matches


#splitting the data by stratifying the type col
def get_split(df,path=SPLIT_PATH):

    split_file=Path(path)

    if split_file.exists():
        split_df=pd.read_csv(split_file)
        print("Loaded existing split")
    else:
        val_ids, test_ids= train_test_split( df[ID_COL],test_size=0.5, random_state=SEED,stratify=df[TYPE_COL])

        split_df=pd.concat(
            [
                pd.DataFrame({ID_COL:val_ids,"split":"val"}),
                pd.DataFrame({ID_COL:test_ids,"split":"test"}),
            ], ignore_index=True,
        )
#save to csv
        split_df.to_csv(split_file,index=False)
        print("Craeted and saved split")

#get row lookups and value and test ids
    row_lookup=pd.Series(np.arange(len(df)),index=df[ID_COL])
    val_ids_order=split_df.loc[split_df["split"]== "val",ID_COL]
    test_ids_order= split_df.loc[split_df["split"]== "test",ID_COL]
    value_index=row_lookup.loc[val_ids_order].to_numpy()
    test_index= row_lookup.loc[test_ids_order].to_numpy()
#return value and test indexes
    return value_index,test_index

#fit the model with the parameters
def fit_tfidf(df,params):
    vec=TfidfVectorizer(**params)
    mat=vec.fit_transform(df[SOUP_COL])
    return vec,mat
#getting recommendations and scores

def rec_by_sim(X,query_index,k=K,batch_size=500):
    query_index=np.asarray(query_index)
    no_of_queries=len(query_index)
#empty arrays initialized for scores and recs
    recs= np.empty((no_of_queries,k),dtype=int)
    scores= np.empty((no_of_queries,k),dtype=np.float32)

    for start in range(0,no_of_queries,batch_size):
        end=min(start+batch_size,no_of_queries)
        q_rows=query_index[start:end]
#get similarities
        sims=(X[q_rows]@X.T).toarray()
        sims[np.arange(len(q_rows)),q_rows]=-np.inf
#get top columns and scores
        top_cols=np.argsort(-sims,axis=1)[:,:k]
        top_scores=np.take_along_axis(sims,top_cols,axis=1)

        recs[start:end]=top_cols
        scores[start:end]=top_scores
#return the recs and scores
    return recs, scores

#saving the recommendations into dataframe and saving it at output path
def save_rec(df,query_index,recs,scores,model_name):
        k=recs.shape[1]
        shows_ids=df[ID_COL].to_numpy()

        out=pd.DataFrame(
            {
                "query_id": np.repeat(shows_ids[query_index],k),
                "rank": np.tile(np.arange(1,k+1),len(query_index)),
                "rec_id":shows_ids[recs.ravel()],
                "score": scores.ravel()
            }
        )

        out_path=RESULTS_DIR/f"recs_{model_name}.parquet"
        out.to_parquet(out_path,index=False)
        print("Saved he output")

        return str(out_path)



if __name__=="__main__":
        #laod data
        df=load_data()
        #cehck for leakage
        leakage(df)
        #get teh split
        val_index,test_index=get_split(df)
#fit the data
        _,X=fit_tfidf(df,{"sublinear_tf":True})
        sample_queries=val_index[:3]
        #get recommendations
        recs,_ =rec_by_sim(X,sample_queries)
#get top titles
        for q_idx,rec_row in zip(sample_queries,recs):
            title=df.loc[q_idx,TITLE_COL]
            top_titles=df.loc[rec_row[:3],TITLE_COL].tolist()
            print(f"{title}-> {top_titles}")

        print("part a complete")


    
        
