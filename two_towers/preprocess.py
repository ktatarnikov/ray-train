from typing import Dict, Any, Tuple, Text

import json
import os
import argparse
from pyarrow import csv
import ray
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def preprocess_dataset(path: str, file: str) -> None:
    decompress(path, file)
    dataset_path = os.path.join(path, os.path.basename(file).split(".")[0])

    load_raw(dataset_path)
    extract_metadata(dataset_path)
    generate_train_val_test(dataset_path)

def decompress(path: str, file: str) -> None:
    os.system(f"unzip {path}/{file} -d {path}")

def load_raw(path: str):
    def skip_comment(row):
        return 'skip'

    parse_options = csv.ParseOptions(delimiter=",", invalid_row_handler=skip_comment)
    movies = ray.data.read_csv(os.path.join(path, "movies.csv"), parse_options=parse_options)
    movies = movies.select_columns(["movieId", "title"])
    movies.write_parquet(os.path.join(path, "movies.parquet"))

    ratings = ray.data.read_csv(os.path.join(path, "ratings.csv"), parse_options=parse_options)
    ratings = ratings.select_columns(["userId", "movieId", "rating"])
    ratings.write_parquet(os.path.join(path, "ratings.parquet"))

def extract_metadata(path: str) -> None:
    ratings_parquet_path = os.path.join(path, "ratings.parquet")
    movies_parquet_path = os.path.join(path, "movies.parquet")
    metadata_path = os.path.join(path, "metadata.json")

    ratings = ray.data.read_parquet(ratings_parquet_path)
    movies = ray.data.read_parquet(movies_parquet_path)

    print(ratings.schema())
    print(ratings.take(2))

    print(movies.schema())
    print(movies.take(2))

    unique_user_ids = ratings.unique("userId")
    unique_movie_ids = movies.unique("movieId")

    metadata = {
        "unique_user_ids": unique_user_ids,
        "unique_movie_ids": unique_movie_ids
    }
    with open(metadata_path, "w") as f:
        f.write(json.dumps(metadata, cls=NpEncoder))
    
def generate_train_val_test(path: str) -> None:
    ratings_parquet_path = os.path.join(path, "ratings.parquet")
    train_parquet_path = os.path.join(path, "train")
    test_parquet_path = os.path.join(path, "test")

    ratings = ray.data.read_parquet(ratings_parquet_path)
    shuffled = ratings.random_shuffle(seed=42)
    train, test = shuffled.split_proportionately([0.8])

    train.write_parquet(train_parquet_path)
    test.write_parquet(test_parquet_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", required=True, type=str, help="Data path"
    )
    parser.add_argument(
        "--file", required=True, type=str, help="Dataset zip"
    )
    args, _ = parser.parse_known_args()

    preprocess_dataset(args.path, args.file)