from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.export_dashboard_artifacts import export_dashboard_artifacts
from src.local_data_config import load_local_data_config

SEED = 42
TOPK_GRID = (5, 10, 20)
MAX_RECOMMEND = max(TOPK_GRID)
DATASET_SPECS = {
    "movielens_20m": {
        "name": "MovieLens 20M",
        "urls": {"ratings_zip": "https://files.grouplens.org/datasets/movielens/ml-20m.zip"},
        "user_sample_mod": 15,
        "user_sample_keep": 1,
        "min_user": 10,
        "min_item": 20,
        "max_users": 5000,
        "max_items": 3200,
    },
    "google_local_sc": {
        "name": "Google Local South Carolina",
        "urls": {
            "ratings": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/rating-South_Carolina.csv.gz",
            "meta": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/meta-South_Carolina.json.gz",
        },
        "user_sample_mod": 10,
        "user_sample_keep": 1,
        "min_user": 5,
        "min_item": 10,
        "max_users": 5000,
        "max_items": 3200,
    },
    "goodreads_fantasy": {
        "name": "Goodreads Fantasy 10K",
        "urls": {
            "reviews": "https://mcauleylab.ucsd.edu/public_datasets/pml_data/fantasy_10000.json",
            "books": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_books_fantasy_paranormal.json.gz",
        },
        "user_sample_mod": None,
        "user_sample_keep": None,
        "min_user": 3,
        "min_item": 3,
        "max_users": 2500,
        "max_items": 2500,
    },
}
HYBRID_TEMPLATES = [
    {"ease": 0.34, "svd": 0.20, "content": 0.12, "popularity": 0.03, "novelty": 0.04, "quality": 0.05, "freshness": 0.05, "tag_overlap": 0.05, "category_match": 0.04, "year_alignment": 0.03, "price_match": 0.02, "richness_alignment": 0.03},
    {"ease": 0.28, "svd": 0.16, "content": 0.18, "popularity": 0.03, "novelty": 0.05, "quality": 0.05, "freshness": 0.04, "tag_overlap": 0.08, "category_match": 0.05, "year_alignment": 0.02, "price_match": 0.02, "richness_alignment": 0.04},
    {"ease": 0.24, "svd": 0.15, "content": 0.22, "popularity": 0.03, "novelty": 0.06, "quality": 0.05, "freshness": 0.05, "tag_overlap": 0.09, "category_match": 0.05, "year_alignment": 0.02, "price_match": 0.01, "richness_alignment": 0.05},
    {"ease": 0.22, "svd": 0.18, "content": 0.18, "popularity": 0.05, "novelty": 0.05, "quality": 0.06, "freshness": 0.05, "tag_overlap": 0.06, "category_match": 0.06, "year_alignment": 0.03, "price_match": 0.02, "richness_alignment": 0.04},
    {"ease": 0.18, "svd": 0.14, "content": 0.28, "popularity": 0.03, "novelty": 0.06, "quality": 0.04, "freshness": 0.04, "tag_overlap": 0.10, "category_match": 0.04, "year_alignment": 0.02, "price_match": 0.01, "richness_alignment": 0.06},
]
ALGO_REASONS = {
    "Popularity": "global popularity prior",
    "TruncatedSVD": "latent factor affinity",
    "EASE": "closed-form co-occurrence signal",
    "TFIDF_Content": "metadata and text similarity",
    "SequentialTransition": "order-aware transition signal from recent user history",
    "HybridFeatureRerank": "candidate rerank with collaborative, content, recency and profile-alignment features",
    "Hybrid_NoContentAblation": "hybrid rerank without content and profile-alignment features",
    "Hybrid_NoCollaborativeAblation": "hybrid rerank without collaborative candidate signals",
}
FEATURE_LABELS = {
    "ease": "strong collaborative co-occurrence",
    "svd": "latent factor match",
    "content": "text and metadata similarity",
    "popularity": "high catalog popularity",
    "novelty": "long-tail novelty",
    "quality": "high average rating",
    "freshness": "recent item activity",
    "tag_overlap": "tag overlap with user profile",
    "category_match": "category affinity",
    "year_alignment": "matching item era",
    "price_match": "matching price tier",
    "richness_alignment": "matching metadata richness",
}
CONTENT_FEATURES = {"content", "tag_overlap", "category_match", "year_alignment", "price_match", "richness_alignment"}
COLLABORATIVE_FEATURES = {"ease", "svd"}
SEQUENTIAL_PARAM_GRID = (
    {"popularity_blend": 0.03, "recent_window": 3, "max_lag": 2},
    {"popularity_blend": 0.05, "recent_window": 5, "max_lag": 3},
    {"popularity_blend": 0.08, "recent_window": 5, "max_lag": 4},
    {"popularity_blend": 0.10, "recent_window": 7, "max_lag": 5},
)
SIGNIFICANCE_METRICS = ("ndcg", "recall")
SIGNIFICANCE_COMPARISONS = (
    ("HybridFeatureRerank", "EASE"),
    ("HybridFeatureRerank", "TFIDF_Content"),
    ("HybridFeatureRerank", "SequentialTransition"),
    ("SequentialTransition", "EASE"),
    ("HybridFeatureRerank", "Hybrid_NoContentAblation"),
    ("HybridFeatureRerank", "Hybrid_NoCollaborativeAblation"),
)
PLANNED_BENCHMARK_STATUS = {
    "NCF": {
        "MovieLens 20M": "planned",
        "Google Local South Carolina": "not_applicable",
        "Goodreads Fantasy 10K": "planned",
    },
    "SASRec": {
        "MovieLens 20M": "planned",
        "Google Local South Carolina": "planned",
        "Goodreads Fantasy 10K": "planned",
    },
    "GRU4Rec": {
        "MovieLens 20M": "not_applicable",
        "Google Local South Carolina": "planned",
        "Goodreads Fantasy 10K": "planned",
    },
    "RBM": {
        "MovieLens 20M": "draft",
        "Google Local South Carolina": "not_applicable",
        "Goodreads Fantasy 10K": "not_applicable",
    },
}


@dataclass
class PreparedDataset:
    name: str
    slug: str
    interactions: pd.DataFrame
    items: pd.DataFrame
    train_base: pd.DataFrame
    valid: pd.DataFrame
    train_full: pd.DataFrame
    test: pd.DataFrame
    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    idx_to_user: list[str]
    idx_to_item: list[str]
    train_base_matrix: sp.csr_matrix
    train_full_matrix: sp.csr_matrix
    item_features: pd.DataFrame
    item_tokens: list[set[str]]
    user_tag_profiles: dict[int, set[str]]
    user_top_categories: dict[int, set[str]]
    user_preferred_year: np.ndarray
    user_preferred_price: np.ndarray
    user_preferred_richness: np.ndarray
    user_activity_norm: np.ndarray
    eval_user_indices: list[int]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    mask = np.isfinite(values)
    if not mask.any():
        return np.zeros_like(values, dtype=float)
    min_value = values[mask].min()
    max_value = values[mask].max()
    if not np.isfinite(min_value) or not np.isfinite(max_value) or math.isclose(max_value, min_value):
        output = np.zeros_like(values, dtype=float)
        output[~mask] = 0.0
        return output
    output = np.zeros_like(values, dtype=float)
    output[mask] = (values[mask] - min_value) / (max_value - min_value)
    return output


def safe_alignment(values: np.ndarray, preference: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.zeros(len(values), dtype=float)
    if not np.isfinite(preference):
        return output
    mask = np.isfinite(values)
    if not mask.any():
        return output
    output[mask] = 1.0 - np.clip(np.abs(values[mask] - preference), 0.0, 1.0)
    return output


def timestamps_to_datetime(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    unit = "ms" if numeric.dropna().median() > 10**11 else "s"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def bucketize_activity(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 5:
        return "4-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    if value <= 50:
        return "21-50"
    return "51+"


def download_file(url: str, target_path: Path) -> Path:
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[download] reuse {target_path.name}")
        return target_path
    ensure_directory(target_path.parent)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"  {target_path.name}: {downloaded / total:.0%}", end="\r")
    print(f"  {target_path.name}: done{' ' * 12}")
    tmp_path.replace(target_path)
    return target_path


def iterative_k_core(frame: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    current = frame.copy()
    while True:
        before = len(current)
        user_counts = current["user_id"].value_counts()
        item_counts = current["item_id"].value_counts()
        current = current[current["user_id"].isin(user_counts[user_counts >= min_user].index)]
        current = current[current["item_id"].isin(item_counts[item_counts >= min_item].index)]
        if len(current) == before:
            return current


def stable_user_sample(frame: pd.DataFrame, mod: int | None, keep: int | None) -> pd.DataFrame:
    if not mod or keep is None:
        return frame
    hashes = pd.util.hash_pandas_object(frame["user_id"].astype("string"), index=False).astype("uint64")
    return frame[(hashes % mod) < keep].copy()


def top_frequency_subset(frame: pd.DataFrame, max_users: int, max_items: int, min_user: int, min_item: int) -> pd.DataFrame:
    current = frame.copy()
    current = iterative_k_core(current, min_user=min_user, min_item=min_item)
    if current["user_id"].nunique() > max_users:
        keep_users = current["user_id"].value_counts().head(max_users).index
        current = current[current["user_id"].isin(keep_users)].copy()
        current = iterative_k_core(current, min_user=min_user, min_item=min_item)
    if current["item_id"].nunique() > max_items:
        keep_items = current["item_id"].value_counts().head(max_items).index
        current = current[current["item_id"].isin(keep_items)].copy()
        current = iterative_k_core(current, min_user=min_user, min_item=min_item)
    return current


def parse_google_meta(meta_path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(meta_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            categories = obj.get("category") or []
            rows.append(
                {
                    "item_id": str(obj["gmap_id"]),
                    "item_name": obj.get("name") or str(obj["gmap_id"]),
                    "item_text": " ".join([obj.get("name") or "", obj.get("description") or "", " ".join(categories)]).strip(),
                    "tag_text": " ".join(categories).strip(),
                    "price_bucket": len(obj.get("price") or ""),
                    "metadata_count": len(categories),
                }
            )
    return pd.DataFrame(rows)


def parse_goodreads_books(books_path: Path, item_ids: set[str]) -> pd.DataFrame:
    rows = []
    with gzip.open(books_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            book_id = str(obj["book_id"])
            if book_id not in item_ids:
                continue
            shelves = [entry["name"] for entry in (obj.get("popular_shelves") or [])[:10]]
            rows.append(
                {
                    "item_id": book_id,
                    "item_name": obj.get("title") or book_id,
                    "item_text": " ".join([obj.get("title") or "", obj.get("description") or "", " ".join(shelves)]).strip(),
                    "tag_text": " ".join(shelves).strip(),
                    "year_hint": pd.to_numeric(obj.get("publication_year"), errors="coerce"),
                    "metadata_count": len(shelves),
                }
            )
    return pd.DataFrame(rows)


def load_movielens(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    zip_path = download_file(DATASET_SPECS["movielens_20m"]["urls"]["ratings_zip"], raw_root / "movielens_20m" / "ml-20m.zip")
    with zipfile.ZipFile(zip_path) as archive:
        ratings = pd.read_csv(
            archive.open("ml-20m/ratings.csv"),
            usecols=["userId", "movieId", "rating", "timestamp"],
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
        )
        movies = pd.read_csv(
            archive.open("ml-20m/movies.csv"),
            usecols=["movieId", "title", "genres"],
            dtype={"movieId": "int32", "title": "string", "genres": "string"},
        )
    ratings = ratings[ratings["rating"] >= 4.0].copy()
    ratings["user_id"] = ratings["userId"].astype(str)
    ratings["item_id"] = ratings["movieId"].astype(str)
    interactions = ratings[["user_id", "item_id", "rating", "timestamp"]].copy()
    interactions["dataset"] = DATASET_SPECS["movielens_20m"]["name"]
    interactions["target"] = 1
    movies["year_hint"] = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    movies["item_id"] = movies["movieId"].astype(str)
    movies["item_name"] = movies["title"].fillna(movies["item_id"])
    movies["item_text"] = (movies["title"].fillna("") + " " + movies["genres"].fillna("").str.replace("|", " ", regex=False)).str.strip()
    movies["tag_text"] = movies["genres"].fillna("").str.replace("|", " ", regex=False).str.strip()
    movies["metadata_count"] = movies["genres"].fillna("(no genres listed)").str.count(r"\|") + 1
    return interactions, movies[["item_id", "item_name", "item_text", "tag_text", "year_hint", "metadata_count"]]


def load_google_local(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path = download_file(DATASET_SPECS["google_local_sc"]["urls"]["ratings"], raw_root / "google_local_sc" / "rating-South_Carolina.csv.gz")
    meta_path = download_file(DATASET_SPECS["google_local_sc"]["urls"]["meta"], raw_root / "google_local_sc" / "meta-South_Carolina.json.gz")
    chunks = []
    for chunk in pd.read_csv(
        ratings_path,
        compression="gzip",
        usecols=["business", "user", "rating", "timestamp"],
        dtype={"business": "string", "user": "string", "rating": "int8", "timestamp": "int64"},
        chunksize=400_000,
    ):
        chunk = chunk[chunk["rating"] >= 4].copy()
        chunk["user_id"] = chunk["user"].astype(str)
        chunk = stable_user_sample(chunk, DATASET_SPECS["google_local_sc"]["user_sample_mod"], DATASET_SPECS["google_local_sc"]["user_sample_keep"])
        chunk["item_id"] = chunk["business"].astype(str)
        chunks.append(chunk[["user_id", "item_id", "rating", "timestamp"]])
    interactions = pd.concat(chunks, ignore_index=True)
    interactions["dataset"] = DATASET_SPECS["google_local_sc"]["name"]
    interactions["target"] = 1
    return interactions, parse_google_meta(meta_path)


def load_goodreads(raw_root: Path) -> tuple[pd.DataFrame, Path]:
    reviews_path = download_file(DATASET_SPECS["goodreads_fantasy"]["urls"]["reviews"], raw_root / "goodreads_fantasy" / "fantasy_10000.json")
    books_path = download_file(DATASET_SPECS["goodreads_fantasy"]["urls"]["books"], raw_root / "goodreads_fantasy" / "goodreads_books_fantasy_paranormal.json.gz")
    reviews = pd.read_json(reviews_path, lines=True)
    reviews = reviews[reviews["rating"] >= 4].copy()
    timestamps = pd.to_datetime(
        reviews["date_updated"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
        utc=True,
    )
    reviews["timestamp"] = timestamps.map(lambda value: int(value.timestamp()) if pd.notna(value) else np.nan)
    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews["item_id"] = reviews["book_id"].astype(str)
    interactions = reviews[["user_id", "item_id", "rating", "timestamp"]].copy()
    interactions["dataset"] = DATASET_SPECS["goodreads_fantasy"]["name"]
    interactions["target"] = 1
    return interactions, books_path


def build_category_group(category_text: str) -> str:
    value = (category_text or "").lower()
    if any(token in value for token in ["restaurant", "cafe", "coffee", "bar", "bakery", "pizza", "food"]):
        return "Dining"
    if any(token in value for token in ["hotel", "resort", "motel", "lodging"]):
        return "Travel"
    if any(token in value for token in ["store", "shop", "market", "mall", "boutique", "retail"]):
        return "Retail"
    if any(token in value for token in ["doctor", "dentist", "salon", "repair", "service", "clinic", "auto"]):
        return "Services"
    return "Other"


def prepare_items(interactions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    item_stats = interactions.groupby("item_id").agg(
        item_interactions=("item_id", "size"),
        item_avg_rating=("rating", "mean"),
        item_last_timestamp=("timestamp", "max"),
    ).reset_index()
    items = item_stats.merge(metadata, on="item_id", how="left")
    items["item_name"] = items["item_name"].fillna(items["item_id"])
    items["item_text"] = items["item_text"].fillna(items["item_name"])
    items["tag_text"] = items["tag_text"].fillna("")
    if "year_hint" not in items.columns:
        items["year_hint"] = np.nan
    if "price_bucket" not in items.columns:
        items["price_bucket"] = 0.0
    items["year_hint"] = pd.to_numeric(items["year_hint"], errors="coerce")
    items["price_bucket"] = pd.to_numeric(items["price_bucket"], errors="coerce").fillna(0.0)
    items["metadata_count"] = pd.to_numeric(items.get("metadata_count"), errors="coerce").fillna(0)
    items["text_length"] = items["item_text"].astype(str).str.len()
    items["item_quality"] = normalize_array(items["item_avg_rating"].to_numpy())
    items["item_popularity_norm"] = normalize_array(np.log1p(items["item_interactions"].to_numpy()))
    items["item_freshness"] = normalize_array(items["item_last_timestamp"].to_numpy())
    items["metadata_richness"] = normalize_array(items["metadata_count"].to_numpy() + np.log1p(items["text_length"].to_numpy()))
    items["year_norm"] = normalize_array(items["year_hint"].to_numpy())
    items["price_bucket_norm"] = normalize_array(items["price_bucket"].to_numpy())
    items["proxy_category"] = items["tag_text"].map(build_category_group)
    return items


def split_user_histories(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_base_parts, valid_parts, train_full_parts, test_parts = [], [], [], []
    for _, group in frame.sort_values(["user_id", "timestamp", "item_id"]).groupby("user_id", sort=False):
        size = len(group)
        if size < 5:
            continue
        test_size = max(1, int(math.ceil(size * 0.25)))
        train_full_size = size - test_size
        if train_full_size < 3:
            continue
        valid_size = max(1, int(math.ceil(train_full_size * 0.2)))
        train_base_size = train_full_size - valid_size
        if train_base_size < 2:
            continue
        train_base_parts.append(group.iloc[:train_base_size])
        valid_parts.append(group.iloc[train_base_size:train_full_size])
        train_full_parts.append(group.iloc[:train_full_size])
        test_parts.append(group.iloc[train_full_size:])
    train_base = pd.concat(train_base_parts, ignore_index=True)
    valid = pd.concat(valid_parts, ignore_index=True)
    train_full = pd.concat(train_full_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    valid = valid[valid["item_id"].isin(set(train_base["item_id"]))].copy()
    test = test[test["item_id"].isin(set(train_full["item_id"]))].copy()
    valid = valid[valid["user_id"].isin(set(train_base["user_id"]))].copy()
    test = test[test["user_id"].isin(set(train_full["user_id"]))].copy()
    return train_base, valid, train_full, test


def build_sparse_matrix(frame: pd.DataFrame, user_to_idx: dict[str, int], item_to_idx: dict[str, int]) -> sp.csr_matrix:
    rows = frame["user_id"].map(user_to_idx)
    cols = frame["item_id"].map(item_to_idx)
    values = np.ones(len(frame), dtype=np.float32)
    return sp.csr_matrix((values, (rows, cols)), shape=(len(user_to_idx), len(item_to_idx)))


def top_indices(scores: np.ndarray, k: int) -> list[int]:
    if k >= len(scores):
        order = np.argsort(scores)[::-1]
    else:
        candidate = np.argpartition(scores, -k)[-k:]
        order = candidate[np.argsort(scores[candidate])[::-1]]
    return [int(index) for index in order if np.isfinite(scores[index])]


def precision_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    return sum(1 for item in recs[:k] if item in truth) / k


def recall_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    return sum(1 for item in recs[:k] if item in truth) / len(truth)


def map_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(recs[:k], start=1):
        if item in truth:
            hits += 1
            total += hits / rank
    return total / min(len(truth), k)


def ndcg_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recs[:k], start=1):
        if item in truth:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(truth), k) + 1))
    return dcg / ideal if ideal else 0.0


def build_prepared_dataset(name: str, slug: str, interactions: pd.DataFrame, metadata: pd.DataFrame, spec: dict) -> PreparedDataset:
    interactions = stable_user_sample(interactions, spec["user_sample_mod"], spec["user_sample_keep"])
    interactions = interactions.sort_values(["user_id", "timestamp", "item_id"]).drop_duplicates(["user_id", "item_id"], keep="last")
    interactions = top_frequency_subset(interactions, spec["max_users"], spec["max_items"], spec["min_user"], spec["min_item"])
    train_base, valid, train_full, test = split_user_histories(interactions)
    interactions = interactions[interactions["user_id"].isin(set(train_full["user_id"])) & interactions["item_id"].isin(set(train_full["item_id"]))].copy()
    items = prepare_items(interactions, metadata[metadata["item_id"].isin(set(interactions["item_id"]))].copy())
    item_order = items.sort_values("item_interactions", ascending=False)["item_id"].astype(str).tolist()
    user_order = sorted(train_full["user_id"].astype(str).unique().tolist())
    user_to_idx = {user_id: index for index, user_id in enumerate(user_order)}
    item_to_idx = {item_id: index for index, item_id in enumerate(item_order)}
    items = items.set_index("item_id").loc[item_order].reset_index()
    train_base_matrix = build_sparse_matrix(train_base, user_to_idx, item_to_idx)
    train_full_matrix = build_sparse_matrix(train_full, user_to_idx, item_to_idx)
    token_sets = [set(re.findall(r"[a-z0-9]+", value.lower())) for value in items["tag_text"].fillna("").astype(str)]
    tag_map = dict(zip(items["item_id"], token_sets))
    user_tag_profiles = {}
    user_top_categories: dict[int, set[str]] = {}
    user_preferred_year = np.full(len(user_order), np.nan, dtype=float)
    user_preferred_price = np.full(len(user_order), np.nan, dtype=float)
    user_preferred_richness = np.zeros(len(user_order), dtype=float)
    # These user-level preference summaries are reused by the hybrid reranker so that
    # explanations and profile-alignment features come from the same source of truth.
    interaction_features = train_full.merge(
        items[["item_id", "proxy_category", "year_norm", "price_bucket_norm", "metadata_richness"]],
        on="item_id",
        how="left",
    )
    for user_id, group in train_full.groupby("user_id"):
        counts = Counter()
        for item_id in group["item_id"]:
            counts.update(tag_map.get(item_id, set()))
        user_idx = user_to_idx[user_id]
        user_tag_profiles[user_idx] = {token for token, _ in counts.most_common(15)}
    for user_id, group in interaction_features.groupby("user_id"):
        user_idx = user_to_idx[user_id]
        categories = group["proxy_category"].dropna().astype(str)
        user_top_categories[user_idx] = set(categories.value_counts().head(2).index.tolist())
        year_values = pd.to_numeric(group["year_norm"], errors="coerce").dropna()
        if not year_values.empty:
            user_preferred_year[user_idx] = float(year_values.median())
        price_values = pd.to_numeric(group["price_bucket_norm"], errors="coerce").dropna()
        if not price_values.empty:
            user_preferred_price[user_idx] = float(price_values.median())
        user_preferred_richness[user_idx] = float(pd.to_numeric(group["metadata_richness"], errors="coerce").fillna(0.0).mean())
    user_activity = train_full.groupby("user_id").size().reindex(user_order).fillna(0).to_numpy()
    eval_users = test.groupby("user_id").size().sort_values(ascending=False).index.tolist()
    eval_user_indices = [user_to_idx[user_id] for user_id in eval_users[: min(len(eval_users), 1500)]]
    return PreparedDataset(
        name=name,
        slug=slug,
        interactions=interactions,
        items=items,
        train_base=train_base,
        valid=valid,
        train_full=train_full,
        test=test,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_user=user_order,
        idx_to_item=item_order,
        train_base_matrix=train_base_matrix,
        train_full_matrix=train_full_matrix,
        item_features=items,
        item_tokens=token_sets,
        user_tag_profiles=user_tag_profiles,
        user_top_categories=user_top_categories,
        user_preferred_year=user_preferred_year,
        user_preferred_price=user_preferred_price,
        user_preferred_richness=user_preferred_richness,
        user_activity_norm=normalize_array(user_activity),
        eval_user_indices=eval_user_indices,
    )


class PopularityModel:
    name = "Popularity"

    def fit(self, data: PreparedDataset) -> None:
        self.scores = np.asarray(data.train_full_matrix.sum(axis=0)).ravel().astype(float)

    def score_user(self, user_idx: int) -> np.ndarray:
        return self.scores.copy()


class TruncatedSVDModel:
    name = "TruncatedSVD"

    def fit(self, data: PreparedDataset) -> None:
        n_components = max(8, min(64, data.train_full_matrix.shape[1] - 1))
        self.model = TruncatedSVD(n_components=n_components, random_state=SEED)
        self.user_factors = self.model.fit_transform(data.train_full_matrix)
        self.item_factors = self.model.components_.T

    def score_user(self, user_idx: int) -> np.ndarray:
        return self.user_factors[user_idx] @ self.item_factors.T


class EASEModel:
    name = "EASE"

    def fit(self, data: PreparedDataset) -> None:
        binary = (data.train_full_matrix > 0).astype(np.float32).tocsr()
        gram = (binary.T @ binary).toarray().astype(np.float64)
        diagonal = np.arange(gram.shape[0])
        gram[diagonal, diagonal] += 300.0
        inverse = np.linalg.inv(gram)
        coefficients = -inverse / np.diag(inverse)
        coefficients[diagonal, diagonal] = 0.0
        self.binary = binary
        self.coefficients = coefficients.astype(np.float32)

    def score_user(self, user_idx: int) -> np.ndarray:
        return self.binary[user_idx].toarray().ravel().astype(np.float32) @ self.coefficients


class TfidfContentModel:
    name = "TFIDF_Content"

    def fit(self, data: PreparedDataset) -> None:
        texts = data.item_features["item_text"].fillna(data.item_features["item_name"]).astype(str)
        self.vectorizer = TfidfVectorizer(max_features=20000, min_df=1, ngram_range=(1, 2))
        item_matrix = self.vectorizer.fit_transform(texts)
        self.item_matrix = normalize(item_matrix)
        self.user_profiles = normalize(data.train_full_matrix @ self.item_matrix)

    def score_user(self, user_idx: int) -> np.ndarray:
        return (self.user_profiles[user_idx] @ self.item_matrix.T).toarray().ravel()


class SequentialTransitionModel:
    name = "SequentialTransition"

    def __init__(self, popularity_blend: float = 0.05, recent_window: int = 5, max_lag: int = 3) -> None:
        self.popularity_blend = float(popularity_blend)
        self.recent_window = int(recent_window)
        self.max_lag = int(max_lag)

    def fit(self, data: PreparedDataset) -> None:
        n_items = len(data.idx_to_item)
        self.popularity = normalize_array(np.asarray(data.train_full_matrix.sum(axis=0)).ravel().astype(float))
        self.transitions = np.zeros((n_items, n_items), dtype=np.float32)
        self.user_recent_items: dict[int, list[int]] = {}
        ordered = data.train_full.sort_values(["user_id", "timestamp", "item_id"])
        for user_id, group in ordered.groupby("user_id", sort=False):
            item_indices = [data.item_to_idx[item_id] for item_id in group["item_id"].astype(str).tolist()]
            user_idx = data.user_to_idx[user_id]
            self.user_recent_items[user_idx] = item_indices[-self.recent_window :]
            # Weighted item-to-item transitions give us an order-aware baseline without
            # introducing a heavy deep-learning stack into the main benchmark.
            for position in range(1, len(item_indices)):
                current_item = item_indices[position]
                max_back = min(self.max_lag, position)
                for lag in range(1, max_back + 1):
                    previous_item = item_indices[position - lag]
                    self.transitions[previous_item, current_item] += 1.0 / lag
        row_sums = self.transitions.sum(axis=1, keepdims=True)
        non_zero = row_sums[:, 0] > 0
        self.transitions[non_zero] = self.transitions[non_zero] / row_sums[non_zero]

    def score_user(self, user_idx: int) -> np.ndarray:
        score = self.popularity_blend * self.popularity.copy()
        recent_items = self.user_recent_items.get(user_idx, [])
        if not recent_items:
            return score
        recency_weights = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12], dtype=np.float32)
        for rank, item_idx in enumerate(reversed(recent_items)):
            score += recency_weights[min(rank, len(recency_weights) - 1)] * self.transitions[item_idx]
        return score


def build_truth_map(frame: pd.DataFrame, user_to_idx: dict[str, int], item_to_idx: dict[str, int]) -> dict[int, set[int]]:
    truth = defaultdict(set)
    for row in frame.itertuples(index=False):
        truth[user_to_idx[row.user_id]].add(item_to_idx[row.item_id])
    return truth


def build_seen_map(frame: pd.DataFrame, user_to_idx: dict[str, int], item_to_idx: dict[str, int]) -> dict[int, set[int]]:
    seen = defaultdict(set)
    for row in frame.itertuples(index=False):
        seen[user_to_idx[row.user_id]].add(item_to_idx[row.item_id])
    return seen


def build_candidate_features(
    data: PreparedDataset,
    score_map: dict[str, np.ndarray],
    candidate_list: list[int],
    user_idx: int,
) -> dict[str, np.ndarray]:
    item_frame = data.item_features.iloc[candidate_list]
    preferred_categories = data.user_top_categories.get(user_idx, set())
    category_values = item_frame["proxy_category"].fillna("").astype(str).tolist()
    return {
        "ease": normalize_array(score_map["EASE"][candidate_list]),
        "svd": normalize_array(score_map["TruncatedSVD"][candidate_list]),
        "content": normalize_array(score_map["TFIDF_Content"][candidate_list]),
        "popularity": item_frame["item_popularity_norm"].to_numpy(dtype=float),
        "novelty": 1.0 - item_frame["item_popularity_norm"].to_numpy(dtype=float),
        "quality": item_frame["item_quality"].to_numpy(dtype=float),
        "freshness": item_frame["item_freshness"].to_numpy(dtype=float),
        "tag_overlap": np.array(
            [
                len(data.item_tokens[item_idx] & data.user_tag_profiles.get(user_idx, set())) / max(1, len(data.item_tokens[item_idx]))
                for item_idx in candidate_list
            ],
            dtype=float,
        ),
        "category_match": np.array(
            [1.0 if category and category in preferred_categories else 0.0 for category in category_values],
            dtype=float,
        ),
        "year_alignment": safe_alignment(item_frame["year_norm"].to_numpy(dtype=float), data.user_preferred_year[user_idx]),
        "price_match": safe_alignment(item_frame["price_bucket_norm"].to_numpy(dtype=float), data.user_preferred_price[user_idx]),
        "richness_alignment": safe_alignment(item_frame["metadata_richness"].to_numpy(dtype=float), data.user_preferred_richness[user_idx]),
    }


def combine_feature_scores(features: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = np.zeros(len(next(iter(features.values()))), dtype=float)
    for key, values in features.items():
        total += weights.get(key, 0.0) * values
    return total


def ablated_weights(weights: dict[str, float], dropped_features: set[str]) -> dict[str, float]:
    filtered = {feature: value for feature, value in weights.items() if feature not in dropped_features}
    total = sum(filtered.values())
    if total <= 0:
        return {feature: 0.0 for feature in weights}
    normalized = {feature: value / total for feature, value in filtered.items()}
    return {feature: normalized.get(feature, 0.0) for feature in weights}


def matched_tag_string(data: PreparedDataset, user_idx: int, item_idx: int) -> str:
    overlap = sorted(data.item_tokens[item_idx] & data.user_tag_profiles.get(user_idx, set()))
    return ", ".join(overlap[:3])


def explain_hybrid_recommendation(
    data: PreparedDataset,
    user_idx: int,
    item_idx: int,
    item_row: pd.Series,
    feature_values: dict[str, float],
    weights: dict[str, float],
) -> tuple[str, str]:
    contributions = {
        feature: weights.get(feature, 0.0) * value
        for feature, value in feature_values.items()
        if weights.get(feature, 0.0) > 0 and value > 0
    }
    ordered = [feature for feature, _ in sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)]
    main_driver = ordered[0] if ordered else "ease"
    phrases: list[str] = []
    for feature in ordered[:3]:
        if feature == "tag_overlap":
            matched_tags = matched_tag_string(data, user_idx, item_idx)
            phrases.append(f"shares profile tags ({matched_tags})" if matched_tags else FEATURE_LABELS[feature])
        elif feature == "category_match" and item_row.get("proxy_category"):
            phrases.append(f"matches preferred category {item_row['proxy_category']}")
        elif feature == "year_alignment" and pd.notna(item_row.get("year_hint")):
            phrases.append(f"fits preferred era around {int(item_row['year_hint'])}")
        elif feature == "price_match" and pd.notna(item_row.get("price_bucket")) and int(item_row["price_bucket"]) > 0:
            phrases.append(f"fits preferred price tier {int(item_row['price_bucket'])}")
        else:
            phrases.append(FEATURE_LABELS.get(feature, feature))
    if not phrases:
        phrases = [ALGO_REASONS["HybridFeatureRerank"]]
    return main_driver, "; ".join(phrases)


def mean_confidence_interval(values: np.ndarray, seed: int = SEED, n_bootstrap: int = 1200) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    boot_means = values[sample_idx].mean(axis=1)
    return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def paired_delta_interval(reference: np.ndarray, comparison: np.ndarray, seed: int = SEED, n_bootstrap: int = 1200) -> tuple[float, float]:
    diff = np.asarray(reference, dtype=float) - np.asarray(comparison, dtype=float)
    return mean_confidence_interval(diff, seed=seed, n_bootstrap=n_bootstrap)


def wilcoxon_p_value(reference: np.ndarray, comparison: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(comparison)
    reference = reference[mask]
    comparison = comparison[mask]
    if reference.size < 5:
        return np.nan
    if np.allclose(reference, comparison):
        return 1.0
    try:
        return float(stats.wilcoxon(reference, comparison, zero_method="wilcox", correction=False, alternative="two-sided", method="auto").pvalue)
    except ValueError:
        return 1.0


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.dropna().sort_values()
    if valid.empty:
        return result
    n = len(valid)
    adjusted = pd.Series(index=valid.index, dtype=float)
    running = 1.0
    for rank, (index, value) in enumerate(reversed(list(valid.items())), start=1):
        denom = n - rank + 1
        candidate = min(running, float(value) * n / denom)
        adjusted.loc[index] = candidate
        running = candidate
    result.loc[adjusted.index] = adjusted
    return result


def build_statistical_artifacts(user_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if user_metrics.empty:
        return (
            pd.DataFrame(columns=["dataset", "algorithm", "metric", "mean", "ci_low", "ci_high", "n_users"]),
            pd.DataFrame(
                columns=[
                    "dataset",
                    "metric",
                    "reference_algorithm",
                    "comparison_algorithm",
                    "reference_mean",
                    "comparison_mean",
                    "delta_mean",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "p_value_adj",
                    "significant",
                    "n_users",
                ]
            ),
        )

    interval_rows: list[dict[str, object]] = []
    significance_rows: list[dict[str, object]] = []
    for dataset_name, dataset_slice in user_metrics.groupby("dataset"):
        for algorithm_name, algo_slice in dataset_slice.groupby("algorithm"):
            for metric in SIGNIFICANCE_METRICS:
                values = pd.to_numeric(algo_slice[metric], errors="coerce").dropna().to_numpy()
                if values.size == 0:
                    continue
                ci_low, ci_high = mean_confidence_interval(values, seed=SEED + len(interval_rows))
                interval_rows.append(
                    {
                        "dataset": dataset_name,
                        "algorithm": algorithm_name,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_users": int(values.size),
                    }
                )

        for metric in SIGNIFICANCE_METRICS:
            metric_slice = dataset_slice[["user_id", "algorithm", metric]].copy()
            # Pairwise tests operate on the same users so that significance reflects
            # algorithm differences rather than user mix differences across samples.
            for reference_algorithm, comparison_algorithm in SIGNIFICANCE_COMPARISONS:
                pair_slice = metric_slice[metric_slice["algorithm"].isin([reference_algorithm, comparison_algorithm])]
                if pair_slice.empty:
                    continue
                pivot = pair_slice.pivot(index="user_id", columns="algorithm", values=metric).dropna()
                if not {reference_algorithm, comparison_algorithm}.issubset(pivot.columns):
                    continue
                if len(pivot) < 10:
                    continue
                reference_values = pivot[reference_algorithm].to_numpy(dtype=float)
                comparison_values = pivot[comparison_algorithm].to_numpy(dtype=float)
                ci_low, ci_high = paired_delta_interval(reference_values, comparison_values, seed=SEED + len(significance_rows))
                significance_rows.append(
                    {
                        "dataset": dataset_name,
                        "metric": metric,
                        "reference_algorithm": reference_algorithm,
                        "comparison_algorithm": comparison_algorithm,
                        "reference_mean": float(np.mean(reference_values)),
                        "comparison_mean": float(np.mean(comparison_values)),
                        "delta_mean": float(np.mean(reference_values - comparison_values)),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": wilcoxon_p_value(reference_values, comparison_values),
                        "n_users": int(len(pivot)),
                    }
                )

    metric_intervals = pd.DataFrame(interval_rows)
    significance_tests = pd.DataFrame(significance_rows)
    if not significance_tests.empty:
        significance_tests["p_value_adj"] = significance_tests.groupby(["dataset", "metric"])["p_value"].transform(benjamini_hochberg)
        significance_tests["significant"] = (
            significance_tests["p_value_adj"].fillna(1.0).lt(0.05)
            & (significance_tests["ci_low"] * significance_tests["ci_high"] > 0)
        )
    return metric_intervals, significance_tests


def tune_hybrid_weights(
    data: PreparedDataset,
    base_models: dict[str, object],
    seen_map: dict[int, set[int]],
    valid_truth: dict[int, set[int]],
) -> dict[str, float]:
    sampled_users = [user_idx for user_idx in data.eval_user_indices if user_idx in valid_truth][:300]
    feature_cache = {}
    for user_idx in sampled_users:
        score_map = {name: model.score_user(user_idx) for name, model in base_models.items()}
        candidates = set()
        for scores in score_map.values():
            masked = scores.copy()
            masked[list(seen_map[user_idx])] = -np.inf
            candidates.update(top_indices(masked, 60))
        candidate_list = sorted(candidates)
        feature_cache[user_idx] = {"candidates": candidate_list, "features": build_candidate_features(data, score_map, candidate_list, user_idx)}
    best_score = -1.0
    best_weights = HYBRID_TEMPLATES[0]
    for weights in HYBRID_TEMPLATES:
        scores = []
        for user_idx, payload in feature_cache.items():
            final_score = combine_feature_scores(payload["features"], weights)
            ranked = [payload["candidates"][index] for index in np.argsort(final_score)[::-1][:10]]
            scores.append(ndcg_at_k(ranked, valid_truth[user_idx], 10))
        mean_score = float(np.mean(scores)) if scores else 0.0
        if mean_score > best_score:
            best_score = mean_score
            best_weights = weights
    return best_weights


def tune_sequential_model(data: PreparedDataset) -> SequentialTransitionModel:
    if data.valid.empty:
        model = SequentialTransitionModel()
        model.fit(data)
        return model
    valid_base = PreparedDataset(**{**data.__dict__, "train_full": data.train_base, "train_full_matrix": data.train_base_matrix})
    valid_truth = build_truth_map(data.valid, data.user_to_idx, data.item_to_idx)
    seen_base = build_seen_map(data.train_base, data.user_to_idx, data.item_to_idx)
    sampled_users = [user_idx for user_idx in data.eval_user_indices if user_idx in valid_truth][:300]
    best_score = -1.0
    best_model = SequentialTransitionModel()
    for params in SEQUENTIAL_PARAM_GRID:
        candidate = SequentialTransitionModel(**params)
        candidate.fit(valid_base)
        scores = []
        for user_idx in sampled_users:
            user_scores = candidate.score_user(user_idx)
            masked = user_scores.copy()
            masked[list(seen_base[user_idx])] = -np.inf
            ranked = top_indices(masked, 10)
            scores.append(ndcg_at_k(ranked, valid_truth[user_idx], 10))
        mean_score = float(np.mean(scores)) if scores else 0.0
        if mean_score > best_score:
            best_score = mean_score
            best_model = SequentialTransitionModel(**params)
    best_model.fit(data)
    return best_model


def recommend_hybrid(
    data: PreparedDataset,
    base_models: dict[str, object],
    weights: dict[str, float],
    seen_map: dict[int, set[int]],
    user_idx: int,
    exclude_seen: bool,
) -> tuple[list[int], dict[int, dict[str, str]]]:
    score_map = {name: model.score_user(user_idx) for name, model in base_models.items()}
    candidates = set()
    for scores in score_map.values():
        masked = scores.copy()
        if exclude_seen:
            masked[list(seen_map[user_idx])] = -np.inf
        candidates.update(top_indices(masked, 80))
    candidate_list = sorted(candidates)
    candidate_features = build_candidate_features(data, score_map, candidate_list, user_idx)
    final_score = combine_feature_scores(candidate_features, weights)
    ranking = np.argsort(final_score)[::-1][:MAX_RECOMMEND]
    explanations: dict[int, dict[str, str]] = {}
    item_frame = data.item_features.iloc[candidate_list].reset_index(drop=True)
    for rank_index in ranking:
        item_idx = candidate_list[rank_index]
        feature_values = {feature: float(values[rank_index]) for feature, values in candidate_features.items()}
        main_driver, explanation = explain_hybrid_recommendation(
            data,
            user_idx,
            item_idx,
            item_frame.iloc[rank_index],
            feature_values,
            weights,
        )
        explanations[item_idx] = {
            "main_driver": main_driver,
            "explanation": explanation,
            "matched_tags": matched_tag_string(data, user_idx, item_idx),
        }
    return [candidate_list[index] for index in ranking], explanations


def evaluate_dataset(
    data: PreparedDataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    algorithms = {
        "Popularity": PopularityModel(),
        "TruncatedSVD": TruncatedSVDModel(),
        "EASE": EASEModel(),
        "TFIDF_Content": TfidfContentModel(),
        "SequentialTransition": tune_sequential_model(data),
    }
    train_time = {}
    for name, model in algorithms.items():
        started = time.perf_counter()
        model.fit(data)
        train_time[name] = time.perf_counter() - started
    valid_base = PreparedDataset(**{**data.__dict__, "train_full": data.train_base, "train_full_matrix": data.train_base_matrix})
    valid_models = {
        "Popularity": PopularityModel(),
        "TruncatedSVD": TruncatedSVDModel(),
        "EASE": EASEModel(),
        "TFIDF_Content": TfidfContentModel(),
    }
    for model in valid_models.values():
        model.fit(valid_base)
    valid_truth = build_truth_map(data.valid, data.user_to_idx, data.item_to_idx)
    seen_base = build_seen_map(data.train_base, data.user_to_idx, data.item_to_idx)
    hybrid_weights = tune_hybrid_weights(data, valid_models, seen_base, valid_truth)
    hybrid_no_content_weights = ablated_weights(hybrid_weights, CONTENT_FEATURES)
    hybrid_no_collab_weights = ablated_weights(hybrid_weights, COLLABORATIVE_FEATURES)
    test_truth = build_truth_map(data.test, data.user_to_idx, data.item_to_idx)
    seen_map = build_seen_map(data.train_full, data.user_to_idx, data.item_to_idx)
    item_popularity = data.item_features.set_index("item_id")["item_popularity_norm"].reindex(data.idx_to_item).fillna(0).to_numpy()
    results_rows, slice_rows, eff_rows, rec_rows, user_metric_rows = [], [], [], [], []
    sample_users = [user_idx for user_idx in data.eval_user_indices if user_idx in test_truth]
    history_user_idx = sample_users[0]
    recommendation_bank: dict[tuple[str, str], dict[int, list[int]]] = {}
    explanation_bank: dict[tuple[str, str], dict[int, dict[int, dict[str, str]]]] = {}
    hybrid_variants = {
        "HybridFeatureRerank": hybrid_weights,
        "Hybrid_NoContentAblation": hybrid_no_content_weights,
        "Hybrid_NoCollaborativeAblation": hybrid_no_collab_weights,
    }
    # Core benchmark, ablation variants and the lightweight sequential baseline all go
    # through the same evaluation loop so their metrics remain directly comparable.
    for algo_name in list(algorithms) + list(hybrid_variants):
        per_mode_recs = {"exc": {}, "inc": {}}
        per_mode_explanations = {"exc": {}, "inc": {}}
        started = time.perf_counter()
        for user_idx in sample_users:
            if algo_name in hybrid_variants:
                variant_weights = hybrid_variants[algo_name]
                per_mode_recs["exc"][user_idx], per_mode_explanations["exc"][user_idx] = recommend_hybrid(data, algorithms, variant_weights, seen_map, user_idx, True)
                per_mode_recs["inc"][user_idx], per_mode_explanations["inc"][user_idx] = recommend_hybrid(data, algorithms, variant_weights, seen_map, user_idx, False)
            else:
                scores = algorithms[algo_name].score_user(user_idx)
                per_mode_recs["inc"][user_idx] = top_indices(scores.copy(), MAX_RECOMMEND)
                masked = scores.copy()
                masked[list(seen_map[user_idx])] = -np.inf
                per_mode_recs["exc"][user_idx] = top_indices(masked, MAX_RECOMMEND)
        recommendation_bank[(algo_name, "exc")] = per_mode_recs["exc"]
        explanation_bank[(algo_name, "exc")] = per_mode_explanations["exc"]
        inference_ms = (time.perf_counter() - started) * 1000 / max(1, len(sample_users))
        eff_rows.append(
            {
                "dataset": data.name,
                "algorithm": algo_name,
                "train_seconds": round(float(train_time.get(algo_name, 0.0)), 3),
                "inference_ms_per_user": round(float(inference_ms), 3),
                "memory_mb": round(float(data.train_full_matrix.nnz * 8 / (1024 * 1024)), 2),
            }
        )
        for mode, recs_by_user in per_mode_recs.items():
            for topk in TOPK_GRID:
                ndcgs, recalls, precisions, maps, coverage_items, novelty_scores = [], [], [], [], set(), []
                for user_idx, recs in recs_by_user.items():
                    truth = test_truth[user_idx]
                    subset = recs[:topk]
                    ndcgs.append(ndcg_at_k(subset, truth, topk))
                    recalls.append(recall_at_k(subset, truth, topk))
                    precisions.append(precision_at_k(subset, truth, topk))
                    maps.append(map_at_k(subset, truth, topk))
                    coverage_items.update(subset)
                    novelty_scores.extend(1.0 - item_popularity[item_idx] for item_idx in subset)
                    if topk == 10 and mode == "exc":
                        user_metric_rows.append(
                            {
                                "dataset": data.name,
                                "algorithm": algo_name,
                                "user_idx": user_idx,
                                "user_id": data.idx_to_user[user_idx],
                                "ndcg": ndcgs[-1],
                                "recall": recalls[-1],
                                "precision": precisions[-1],
                                "map": maps[-1],
                                "activity": len(seen_map[user_idx]),
                            }
                        )
                results_rows.append(
                    {
                        "dataset": data.name,
                        "algorithm": algo_name,
                        "topk": topk,
                        "mode": mode,
                        "ndcg": round(float(np.mean(ndcgs)), 4),
                        "recall": round(float(np.mean(recalls)), 4),
                        "precision": round(float(np.mean(precisions)), 4),
                        "map": round(float(np.mean(maps)), 4),
                        "coverage": round(len(coverage_items) / max(1, len(data.idx_to_item)), 4),
                        "novelty": round(float(np.mean(novelty_scores)) if novelty_scores else 0.0, 4),
                        "train_seconds": round(float(train_time.get(algo_name, 0.0)), 3),
                        "inference_ms_per_user": round(float(inference_ms), 3),
                        "run_status": (
                            "ablation"
                            if algo_name in {"Hybrid_NoContentAblation", "Hybrid_NoCollaborativeAblation"}
                            else "done"
                        ),
                    }
                )
    user_metrics = pd.DataFrame(user_metric_rows)
    hybrid_metrics = user_metrics[user_metrics["algorithm"] == "HybridFeatureRerank"].sort_values(["ndcg", "recall"], ascending=False)
    if not hybrid_metrics.empty:
        history_user_idx = int(hybrid_metrics.iloc[0]["user_idx"])
    cold_threshold = user_metrics["activity"].quantile(0.33) if not user_metrics.empty else 0
    hot_threshold = user_metrics["activity"].quantile(0.67) if not user_metrics.empty else 0
    for algo_name, group in user_metrics.groupby("algorithm"):
        for slice_name, mask in {
            "Cold users": group["activity"] <= cold_threshold,
            "Frequent users": group["activity"] >= hot_threshold,
        }.items():
            subset = group[mask]
            if subset.empty:
                continue
            slice_rows.append(
                {
                    "dataset": data.name,
                    "algorithm": algo_name,
                    "slice_name": slice_name,
                    "topk": 10,
                    "mode": "exc",
                    "ndcg": round(float(subset["ndcg"].mean()), 4),
                    "recall": round(float(subset["recall"].mean()), 4),
                }
            )
    history = data.train_full[data.train_full["user_id"] == data.idx_to_user[history_user_idx]].sort_values("timestamp").tail(8)
    for row in history.itertuples(index=False):
        rec_rows.append(
            {
                "dataset": data.name,
                "user_id": row.user_id,
                "algorithm": "history",
                "rank": len(rec_rows) + 1,
                "item_id": row.item_id,
                "item_name": data.item_features.set_index("item_id").at[row.item_id, "item_name"],
                "score": np.nan,
                "seen_before": True,
                "reason": "observed interaction",
                "main_driver": "history",
                "explanation": "observed interaction from the user history",
                "matched_tags": matched_tag_string(data, history_user_idx, data.item_to_idx[row.item_id]),
                "proxy_category": data.item_features.set_index("item_id").at[row.item_id, "proxy_category"],
                "kind": "history",
            }
        )
    for algo_name in list(algorithms) + list(hybrid_variants):
        for rank, item_idx in enumerate(recommendation_bank[(algo_name, "exc")][history_user_idx][:10], start=1):
            item_id = data.idx_to_item[item_idx]
            item_row = data.item_features.set_index("item_id").loc[item_id]
            explanation_payload = explanation_bank.get((algo_name, "exc"), {}).get(history_user_idx, {}).get(
                item_idx,
                {
                    "main_driver": algo_name,
                    "explanation": ALGO_REASONS[algo_name],
                    "matched_tags": matched_tag_string(data, history_user_idx, item_idx),
                },
            )
            rec_rows.append(
                {
                    "dataset": data.name,
                    "user_id": data.idx_to_user[history_user_idx],
                    "algorithm": algo_name,
                    "rank": rank,
                    "item_id": item_id,
                    "item_name": data.item_features.set_index("item_id").at[item_id, "item_name"],
                    "score": float(MAX_RECOMMEND - rank + 1),
                    "seen_before": False,
                    "reason": ALGO_REASONS[algo_name],
                    "main_driver": explanation_payload["main_driver"],
                    "explanation": explanation_payload["explanation"],
                    "matched_tags": explanation_payload["matched_tags"],
                    "proxy_category": item_row.get("proxy_category", ""),
                    "kind": "recommendation",
                }
            )
    importance_rows = []
    for algo_name, weights in hybrid_variants.items():
        importance_rows.extend(
            [{"dataset": data.name, "algorithm": algo_name, "feature": key, "importance": value} for key, value in weights.items()]
        )
    feature_importance = pd.DataFrame(importance_rows)
    return pd.DataFrame(results_rows), pd.DataFrame(slice_rows), pd.DataFrame(eff_rows), pd.DataFrame(rec_rows), feature_importance, user_metrics


def build_supporting_artifacts(data: PreparedDataset, rec_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = data.interactions.copy()
    dt = timestamps_to_datetime(frame["timestamp"])
    min_dt = dt.min()
    max_dt = dt.max()
    datasets_summary = pd.DataFrame(
        [
            {
                "dataset": data.name,
                "users": frame["user_id"].nunique(),
                "items": frame["item_id"].nunique(),
                "events": len(frame),
                "sparsity_pct": round(100 * (1 - len(frame) / (frame["user_id"].nunique() * frame["item_id"].nunique())), 2),
                "mean_events_per_user": round(len(frame) / frame["user_id"].nunique(), 2),
                "mean_events_per_item": round(len(frame) / frame["item_id"].nunique(), 2),
                "time_start": f"{int(min_dt.year):04d}-{int(min_dt.month):02d}",
                "time_end": f"{int(max_dt.year):04d}-{int(max_dt.month):02d}",
            }
        ]
    )
    monthly = frame.assign(month=dt.dt.tz_localize(None).dt.to_period("M").astype(str)).groupby("month").agg(
        interactions=("item_id", "size"),
        active_users=("user_id", "nunique"),
        active_items=("item_id", "nunique"),
    ).reset_index()
    monthly["dataset"] = data.name
    activity_rows = []
    for entity, column in [("user", "user_id"), ("item", "item_id")]:
        counts = frame.groupby(column).size()
        buckets = counts.map(bucketize_activity)
        for bucket, count in buckets.value_counts().sort_index().items():
            activity_rows.append({"dataset": data.name, "entity_type": entity, "bucket": str(bucket), "count": int(count)})
    rating_distribution = frame.groupby("rating").size().reset_index(name="count")
    rating_distribution["dataset"] = data.name
    popularity = data.train_full["item_id"].value_counts().sort_values(ascending=False)
    cumulative = popularity.cumsum() / popularity.sum() * 100
    checkpoints = np.array([1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    curve = []
    for checkpoint in checkpoints:
        index = min(len(cumulative) - 1, max(0, math.ceil(len(popularity) * checkpoint / 100) - 1))
        curve.append({"dataset": data.name, "item_share_pct": checkpoint, "interaction_share_pct": round(float(cumulative.iloc[index]), 2)})
    history_rows = rec_rows[rec_rows["kind"] == "history"][["dataset", "user_id", "rank", "item_id", "item_name"]].rename(columns={"rank": "interaction_rank"})
    history_rows["interaction_type"] = "history"
    recommendations = rec_rows[
        rec_rows["kind"] == "recommendation"
    ][
        [
            "dataset",
            "user_id",
            "algorithm",
            "rank",
            "item_id",
            "item_name",
            "score",
            "seen_before",
            "reason",
            "main_driver",
            "explanation",
            "matched_tags",
            "proxy_category",
        ]
    ]
    return {
        "datasets_summary": datasets_summary,
        "monthly_interactions": monthly[["dataset", "month", "interactions", "active_users", "active_items"]],
        "activity_distribution": pd.DataFrame(activity_rows),
        "rating_distribution": rating_distribution[["dataset", "rating", "count"]],
        "long_tail_curve": pd.DataFrame(curve),
        "recommendations": recommendations,
        "user_history": history_rows[["dataset", "user_id", "interaction_rank", "item_id", "item_name", "interaction_type"]],
    }


def build_global_artifacts(
    prepared: list[PreparedDataset],
    results: pd.DataFrame,
    slice_metrics: pd.DataFrame,
    efficiency: pd.DataFrame,
    feature_importance: pd.DataFrame,
    user_metrics: pd.DataFrame,
    support: list[dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    pending_rows = []
    dataset_names = [dataset.name for dataset in prepared]
    for algorithm, dataset_statuses in PLANNED_BENCHMARK_STATUS.items():
        for dataset_name in dataset_names:
            status = dataset_statuses.get(dataset_name, "planned")
            pending_rows.append(
                {
                    "dataset": dataset_name,
                    "algorithm": algorithm,
                    "topk": 10,
                    "mode": "exc",
                    "ndcg": np.nan,
                    "recall": np.nan,
                    "precision": np.nan,
                    "map": np.nan,
                    "coverage": np.nan,
                    "novelty": np.nan,
                    "train_seconds": np.nan,
                    "inference_ms_per_user": np.nan,
                    "run_status": status,
                    "note": "phase-2 backlog" if status == "planned" else "outside phase-1 scope",
                }
            )
    # Planned rows keep the dashboard honest about scope without polluting the visible
    # leaderboard because views filter to completed runs by run_status.
    results = pd.concat([results, pd.DataFrame(pending_rows)], ignore_index=True)
    metric_intervals, significance_tests = build_statistical_artifacts(user_metrics)

    frames = {
        "datasets_summary": pd.concat([payload["datasets_summary"] for payload in support], ignore_index=True),
        "results": results,
        "monthly_interactions": pd.concat([payload["monthly_interactions"] for payload in support], ignore_index=True),
        "activity_distribution": pd.concat([payload["activity_distribution"] for payload in support], ignore_index=True),
        "rating_distribution": pd.concat([payload["rating_distribution"] for payload in support], ignore_index=True),
        "long_tail_curve": pd.concat([payload["long_tail_curve"] for payload in support], ignore_index=True),
        "recommendations": pd.concat([payload["recommendations"] for payload in support], ignore_index=True),
        "user_history": pd.concat([payload["user_history"] for payload in support], ignore_index=True),
        "slice_metrics": slice_metrics,
        "efficiency": efficiency,
        "metric_intervals": metric_intervals,
        "significance_tests": significance_tests,
        "feature_importance": (
            feature_importance.groupby(["dataset", "algorithm", "feature"], as_index=False)["importance"].mean()
            if "dataset" in feature_importance.columns
            else feature_importance.groupby(["algorithm", "feature"], as_index=False)["importance"].mean()
        ),
    }
    cannibalization_rows = []
    for (dataset, algorithm, topk), group in results.groupby(["dataset", "algorithm", "topk"]):
        inc = group[group["mode"] == "inc"]
        exc = group[group["mode"] == "exc"]
        if inc.empty or exc.empty:
            continue
        inc_row, exc_row = inc.iloc[0], exc.iloc[0]
        cannibalization_rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "topk": topk,
                "ndcg_exc": exc_row["ndcg"],
                "ndcg_inc": inc_row["ndcg"],
                "recall_exc": exc_row["recall"],
                "recall_inc": inc_row["recall"],
                "precision_exc": exc_row["precision"],
                "precision_inc": inc_row["precision"],
                "map_exc": exc_row["map"],
                "map_inc": inc_row["map"],
                "ndcg_gap": round(float(inc_row["ndcg"] - exc_row["ndcg"]), 4),
                "recall_gap": round(float(inc_row["recall"] - exc_row["recall"]), 4),
                "precision_gap": round(float(inc_row["precision"] - exc_row["precision"]), 4),
                "map_gap": round(float(inc_row["map"] - exc_row["map"]), 4),
            }
        )
    frames["cannibalization"] = pd.DataFrame(cannibalization_rows)
    rfm_rows, revenue_rows = [], []
    for dataset in prepared:
        user_stats = dataset.train_full.groupby("user_id").agg(recency=("timestamp", "max"), frequency=("item_id", "size"), monetary=("rating", "mean")).reset_index()
        user_stats["recency_score"] = pd.qcut(user_stats["recency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        user_stats["frequency_score"] = pd.qcut(user_stats["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        user_stats["monetary_score"] = pd.qcut(user_stats["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        user_stats["RFM_score"] = user_stats["recency_score"] + user_stats["frequency_score"] + user_stats["monetary_score"]
        user_stats["rfm_segment"] = np.select(
            [user_stats["RFM_score"] >= 10, user_stats["RFM_score"].between(8, 9), user_stats["RFM_score"].between(6, 7)],
            ["Champions", "Loyal", "Potential"],
            default="At Risk",
        )
        metric_slice = user_metrics[user_metrics["dataset"] == dataset.name].merge(
            user_stats[["user_id", "rfm_segment"]],
            on="user_id",
            how="left",
        )
        if not metric_slice.empty:
            grouped = metric_slice.groupby(["algorithm", "rfm_segment"], as_index=False).agg(
                ndcg=("ndcg", "mean"),
                recall=("recall", "mean"),
                n_users=("user_id", "nunique"),
            )
            grouped["dataset"] = dataset.name
            rfm_rows.append(grouped)
        merged = dataset.train_full.merge(user_stats[["user_id", "RFM_score"]], on="user_id", how="left").merge(
            dataset.item_features[["item_id", "proxy_category"]], on="item_id", how="left"
        )
        share = merged.groupby(["RFM_score", "proxy_category"]).size().reset_index(name="events")
        share["share"] = share.groupby("RFM_score")["events"].transform(lambda column: 100 * column / column.sum())
        share["revenue"] = share["events"]
        share["dataset"] = dataset.name
        revenue_rows.append(share[["dataset", "RFM_score", "proxy_category", "share", "revenue"]])
    frames["rfm_metrics"] = (
        pd.concat(rfm_rows, ignore_index=True).groupby(["dataset", "algorithm", "rfm_segment"], as_index=False).agg({"ndcg": "mean", "recall": "mean", "n_users": "sum"})
        if rfm_rows
        else pd.DataFrame(columns=["dataset", "algorithm", "rfm_segment", "ndcg", "recall", "n_users"])
    )
    frames["revenue_mix"] = pd.concat(revenue_rows, ignore_index=True) if revenue_rows else pd.DataFrame(columns=["dataset", "RFM_score", "proxy_category", "share", "revenue"])
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Download real datasets, run a lightweight benchmark and export dashboard artifacts.")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    config = load_local_data_config()
    raw_root = Path(args.raw_root) if args.raw_root else config.raw_data_root / "real_benchmark"
    output_dir = Path(args.output_dir) if args.output_dir else config.dashboard_artifacts_root
    if args.refresh and output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_directory(raw_root)
    ensure_directory(output_dir)

    movielens_interactions, movielens_meta = load_movielens(raw_root)
    google_interactions, google_meta = load_google_local(raw_root)
    goodreads_interactions, goodreads_books_path = load_goodreads(raw_root)
    goodreads_meta = parse_goodreads_books(goodreads_books_path, set(goodreads_interactions["item_id"].astype(str)))
    prepared = [
        build_prepared_dataset(DATASET_SPECS["movielens_20m"]["name"], "movielens_20m", movielens_interactions, movielens_meta, DATASET_SPECS["movielens_20m"]),
        build_prepared_dataset(DATASET_SPECS["google_local_sc"]["name"], "google_local_sc", google_interactions, google_meta, DATASET_SPECS["google_local_sc"]),
        build_prepared_dataset(DATASET_SPECS["goodreads_fantasy"]["name"], "goodreads_fantasy", goodreads_interactions, goodreads_meta, DATASET_SPECS["goodreads_fantasy"]),
    ]
    all_results, all_slices, all_eff, all_support, all_importance, all_user_metrics = [], [], [], [], [], []
    for dataset in prepared:
        print(f"[benchmark] {dataset.name}: users={dataset.train_full['user_id'].nunique()} items={dataset.train_full['item_id'].nunique()} events={len(dataset.train_full)}")
        results, slice_metrics, efficiency, rec_rows, feature_importance, user_metrics = evaluate_dataset(dataset)
        all_results.append(results)
        all_slices.append(slice_metrics)
        all_eff.append(efficiency)
        all_support.append(build_supporting_artifacts(dataset, rec_rows))
        all_importance.append(feature_importance)
        all_user_metrics.append(user_metrics)
    frames = build_global_artifacts(
        prepared,
        pd.concat(all_results, ignore_index=True),
        pd.concat(all_slices, ignore_index=True),
        pd.concat(all_eff, ignore_index=True),
        pd.concat(all_importance, ignore_index=True),
        pd.concat(all_user_metrics, ignore_index=True),
        all_support,
    )
    export_dashboard_artifacts(output_dir, frames)
    print(f"[export] dashboard artifacts -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
