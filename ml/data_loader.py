"""
GuardURL - Phase 2: Data Loader
Loads and prepares dataset for feature extraction and model training.
Removes duplicate URLs to prevent train/test data leakage.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.feature_extraction import extract_features_batch, build_tld_probability_map

DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
CACHED_FEATURES_PATH = "data/processed_features.parquet"


def load_raw_dataset(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Loads the raw PhiUSIIL CSV dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def prepare_dataset(filepath: str = DATA_PATH, use_cache: bool = True, sample_size: int = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Loads raw dataset, cleans duplicates, computes empirical TLD probability map,
    extracts standardized features from the URL strings, and returns (X, y, tld_prob_map).
    
    Target:
      1 = Legitimate
      0 = Phishing
    """
    if use_cache and os.path.exists(CACHED_FEATURES_PATH) and sample_size is None:
        print(f"Loading pre-extracted features from cache: {CACHED_FEATURES_PATH}...")
        cached_df = pd.read_parquet(CACHED_FEATURES_PATH)
        y = cached_df["label"]
        X = cached_df.drop(columns=["label"])
        # Load empirical TLD map if cached
        tld_prob_map_path = "models/tld_prob_map.json"
        import json
        if os.path.exists(tld_prob_map_path):
            with open(tld_prob_map_path, "r") as f:
                tld_prob_map = json.load(f)
        else:
            tld_prob_map = {}
        return X, y, tld_prob_map

    print(f"Loading raw dataset from {filepath}...")
    df = load_raw_dataset(filepath)

    # Deduplicate based on URL column
    initial_len = len(df)
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    dedup_len = len(df)
    print(f"Deduplicated dataset: {initial_len:,} -> {dedup_len:,} rows ({initial_len - dedup_len} duplicates removed).")

    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size:,} records with stratified class distribution...")
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(int(sample_size * len(x) / len(df)), random_state=42),
            include_groups=True
        ).reset_index(drop=True)

    # Build empirical TLD lookup from clean dataset
    print("Building empirical TLD legitimacy distribution map...")
    tld_prob_map = build_tld_probability_map(df, url_col="URL", label_col="label")
    os.makedirs("models", exist_ok=True)
    import json
    with open("models/tld_prob_map.json", "w") as f:
        json.dump(tld_prob_map, f, indent=2)

    print(f"Extracting features from {len(df):,} URLs...")
    X = extract_features_batch(df["URL"].tolist(), tld_prob_map=tld_prob_map)
    y = df["label"].astype(int)

    # Cache processed dataset for speed
    if sample_size is None:
        cached_df = X.copy()
        cached_df["label"] = y
        try:
            cached_df.to_parquet(CACHED_FEATURES_PATH, index=False)
            print(f"Cached processed features to {CACHED_FEATURES_PATH}")
        except Exception as e:
            print(f"Notice: Parquet caching skipped ({e}), continuing with in-memory features.")

    return X, y, tld_prob_map


if __name__ == "__main__":
    X, y, tld_map = prepare_dataset(sample_size=1000)
    print(f"Prepared sample: X shape={X.shape}, y distribution:\n{y.value_counts()}")
