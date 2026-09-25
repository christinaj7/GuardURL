"""
GuardURL - Phase A: Dataset Inspection Script
Analyzes the PhiUSIIL Phishing URL Dataset.
"""
import os
import json
import pandas as pd
import numpy as np

DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"

def inspect_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    rows, cols = df.shape
    print(f"Total Rows: {rows:,}")
    print(f"Total Columns: {cols}")

    # Column names
    columns = list(df.columns)
    print("\n--- Columns ---")
    for i, col in enumerate(columns, 1):
        print(f"{i:2d}. {col} ({df[col].dtype})")

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    print(f"\nMissing values total: {missing.sum()}")
    if not missing_cols.empty:
        print("Columns with missing values:")
        for col, cnt in missing_cols.items():
            print(f"  {col}: {cnt} ({cnt/rows*100:.2f}%)")
    else:
        print("No missing values found across all columns.")

    # Duplicate records
    full_duplicates = df.duplicated().sum()
    print(f"\nFull row duplicate records: {full_duplicates}")
    
    url_col = "URL" if "URL" in df.columns else None
    if url_col:
        url_duplicates = df.duplicated(subset=[url_col]).sum()
        print(f"Duplicate URLs: {url_duplicates} ({url_duplicates/rows*100:.2f}%)")

    # Target column analysis
    target_col = "label" if "label" in df.columns else None
    print(f"\nTarget Column: '{target_col}'")
    if target_col:
        label_counts = df[target_col].value_counts().to_dict()
        label_props = df[target_col].value_counts(normalize=True).to_dict()
        print("Label Distribution:")
        print(f"  1 (Legitimate): {label_counts.get(1, 0):,} ({label_props.get(1, 0)*100:.2f}%)")
        print(f"  0 (Phishing):   {label_counts.get(0, 0):,} ({label_props.get(0, 0)*100:.2f}%)")
        
        # Check class imbalance
        ratio = max(label_counts.values()) / min(label_counts.values())
        print(f"  Class Ratio (Majority/Minority): {ratio:.2f}:1")
        if ratio > 1.5:
            print("  Note: Noticeable class imbalance detected.")
        else:
            print("  Note: Well-balanced dataset.")

    # Save summary dictionary
    summary = {
        "num_rows": rows,
        "num_columns": cols,
        "columns": columns,
        "missing_values": {k: int(v) for k, v in missing_cols.items()},
        "full_duplicates": int(full_duplicates),
        "url_duplicates": int(url_duplicates) if url_col else 0,
        "target_col": target_col,
        "url_col": url_col,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "label_percentages": {str(k): round(float(v) * 100, 2) for k, v in label_props.items()},
        "dtypes": {k: str(v) for k, v in df.dtypes.items()}
    }

    os.makedirs("results", exist_ok=True)
    with open("results/dataset_inspection.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDataset inspection summary saved to results/dataset_inspection.json")

    return summary

if __name__ == "__main__":
    inspect_dataset()
