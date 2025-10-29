"""
I/O utilities for loading and saving data.
"""
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Any


def load_lending_club_data(file_path: str, row_limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load LendingClub data with consistent preprocessing.

    Args:
        file_path: Path to CSV/gzip file
        row_limit: Maximum number of rows to load (None for all)

    Returns:
        DataFrame with loaded data
    """
    read_kwargs = dict(low_memory=False, compression="infer")
    if row_limit is not None:
        read_kwargs["nrows"] = row_limit

    df = pd.read_csv(file_path, **read_kwargs)
    print(f"[IO] Loaded {len(df)} rows from {Path(file_path).name}")

    return df


def prepare_binary_target(df: pd.DataFrame, target_col: str = "loan_status") -> pd.DataFrame:
    """
    Create binary target variable from loan status.

    Args:
        df: Input DataFrame
        target_col: Name of loan status column

    Returns:
        DataFrame with 'target' column (1=Charged Off, 0=Fully Paid)
    """
    keep_status = ["Fully Paid", "Charged Off"]
    df = df[df[target_col].isin(keep_status)].copy()
    df["target"] = (df[target_col] == "Charged Off").astype(int)
    df.drop(columns=[target_col], inplace=True)

    n_default = (df["target"] == 1).sum()
    n_paid = (df["target"] == 0).sum()
    print(f"[IO] Target distribution: {n_paid} Fully Paid, {n_default} Charged Off ({n_default/(n_paid+n_default)*100:.2f}% default)")

    return df


def save_json(data: Dict[Any, Any], file_path: str):
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[IO] Saved JSON to {file_path}")


def load_json(file_path: str) -> Dict[Any, Any]:
    """
    Load dictionary from JSON file.

    Args:
        file_path: Input file path

    Returns:
        Dictionary with loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def merge_text_fields(df: pd.DataFrame,
                     fields: list = ["desc", "title", "purpose"],
                     output_col: str = "text_merged",
                     min_length: int = 10) -> pd.DataFrame:
    """
    Merge multiple text fields into a single column.

    Args:
        df: Input DataFrame
        fields: List of text column names to merge
        output_col: Name of output merged column
        min_length: Minimum character length to consider valid

    Returns:
        DataFrame with merged text column
    """
    df = df.copy()

    # Merge available fields
    text_parts = []
    for field in fields:
        if field in df.columns:
            text_parts.append(df[field].fillna("").astype(str))

    if text_parts:
        df[output_col] = " ".join(text_parts).str.strip()
        # Mark as valid only if meets minimum length
        df[output_col] = df[output_col].apply(lambda x: x if len(x) >= min_length else None)

        n_valid = df[output_col].notna().sum()
        coverage = n_valid / len(df) * 100
        print(f"[IO] Text coverage: {n_valid}/{len(df)} ({coverage:.2f}%) samples with ≥{min_length} chars")
    else:
        df[output_col] = None
        print(f"[IO] Warning: None of the specified text fields {fields} found in data")

    return df


def get_feature_lists(df: pd.DataFrame, exclude_cols: list = ["target"]) -> tuple:
    """
    Automatically identify numeric and categorical features.

    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude (e.g., target, IDs)

    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    import numpy as np

    # Exclude specified columns
    df_features = df.drop(columns=[c for c in exclude_cols if c in df.columns])

    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_features.select_dtypes(include=["object"]).columns.tolist()

    print(f"[IO] Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features")

    return numeric_features, categorical_features
