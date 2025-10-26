"""
Shared utility functions for EDA pipelines.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Base paths
BASE_DIR = Path(__file__).parent.parent
FIG_DIR = BASE_DIR / "reports" / "figures"
PROC_DIR = BASE_DIR / "data" / "processed"


def ensure_dirs():
    """Create output directories if they don't exist."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)


def standardize_columns(df):
    """Standardize column names: lowercase with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df


def save_fig(filename, dpi=100, bbox_inches='tight'):
    """Save the current matplotlib figure to the figures directory."""
    ensure_dirs()
    filepath = FIG_DIR / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure: {filepath}")
    plt.close()


def iqr_filter(df, column):
    """
    Apply IQR-based outlier filtering to a numeric column.
    Returns a boolean mask for rows to keep.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    n_outliers = (~mask).sum()
    print(f"  {column}: removed {n_outliers} outliers (IQR method)")
    return mask


def split_train_test_stratified_classification(df, target_col, test_size=0.2, random_state=42):
    """
    Split data for classification with stratification by target class.
    Returns train and test DataFrames.
    """
    train, test = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[target_col], 
        random_state=random_state
    )
    print(f"\nTrain/Test split (stratified by {target_col}):")
    print(f"  Train size: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Test size: {len(test)} ({len(test)/len(df)*100:.1f}%)")
    return train, test


def split_train_test_stratified_regression_by_quantiles(df, target_col, q=5, test_size=0.2, random_state=42):
    """
    Split data for regression with stratification by target quantiles.
    Returns train and test DataFrames.
    """
    # Create quantile bins for stratification
    bins = pd.qcut(df[target_col], q=q, labels=False, duplicates='drop')
    
    train, test = train_test_split(
        df, 
        test_size=test_size, 
        stratify=bins, 
        random_state=random_state
    )
    
    print(f"\nTrain/Test split (stratified by {target_col} quantiles, q={q}):")
    print(f"  Train size: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Test size: {len(test)} ({len(test)/len(df)*100:.1f}%)")
    
    # Show quantile distribution in train and test
    train_bins = pd.qcut(train[target_col], q=q, duplicates='drop')
    test_bins = pd.qcut(test[target_col], q=q, duplicates='drop')
    
    print(f"\n  Train quantile distribution:")
    print(train_bins.value_counts(normalize=True).sort_index())
    print(f"\n  Test quantile distribution:")
    print(test_bins.value_counts(normalize=True).sort_index())
    
    return train, test


def check_class_distribution(df, target_col, label="Dataset"):
    """
    Print class distribution for a categorical target.
    """
    print(f"\n{label} - {target_col} distribution:")
    dist = df[target_col].value_counts()
    print(dist)
    print("\nProportions:")
    print(df[target_col].value_counts(normalize=True))
