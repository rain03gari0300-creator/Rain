"""
Full EDA pipeline for Sleep Health and Lifestyle dataset (classification).

Expected dataset: data/raw/Sleep_health_and_lifestyle_dataset.csv

This script performs:
1. Data loading and preprocessing
2. Univariate analysis
3. Outlier detection and removal (IQR)
4. Bivariate analysis
5. Correlation analysis with feature selection
6. Train/test split with stratification
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    standardize_columns, save_fig, iqr_filter,
    split_train_test_stratified_classification, 
    check_class_distribution, ensure_dirs, BASE_DIR, PROC_DIR
)

# Configuration
sns.set_theme(style="whitegrid")
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Sleep_health_and_lifestyle_dataset.csv"


def load_and_preprocess(filepath):
    """Load and perform initial preprocessing."""
    print("="*60)
    print("1. LOADING AND PREPROCESSING")
    print("="*60)
    
    if not filepath.exists():
        print(f"\nERROR: Dataset not found at {filepath}")
        print("Please add the CSV file to data/raw/ directory")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Standardize column names
    df = standardize_columns(df)
    print(f"\nColumns: {list(df.columns)}")
    
    # Parse blood_pressure "S/D" into numeric columns
    if 'blood_pressure' in df.columns:
        print("\nParsing blood_pressure column...")
        bp_split = df['blood_pressure'].str.split('/', expand=True)
        df['bp_systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
        df['bp_diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
        df = df.drop('blood_pressure', axis=1)
        print("  Created: bp_systolic, bp_diastolic")
    
    # Drop duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    print(f"\nDuplicates removed: {n_before - n_after}")
    
    # Cast categorical columns
    categorical_cols = ['gender', 'occupation', 'bmi_category', 'sleep_disorder']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Check for missing values
    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  None")
    
    # Impute if needed (median for numeric, mode for categorical)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  Imputed {col} with median")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  Imputed {col} with mode")
    
    print(f"\nFinal shape: {df.shape}")
    print(df.info())
    
    return df


def univariate_analysis(df):
    """Perform univariate analysis and visualization."""
    print("\n" + "="*60)
    print("2. UNIVARIATE ANALYSIS")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    
    print(f"\nNumeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Histograms and boxplots for numeric
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Histogram: {col}')
        
        # Boxplot
        axes[1].boxplot(df[col].dropna(), vert=True)
        axes[1].set_ylabel(col)
        axes[1].set_title(f'Boxplot: {col}')
        
        plt.tight_layout()
        save_fig(f'sleep_univariate_{col}.png')
    
    # Count plots for categorical
    for col in categorical_cols:
        plt.figure(figsize=(10, 5))
        value_counts = df[col].value_counts()
        plt.bar(range(len(value_counts)), value_counts.values)
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.title(f'Count Plot: {col}')
        plt.tight_layout()
        save_fig(f'sleep_univariate_{col}.png')
    
    print("\nUnivariante analysis complete. Figures saved.")


def outlier_detection(df):
    """Detect and remove outliers using IQR method."""
    print("\n" + "="*60)
    print("3. OUTLIER DETECTION AND REMOVAL (IQR)")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nApplying IQR filter to numeric columns...")
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in numeric_cols:
        col_mask = iqr_filter(df, col)
        mask &= col_mask
    
    df_clean = df[mask].copy()
    n_removed = len(df) - len(df_clean)
    print(f"\nTotal rows removed: {n_removed} ({n_removed/len(df)*100:.1f}%)")
    print(f"Remaining rows: {len(df_clean)}")
    
    return df_clean


def bivariate_analysis(df):
    """Perform bivariate analysis with target variable."""
    print("\n" + "="*60)
    print("4. BIVARIATE ANALYSIS")
    print("="*60)
    
    target = 'sleep_disorder'
    if target not in df.columns:
        print(f"Warning: Target column '{target}' not found")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.select_dtypes(include=['category', 'object']).columns 
                       if col != target]
    
    # Numeric vs target (boxen plots)
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.boxenplot(data=df, x=target, y=col)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{col} by {target}')
        plt.tight_layout()
        save_fig(f'sleep_bivariate_{col}_by_{target}.png')
    
    # Categorical vs target (stacked proportions)
    for col in categorical_cols:
        ct = pd.crosstab(df[col], df[target], normalize='index')
        
        plt.figure(figsize=(10, 6))
        ct.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.xlabel(col)
        plt.ylabel('Proportion')
        plt.title(f'{target} proportions by {col}')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=target)
        plt.tight_layout()
        save_fig(f'sleep_bivariate_{col}_by_{target}.png')
    
    print("\nBivariate analysis complete. Figures saved.")


def correlation_analysis(df):
    """Perform correlation analysis and feature selection."""
    print("\n" + "="*60)
    print("5. CORRELATION ANALYSIS")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation analysis")
        return df
    
    # Spearman correlation
    corr_matrix = df[numeric_cols].corr(method='spearman')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Spearman Correlation Matrix')
    plt.tight_layout()
    save_fig('sleep_correlation_matrix.png')
    
    # Check for high correlation between bp_systolic and bp_diastolic
    if 'bp_systolic' in df.columns and 'bp_diastolic' in df.columns:
        corr = corr_matrix.loc['bp_systolic', 'bp_diastolic']
        print(f"\nCorrelation between bp_systolic and bp_diastolic: {corr:.3f}")
        
        if abs(corr) > 0.8:
            print(f"  High correlation detected (|r| > 0.8)")
            print(f"  Decision: Dropping bp_diastolic to reduce multicollinearity")
            df = df.drop('bp_diastolic', axis=1)
        else:
            print(f"  Correlation acceptable, keeping both features")
    
    return df


def save_and_split(df):
    """Save processed data and create train/test splits."""
    print("\n" + "="*60)
    print("6. SAVE AND SPLIT DATA")
    print("="*60)
    
    ensure_dirs()
    
    # Save processed data
    processed_path = PROC_DIR / "sleep_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"\nSaved processed data: {processed_path}")
    
    # Split data
    target = 'sleep_disorder'
    if target not in df.columns:
        print(f"Warning: Target column '{target}' not found, skipping split")
        return
    
    train_df, test_df = split_train_test_stratified_classification(
        df, target, test_size=0.2, random_state=42
    )
    
    # Check distributions
    check_class_distribution(df, target, "Full dataset")
    check_class_distribution(train_df, target, "Train set")
    check_class_distribution(test_df, target, "Test set")
    
    # Save splits
    train_path = PROC_DIR / "sleep_train.csv"
    test_path = PROC_DIR / "sleep_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nSaved train split: {train_path}")
    print(f"Saved test split: {test_path}")


def main():
    """Run full EDA pipeline."""
    print("\n" + "="*60)
    print("SLEEP HEALTH AND LIFESTYLE - EDA PIPELINE")
    print("="*60)
    
    # Pipeline steps
    df = load_and_preprocess(RAW_DATA_PATH)
    univariate_analysis(df)
    df = outlier_detection(df)
    bivariate_analysis(df)
    df = correlation_analysis(df)
    save_and_split(df)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Figures: {BASE_DIR / 'reports' / 'figures'}")
    print(f"  Processed data: {PROC_DIR}")


if __name__ == "__main__":
    main()
