"""
Full EDA pipeline for Medical Insurance Cost dataset (regression).

Expected dataset: data/raw/insurance.csv

This script performs:
1. Data loading and preprocessing
2. Univariate analysis
3. Log transformation of target
4. Outlier detection and removal (IQR on features only)
5. Bivariate analysis
6. Correlation analysis with feature selection
7. Train/test split with stratification by target quantiles
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
    split_train_test_stratified_regression_by_quantiles,
    ensure_dirs, BASE_DIR, PROC_DIR
)

# Configuration
sns.set_theme(style="whitegrid")
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "insurance.csv"


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
    
    # Cast categorical columns
    categorical_cols = ['sex', 'smoker', 'region']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"  Cast {col} as category: {df[col].unique().tolist()}")
    
    # Check for missing values
    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  None")
    
    print(f"\nFinal shape: {df.shape}")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
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
    
    # Summary statistics
    print("\nNumeric summary statistics:")
    print(df[numeric_cols].describe())
    
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
        save_fig(f'insurance_univariate_{col}.png')
    
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
        save_fig(f'insurance_univariate_{col}.png')
    
    print("\nUnivariante analysis complete. Figures saved.")


def create_log_target(df):
    """Create log-transformed target for modeling stability."""
    print("\n" + "="*60)
    print("3. LOG TRANSFORMATION OF TARGET")
    print("="*60)
    
    if 'charges' not in df.columns:
        print("Warning: 'charges' column not found")
        return df
    
    # Create log-transformed target
    df['charges_log'] = np.log1p(df['charges'])
    
    print(f"\nCreated charges_log = log1p(charges)")
    print(f"  Original charges range: [{df['charges'].min():.2f}, {df['charges'].max():.2f}]")
    print(f"  Log charges range: [{df['charges_log'].min():.2f}, {df['charges_log'].max():.2f}]")
    
    # Visualize transformation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(df['charges'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('charges')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Original charges distribution')
    
    axes[1].hist(df['charges_log'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('charges_log')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-transformed charges distribution')
    
    plt.tight_layout()
    save_fig('insurance_log_transformation.png')
    
    return df


def outlier_detection(df):
    """Detect and remove outliers using IQR method (features only, not target)."""
    print("\n" + "="*60)
    print("4. OUTLIER DETECTION AND REMOVAL (IQR)")
    print("="*60)
    
    # Apply IQR only to features, not target
    feature_cols = ['age', 'bmi']
    
    print(f"\nApplying IQR filter to feature columns: {feature_cols}")
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in feature_cols:
        if col in df.columns:
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
    print("5. BIVARIATE ANALYSIS")
    print("="*60)
    
    target = 'charges'
    if target not in df.columns:
        print(f"Warning: Target column '{target}' not found")
        return
    
    # Charges by smoker (boxplot)
    if 'smoker' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='smoker', y=target)
        plt.title(f'{target} by smoker status')
        plt.tight_layout()
        save_fig('insurance_bivariate_charges_by_smoker.png')
    
    # Age vs charges (scatter, colored by smoker)
    if 'age' in df.columns and 'smoker' in df.columns:
        plt.figure(figsize=(10, 6))
        for smoker_status in df['smoker'].unique():
            subset = df[df['smoker'] == smoker_status]
            plt.scatter(subset['age'], subset[target], 
                       label=f'Smoker: {smoker_status}', alpha=0.6)
        plt.xlabel('Age')
        plt.ylabel(target)
        plt.title(f'Age vs {target} (by smoker status)')
        plt.legend()
        plt.tight_layout()
        save_fig('insurance_bivariate_age_vs_charges.png')
    
    # BMI vs charges (scatter, colored by smoker)
    if 'bmi' in df.columns and 'smoker' in df.columns:
        plt.figure(figsize=(10, 6))
        for smoker_status in df['smoker'].unique():
            subset = df[df['smoker'] == smoker_status]
            plt.scatter(subset['bmi'], subset[target], 
                       label=f'Smoker: {smoker_status}', alpha=0.6)
        plt.xlabel('BMI')
        plt.ylabel(target)
        plt.title(f'BMI vs {target} (by smoker status)')
        plt.legend()
        plt.tight_layout()
        save_fig('insurance_bivariate_bmi_vs_charges.png')
    
    # Charges by sex
    if 'sex' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='sex', y=target)
        plt.title(f'{target} by sex')
        plt.tight_layout()
        save_fig('insurance_bivariate_charges_by_sex.png')
    
    # Charges by region
    if 'region' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='region', y=target)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{target} by region')
        plt.tight_layout()
        save_fig('insurance_bivariate_charges_by_region.png')
    
    print("\nBivariate analysis complete. Figures saved.")


def correlation_analysis(df):
    """Perform correlation analysis and feature selection."""
    print("\n" + "="*60)
    print("6. CORRELATION ANALYSIS")
    print("="*60)
    
    # Create dummy variables for categorical features
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Get numeric columns (including dummies)
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation analysis")
        return df
    
    # Spearman correlation
    corr_matrix = df_encoded[numeric_cols].corr(method='spearman')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, annot_kws={'size': 8})
    plt.title('Spearman Correlation Matrix (with dummy variables)')
    plt.tight_layout()
    save_fig('insurance_correlation_matrix.png')
    
    # Analyze correlations with target
    if 'charges' in corr_matrix.columns:
        target_corr = corr_matrix['charges'].sort_values(ascending=False)
        print("\nCorrelations with charges (sorted):")
        print(target_corr)
        
        print("\nKey observations:")
        print("  - Region variables show low correlation with charges")
        print("  - Children shows low correlation with charges")
        print("  - Decision: Dropping 'region' to simplify the model")
        print("  - Keeping 'children' as it may have interaction effects")
        
        if 'region' in df.columns:
            df = df.drop('region', axis=1)
            print("\n  Dropped: region")
    
    return df


def save_and_split(df):
    """Save processed data and create train/test splits."""
    print("\n" + "="*60)
    print("7. SAVE AND SPLIT DATA")
    print("="*60)
    
    ensure_dirs()
    
    # Save processed data
    processed_path = PROC_DIR / "insurance_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"\nSaved processed data: {processed_path}")
    
    # Split data stratified by target quantiles
    target = 'charges'
    if target not in df.columns:
        print(f"Warning: Target column '{target}' not found, skipping split")
        return
    
    train_df, test_df = split_train_test_stratified_regression_by_quantiles(
        df, target, q=5, test_size=0.2, random_state=42
    )
    
    # Save splits
    train_path = PROC_DIR / "insurance_train.csv"
    test_path = PROC_DIR / "insurance_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nSaved train split: {train_path}")
    print(f"Saved test split: {test_path}")


def main():
    """Run full EDA pipeline."""
    print("\n" + "="*60)
    print("MEDICAL INSURANCE COST - EDA PIPELINE")
    print("="*60)
    
    # Pipeline steps
    df = load_and_preprocess(RAW_DATA_PATH)
    univariate_analysis(df)
    df = create_log_target(df)
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
