# Sleep Health and Insurance EDA Project

A complete, professional exploratory data analysis (EDA) project covering two datasets:
1. **Sleep Health and Lifestyle** (Classification)
2. **Medical Insurance Cost** (Regression)

This project provides runnable scripts, lightweight notebooks, documentation for GitHub Pages, and a clean repository structure.

## 📁 Project Structure

```
sleep-insurance-eda/
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   ├── Sleep_health_and_lifestyle_dataset.csv  # (user adds this)
│   │   └── insurance.csv                          # (user adds this)
│   └── processed/
│       ├── .gitkeep
│       └── [generated train/test CSV files]
├── reports/
│   ├── figures/
│   │   ├── .gitkeep
│   │   └── [generated PNG plots]
│   └── notebooks/
│       ├── sleep_eda.ipynb                        # Summary notebook
│       └── insurance_eda.ipynb                    # Summary notebook
├── src/
│   ├── utils.py                                   # Shared helper functions
│   ├── eda_sleep.py                               # Sleep dataset EDA pipeline
│   └── eda_insurance.py                           # Insurance dataset EDA pipeline
├── docs/
│   └── index.html                                 # GitHub Pages landing page
├── README.md                                      # This file
├── requirements.txt                               # Python dependencies (pinned)
└── .gitignore                                     # Ignore caches and generated files
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- pip package manager

### 2. Installation

```bash
cd sleep-insurance-eda
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add Datasets

Place the following CSV files in the `data/raw/` directory:

- `Sleep_health_and_lifestyle_dataset.csv` - Sleep health dataset
- `insurance.csv` - Medical insurance cost dataset

**Note:** Raw datasets are not included in this repository. You must obtain them separately and place them in `data/raw/`.

### 4. Run EDA Pipelines

```bash
# Run sleep health EDA
python src/eda_sleep.py

# Run insurance cost EDA
python src/eda_insurance.py
```

## 📊 Dataset Descriptions

### Sleep Health and Lifestyle (Classification)

**Task:** Multi-class classification  
**Target:** `sleep_disorder` (categories: None, Sleep Apnea, Insomnia)  
**Features:**
- Demographics: gender, age, occupation
- Sleep metrics: sleep_duration, quality_of_sleep
- Lifestyle: physical_activity_level, stress_level
- Health: bmi_category, heart_rate, daily_steps
- Medical: blood_pressure (parsed into systolic/diastolic)

**Expected path:** `data/raw/Sleep_health_and_lifestyle_dataset.csv`

### Medical Insurance Cost (Regression)

**Task:** Regression  
**Target:** `charges` (continuous, USD)  
**Features:**
- age: Age of beneficiary
- sex: Gender (male/female)
- bmi: Body mass index
- children: Number of dependents
- smoker: Smoking status (yes/no)
- region: Geographic region (northeast, southeast, southwest, northwest)

**Expected path:** `data/raw/insurance.csv`

## 🔧 Pipeline Details

### Sleep Health EDA (`src/eda_sleep.py`)

1. **Data Loading and Preprocessing**
   - Standardize column names (lowercase, underscores)
   - Parse `blood_pressure` "S/D" format → `bp_systolic`, `bp_diastolic`
   - Drop duplicate rows
   - Cast categorical variables
   - Check for missing values and impute (median for numeric, mode for categorical)

2. **Univariate Analysis**
   - Histograms and boxplots for numeric features
   - Count plots for categorical features
   - Saves figures to `reports/figures/sleep_univariate_*.png`

3. **Outlier Detection**
   - Apply IQR method (1.5 × IQR) to all numeric features
   - Remove rows with outliers

4. **Bivariate Analysis**
   - Boxen plots: numeric features vs. `sleep_disorder`
   - Stacked bar charts: categorical features vs. `sleep_disorder`
   - Saves figures to `reports/figures/sleep_bivariate_*.png`

5. **Correlation Analysis**
   - Compute Spearman correlation matrix
   - **Decision rule:** Drop `bp_diastolic` if |correlation| > 0.8 with `bp_systolic`
   - Saves figure to `reports/figures/sleep_correlation_matrix.png`

6. **Train/Test Split**
   - 80/20 split with stratification by `sleep_disorder` class
   - Saves: `data/processed/sleep_train.csv`, `data/processed/sleep_test.csv`
   - Prints class distributions to verify balance

**Outputs:**
- Processed data: `data/processed/sleep_processed.csv`
- Train set: `data/processed/sleep_train.csv`
- Test set: `data/processed/sleep_test.csv`
- Figures: `reports/figures/sleep_*.png`

### Insurance Cost EDA (`src/eda_insurance.py`)

1. **Data Loading and Preprocessing**
   - Standardize column names
   - Cast categorical variables (sex, smoker, region)
   - Check for missing values (none expected)

2. **Univariate Analysis**
   - Histograms and boxplots for numeric features
   - Count plots for categorical features
   - Saves figures to `reports/figures/insurance_univariate_*.png`

3. **Log Transformation**
   - Create `charges_log = log1p(charges)` for modeling stability
   - Reduces skewness in target distribution
   - Saves figure to `reports/figures/insurance_log_transformation.png`

4. **Outlier Detection**
   - Apply IQR method **only to features** (age, bmi), not to target
   - Preserves natural variation in insurance charges

5. **Bivariate Analysis**
   - Boxplots: charges by smoker, sex, region
   - Scatter plots: age vs. charges, bmi vs. charges (colored by smoker)
   - Saves figures to `reports/figures/insurance_bivariate_*.png`

6. **Correlation Analysis**
   - Compute Spearman correlation with dummy variables (`get_dummies(drop_first=True)`)
   - **Key findings:**
     - Region variables: low correlation with charges
     - Children: low correlation but may have interaction effects
   - **Decision:** Drop `region` to simplify model (weak predictor)
   - Saves figure to `reports/figures/insurance_correlation_matrix.png`

7. **Train/Test Split**
   - 80/20 split with stratification by `charges` quantiles (q=5)
   - Ensures representative distribution of target values
   - Saves: `data/processed/insurance_train.csv`, `data/processed/insurance_test.csv`
   - Prints quantile distributions to verify balance

**Outputs:**
- Processed data: `data/processed/insurance_processed.csv`
- Train set: `data/processed/insurance_train.csv`
- Test set: `data/processed/insurance_test.csv`
- Figures: `reports/figures/insurance_*.png`

## 📈 Key Decisions and Justifications

### IQR Outlier Filtering

**Method:** Interquartile Range (IQR)  
**Rule:** Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] are considered outliers

**Rationale:**
- Robust to non-normal distributions
- Well-established statistical method
- Balances outlier removal with data retention
- Applied to features only in regression (not target) to preserve natural variation

### Correlation-Based Feature Selection

**Sleep Dataset:**
- **Rule:** Drop `bp_diastolic` if |correlation with `bp_systolic`| > 0.8
- **Rationale:** Blood pressure components are mechanically related; high correlation indicates redundancy and potential multicollinearity

**Insurance Dataset:**
- **Rule:** Drop `region` due to low correlation with target (< 0.1)
- **Rationale:** Region shows minimal predictive value; simplifies model without sacrificing performance
- **Note:** Keep `children` despite low correlation due to potential interaction effects with other features

### Train/Test Stratification

**Classification (Sleep):**
- **Method:** Stratify by target class (`sleep_disorder`)
- **Split:** 80% train / 20% test
- **Rationale:** Ensures balanced class distributions in both sets, critical for imbalanced datasets

**Regression (Insurance):**
- **Method:** Stratify by target quantiles (q=5)
- **Split:** 80% train / 20% test
- **Rationale:** Ensures representative distribution of target values across the range, prevents train/test mismatch

## 📚 Notebooks

Lightweight summary notebooks are provided in `reports/notebooks/`:
- `sleep_eda.ipynb` - Markdown-only summary of sleep health analysis
- `insurance_eda.ipynb` - Markdown-only summary of insurance cost analysis

These notebooks contain key insights and pointers to generated figures, keeping the repository lightweight.

## 🌐 GitHub Pages Setup

To publish the documentation using GitHub Pages:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select branch `main`
4. Under **Folder**, select `/sleep-insurance-eda/docs`
5. Click **Save**
6. Your site will be available at: `https://[username].github.io/[repository]/`

The landing page (`docs/index.html`) provides an overview of the project, instructions, and links to results.

## 🛠️ Utilities (`src/utils.py`)

The `utils.py` module provides shared helper functions:

- `standardize_columns(df)` - Standardize column names
- `save_fig(filename)` - Save matplotlib figure to reports/figures/
- `iqr_filter(df, column)` - Apply IQR outlier filter
- `split_train_test_stratified_classification(df, target_col)` - Stratified split for classification
- `split_train_test_stratified_regression_by_quantiles(df, target_col, q)` - Stratified split for regression
- `check_class_distribution(df, target_col)` - Print class distribution
- `ensure_dirs()` - Create output directories if needed

## 📦 Dependencies

```
pandas==2.2.2
numpy==2.1.2
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.5.2
scipy==1.13.1
```

All dependencies are pinned to specific versions for reproducibility.

## 🧪 Testing the Scripts

You can test that the scripts import correctly without running the full pipeline:

```bash
# Test imports
python -c "from src.utils import *; print('utils.py OK')"
python -c "import src.eda_sleep; print('eda_sleep.py OK')"
python -c "import src.eda_insurance; print('eda_insurance.py OK')"
```

To run the full pipelines, ensure datasets are in place first.

## 🚫 What's Not Included

- **Raw datasets** - Not included due to size/licensing. User must add them.
- **Generated figures** - Generated when you run the scripts
- **Processed data** - Generated when you run the scripts
- **Python caches** - Ignored via `.gitignore`

## 📝 Notes

- All code is Python 3.10+ compatible
- Uses seaborn default styling for plots
- Figures are saved at 100 DPI by default
- All paths are relative to the `sleep-insurance-eda/` directory
- Scripts are idempotent - can be run multiple times safely

## 🤝 Contributing

This is a standalone EDA project. To extend:
- Add new analysis scripts to `src/`
- Add new notebooks to `reports/notebooks/`
- Update `docs/index.html` with new content

## 📄 License

Not specified. Recommended: Add MIT license for academic work.

## 👤 Author

This project is part of the Rain repository by @rain03gari0300-creator.

---

**Last updated:** 2025-10-26
