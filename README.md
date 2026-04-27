# GR5243 Project 4

End-to-end machine learning project using Compustat and CRSP data to study whether firm fundamentals and recent market behavior can predict next-quarter stock outperformance.

## Main Files

- `compustat_cleaning_merging.ipynb`: cleans Compustat data and merges it to monthly CRSP observations.
- `eda_unsupervised.py`: creates EDA tables, figures, PCA results, and exploratory KMeans clusters.
- `feature_engineering.py`: builds the modeling target, advanced engineered features, preprocessing pipeline, PCA/KMeans feature artifacts, and chronological splits.
- `feature_engineering_report.md`: written feature engineering section aligned to the grading rubric.
- `EDA_Report.md`: EDA and unsupervised learning write-up.
- `preprocessing_report.md`: data cleaning and preprocessing write-up.

## Feature Engineering

Expected input:

```bash
compustat_crsp_merged_matched_only.csv
```

If that exact file is not in the repository, `feature_engineering.py` will also look for the generated Compustat-CRSP merge, including `merged_crsp_compustat.csv.gz` in the project `outputs/` folder or Downloads. On this machine it auto-detects:

```bash
/Users/yiyi/Downloads/New Folder With Items 2/outputs/merged_crsp_compustat.csv.gz
```

Run:

```bash
python3 feature_engineering.py --input compustat_crsp_merged_matched_only.csv
```

Outputs are written to:

```bash
outputs/feature_engineering/
```

Key outputs include the full engineered dataset, train/validation/test splits, `feature_preprocessor.joblib`, PCA/KMeans artifacts, feature metadata, and an automatically generated output report.

The feature table keeps the original equal-weight benchmark target and adds S&P 500 comparison targets:

- `target_excess_return_3m`: stock return minus equal-weight market return.
- `target_outperform_next_quarter`: 1 if the stock beats the equal-weight market proxy.
- `target_sp500_excess_return_3m`: stock return minus S&P 500 return.
- `target_outperform_sp500_next_quarter`: 1 if the stock beats the S&P 500.

## Environment

Install the analysis dependencies:

```bash
python3 -m pip install -r requirements-eda.txt
```

Recommended local environment for VS Code:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eda.txt
python feature_engineering.py --input compustat_crsp_merged_matched_only.csv
```

Then choose `.venv/bin/python` as the Python interpreter in VS Code.
