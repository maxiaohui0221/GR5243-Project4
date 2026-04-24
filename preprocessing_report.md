# Project 4 Data Cleaning and Preprocessing Report

## Rubric Alignment

The Project 4 rubric asks for an end-to-end machine learning workflow: data collection and cleaning, EDA, unsupervised learning, feature engineering, supervised modeling, model comparison, communication, and reproducibility. This preprocessing deliverable focuses on the data-preparation foundation and intentionally includes EDA outputs, unsupervised KMeans/PCA features, chronological splitting, and a reusable sklearn pipeline so the modeling stage can build directly on it.

## Source Data

- Raw file: `zyukuvp88bxlctvl.csv`
- Exact raw-data duplicate files: None detected
- Raw shape: 765,204 rows x 29 columns
- Date range after parsing: 2010-01-31 to 2026-03-31
- Unique companies (`gvkey`): 25,589
- Raw SHA-256: `1ff075cb89934c0f39a7354474499f2a754dc4f3cfde4dc04ebc6943e22c5554`

## Cleaning Decisions

- Removed 30 exact duplicate rows.
- Resolved 262 duplicate `gvkey + datadate` rows by keeping the most complete record in each duplicate group.
- Dropped 0 rows with unusable keys or dates.
- Standardized ticker, CUSIP, currency, format, consolidation, and status strings using trimming and uppercase conversion.
- Converted `datadate` to datetime and numeric accounting fields to numeric dtypes.
- Set impossible negative balance-sheet stock values (assets, liabilities, current assets/current liabilities, long-term debt, cash-like fields) to missing.
- Set negative values in accounting flows that should be nonnegative for modeling ratios (sales, COGS, SG&A, capex) to missing. Negative income and operating cash flow were preserved because losses and negative cash flow are economically meaningful.
- Fields with more than 90% missingness were excluded from model features but retained in raw-source documentation.

High-missing columns excluded from model features:

uceqq, npatq, dvintfq, rstcheltq

## Feature Engineering

- Created date features: calendar year, quarter, month, and year-end indicator.
- Created SIC industry features: two-digit SIC and broad SIC division.
- Converted year-to-date `oancfy` and `capxy` into approximate quarterly `oancfq_approx` and `capxq_approx` by differencing within company-year.
- Created financial ratios: log assets, leverage, debt-to-assets, equity-to-assets, current ratio, ROA, ROE proxy, margins, asset turnover, operating-cash-flow-to-assets, and capex-to-assets.
- Created lagged growth features by company: sales growth, earnings growth, and asset growth.
- Created the supervised target `target_next_niq_growth`, a next-quarter earnings-growth target. CRSP excess-return prediction still requires adding CRSP returns and a WRDS linking table.
- Added missingness indicators for core accounting fields so missing-data patterns can remain predictive after imputation.
- Fitted KMeans clusters and two PCA components on training data only, then assigned cluster/PCA features to all splits.

## Preprocessing Pipeline

- Numeric pipeline: median imputation, train-fitted 1st/99th percentile clipping, standard scaling.
- Categorical pipeline: one-hot encoding with unknown-category handling.
- Split strategy: chronological train/validation/test split to reduce future-data leakage.
- Numeric feature count: 49
- Categorical feature count: 6
- Clustering feature count: 14

## Split Summary

| split      |   rows |   companies | start_date   | end_date   |   target_mean |   target_median |
|:-----------|-------:|------------:|:-------------|:-----------|--------------:|----------------:|
| test       |  61473 |        7478 | 2023-08-31   | 2025-12-31 |   -0.0475018  |     0.00649309  |
| train      | 358901 |       13988 | 2010-01-31   | 2021-02-28 |   -0.0547878  |     0.00649954  |
| validation |  75914 |        8498 | 2021-03-31   | 2023-07-31 |   -0.00724164 |     0.000998004 |

## CRSP + Compustat Merge

- CRSP daily file: `o42vryezwnel4m8s 4.csv`
- CRSP raw shape: 13,835,431 rows x 11 columns
- CRSP date range: 2020-01-02 to 2025-12-31
- Unique PERMNOs: 14,559
- CRSP monthly rows after aggregation and CUSIP-month dedupe: 665,514
- Merge key used here: Compustat `cusip` first 8 characters + month-end date matched to CRSP `HdrCUSIP` + month-end date.
- Important limitation: the official WRDS CRSP-Compustat linking table was not included in the folder, so this is a transparent CUSIP-based merge. Replace it with the linking table when available for production-quality entity resolution.
- Created CRSP features: 1/3/6/12 month returns, one-month volatility, average volume, average dollar volume, last price, and last market capitalization.
- Created return target: future 3-month CRSP return minus an equal-weight CRSP-universe future 3-month market proxy.
- Merged rows: 764,912
- Rows matched to CRSP: 202,571
- Rows with nonmissing CRSP excess-return target: 191,513

CRSP target split summary:

| crsp_split   |   rows |   companies |   permnos | start_date   | end_date   |   target_mean |   target_median |
|:-------------|-------:|------------:|----------:|:-------------|:-----------|--------------:|----------------:|
| test         |  36803 |       10036 |     10036 | 2024-11-30   | 2025-09-30 |    0.0116127  |     -0.00473725 |
| train        | 127960 |       10328 |     10328 | 2020-01-31   | 2023-12-31 |    0.00167298 |     -0.00367693 |
| validation   |  26750 |        9183 |      9183 | 2024-01-31   | 2024-10-31 |    0.00549628 |     -0.00396652 |

## Output Files

- `outputs/cleaned_compustat.csv.gz`: cleaned and feature-engineered dataset.
- `outputs/train_clean.csv.gz`, `outputs/validation_clean.csv.gz`, `outputs/test_clean.csv.gz`: chronological modeling splits.
- `outputs/preprocessing_pipeline.joblib`: sklearn transformer fitted on training data only.
- `outputs/cluster_artifacts.joblib`: KMeans/PCA/clustering preprocessing artifacts fitted on training data only.
- `outputs/data_profile.json`, `outputs/missingness.csv`, `outputs/invalid_values.csv`, `outputs/cluster_profiles.csv`: audit tables.
- `outputs/figures/`: EDA and unsupervised-learning figures.
- If CRSP data is present: `outputs/merged_crsp_compustat.csv.gz`, `outputs/crsp_train_clean.csv.gz`, `outputs/crsp_validation_clean.csv.gz`, `outputs/crsp_test_clean.csv.gz`, and `outputs/crsp_preprocessing_pipeline.joblib`.

## Notes for Modeling

Two modeling targets are now available when both datasets are present: `target_next_niq_growth` for Compustat earnings-growth regression/classification and `target_crsp_excess_return_3m` for CRSP-based future excess-return prediction. For the strongest final project version, use the official WRDS linking table instead of the CUSIP approximation and compare at least three supervised models using the chronological validation/test splits.
