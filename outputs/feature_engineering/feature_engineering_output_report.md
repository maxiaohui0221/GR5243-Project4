# Feature Engineering Output Report

## Rubric Alignment

This feature engineering step is designed for the advanced rubric tier: it creates predictive market, accounting, industry-relative, composite, and unsupervised features while fitting all learned preprocessing artifacts on the training period only.

## Target

- Regression target: `target_excess_return_3m` = next 3-month stock return minus next 3-month equal-weight market return.
- Classification target: `target_outperform_next_quarter` = 1 when the stock outperforms the equal-weight market proxy over the next quarter.
- Additional benchmark target: `target_sp500_excess_return_3m` = next 3-month stock return minus next 3-month S&P 500 return.
- Additional classification target: `target_outperform_sp500_next_quarter` = 1 when the stock outperforms the S&P 500 over the next quarter.
- The target uses future months only; current-month features are not used in target construction.

## Engineered Feature Families

- Market behavior: 1/3/6/12-month return, excess-return momentum, volatility, return consistency, reversal, liquidity, and Amihud-style illiquidity.
- Accounting strength: profitability, leverage, liquidity, valuation, efficiency, balance-sheet composition, and growth/change features.
- Relative positioning: monthly cross-sectional ranks and industry-relative deviations for size, value, profitability, leverage, momentum, and risk.
- Composite signals: quality, value, momentum, risk, risk-adjusted momentum, interaction terms, and a distress flag.
- Unsupervised features: PCA components and KMeans profile clusters fitted on the chronological training split only.

## Preprocessing

- Numeric features use median imputation, train-fitted 1st/99th percentile clipping, and standard scaling.
- Categorical features use missing-category handling and one-hot encoding with unknown-category support.
- Train/validation/test splits are chronological to reduce look-ahead bias.

## Output Summary

- Input rows: 764,912
- Modelable rows: 191,513
- Numeric features: 184
- Categorical features: 12
- S&P 500 source: `/Users/yiyi/Documents/GitHub/GR5243-Project4/data/sp500_daily.csv`
- Rows with S&P 500 benchmark target: 191,513
- Output directory: `outputs/feature_engineering`

## Split Summary

| split | rows | firms | start_date | end_date | target_excess_mean | target_excess_median | outperformance_rate | sp500_excess_mean | sp500_outperformance_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 36803 | 10036 | 2024-11-30 | 2025-09-30 | 0.011612731232404984 | -0.0047372530488398 | 0.48895470477950165 | -0.0016922679080213368 | 0.41167839578295246 |
| train | 127960 | 10328 | 2020-01-31 | 2023-12-31 | 0.0016729808679831297 | -0.003676933816665 | 0.4871131603626133 | -0.011300526217926208 | 0.4054157549234136 |
| validation | 26750 | 9183 | 2024-01-31 | 2024-10-31 | 0.005496283973809785 | -0.003966521117675649 | 0.4823551401869159 | -0.017197344479689472 | 0.3603738317757009 |

## Key Output Files

- `engineered_features.csv.gz`: full engineered dataset with targets and split labels.
- `feature_train.csv.gz`, `feature_validation.csv.gz`, `feature_test.csv.gz`: chronological modeling splits.
- `feature_preprocessor.joblib`: sklearn preprocessing transformer fitted on training data only.
- `feature_unsupervised_artifacts.joblib`: PCA/KMeans artifacts fitted on training data only.
- `feature_metadata.json`: feature lists, counts, split dates, and configuration.
