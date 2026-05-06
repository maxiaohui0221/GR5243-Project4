# Feature Engineering and Preprocessing


## Predictive Target

The modeling question is:

Can current financial statement information, market behavior, and firm context predict whether a stock will outperform the market over the next quarter?

Two targets are created:

- `target_excess_return_3m`: next 3-month compounded stock return minus the next 3-month equal-weight market return from the sample.
- `target_outperform_next_quarter`: binary target equal to 1 when `target_excess_return_3m` is positive.
- `target_sp500_excess_return_3m`: next 3-month compounded stock return minus the next 3-month S&P 500 return.
- `target_outperform_sp500_next_quarter`: binary target equal to 1 when `target_sp500_excess_return_3m` is positive.

The equal-weight benchmark remains the primary target, and the S&P 500 benchmark is added as an additional comparison. Both targets use months after the observation date only. Chronological train/validation/test splits are used so preprocessing and unsupervised transformations are fitted on the training period before being applied to later periods.

Forward-looking helper columns used during target construction, including `market_return_fwd_3m` and `days_to_next_observation`, are excluded from the final predictor list. This keeps future market information and future data-availability information out of the supervised feature matrix.

## Feature Families

The engineered features are intentionally grouped around economic mechanisms that should matter for future relative stock performance.

Market behavior features:

- 1-month return, short-term reversal, 3/6/12-month momentum, and market-excess momentum.
- 3/6/12-month volatility and return consistency.
- Liquidity and trading features including log market capitalization, log price, log volume, log dollar volume, and Amihud-style illiquidity.

Accounting and fundamental features:

- Profitability: ROA, ROE, profit margin, gross margin, operating margin, and operating return on assets when available.
- Leverage and liquidity: debt-to-assets, long-term debt-to-assets, current debt-to-assets, liabilities-to-assets, net debt-to-assets, equity-to-assets, current ratio, cash-to-assets, and sales-to-debt.
- Valuation and scale: book-to-market, earnings-to-market, sales-to-market, log assets, log sales, and log book equity.
- Efficiency: asset turnover and changes in asset turnover.
- Growth and change: quarter-over-quarter and year-over-year growth in sales, earnings, assets, equity, cash, liabilities, debt, COGS, and market value; changes in profitability, margins, turnover, and leverage.

Relative and contextual features:

- Monthly cross-sectional percentile ranks for size, value, profitability, leverage, momentum, volatility, and illiquidity.
- Industry-relative deviations and ranks within SIC two-digit groups.
- Calendar features for year, quarter, month, year-end months, and earnings-season months.
- SIC two-digit industry and broad SIC division encodings.

Composite features:

- `quality_score`: combines profitability, margin, efficiency, and low leverage ranks.
- `value_score`: combines book-to-market, earnings-to-market, and sales-to-market ranks.
- `momentum_score`: combines momentum, market-excess momentum, and low volatility information.
- `risk_score`: combines leverage, volatility, and illiquidity.
- Interaction features: quality-value, quality-momentum, value-momentum, risk-adjusted momentum, and a distress flag.

Unsupervised feature engineering:

- PCA components are fitted on the training period using quality, value, momentum, risk, size, leverage, profitability, and growth signals.
- KMeans clusters are fitted on the same training-period feature space to create `firm_profile_cluster`.
- These transformations are then applied to validation and test rows without refitting, preserving a fair modeling workflow.

## Missing Values and Scaling

Missingness is handled as signal rather than only as a nuisance:

- Core accounting fields receive missingness indicator columns.
- Aggregate accounting and market missingness counts are included.
- Numeric modeling features use median imputation.
- Numeric outliers are clipped at train-fitted 1st and 99th percentiles before standardization.
- Categorical variables use unknown-category handling and one-hot encoding.

This combination is appropriate for financial data because firm fundamentals and market variables are heavy-tailed, sparse for some firms, and sensitive to extreme outliers.

## Outputs

Running the script creates:

- `outputs/feature_engineering/engineered_features.csv.gz`
- `outputs/feature_engineering/feature_train.csv.gz`
- `outputs/feature_engineering/feature_validation.csv.gz`
- `outputs/feature_engineering/feature_test.csv.gz`
- `outputs/feature_engineering/feature_preprocessor.joblib`
- `outputs/feature_engineering/feature_unsupervised_artifacts.joblib`
- `outputs/feature_engineering/feature_metadata.json`
- `outputs/feature_engineering/feature_engineering_output_report.md`
- `data/sp500_daily.csv`: S&P 500 price-index data used to construct the additional S&P 500 benchmark target.

These files give the supervised modeling section a clean, reproducible input table with transparent targets, leakage-controlled engineered predictors, preprocessing artifacts, and split documentation.
