# Exploratory Data Analysis and Unsupervised Learning

## Research Context

The broader project asks whether financial statement information can help predict whether a stock will outperform an index in the following quarter. This EDA section does not build the predictive target or train supervised models. Instead, it studies the cleaned Compustat-CRSP matched dataset to understand data quality, variable distributions, market/accounting relationships, and latent firm-month structure before downstream feature engineering and modeling.

## Dataset Overview

The cleaned dataset, `compustat_crsp_merged_matched_only.csv`, contains 253,281 firm-month observations and 37 variables. It covers 5,811 unique `PERMNO` stock identifiers and 5,877 tickers from January 2020 through December 2025. The reporting-date variable `rdq` ranges from July 23, 2019 to December 31, 2025, reflecting the quarterly accounting announcements matched to monthly CRSP market data.

The dataset combines three broad categories of information:

- Market variables: monthly return (`ret_m`), daily volatility proxy (`vol_d`), ending price (`prc_end`), market value (`mv_end`), and average volume (`vol_avg`).
- Accounting variables: assets, equity, cash, cost of goods sold, debt, liabilities, net income, sales, and quarter-end prices.
- Firm/date identifiers: `PERMNO`, `gvkey`, ticker, CUSIP, fiscal quarter/year, announcement dates, SIC, and NAICS codes.

Useful generated outputs:

- `outputs/eda_tables/dataset_overview.csv`
- `outputs/eda_tables/numeric_descriptive_statistics.csv`
- `outputs/eda_figures/observations_by_month.png`

## Missingness and Data Quality

The cleaned dataset is mostly complete for identifiers, dates, returns, prices, and core market variables. Missingness is concentrated in accounting fields. The highest missingness rates are:

- `dlcq`: 14.22%
- `dlttq`: 6.03%
- `asset_turnover`: 5.40%
- `roa`: 5.38%
- `ceqq`: 5.38%
- `atq`: 5.30%
- `ltq`: 5.30%
- `cheq`: 5.30%
- `cogsq`: 5.02%
- `saleq`: 4.91%
- `niq`: 4.89%

This pattern is reasonable for a merged financial-market dataset because debt and accounting statement items are not always reported or matched cleanly for every firm-month. For EDA charts, missingness is reported directly rather than hidden. For PCA and KMeans, missing values in the unsupervised-learning input variables are filled using median imputation inside the modeling pipeline. Median imputation is appropriate here because financial variables are highly skewed and sensitive to extreme observations; the median is less distorted by very large firms or unusual accounting values.

Useful generated outputs:

- `outputs/eda_tables/missingness.csv`
- `outputs/eda_figures/missingness_top_columns.png`

## Distributional Patterns

Monthly returns are centered near zero but have wide tails, which is expected in stock return data. The return histogram is clipped at the 1st and 99th percentiles for visualization so that the central distribution is readable without letting extreme return months dominate the plot.

The market and accounting variables are strongly right-skewed. Variables such as market value, total assets, sales, liabilities, and trading volume differ by orders of magnitude across firms. This matters for both EDA and unsupervised learning because raw-scale PCA would mostly identify the largest firms rather than broader financial structure.

Useful generated outputs:

- `outputs/eda_figures/monthly_return_distribution.png`
- `outputs/eda_figures/correlation_heatmap.png`
- `outputs/eda_figures/sic_sector_counts.png`

## PCA and KMeans Method

The unsupervised learning portion uses PCA followed by KMeans clustering. The goal is not prediction, but exploratory segmentation of firm-month observations based on accounting and market characteristics.

The variables used for clustering are continuous market/accounting variables:

`vol_d`, `prc_end`, `mv_end`, `vol_avg`, `atq`, `ceqq`, `cheq`, `cogsq`, `dlcq`, `dlttq`, `ltq`, `niq`, `saleq`, `prccq`, `prclq`, `roa`, and `asset_turnover`.

Identifiers, dates, SIC/NAICS codes, fiscal year, and the monthly return variable `ret_m` are excluded from fitting PCA/KMeans. `ret_m` is held out and used only after clustering to describe the groups. This avoids using return behavior as a direct clustering input and keeps the unsupervised analysis focused on firm characteristics available around the reporting period.

The preprocessing pipeline is:

1. Signed `log1p` transformation for highly skewed scale variables such as assets, market value, sales, debt, income, and price.
2. 1st/99th percentile winsorization to reduce the influence of extreme outliers.
3. Median imputation for missing values.
4. Robust scaling using median and interquartile range.
5. PCA reduction.
6. KMeans clustering on the first five principal components.

PCA requires scaling because otherwise variables with large units, such as total assets or market value, would dominate the components. Robust scaling was used instead of ordinary z-score scaling because the data contain large outliers and heavy-tailed financial variables.

## PCA Results

The first five principal components explain 88.36% of the variance in the processed input variables:

- PC1: 48.90%
- PC2: 20.79%
- PC3: 7.97%
- PC4: 5.70%
- PC5: 5.01%

The strongest loading patterns suggest that:

- PC1 is heavily associated with profitability and broad firm scale variables such as sales, equity, assets, liabilities, net income, and long-term debt.
- PC2 separates firms based on profitability and balance-sheet/liquidity characteristics.
- PC3 is strongly related to equity and operating efficiency, especially asset turnover.

Useful generated outputs:

- `outputs/eda_tables/pca_explained_variance.csv`
- `outputs/eda_tables/pca_loadings.csv`
- `outputs/eda_figures/pca_explained_variance.png`

## KMeans Results

KMeans was evaluated across 2 to 8 clusters using inertia and silhouette score. A 4-cluster solution was selected as a practical balance: it provides more detail than a simple 2-cluster split while keeping the groups interpretable for reporting.

The four clusters can be summarized as follows:

- Cluster 0: Smaller, weaker-profitability firms. These observations have low median sales, negative median ROA, relatively high volatility, and slightly negative median monthly returns.
- Cluster 1: Large, established firms. This group has the highest median market value, assets, sales, and long-term debt, with positive median ROA and the lowest median volatility among the four clusters.
- Cluster 2: Mid-sized operating firms. This is the largest group, with moderate assets and sales, slightly positive ROA, and moderate volatility.
- Cluster 3: Very small, distressed or low-revenue firms. This group has the lowest median assets and sales, strongly negative median ROA, very low asset turnover, and the highest volatility.

Cluster summary:

| Cluster | Share of Rows | Unique Stocks | Median Return | Median Market Value | Median Assets | Median Sales | Median ROA | Median Volatility |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.71% | 2,465 | -0.95% | 738,964 | 379 | 13.7 | -6.40% | 4.01% |
| 1 | 36.77% | 1,921 | 0.82% | 9,348,930 | 11,281 | 1,250.0 | 1.11% | 1.89% |
| 2 | 49.36% | 3,972 | 0.36% | 605,101 | 1,151 | 100.8 | 0.40% | 2.28% |
| 3 | 3.15% | 1,274 | -0.74% | 399,154 | 135 | 0.4 | -19.85% | 4.47% |

The cluster profiles show that firm size, profitability, sales scale, leverage, and volatility are major sources of structure in the dataset. These latent groups may be useful later for feature engineering or model interpretation, but this EDA stops short of creating predictive features.

Useful generated outputs:

- `outputs/eda_tables/kmeans_k_selection.csv`
- `outputs/eda_tables/cluster_summary.csv`
- `outputs/eda_tables/cluster_profile_standardized_medians.csv`
- `outputs/eda_figures/kmeans_elbow_silhouette.png`
- `outputs/eda_figures/pca_kmeans_scatter.png`
- `outputs/eda_figures/cluster_profile_heatmap.png`

## Takeaways for the Next Modeling Stage

The dataset has enough breadth and complexity for the project: it includes thousands of firms, several years of monthly observations, and both market and accounting information. The main data quality issue is missingness in accounting variables, especially debt fields, which should be handled carefully in downstream modeling.

The unsupervised analysis suggests that the sample is not homogeneous. PCA and KMeans reveal distinct firm-month profiles related to scale, profitability, operating efficiency, and volatility. This matters for the later supervised task because the relationship between financial statement data and next-quarter index outperformance may differ across large stable firms, mid-sized operating firms, and distressed low-revenue firms.

For the final report, the strongest EDA story is: before predicting future relative performance, we first show that the merged Compustat-CRSP data contain meaningful cross-sectional structure. PCA summarizes most of that structure in a small number of components, and KMeans identifies interpretable groups that align with economic intuition about firm size, profitability, and risk.
