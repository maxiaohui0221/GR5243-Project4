# Supervised Model Development and Evaluation

## Modeling Target

The primary supervised task predicts `target_outperform_sp500_next_quarter`, a binary indicator for whether a stock outperforms the S&P 500 over the next quarter. The corresponding return diagnostic is `target_sp500_excess_return_3m`. The feature table uses only information available at the firm-month observation date, while the target uses future three-month returns.

## Validation Design

Model development uses the chronological train, validation, and test files created during feature engineering. Hyperparameters are selected on the validation period, and the held-out test period is used only for final comparison. After selecting the final model, the selected specification is refit on train plus validation and evaluated on the test period.

### Split Summary

| split      |   rows |   firms | start_date   | end_date   | sp500_outperformance_rate   |
|:-----------|-------:|--------:|:-------------|:-----------|:----------------------------|
| test       |  36803 |   10036 | 2024-11-30   | 2025-09-30 | 41.17%                      |
| train      | 127960 |   10328 | 2020-01-31   | 2023-12-31 | 40.54%                      |
| validation |  26750 |    9183 | 2024-01-31   | 2024-10-31 | 36.04%                      |

## Models Compared

Three distinct supervised-learning families are trained and tuned:

- Logistic Regression: a regularized linear baseline with balanced class weights.
- Random Forest: a nonlinear bagged-tree model that captures feature interactions while controlling tree depth and leaf size.
- HistGradientBoosting: a boosted-tree model that sequentially learns nonlinear corrections and uses early stopping.

The tuning grid is intentionally moderate because the project prioritizes leakage control, interpretability, and stable validation over brute-force search.

## Model Comparison

| model                |   average_precision_test |   average_precision_train |   average_precision_validation |   balanced_accuracy_test |   balanced_accuracy_train |   balanced_accuracy_validation |   f1_test |   f1_train |   f1_validation |   roc_auc_test |   roc_auc_train |   roc_auc_validation |
|:---------------------|-------------------------:|--------------------------:|-------------------------------:|-------------------------:|--------------------------:|-------------------------------:|----------:|-----------:|----------------:|---------------:|----------------:|---------------------:|
| HistGradientBoosting |                   0.3819 |                    0.7532 |                         0.448  |                   0.4872 |                    0.7156 |                         0.609  |    0.4372 |     0.692  |          0.5438 |         0.4789 |          0.8151 |               0.6352 |
| Logistic Regression  |                   0.4424 |                    0.5506 |                         0.4498 |                   0.5282 |                    0.6015 |                         0.5843 |    0.4356 |     0.5437 |          0.47   |         0.5361 |          0.6459 |               0.6086 |
| Random Forest        |                   0.4337 |                    0.7082 |                         0.4701 |                   0.5047 |                    0.6738 |                         0.6054 |    0.4924 |     0.6606 |          0.5449 |         0.5202 |          0.7759 |               0.6339 |

## Final Model Selection

The selected final model is **Logistic Regression**. It achieved validation ROC AUC 0.6086 with a train-validation AUC gap of 0.0373; its robustness-adjusted selection score is 0.5993. After refitting on train plus validation, it reached held-out test ROC AUC 0.5502. On the test set, its balanced accuracy is 0.5306, F1 score is 0.4113, and average precision is 0.4500.

Final selected hyperparameters:

```json
{
  "C": 0.1,
  "class_weight": "balanced"
}
```

The final model is selected using validation ROC AUC with a penalty for the train-validation AUC gap. This keeps the selection rule based only on pre-test information while recognizing that the project is a noisy, nonstationary financial prediction problem where robustness and interpretability matter alongside raw validation performance.

## Economic Diagnostics

Top-probability portfolios provide an economic ranking diagnostic:

| portfolio   |   rows | precision   | mean_predicted_probability   |
|:------------|-------:|:------------|:-----------------------------|
| Top 5%      |   1840 | 48.86%      | 69.71%                       |
| Top 10%     |   3680 | 47.55%      | 66.18%                       |
| Top 20%     |   7360 | 46.96%      | 61.84%                       |

The held-out test base outperformance rate is 41.17%, while the top-decile precision is 47.55%. This comparison is included to avoid overstating model lift: in this leakage-controlled version, classification signal is modest.

A monthly top-decile long-short diagnostic produces an average long-short excess return of 3.29% per three-month target window, with an approximate annualized Sharpe ratio of 0.870. This is an evaluation diagnostic, not a transaction-cost-adjusted trading strategy.

## Interpretation

The most important final-model features are:

| clean_feature                    | family                 | importance_share   |
|:---------------------------------|:-----------------------|:-------------------|
| momentum_score                   | Composite/unsupervised | 4.01%              |
| log_market_cap                   | Market behavior        | 2.96%              |
| log_market_cap_month_rank        | Relative position      | 2.00%              |
| sic2_24                          | Context/categorical    | 1.91%              |
| low_volatility_6m_month_rank     | Relative position      | 1.60%              |
| volatility_6m_month_rank         | Relative position      | 1.60%              |
| log_market_cap_industry_relative | Relative position      | 1.52%              |
| sic2_17                          | Context/categorical    | 1.52%              |
| sic2_23                          | Context/categorical    | 1.49%              |
| sic2_48                          | Context/categorical    | 1.45%              |
| risk_adjusted_momentum           | Composite/unsupervised | 1.32%              |
| sic2_29                          | Context/categorical    | 1.27%              |
| sic2_22                          | Context/categorical    | 1.26%              |
| momentum_6m_month_rank           | Relative position      | 1.25%              |
| market_excess_momentum_6m        | Market behavior        | 1.22%              |

The importance profile is consistent with the earlier feature-engineering logic: the model uses a mixture of market behavior, accounting fundamentals, relative-positioning variables, and unsupervised firm-profile summaries rather than relying on a single raw accounting item.

## Output Files

- `model_tuning_results.csv`: all tuned model variants and validation scores.
- `model_metrics.csv`: train/validation/test metrics for each selected model family.
- `final_model.joblib` and `final_preprocessor.joblib`: final train+validation fitted artifacts.
- `final_test_predictions.csv.gz`: held-out test predictions for audit and dashboard use.
- `interactive_dashboard.html`: self-contained dashboard for model comparison, feature importance, and portfolio diagnostics.
- `*.png`: report-ready evaluation figures.
