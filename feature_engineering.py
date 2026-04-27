#!/usr/bin/env python3
"""Advanced feature engineering for the Project 4 Compustat-CRSP model.

The script expects the cleaned matched firm-month file created by the cleaning
notebook, `compustat_crsp_merged_matched_only.csv`. It creates a next-quarter
excess-return target, engineered market/accounting/industry features, train-only
PCA and KMeans features, chronological train/validation/test splits, and a
reusable sklearn preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
warnings.filterwarnings(
    "ignore",
    message=".*Could not find the number of physical cores.*",
    category=UserWarning,
)

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:
    missing_package = exc.name
    raise SystemExit(
        f"Missing required Python package: {missing_package}\n"
        "Create and use the project virtual environment with:\n"
        "  /opt/homebrew/bin/python3.13 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  python -m pip install -r requirements-eda.txt\n"
        "Then choose .venv/bin/python as the VS Code interpreter."
    ) from exc


if __name__ == "__main__":
    # Keep joblib artifacts loadable when this file is run as a script.
    sys.modules["feature_engineering"] = sys.modules[__name__]


DEFAULT_INPUT = Path("compustat_crsp_merged_matched_only.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/feature_engineering")
RANDOM_STATE = 42

TARGET_RETURN_COL = "target_excess_return_3m"
TARGET_BINARY_COL = "target_outperform_next_quarter"
TARGET_RAW_RETURN_COL = "target_stock_return_3m"
TARGET_MARKET_RETURN_COL = "target_market_return_3m"
SP500_RETURN_COL = "target_sp500_return_3m"
SP500_EXCESS_RETURN_COL = "target_sp500_excess_return_3m"
SP500_BINARY_COL = "target_outperform_sp500_next_quarter"

INPUT_CANDIDATE_NAMES = [
    "compustat_crsp_merged_matched_only.csv",
    "merged_crsp_compustat.csv.gz",
    "merged_crsp_compustat.csv",
]

COLUMN_ALIASES = {
    "crsp_ret_1m": "ret_m",
    "crsp_vol_1m": "vol_d",
    "crsp_price_last": "prc_end",
    "crsp_market_cap_last": "mv_end",
    "crsp_avg_volume_1m": "vol_avg",
    "capxq_approx": "capxq",
    "oancfq_approx": "oancfq",
}

TARGET_ALIASES = {
    "target_crsp_excess_return_3m": TARGET_RETURN_COL,
    "target_crsp_return_3m": TARGET_RAW_RETURN_COL,
    "market_return_fwd_3m": TARGET_MARKET_RETURN_COL,
}

SP500_INPUT_CANDIDATE_NAMES = [
    "sp500_daily.csv",
    "sp500.csv",
    "SP500.csv",
    "gspc.csv",
    "GSPC.csv",
    "spy.csv",
    "SPY.csv",
]

IDENTIFIER_COLUMNS = {
    "PERMNO",
    "LPERMNO",
    "PERMCO",
    "gvkey",
    "gvkey_str",
    "cusip",
    "tic",
    "Ticker",
    "crsp_ticker",
    "firm_id",
}

DATE_COLUMNS = {
    "date",
    "datadate",
    "rdq",
    "rdq_month",
    "month_end",
    "model_date",
}

CATEGORICAL_FEATURE_CANDIDATES = [
    "sic2",
    "sic_division",
    "calendar_quarter",
    "calendar_month_name",
    "costat",
    "curcdq",
    "datafmt",
    "indfmt",
    "consol",
    "primary_exch",
    "share_type",
    "firm_profile_cluster",
]

NUMERIC_SOURCE_CANDIDATES = [
    "ret_m",
    "vol_d",
    "prc_end",
    "mv_end",
    "vol_avg",
    "atq",
    "ceqq",
    "cheq",
    "cogsq",
    "dlcq",
    "dlttq",
    "ltq",
    "niq",
    "saleq",
    "prccq",
    "prclq",
    "actq",
    "lctq",
    "oibdpq",
    "xsgaq",
    "capxy",
    "oancfy",
    "capxq",
    "oancfq",
    "sic",
    "naics",
]

ACCOUNTING_MISSINGNESS_COLUMNS = [
    "atq",
    "ceqq",
    "cheq",
    "cogsq",
    "dlcq",
    "dlttq",
    "ltq",
    "niq",
    "saleq",
    "prccq",
    "prclq",
]

MONTHLY_RANK_FEATURES = [
    "log_market_cap",
    "book_to_market",
    "earnings_to_market",
    "sales_to_market",
    "roa",
    "profit_margin",
    "gross_margin",
    "asset_turnover",
    "debt_to_assets",
    "liabilities_to_assets",
    "momentum_6m",
    "momentum_12m",
    "volatility_6m",
    "amihud_illiq",
]

INDUSTRY_RELATIVE_FEATURES = [
    "log_market_cap",
    "book_to_market",
    "roa",
    "profit_margin",
    "gross_margin",
    "asset_turnover",
    "debt_to_assets",
    "momentum_12m",
    "volatility_6m",
]

UNSUPERVISED_FEATURE_CANDIDATES = [
    "quality_score",
    "value_score",
    "momentum_score",
    "risk_score",
    "log_market_cap",
    "book_to_market",
    "debt_to_assets",
    "roa",
    "gross_margin",
    "asset_turnover",
    "momentum_12m",
    "volatility_6m",
    "amihud_illiq",
    "sales_growth_yoy",
    "niq_growth_yoy",
    "atq_growth_yoy",
]


@dataclass
class FeatureEngineeringConfig:
    input_path: str = str(DEFAULT_INPUT)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    sp500_input_path: str = ""
    target_horizon_months: int = 3
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    min_feature_nonmissing_rate: float = 0.02
    quantile_clip_lower: float = 0.01
    quantile_clip_upper: float = 0.99
    pca_components: int = 5
    kmeans_clusters: int = 5


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns to train-fitted quantile bounds."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)


QuantileClipper.__module__ = "feature_engineering"


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Fill missing categories and convert values to strings."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            out = X.copy()
        else:
            out = pd.DataFrame(X)
        out = out.astype("object").where(pd.notna(out), "Unknown")
        return out.astype(str)


CategoricalCleaner.__module__ = "feature_engineering"


def first_existing(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in frame.columns]


def col_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def safe_divide(numerator, denominator, min_abs: float = 1e-9) -> pd.Series:
    if isinstance(numerator, pd.Series):
        num = numerator
    elif isinstance(denominator, pd.Series):
        num = pd.Series(numerator, index=denominator.index)
    else:
        num = pd.Series(numerator)

    if isinstance(denominator, pd.Series):
        den = denominator
    elif isinstance(numerator, pd.Series):
        den = pd.Series(denominator, index=numerator.index)
    else:
        den = pd.Series(denominator)

    den = den.where(den.abs() > min_abs)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def signed_log1p(series: pd.Series) -> pd.Series:
    return np.sign(series) * np.log1p(series.abs())


def stable_growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    out = (current - previous) / (previous.abs() + 1.0)
    return out.replace([np.inf, -np.inf], np.nan)


def sic_division(sic_value) -> str:
    if pd.isna(sic_value):
        return "Unknown"
    sic = int(sic_value)
    if 100 <= sic <= 999:
        return "Agriculture"
    if 1000 <= sic <= 1499:
        return "Mining"
    if 1500 <= sic <= 1799:
        return "Construction"
    if 2000 <= sic <= 3999:
        return "Manufacturing"
    if 4000 <= sic <= 4999:
        return "Transportation"
    if 5000 <= sic <= 5199:
        return "Wholesale"
    if 5200 <= sic <= 5999:
        return "Retail"
    if 6000 <= sic <= 6799:
        return "Finance"
    if 7000 <= sic <= 8999:
        return "Services"
    if 9100 <= sic <= 9729:
        return "Public Administration"
    return "Other"


def make_one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)


def resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path

    if path != DEFAULT_INPUT and path.name != DEFAULT_INPUT.name:
        raise FileNotFoundError(
            f"Input file not found: {path}. Pass --input with the full path to the matched CSV."
        )

    search_roots = [
        Path.cwd(),
        Path.cwd() / "outputs",
        Path.home() / "Downloads",
        Path.home() / "Documents" / "GitHub",
    ]
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        for name in INPUT_CANDIDATE_NAMES:
            direct_candidates = [
                root / name,
                root / "outputs" / name,
                root / "New Folder With Items 2" / "outputs" / name,
            ]
            for candidate in direct_candidates:
                if candidate.is_file():
                    return candidate
            try:
                matches = sorted(root.rglob(name), key=lambda item: item.stat().st_size, reverse=True)
            except (OSError, PermissionError):
                matches = []
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"Input file not found: {path}. Run the cleaning/merging notebook first, "
        "or pass --input /path/to/compustat_crsp_merged_matched_only.csv."
    )


def resolve_sp500_path(path_text: str) -> Path | None:
    if path_text:
        path = Path(path_text)
        if path.exists():
            return path
        raise FileNotFoundError(f"S&P 500 input file not found: {path}")

    search_roots = [
        Path.cwd(),
        Path.cwd() / "data",
        Path.cwd() / "outputs",
        Path.home() / "Downloads",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for name in SP500_INPUT_CANDIDATE_NAMES:
            direct_candidates = [root / name, root / "data" / name]
            for candidate in direct_candidates:
                if candidate.is_file():
                    return candidate
            try:
                matches = sorted(root.rglob(name), key=lambda item: item.stat().st_size, reverse=True)
            except (OSError, PermissionError):
                matches = []
            if matches:
                return matches[0]
    return None


def read_modeling_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def read_sp500_monthly(path: Path) -> pd.DataFrame:
    sp500 = pd.read_csv(path, low_memory=False)
    sp500.columns = [str(col).strip() for col in sp500.columns]
    lower_map = {col.lower(): col for col in sp500.columns}

    date_col = next(
        (
            lower_map[name]
            for name in ["date", "observation_date", "month_end", "datetime"]
            if name in lower_map
        ),
        None,
    )
    if date_col is None:
        raise ValueError(
            f"S&P 500 file {path} needs a date column such as Date or observation_date."
        )

    return_col = next(
        (
            lower_map[name]
            for name in ["sp500_ret_1m", "sp500_return_1m", "return", "ret", "monthly_return"]
            if name in lower_map
        ),
        None,
    )
    price_col = next(
        (
            lower_map[name]
            for name in ["sp500", "close", "adj close", "adj_close", "price", "^gspc", "gspc"]
            if name in lower_map
        ),
        None,
    )

    sp500["date"] = pd.to_datetime(sp500[date_col], errors="coerce")
    sp500 = sp500.dropna(subset=["date"]).copy()
    sp500["month_end"] = sp500["date"].dt.to_period("M").dt.to_timestamp("M")

    if return_col is not None:
        sp500["sp500_ret_1m"] = pd.to_numeric(sp500[return_col], errors="coerce")
        monthly = sp500.groupby("month_end", sort=True)["sp500_ret_1m"].last().reset_index()
    elif price_col is not None:
        sp500["sp500_price"] = pd.to_numeric(sp500[price_col], errors="coerce")
        monthly = (
            sp500.dropna(subset=["sp500_price"])
            .sort_values("date")
            .groupby("month_end", sort=True)["sp500_price"]
            .last()
            .pct_change()
            .rename("sp500_ret_1m")
            .reset_index()
        )
    else:
        raise ValueError(
            f"S&P 500 file {path} needs either a return column or a price/close column."
        )

    monthly["sp500_ret_1m"] = monthly["sp500_ret_1m"].clip(lower=-0.999999)
    monthly["sp500_log_ret_1m"] = np.log1p(monthly["sp500_ret_1m"])
    return monthly.dropna(subset=["sp500_ret_1m"]).copy()


def add_sp500_targets(frame: pd.DataFrame, sp500_path: Path, horizon_months: int) -> pd.DataFrame:
    df = frame.copy()
    monthly = read_sp500_monthly(sp500_path)
    monthly = monthly.sort_values("month_end").reset_index(drop=True)
    future_logs = [
        monthly["sp500_log_ret_1m"].shift(-step) for step in range(1, horizon_months + 1)
    ]
    monthly[SP500_RETURN_COL] = np.expm1(sum(future_logs))

    df[SP500_RETURN_COL] = df["model_date"].map(monthly.set_index("month_end")[SP500_RETURN_COL])
    df[SP500_EXCESS_RETURN_COL] = df[TARGET_RAW_RETURN_COL] - df[SP500_RETURN_COL]
    df[SP500_BINARY_COL] = (df[SP500_EXCESS_RETURN_COL] > 0).astype("Int64")
    df.loc[df[SP500_EXCESS_RETURN_COL].isna(), SP500_BINARY_COL] = pd.NA
    return df


def standardize_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [str(col).strip() for col in df.columns]

    for source, target in COLUMN_ALIASES.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]
    for source, target in TARGET_ALIASES.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]

    for col in first_existing(NUMERIC_SOURCE_CANDIDATES, df):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    date_source = next((col for col in ["date", "month_end", "rdq_month", "datadate"] if col in df.columns), None)
    if date_source is None:
        raise ValueError("Expected a monthly date column named date, month_end, rdq_month, or datadate.")
    df["model_date"] = pd.to_datetime(df[date_source], errors="coerce").dt.to_period("M").dt.to_timestamp("M")

    firm_source = next((col for col in ["PERMNO", "LPERMNO", "gvkey", "tic"] if col in df.columns), None)
    if firm_source is None:
        raise ValueError("Expected a firm identifier column such as PERMNO, LPERMNO, gvkey, or tic.")
    df["firm_id"] = df[firm_source].astype("string").str.strip()

    before = len(df)
    df = df.dropna(subset=["model_date", "firm_id"]).copy()
    if len(df) == 0:
        raise ValueError(f"All {before:,} rows were dropped because date or firm identifiers were missing.")

    if "ret_m" not in df.columns:
        raise ValueError("Feature engineering requires monthly stock return column ret_m.")
    df["ret_m"] = pd.to_numeric(df["ret_m"], errors="coerce")
    df["ret_m_clean"] = df["ret_m"].clip(lower=-0.999999)
    df["log_ret_m"] = np.log1p(df["ret_m_clean"])

    for col in ["tic", "cusip", "costat", "curcdq", "datafmt", "indfmt", "consol", "primary_exch", "share_type"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.upper()

    return df.sort_values(["firm_id", "model_date"]).reset_index(drop=True)


def add_target_variables(frame: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    df = frame.sort_values(["firm_id", "model_date"]).copy()

    if TARGET_RETURN_COL in df.columns:
        df[TARGET_RETURN_COL] = pd.to_numeric(df[TARGET_RETURN_COL], errors="coerce")
        if TARGET_RAW_RETURN_COL in df.columns:
            df[TARGET_RAW_RETURN_COL] = pd.to_numeric(df[TARGET_RAW_RETURN_COL], errors="coerce")
        else:
            df[TARGET_RAW_RETURN_COL] = np.nan
        if TARGET_MARKET_RETURN_COL in df.columns:
            df[TARGET_MARKET_RETURN_COL] = pd.to_numeric(df[TARGET_MARKET_RETURN_COL], errors="coerce")
        else:
            df[TARGET_MARKET_RETURN_COL] = np.nan
        df[TARGET_BINARY_COL] = (df[TARGET_RETURN_COL] > 0).astype("Int64")
        df.loc[df[TARGET_RETURN_COL].isna(), TARGET_BINARY_COL] = pd.NA
        return df

    group = df.groupby("firm_id", sort=False)

    future_log_cols: list[str] = []
    for step in range(1, horizon_months + 1):
        col = f"_future_log_ret_{step}"
        df[col] = group["log_ret_m"].shift(-step)
        future_log_cols.append(col)

    future_log_sum = df[future_log_cols].sum(axis=1, min_count=horizon_months)
    df[TARGET_RAW_RETURN_COL] = np.expm1(future_log_sum)

    lead_date = group["model_date"].shift(-horizon_months)
    current_month = df["model_date"].dt.to_period("M").astype("int64")
    lead_month = lead_date.dt.to_period("M").astype("int64")
    valid_consecutive_horizon = (lead_month - current_month) == horizon_months
    df.loc[~valid_consecutive_horizon, TARGET_RAW_RETURN_COL] = np.nan

    market = (
        df.groupby("model_date", sort=True)["ret_m_clean"]
        .mean()
        .clip(lower=-0.999999)
        .rename("market_ret_1m")
        .to_frame()
    )
    market["market_log_ret_1m"] = np.log1p(market["market_ret_1m"])
    market_future_logs = [
        market["market_log_ret_1m"].shift(-step) for step in range(1, horizon_months + 1)
    ]
    market[TARGET_MARKET_RETURN_COL] = np.expm1(sum(market_future_logs))
    df[TARGET_MARKET_RETURN_COL] = df["model_date"].map(market[TARGET_MARKET_RETURN_COL])
    df[TARGET_RETURN_COL] = df[TARGET_RAW_RETURN_COL] - df[TARGET_MARKET_RETURN_COL]
    df[TARGET_BINARY_COL] = (df[TARGET_RETURN_COL] > 0).astype("Int64")
    df.loc[df[TARGET_RETURN_COL].isna(), TARGET_BINARY_COL] = pd.NA

    return df.drop(columns=future_log_cols)


def trailing_compound_return(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    logs = np.log1p(series.clip(lower=-0.999999))
    return np.expm1(logs.rolling(window=window, min_periods=min_periods).sum())


def add_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.sort_values(["firm_id", "model_date"]).copy()
    group = df.groupby("firm_id", sort=False)

    monthly_market_ret = df.groupby("model_date")["ret_m_clean"].transform("mean")
    df["return_1m"] = df["ret_m_clean"]
    df["market_excess_return_1m"] = df["ret_m_clean"] - monthly_market_ret
    df["short_term_reversal"] = -df["return_1m"]

    for window, min_periods in [(3, 2), (6, 3), (12, 6)]:
        df[f"momentum_{window}m"] = group["ret_m_clean"].transform(
            lambda s, w=window, m=min_periods: trailing_compound_return(s, w, m)
        )
        df[f"market_excess_momentum_{window}m"] = group["market_excess_return_1m"].transform(
            lambda s, w=window, m=min_periods: s.rolling(window=w, min_periods=m).sum()
        )
        df[f"volatility_{window}m"] = group["ret_m_clean"].transform(
            lambda s, w=window, m=min_periods: s.rolling(window=w, min_periods=m).std()
        )
        df[f"return_consistency_{window}m"] = group["ret_m_clean"].transform(
            lambda s, w=window, m=min_periods: s.rolling(window=w, min_periods=m).apply(
                lambda values: np.mean(values > 0), raw=True
            )
        )

    df["momentum_acceleration"] = df["momentum_3m"] - safe_divide(df["momentum_12m"], 4.0)
    df["volatility_change"] = df["volatility_3m"] - df["volatility_12m"]

    market_value = col_or_nan(df, "mv_end").abs()
    price = col_or_nan(df, "prc_end").abs()
    volume = col_or_nan(df, "vol_avg")
    dollar_volume = price * volume
    df["log_market_cap"] = np.log1p(market_value.clip(lower=0))
    df["log_price"] = np.log1p(price.clip(lower=0))
    df["log_avg_volume"] = np.log1p(volume.clip(lower=0))
    df["log_dollar_volume"] = np.log1p(dollar_volume.clip(lower=0))
    df["amihud_illiq"] = safe_divide(df["return_1m"].abs(), dollar_volume + 1.0)

    if "vol_d" in df.columns:
        df["daily_vol_to_monthly_vol"] = safe_divide(df["vol_d"], df["volatility_3m"])

    return df.replace([np.inf, -np.inf], np.nan)


def add_accounting_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.sort_values(["firm_id", "model_date"]).copy()
    group = df.groupby("firm_id", sort=False)

    assets = col_or_nan(df, "atq")
    liabilities = col_or_nan(df, "ltq")
    equity = col_or_nan(df, "ceqq")
    equity = equity.where(equity.notna(), assets - liabilities)
    debt = col_or_nan(df, "dlttq") + col_or_nan(df, "dlcq")
    cash = col_or_nan(df, "cheq")
    sales = col_or_nan(df, "saleq")
    cogs = col_or_nan(df, "cogsq")
    income = col_or_nan(df, "niq")
    market_value = col_or_nan(df, "mv_end").abs()

    df["book_equity"] = equity
    df["total_debt"] = debt
    df["net_debt"] = debt - cash
    df["log_assets"] = np.log1p(assets.clip(lower=0))
    df["log_sales"] = signed_log1p(sales)
    df["log_book_equity"] = signed_log1p(equity)
    df["cash_to_assets"] = safe_divide(cash, assets)
    df["debt_to_assets"] = safe_divide(debt, assets)
    df["long_term_debt_to_assets"] = safe_divide(col_or_nan(df, "dlttq"), assets)
    df["current_debt_to_assets"] = safe_divide(col_or_nan(df, "dlcq"), assets)
    df["liabilities_to_assets"] = safe_divide(liabilities, assets)
    df["net_debt_to_assets"] = safe_divide(df["net_debt"], assets)
    df["equity_to_assets"] = safe_divide(equity, assets)
    df["current_ratio"] = safe_divide(col_or_nan(df, "actq"), col_or_nan(df, "lctq"))
    df["book_to_market"] = safe_divide(equity, market_value)
    df["earnings_to_market"] = safe_divide(income, market_value)
    df["sales_to_market"] = safe_divide(sales, market_value)
    df["roa"] = safe_divide(income, assets)
    df["roe"] = safe_divide(income, equity)
    df["profit_margin"] = safe_divide(income, sales)
    df["gross_margin"] = safe_divide(sales - cogs, sales)
    df["asset_turnover"] = safe_divide(sales, assets)
    df["equity_multiplier"] = safe_divide(assets, equity)
    df["sales_to_debt"] = safe_divide(sales, debt)

    if "oibdpq" in df.columns:
        df["operating_margin"] = safe_divide(df["oibdpq"], sales)
        df["operating_return_on_assets"] = safe_divide(df["oibdpq"], assets)
    if "xsgaq" in df.columns:
        df["sgna_to_sales"] = safe_divide(df["xsgaq"], sales)
    if "capxq" in df.columns:
        df["capex_to_assets"] = safe_divide(df["capxq"], assets)
    if "oancfq" in df.columns:
        df["operating_cf_to_assets"] = safe_divide(df["oancfq"], assets)

    growth_sources = first_existing(
        ["saleq", "niq", "atq", "ceqq", "cheq", "ltq", "dlttq", "cogsq", "mv_end"],
        df,
    )
    for col in growth_sources:
        df[f"{col}_growth_qoq"] = stable_growth(df[col], group[col].shift(3))
        df[f"{col}_growth_yoy"] = stable_growth(df[col], group[col].shift(12))

    for col in ["roa", "profit_margin", "gross_margin", "asset_turnover", "debt_to_assets"]:
        df[f"{col}_change_qoq"] = df[col] - group[col].shift(3)
        df[f"{col}_change_yoy"] = df[col] - group[col].shift(12)

    if "saleq_growth_qoq" in df.columns and "saleq_growth_yoy" in df.columns:
        df["sales_growth_acceleration"] = df["saleq_growth_qoq"] - safe_divide(df["saleq_growth_yoy"], 4.0)
    if "niq_growth_qoq" in df.columns and "niq_growth_yoy" in df.columns:
        df["earnings_growth_acceleration"] = df["niq_growth_qoq"] - safe_divide(df["niq_growth_yoy"], 4.0)

    missing_cols = first_existing(ACCOUNTING_MISSINGNESS_COLUMNS, df)
    for col in missing_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    df["accounting_missing_count"] = df[missing_cols].isna().sum(axis=1) if missing_cols else 0
    market_missing_cols = first_existing(["ret_m", "vol_d", "prc_end", "mv_end", "vol_avg"], df)
    df["market_missing_count"] = df[market_missing_cols].isna().sum(axis=1) if market_missing_cols else 0

    return df.replace([np.inf, -np.inf], np.nan)


def add_time_and_industry_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["calendar_year"] = df["model_date"].dt.year.astype("Int64")
    df["calendar_quarter"] = "Q" + df["model_date"].dt.quarter.astype("string")
    df["calendar_month"] = df["model_date"].dt.month.astype("Int64")
    df["calendar_month_name"] = df["model_date"].dt.month_name().astype("string")
    df["is_year_end_month"] = (df["calendar_month"] == 12).astype(int)
    df["is_earnings_season_month"] = df["calendar_month"].isin([1, 4, 7, 10]).astype(int)

    if "sic" in df.columns:
        sic = pd.to_numeric(df["sic"], errors="coerce")
    elif "naics" in df.columns:
        sic = pd.to_numeric(df["naics"], errors="coerce").floordiv(100)
    else:
        sic = pd.Series(np.nan, index=df.index)

    df["sic2"] = sic.floordiv(100).astype("Int64").astype("string").str.zfill(2)
    df["sic2"] = df["sic2"].mask(df["sic2"].isin(["<NA>", "nan", "NaN"]), "Unknown")
    df["sic_division"] = sic.apply(sic_division)
    return df


def add_cross_sectional_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    rank_group = df.groupby("model_date", sort=False)

    for col in first_existing(MONTHLY_RANK_FEATURES, df):
        df[f"{col}_month_rank"] = rank_group[col].rank(pct=True)

    low_is_good = ["debt_to_assets", "liabilities_to_assets", "volatility_6m", "amihud_illiq"]
    for col in first_existing(low_is_good, df):
        rank_col = f"{col}_month_rank"
        if rank_col in df.columns:
            df[f"low_{col}_month_rank"] = 1.0 - df[rank_col]

    industry_group = df.groupby(["model_date", "sic2"], sort=False)
    for col in first_existing(INDUSTRY_RELATIVE_FEATURES, df):
        industry_median = industry_group[col].transform("median")
        df[f"{col}_industry_relative"] = df[col] - industry_median
        df[f"{col}_industry_rank"] = industry_group[col].rank(pct=True)

    quality_inputs = first_existing(
        [
            "roa_month_rank",
            "profit_margin_month_rank",
            "gross_margin_month_rank",
            "asset_turnover_month_rank",
            "low_debt_to_assets_month_rank",
        ],
        df,
    )
    value_inputs = first_existing(
        ["book_to_market_month_rank", "earnings_to_market_month_rank", "sales_to_market_month_rank"],
        df,
    )
    momentum_inputs = first_existing(
        [
            "momentum_6m_month_rank",
            "momentum_12m_month_rank",
            "market_excess_momentum_6m",
            "low_volatility_6m_month_rank",
        ],
        df,
    )
    risk_inputs = first_existing(
        [
            "debt_to_assets_month_rank",
            "liabilities_to_assets_month_rank",
            "volatility_6m_month_rank",
            "amihud_illiq_month_rank",
        ],
        df,
    )

    df["quality_score"] = df[quality_inputs].mean(axis=1) if quality_inputs else np.nan
    df["value_score"] = df[value_inputs].mean(axis=1) if value_inputs else np.nan
    df["momentum_score"] = df[momentum_inputs].mean(axis=1) if momentum_inputs else np.nan
    df["risk_score"] = df[risk_inputs].mean(axis=1) if risk_inputs else np.nan
    df["quality_value_interaction"] = df["quality_score"] * df["value_score"]
    df["quality_momentum_interaction"] = df["quality_score"] * df["momentum_score"]
    df["value_momentum_interaction"] = df["value_score"] * df["momentum_score"]
    df["risk_adjusted_momentum"] = df["momentum_score"] - df["risk_score"]
    df["distress_flag"] = (
        (df["risk_score"] >= 0.75) & (df["quality_score"] <= 0.25)
    ).astype(int)

    return df.replace([np.inf, -np.inf], np.nan)


def engineer_features(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
    df = standardize_inputs(frame)
    df = add_target_variables(df, config.target_horizon_months)
    df = add_time_and_industry_features(df)
    df = add_market_features(df)
    df = add_accounting_features(df)
    df = add_cross_sectional_features(df)
    return df.sort_values(["model_date", "firm_id"]).reset_index(drop=True)


def chronological_split(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> tuple[pd.Series, dict]:
    unique_dates = np.array(sorted(frame["model_date"].dropna().unique()))
    if len(unique_dates) < 3:
        raise ValueError("Need at least three distinct model dates for train/validation/test splitting.")

    train_idx = max(0, int(len(unique_dates) * config.train_fraction) - 1)
    validation_idx = max(
        train_idx + 1,
        int(len(unique_dates) * (config.train_fraction + config.validation_fraction)) - 1,
    )
    validation_idx = min(validation_idx, len(unique_dates) - 2)
    train_cut = pd.Timestamp(unique_dates[train_idx])
    validation_cut = pd.Timestamp(unique_dates[validation_idx])

    split = pd.Series("test", index=frame.index, dtype="object")
    split.loc[frame["model_date"] <= train_cut] = "train"
    split.loc[(frame["model_date"] > train_cut) & (frame["model_date"] <= validation_cut)] = "validation"

    notes = {
        "train_end_date": train_cut.strftime("%Y-%m-%d"),
        "validation_end_date": validation_cut.strftime("%Y-%m-%d"),
        "test_start_date": (validation_cut + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
    }
    return split, notes


def build_numeric_pipeline(config: FeatureEngineeringConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("outlier_capper", QuantileClipper(config.quantile_clip_lower, config.quantile_clip_upper)),
            ("scaler", StandardScaler()),
        ]
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    config: FeatureEngineeringConfig,
) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("cleaner", CategoricalCleaner()),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(config), numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def select_feature_columns(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> tuple[list[str], list[str]]:
    excluded = set(IDENTIFIER_COLUMNS) | set(DATE_COLUMNS)
    excluded |= {
        TARGET_RETURN_COL,
        TARGET_BINARY_COL,
        TARGET_RAW_RETURN_COL,
        TARGET_MARKET_RETURN_COL,
        SP500_RETURN_COL,
        SP500_EXCESS_RETURN_COL,
        SP500_BINARY_COL,
        "ret_m_clean",
        "log_ret_m",
    }
    excluded |= {col for col in frame.columns if col.startswith("target_")}

    nonmissing_rate = frame.notna().mean()
    numeric_features = [
        col
        for col in frame.select_dtypes(include=[np.number, "Int64", "Float64"]).columns
        if col not in excluded and nonmissing_rate[col] >= config.min_feature_nonmissing_rate
    ]
    categorical_features = [
        col
        for col in first_existing(CATEGORICAL_FEATURE_CANDIDATES, frame)
        if col not in excluded and nonmissing_rate[col] >= config.min_feature_nonmissing_rate
    ]
    return numeric_features, categorical_features


def add_unsupervised_features(
    model_df: pd.DataFrame,
    config: FeatureEngineeringConfig,
) -> tuple[pd.DataFrame, dict]:
    df = model_df.copy()
    train_mask = df["split"] == "train"
    unsupervised_features = first_existing(UNSUPERVISED_FEATURE_CANDIDATES, df)
    artifacts: dict[str, object] = {
        "unsupervised_features": unsupervised_features,
        "cluster_preprocessor": None,
        "pca": None,
        "kmeans": None,
    }

    if len(unsupervised_features) < 2 or train_mask.sum() < max(10, config.kmeans_clusters):
        df["firm_profile_cluster"] = "cluster_unavailable"
        return df, artifacts

    cluster_preprocessor = build_numeric_pipeline(config)
    train_matrix = cluster_preprocessor.fit_transform(df.loc[train_mask, unsupervised_features])
    all_matrix = cluster_preprocessor.transform(df[unsupervised_features])

    n_components = min(config.pca_components, train_matrix.shape[1], max(1, train_matrix.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(train_matrix)
    pca_values = pca.transform(all_matrix)
    for idx in range(n_components):
        df[f"feature_pca{idx + 1}"] = pca_values[:, idx]

    n_clusters = min(config.kmeans_clusters, train_matrix.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_STATE)
    kmeans.fit(train_matrix)
    df["firm_profile_cluster"] = ["cluster_" + str(label) for label in kmeans.predict(all_matrix)]

    cluster_profile = (
        df.loc[train_mask]
        .groupby("firm_profile_cluster")[unsupervised_features]
        .median(numeric_only=True)
        .round(4)
        .sort_index()
    )
    cluster_profile["row_count"] = df.loc[train_mask].groupby("firm_profile_cluster").size()

    artifacts.update(
        {
            "cluster_preprocessor": cluster_preprocessor,
            "pca": pca,
            "kmeans": kmeans,
            "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "cluster_profile": cluster_profile,
        }
    )
    return df, artifacts


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("split")
        .agg(
            rows=("firm_id", "size"),
            firms=("firm_id", "nunique"),
            start_date=("model_date", "min"),
            end_date=("model_date", "max"),
            target_excess_mean=(TARGET_RETURN_COL, "mean"),
            target_excess_median=(TARGET_RETURN_COL, "median"),
            outperformance_rate=(TARGET_BINARY_COL, "mean"),
            sp500_excess_mean=(SP500_EXCESS_RETURN_COL, "mean")
            if SP500_EXCESS_RETURN_COL in frame.columns
            else (TARGET_RETURN_COL, "mean"),
            sp500_outperformance_rate=(SP500_BINARY_COL, "mean")
            if SP500_BINARY_COL in frame.columns
            else (TARGET_BINARY_COL, "mean"),
        )
        .reset_index()
    )
    summary["start_date"] = summary["start_date"].dt.strftime("%Y-%m-%d")
    summary["end_date"] = summary["end_date"].dt.strftime("%Y-%m-%d")
    return summary


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_dynamic_report(
    output_dir: Path,
    metadata: dict,
    summary: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    lines = [
        "# Feature Engineering Output Report",
        "",
        "## Rubric Alignment",
        "",
        "This feature engineering step is designed for the advanced rubric tier: it creates predictive market, accounting, industry-relative, composite, and unsupervised features while fitting all learned preprocessing artifacts on the training period only.",
        "",
        "## Target",
        "",
        f"- Regression target: `{TARGET_RETURN_COL}` = next 3-month stock return minus next 3-month equal-weight market return.",
        f"- Classification target: `{TARGET_BINARY_COL}` = 1 when the stock outperforms the equal-weight market proxy over the next quarter.",
        f"- Additional benchmark target: `{SP500_EXCESS_RETURN_COL}` = next 3-month stock return minus next 3-month S&P 500 return.",
        f"- Additional classification target: `{SP500_BINARY_COL}` = 1 when the stock outperforms the S&P 500 over the next quarter.",
        "- The target uses future months only; current-month features are not used in target construction.",
        "",
        "## Engineered Feature Families",
        "",
        "- Market behavior: 1/3/6/12-month return, excess-return momentum, volatility, return consistency, reversal, liquidity, and Amihud-style illiquidity.",
        "- Accounting strength: profitability, leverage, liquidity, valuation, efficiency, balance-sheet composition, and growth/change features.",
        "- Relative positioning: monthly cross-sectional ranks and industry-relative deviations for size, value, profitability, leverage, momentum, and risk.",
        "- Composite signals: quality, value, momentum, risk, risk-adjusted momentum, interaction terms, and a distress flag.",
        "- Unsupervised features: PCA components and KMeans profile clusters fitted on the chronological training split only.",
        "",
        "## Preprocessing",
        "",
        "- Numeric features use median imputation, train-fitted 1st/99th percentile clipping, and standard scaling.",
        "- Categorical features use missing-category handling and one-hot encoding with unknown-category support.",
        "- Train/validation/test splits are chronological to reduce look-ahead bias.",
        "",
        "## Output Summary",
        "",
        f"- Input rows: {metadata['input_rows']:,}",
        f"- Modelable rows: {metadata['model_rows']:,}",
        f"- Numeric features: {len(numeric_features):,}",
        f"- Categorical features: {len(categorical_features):,}",
        f"- S&P 500 source: `{metadata.get('sp500_input_path') or 'not provided'}`",
        f"- Rows with S&P 500 benchmark target: {metadata.get('sp500_target_nonmissing_rows', 0):,}",
        f"- Output directory: `{output_dir}`",
        "",
        "## Split Summary",
        "",
        markdown_table(summary),
        "",
        "## Key Output Files",
        "",
        "- `engineered_features.csv.gz`: full engineered dataset with targets and split labels.",
        "- `feature_train.csv.gz`, `feature_validation.csv.gz`, `feature_test.csv.gz`: chronological modeling splits.",
        "- `feature_preprocessor.joblib`: sklearn preprocessing transformer fitted on training data only.",
        "- `feature_unsupervised_artifacts.joblib`: PCA/KMeans artifacts fitted on training data only.",
        "- `feature_metadata.json`: feature lists, counts, split dates, and configuration.",
        "",
    ]
    (output_dir / "feature_engineering_output_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_feature_engineering(config: FeatureEngineeringConfig) -> dict:
    input_path = resolve_input_path(Path(config.input_path))
    sp500_path = resolve_sp500_path(config.sp500_input_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_modeling_data(input_path)
    engineered = engineer_features(raw, config)
    if sp500_path is not None:
        engineered = add_sp500_targets(engineered, sp500_path, config.target_horizon_months)
    model_df = engineered.loc[engineered[TARGET_RETURN_COL].notna()].copy()
    if model_df.empty:
        raise ValueError("No rows have a nonmissing next-quarter excess-return target.")

    split, split_notes = chronological_split(model_df, config)
    model_df["split"] = split
    model_df, unsupervised_artifacts = add_unsupervised_features(model_df, config)

    numeric_features, categorical_features = select_feature_columns(model_df, config)
    feature_columns = numeric_features + categorical_features
    train_df = model_df.loc[model_df["split"] == "train"].copy()
    validation_df = model_df.loc[model_df["split"] == "validation"].copy()
    test_df = model_df.loc[model_df["split"] == "test"].copy()

    preprocessor = build_preprocessor(numeric_features, categorical_features, config)
    preprocessor.fit(train_df[feature_columns], train_df[TARGET_RETURN_COL])

    summary = split_summary(model_df)
    cluster_profile = unsupervised_artifacts.pop("cluster_profile", None)

    model_df.to_csv(output_dir / "engineered_features.csv.gz", index=False, compression="gzip")
    train_df.to_csv(output_dir / "feature_train.csv.gz", index=False, compression="gzip")
    validation_df.to_csv(output_dir / "feature_validation.csv.gz", index=False, compression="gzip")
    test_df.to_csv(output_dir / "feature_test.csv.gz", index=False, compression="gzip")
    summary.to_csv(output_dir / "feature_split_summary.csv", index=False)
    if cluster_profile is not None:
        cluster_profile.to_csv(output_dir / "feature_cluster_profile.csv")

    joblib.dump(preprocessor, output_dir / "feature_preprocessor.joblib")
    joblib.dump(unsupervised_artifacts, output_dir / "feature_unsupervised_artifacts.joblib")

    metadata = {
        "config": asdict(config),
        "resolved_input_path": str(input_path),
        "sp500_input_path": str(sp500_path) if sp500_path is not None else None,
        "input_rows": int(len(raw)),
        "engineered_rows": int(len(engineered)),
        "model_rows": int(len(model_df)),
        "sp500_target_nonmissing_rows": int(model_df[SP500_EXCESS_RETURN_COL].notna().sum())
        if SP500_EXCESS_RETURN_COL in model_df.columns
        else 0,
        "split_notes": split_notes,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_feature_columns": feature_columns,
        "target_columns": [
            TARGET_RETURN_COL,
            TARGET_BINARY_COL,
            SP500_EXCESS_RETURN_COL,
            SP500_BINARY_COL,
        ],
        "pca_explained_variance_ratio": unsupervised_artifacts.get("pca_explained_variance_ratio"),
    }
    with (output_dir / "feature_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    write_dynamic_report(output_dir, metadata, summary, numeric_features, categorical_features)
    return metadata


def parse_args() -> FeatureEngineeringConfig:
    parser = argparse.ArgumentParser(description="Run Project 4 advanced feature engineering.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to cleaned Compustat-CRSP matched CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for engineered outputs.")
    parser.add_argument(
        "--sp500-input",
        default="",
        help="Optional S&P 500 CSV with date plus price/close or monthly return. Defaults to auto-detecting data/sp500_daily.csv.",
    )
    parser.add_argument("--horizon", type=int, default=3, help="Forward return horizon in months.")
    parser.add_argument("--pca-components", type=int, default=5, help="Maximum train-fitted PCA components.")
    parser.add_argument("--kmeans-clusters", type=int, default=5, help="Maximum train-fitted KMeans clusters.")
    args = parser.parse_args()
    return FeatureEngineeringConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        sp500_input_path=args.sp500_input,
        target_horizon_months=args.horizon,
        pca_components=args.pca_components,
        kmeans_clusters=args.kmeans_clusters,
    )


def main() -> None:
    config = parse_args()
    try:
        metadata = run_feature_engineering(config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print("Feature engineering complete.")
    print(f"Input file: {metadata['resolved_input_path']}")
    if metadata.get("sp500_input_path"):
        print(f"S&P 500 file: {metadata['sp500_input_path']}")
    print(f"Model rows: {metadata['model_rows']:,}")
    print(f"S&P 500 target rows: {metadata.get('sp500_target_nonmissing_rows', 0):,}")
    print(f"Numeric/categorical features: {len(metadata['numeric_features']):,} / {len(metadata['categorical_features']):,}")
    print(f"Outputs written to: {Path(config.output_dir).resolve()}")


if __name__ == "__main__":
    main()
