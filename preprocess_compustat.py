#!/usr/bin/env python3
"""Clean and preprocess the Compustat quarterly dataset for Project 4.

The script is intentionally end-to-end and reproducible:
1. inspect and profile the raw data
2. remove exact and key duplicates
3. fix types and inconsistent strings
4. treat invalid financial values
5. engineer finance ratios, growth features, and an earnings-growth target
6. split chronologically into train/validation/test sets
7. fit preprocessing, PCA, and KMeans artifacts on the training split only
8. export cleaned data, split files, figures, and a markdown report
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


if __name__ == "__main__":
    # Make custom classes loadable from joblib artifacts after running as a script.
    sys.modules["preprocess_compustat"] = sys.modules[__name__]


RAW_FILE = Path("zyukuvp88bxlctvl.csv")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
RANDOM_STATE = 42
TARGET_COL = "target_next_niq_growth"
CRSP_TARGET_COL = "target_crsp_excess_return_3m"
CRSP_REQUIRED_COLUMNS = {
    "PERMNO",
    "HdrCUSIP",
    "PrimaryExch",
    "ShareType",
    "Ticker",
    "PERMCO",
    "DlyCalDt",
    "DlyPrc",
    "DlyRet",
    "DlyVol",
    "ShrOut",
}


STRING_COLUMNS = [
    "costat",
    "curcdq",
    "datafmt",
    "indfmt",
    "consol",
    "tic",
    "cusip",
    "gvkey",
]

CORE_NUMERIC_COLUMNS = [
    "actq",
    "atq",
    "cogsq",
    "dlttq",
    "lctq",
    "ltq",
    "niq",
    "oibdpq",
    "saleq",
    "xsgaq",
    "capxy",
    "oancfy",
    "rstcheq",
]

NONNEGATIVE_STOCK_COLUMNS = [
    "actq",
    "atq",
    "dlttq",
    "lctq",
    "ltq",
    "rstcheq",
    "rstcheltq",
]

NONNEGATIVE_FLOW_COLUMNS = [
    "saleq",
    "cogsq",
    "xsgaq",
    "capxy",
    "dvintfq",
    "npatq",
]

HIGH_MISSING_THRESHOLD = 0.90


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


QuantileClipper.__module__ = "preprocess_compustat"


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Convert categorical inputs to strings and fill missing values."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            out = X.copy()
        else:
            out = pd.DataFrame(X)
        out = out.astype("object").where(pd.notna(out), "Unknown")
        return out.astype(str)


CategoricalCleaner.__module__ = "preprocess_compustat"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_header(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return set(handle.readline().strip().split(","))


def detect_raw_duplicates(raw_path: Path, raw_hash: str) -> list[str]:
    duplicates: list[str] = []
    raw_size = raw_path.stat().st_size
    for path in Path(".").glob("*.csv"):
        if path.resolve() == raw_path.resolve():
            continue
        if path.stat().st_size == raw_size and sha256(path) == raw_hash:
            duplicates.append(path.name)
    return duplicates


def detect_crsp_file() -> Path | None:
    candidates: list[Path] = []
    for path in Path(".").glob("*.csv"):
        if path.resolve() == RAW_FILE.resolve():
            continue
        header = csv_header(path)
        if CRSP_REQUIRED_COLUMNS.issubset(header):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def clean_string_series(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    return out.mask(out.isin(["", "NAN", "NONE", "NULL", "<NA>"]))


def safe_divide(numerator: pd.Series, denominator: pd.Series, min_abs: float = 1e-9) -> pd.Series:
    den = denominator.where(denominator.abs() > min_abs)
    out = numerator / den
    return out.replace([np.inf, -np.inf], np.nan)


def stable_growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Growth rate that is stable around zero and negative denominators."""

    out = (current - previous) / (previous.abs() + 1.0)
    return out.replace([np.inf, -np.inf], np.nan)


def first_existing(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in frame.columns]


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


def read_raw_data(path: Path) -> pd.DataFrame:
    dtype = {col: "string" for col in STRING_COLUMNS}
    return pd.read_csv(path, dtype=dtype, low_memory=False)


def standardize_and_type(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = frame.copy()
    notes: dict[str, object] = {}

    for col in first_existing(STRING_COLUMNS, df):
        df[col] = clean_string_series(df[col])

    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    notes["invalid_datadate_rows"] = int(df["datadate"].isna().sum())

    if "gvkey" in df.columns:
        df["gvkey_str"] = (
            df["gvkey"]
            .astype("string")
            .str.extract(r"(\d+)", expand=False)
            .str.zfill(6)
        )
    else:
        df["gvkey_str"] = pd.NA
    notes["missing_gvkey_rows"] = int(df["gvkey_str"].isna().sum())

    for col in df.columns:
        if col in STRING_COLUMNS or col in ["datadate", "gvkey_str"]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "sic" in df.columns:
        df["sic"] = pd.to_numeric(df["sic"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["gvkey_str", "datadate"]).copy()
    notes["rows_dropped_bad_key_or_date"] = int(before - len(df))
    return df, notes


def remove_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = frame.copy()
    notes: dict[str, object] = {}
    notes["exact_duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    key_cols = ["gvkey_str", "datadate"]
    notes["duplicate_key_rows_before_resolution"] = int(df.duplicated(key_cols).sum())
    duplicate_groups = df[df.duplicated(key_cols, keep=False)].groupby(key_cols).size()
    notes["duplicate_key_groups_before_resolution"] = int(len(duplicate_groups))

    df["_non_missing_count"] = df.notna().sum(axis=1)
    df = df.sort_values(
        ["gvkey_str", "datadate", "_non_missing_count"],
        ascending=[True, True, False],
    )
    df = df.drop_duplicates(key_cols, keep="first").drop(columns="_non_missing_count")
    notes["rows_after_duplicate_resolution"] = int(len(df))
    return df, notes


def treat_invalid_values(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = frame.copy()
    records: list[dict[str, object]] = []

    for col in first_existing(NONNEGATIVE_STOCK_COLUMNS, df):
        invalid = df[col].lt(0)
        records.append(
            {
                "column": col,
                "rule": "balance_sheet_stock_must_be_nonnegative",
                "invalid_count": int(invalid.sum()),
            }
        )
        df.loc[invalid, col] = np.nan

    for col in first_existing(NONNEGATIVE_FLOW_COLUMNS, df):
        invalid = df[col].lt(0)
        records.append(
            {
                "column": col,
                "rule": "nonnegative_accounting_flow_set_to_missing",
                "invalid_count": int(invalid.sum()),
            }
        )
        df.loc[invalid, col] = np.nan

    return df, pd.DataFrame(records)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.sort_values(["gvkey_str", "datadate"]).copy()

    df["calendar_year"] = df["datadate"].dt.year.astype("Int64")
    df["calendar_quarter"] = "Q" + df["datadate"].dt.quarter.astype("string")
    df["calendar_month"] = df["datadate"].dt.month.astype("Int64")
    df["is_calendar_year_end"] = (df["calendar_month"] == 12).astype(int)

    df["sic2"] = (
        pd.to_numeric(df["sic"], errors="coerce")
        .floordiv(100)
        .astype("Int64")
        .astype("string")
        .str.zfill(2)
    )
    df["sic2"] = df["sic2"].mask(df["sic2"].isin(["<NA>", "nan"]), "Unknown")
    df["sic_division"] = pd.to_numeric(df["sic"], errors="coerce").apply(sic_division)

    group = df.groupby("gvkey_str", sort=False)
    year_group = df.groupby(["gvkey_str", "calendar_year"], sort=False)

    if "oancfy" in df.columns:
        prev_oancfy = year_group["oancfy"].shift(1)
        df["oancfq_approx"] = df["oancfy"] - prev_oancfy
        df.loc[prev_oancfy.isna(), "oancfq_approx"] = df.loc[prev_oancfy.isna(), "oancfy"]
    else:
        df["oancfq_approx"] = np.nan

    if "capxy" in df.columns:
        prev_capxy = year_group["capxy"].shift(1)
        df["capxq_approx"] = df["capxy"] - prev_capxy
        df.loc[prev_capxy.isna(), "capxq_approx"] = df.loc[prev_capxy.isna(), "capxy"]
        df.loc[df["capxq_approx"].lt(0), "capxq_approx"] = np.nan
    else:
        df["capxq_approx"] = np.nan

    df["book_equity_proxy"] = df["atq"] - df["ltq"]
    df["log_assets"] = np.log1p(df["atq"].clip(lower=0))
    df["liabilities_to_assets"] = safe_divide(df["ltq"], df["atq"])
    df["debt_to_assets"] = safe_divide(df["dlttq"], df["atq"])
    df["equity_to_assets"] = safe_divide(df["book_equity_proxy"], df["atq"])
    df["current_ratio"] = safe_divide(df["actq"], df["lctq"])
    df["cash_like_to_assets"] = safe_divide(df.get("rstcheq", np.nan), df["atq"])
    df["roa"] = safe_divide(df["niq"], df["atq"])
    df["roe_proxy"] = safe_divide(df["niq"], df["book_equity_proxy"])
    df["profit_margin"] = safe_divide(df["niq"], df["saleq"])
    df["ebitda_margin"] = safe_divide(df["oibdpq"], df["saleq"])
    df["gross_margin"] = safe_divide(df["saleq"] - df["cogsq"], df["saleq"])
    df["sgna_to_sales"] = safe_divide(df["xsgaq"], df["saleq"])
    df["asset_turnover"] = safe_divide(df["saleq"], df["atq"])
    df["operating_cf_to_assets"] = safe_divide(df["oancfq_approx"], df["atq"])
    df["capex_to_assets"] = safe_divide(df["capxq_approx"], df["atq"])

    for col in first_existing(CORE_NUMERIC_COLUMNS, df):
        df[f"{col}_missing"] = df[col].isna().astype(int)

    prev_saleq = group["saleq"].shift(1)
    prev_niq = group["niq"].shift(1)
    prev_atq = group["atq"].shift(1)
    df["sales_growth_qoq"] = stable_growth(df["saleq"], prev_saleq)
    df["niq_growth_qoq"] = stable_growth(df["niq"], prev_niq)
    df["asset_growth_qoq"] = stable_growth(df["atq"], prev_atq)

    next_niq = group["niq"].shift(-1)
    next_saleq = group["saleq"].shift(-1)
    next_date = group["datadate"].shift(-1)
    days_to_next = (next_date - df["datadate"]).dt.days
    valid_next_quarter = days_to_next.between(45, 190)
    df["days_to_next_observation"] = days_to_next
    df["target_next_niq"] = next_niq
    df[TARGET_COL] = stable_growth(next_niq, df["niq"]).where(valid_next_quarter)
    df["target_next_sale_growth"] = stable_growth(next_saleq, df["saleq"]).where(valid_next_quarter)
    df["target_earnings_growth_positive"] = (df[TARGET_COL] > 0).astype("Int64")
    df.loc[df[TARGET_COL].isna(), "target_earnings_growth_positive"] = pd.NA

    return df.replace([np.inf, -np.inf], np.nan)


def chronological_split(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    unique_dates = np.array(sorted(frame["datadate"].dropna().unique()))
    train_cut = unique_dates[int(len(unique_dates) * 0.70) - 1]
    val_cut = unique_dates[int(len(unique_dates) * 0.85) - 1]

    split = pd.Series("test", index=frame.index, dtype="object")
    split.loc[frame["datadate"] <= train_cut] = "train"
    split.loc[(frame["datadate"] > train_cut) & (frame["datadate"] <= val_cut)] = "validation"

    notes = {
        "train_end_date": pd.Timestamp(train_cut).strftime("%Y-%m-%d"),
        "validation_end_date": pd.Timestamp(val_cut).strftime("%Y-%m-%d"),
    }
    return split, notes


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric_features = first_existing(
        [
            "actq",
            "atq",
            "cogsq",
            "dlttq",
            "lctq",
            "ltq",
            "niq",
            "oibdpq",
            "saleq",
            "xsgaq",
            "capxq_approx",
            "oancfq_approx",
            "calendar_year",
            "calendar_month",
            "is_calendar_year_end",
            "book_equity_proxy",
            "log_assets",
            "liabilities_to_assets",
            "debt_to_assets",
            "equity_to_assets",
            "current_ratio",
            "cash_like_to_assets",
            "roa",
            "roe_proxy",
            "profit_margin",
            "ebitda_margin",
            "gross_margin",
            "sgna_to_sales",
            "asset_turnover",
            "operating_cf_to_assets",
            "capex_to_assets",
            "sales_growth_qoq",
            "niq_growth_qoq",
            "asset_growth_qoq",
            "cluster_pca1",
            "cluster_pca2",
        ],
        frame,
    )
    numeric_features += [col for col in frame.columns if col.endswith("_missing")]

    categorical_features = first_existing(
        [
            "costat",
            "curcdq",
            "sic2",
            "sic_division",
            "calendar_quarter",
            "finance_cluster",
        ],
        frame,
    )

    cluster_features = first_existing(
        [
            "log_assets",
            "liabilities_to_assets",
            "debt_to_assets",
            "equity_to_assets",
            "current_ratio",
            "roa",
            "profit_margin",
            "ebitda_margin",
            "gross_margin",
            "asset_turnover",
            "operating_cf_to_assets",
            "capex_to_assets",
            "sales_growth_qoq",
            "asset_growth_qoq",
        ],
        frame,
    )
    return numeric_features, categorical_features, cluster_features


def build_numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("outlier_capper", QuantileClipper(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
        ]
    )


def fit_unsupervised_features(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    cluster_features: list[str],
) -> tuple[pd.DataFrame, Pipeline, PCA, KMeans, pd.DataFrame]:
    df = full_df.copy()
    cluster_preprocessor = build_numeric_pipeline()
    train_cluster_matrix = cluster_preprocessor.fit_transform(train_df[cluster_features])
    all_cluster_matrix = cluster_preprocessor.transform(df[cluster_features])

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(train_cluster_matrix)
    pca_values = pca.transform(all_cluster_matrix)
    df["cluster_pca1"] = pca_values[:, 0]
    df["cluster_pca2"] = pca_values[:, 1]

    kmeans = KMeans(n_clusters=6, n_init=20, random_state=RANDOM_STATE)
    kmeans.fit(train_cluster_matrix)
    df["finance_cluster"] = ["cluster_" + str(label) for label in kmeans.predict(all_cluster_matrix)]

    profiled = df.loc[train_df.index].copy()
    cluster_profile = (
        profiled.groupby("finance_cluster")[cluster_features]
        .median(numeric_only=True)
        .round(4)
        .sort_index()
    )
    cluster_profile["row_count"] = profiled.groupby("finance_cluster").size()
    return df, cluster_preprocessor, pca, kmeans, cluster_profile


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("cleaner", CategoricalCleaner()),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def forward_log_return(log_returns: pd.Series, window: int) -> pd.Series:
    return log_returns.shift(-1).rolling(window=window, min_periods=window).sum().shift(-(window - 1))


def compound_log_return(log_returns: pd.Series, window: int, min_periods: int) -> pd.Series:
    return log_returns.rolling(window=window, min_periods=min_periods).sum()


def preprocess_crsp_monthly(path: Path) -> tuple[pd.DataFrame, dict]:
    usecols = list(CRSP_REQUIRED_COLUMNS)
    dtype = {
        "PERMNO": "int32",
        "PERMCO": "int32",
        "HdrCUSIP": "string",
        "PrimaryExch": "string",
        "ShareType": "string",
        "Ticker": "string",
    }
    crsp = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        parse_dates=["DlyCalDt"],
        low_memory=False,
    )
    profile = {
        "crsp_file": path.name,
        "crsp_raw_rows": int(len(crsp)),
        "crsp_raw_columns": int(crsp.shape[1]),
    }

    for col in ["HdrCUSIP", "PrimaryExch", "ShareType", "Ticker"]:
        crsp[col] = clean_string_series(crsp[col])
    crsp["cusip8"] = crsp["HdrCUSIP"].str.extract(r"([A-Z0-9]+)", expand=False).str.zfill(8).str[:8]
    crsp["DlyRet"] = pd.to_numeric(crsp["DlyRet"], errors="coerce")
    crsp["DlyPrc"] = pd.to_numeric(crsp["DlyPrc"], errors="coerce")
    crsp["DlyVol"] = pd.to_numeric(crsp["DlyVol"], errors="coerce")
    crsp["ShrOut"] = pd.to_numeric(crsp["ShrOut"], errors="coerce")
    crsp["price_abs"] = crsp["DlyPrc"].abs()
    crsp["market_cap"] = crsp["price_abs"] * crsp["ShrOut"]
    crsp["dollar_volume"] = crsp["price_abs"] * crsp["DlyVol"]
    crsp["month_end"] = crsp["DlyCalDt"].dt.to_period("M").dt.to_timestamp("M")
    crsp["ret_plus"] = 1.0 + crsp["DlyRet"].clip(lower=-0.999999)

    before = len(crsp)
    crsp = crsp.dropna(subset=["PERMNO", "DlyCalDt", "cusip8", "month_end"]).copy()
    profile["crsp_rows_dropped_bad_key_or_date"] = int(before - len(crsp))
    profile["crsp_min_date"] = crsp["DlyCalDt"].min().strftime("%Y-%m-%d")
    profile["crsp_max_date"] = crsp["DlyCalDt"].max().strftime("%Y-%m-%d")
    profile["crsp_unique_permnos"] = int(crsp["PERMNO"].nunique())
    profile["crsp_unique_cusip8"] = int(crsp["cusip8"].nunique())

    crsp = crsp.sort_values(["PERMNO", "DlyCalDt"])
    grouped = crsp.groupby(["PERMNO", "month_end"], sort=False)
    monthly = grouped.agg(
        cusip8=("cusip8", "last"),
        crsp_ticker=("Ticker", "last"),
        permco=("PERMCO", "last"),
        primary_exch=("PrimaryExch", "last"),
        share_type=("ShareType", "last"),
        crsp_obs_days=("DlyRet", "count"),
        crsp_ret_1m=("ret_plus", "prod"),
        crsp_vol_1m=("DlyRet", "std"),
        crsp_avg_volume_1m=("DlyVol", "mean"),
        crsp_avg_dollar_volume_1m=("dollar_volume", "mean"),
        crsp_price_last=("price_abs", "last"),
        crsp_market_cap_last=("market_cap", "last"),
        crsp_date_last=("DlyCalDt", "last"),
    ).reset_index()
    monthly["crsp_ret_1m"] = monthly["crsp_ret_1m"].where(monthly["crsp_obs_days"] > 0) - 1.0
    monthly["crsp_log_ret_1m"] = np.log1p(monthly["crsp_ret_1m"].clip(lower=-0.999999))

    monthly = monthly.sort_values(["PERMNO", "month_end"])
    permno_group = monthly.groupby("PERMNO", sort=False)
    for window, min_periods in [(3, 2), (6, 3), (12, 6)]:
        log_sum = permno_group["crsp_log_ret_1m"].transform(
            lambda s, w=window, m=min_periods: compound_log_return(s, w, m)
        )
        monthly[f"crsp_ret_{window}m"] = np.expm1(log_sum)

    monthly["target_crsp_return_3m"] = np.expm1(
        permno_group["crsp_log_ret_1m"].transform(lambda s: forward_log_return(s, 3))
    )

    market_monthly = monthly.groupby("month_end", sort=True)["crsp_ret_1m"].mean().reset_index()
    market_monthly["market_log_ret_1m"] = np.log1p(market_monthly["crsp_ret_1m"].clip(lower=-0.999999))
    market_monthly["market_return_fwd_3m"] = np.expm1(
        forward_log_return(market_monthly["market_log_ret_1m"], 3)
    )
    monthly = monthly.merge(
        market_monthly[["month_end", "market_return_fwd_3m"]],
        on="month_end",
        how="left",
    )
    monthly[CRSP_TARGET_COL] = monthly["target_crsp_return_3m"] - monthly["market_return_fwd_3m"]

    duplicate_cusip_month = int(monthly.duplicated(["cusip8", "month_end"]).sum())
    profile["crsp_monthly_rows_before_cusip_dedupe"] = int(len(monthly))
    profile["crsp_duplicate_cusip_month_rows"] = duplicate_cusip_month
    monthly = monthly.sort_values(["cusip8", "month_end", "crsp_market_cap_last"])
    monthly = monthly.drop_duplicates(["cusip8", "month_end"], keep="last").copy()
    profile["crsp_monthly_rows"] = int(len(monthly))
    profile["crsp_target_nonmissing_rows"] = int(monthly[CRSP_TARGET_COL].notna().sum())
    return monthly, profile


def add_crsp_merge_outputs(
    engineered: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    crsp_monthly: pd.DataFrame,
) -> dict:
    comp = engineered.copy()
    comp["cusip8"] = comp["cusip"].astype("string").str.strip().str.upper().str[:8]
    comp["month_end"] = comp["datadate"].dt.to_period("M").dt.to_timestamp("M")

    crsp_feature_cols = [
        "cusip8",
        "month_end",
        "PERMNO",
        "permco",
        "crsp_ticker",
        "primary_exch",
        "share_type",
        "crsp_obs_days",
        "crsp_ret_1m",
        "crsp_ret_3m",
        "crsp_ret_6m",
        "crsp_ret_12m",
        "crsp_vol_1m",
        "crsp_avg_volume_1m",
        "crsp_avg_dollar_volume_1m",
        "crsp_price_last",
        "crsp_market_cap_last",
        "market_return_fwd_3m",
        "target_crsp_return_3m",
        CRSP_TARGET_COL,
    ]
    merged = comp.merge(
        crsp_monthly[first_existing(crsp_feature_cols, crsp_monthly)],
        on=["cusip8", "month_end"],
        how="left",
    )

    crsp_numeric = first_existing(
        [
            "crsp_obs_days",
            "crsp_ret_1m",
            "crsp_ret_3m",
            "crsp_ret_6m",
            "crsp_ret_12m",
            "crsp_vol_1m",
            "crsp_avg_volume_1m",
            "crsp_avg_dollar_volume_1m",
            "crsp_price_last",
            "crsp_market_cap_last",
        ],
        merged,
    )
    crsp_categorical = first_existing(["primary_exch", "share_type"], merged)
    merged_numeric_features = numeric_features + [col for col in crsp_numeric if col not in numeric_features]
    merged_categorical_features = categorical_features + [
        col for col in crsp_categorical if col not in categorical_features
    ]

    merged_model = merged.loc[merged[CRSP_TARGET_COL].notna()].copy()
    split, split_notes = chronological_split(merged_model)
    merged_model["crsp_split"] = split

    train_df = merged_model.loc[merged_model["crsp_split"] == "train"].copy()
    validation_df = merged_model.loc[merged_model["crsp_split"] == "validation"].copy()
    test_df = merged_model.loc[merged_model["crsp_split"] == "test"].copy()

    merged_preprocessor = build_preprocessor(merged_numeric_features, merged_categorical_features)
    merged_preprocessor.fit(
        train_df[merged_numeric_features + merged_categorical_features],
        train_df[CRSP_TARGET_COL],
    )

    split_summary = (
        merged_model.groupby("crsp_split")
        .agg(
            rows=("gvkey_str", "size"),
            companies=("gvkey_str", "nunique"),
            permnos=("PERMNO", "nunique"),
            start_date=("datadate", "min"),
            end_date=("datadate", "max"),
            target_mean=(CRSP_TARGET_COL, "mean"),
            target_median=(CRSP_TARGET_COL, "median"),
        )
        .reset_index()
    )
    split_summary["start_date"] = split_summary["start_date"].dt.strftime("%Y-%m-%d")
    split_summary["end_date"] = split_summary["end_date"].dt.strftime("%Y-%m-%d")

    merged.to_csv(OUTPUT_DIR / "merged_crsp_compustat.csv.gz", index=False, compression="gzip")
    train_df.to_csv(OUTPUT_DIR / "crsp_train_clean.csv.gz", index=False, compression="gzip")
    validation_df.to_csv(OUTPUT_DIR / "crsp_validation_clean.csv.gz", index=False, compression="gzip")
    test_df.to_csv(OUTPUT_DIR / "crsp_test_clean.csv.gz", index=False, compression="gzip")
    split_summary.to_csv(OUTPUT_DIR / "crsp_split_summary.csv", index=False)
    joblib.dump(merged_preprocessor, OUTPUT_DIR / "crsp_preprocessing_pipeline.joblib")

    return {
        "merge_rows": int(len(merged)),
        "merge_matched_rows": int(merged["PERMNO"].notna().sum()),
        "merge_model_rows": int(len(merged_model)),
        "crsp_train_rows": int(len(train_df)),
        "crsp_validation_rows": int(len(validation_df)),
        "crsp_test_rows": int(len(test_df)),
        "crsp_split_notes": split_notes,
        "crsp_numeric_features": merged_numeric_features,
        "crsp_categorical_features": merged_categorical_features,
        "crsp_split_summary": split_summary.to_dict(orient="records"),
    }


def write_figures(
    frame: pd.DataFrame,
    model_frame: pd.DataFrame,
    numeric_features: list[str],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    missing_pct = frame.isna().mean().sort_values(ascending=False).head(20) * 100
    plt.figure(figsize=(10, 7))
    missing_pct.sort_values().plot(kind="barh", color="#4C78A8")
    plt.xlabel("Missing values (%)")
    plt.title("Top 20 Missingness Rates")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "missingness_top20.png", dpi=180)
    plt.close()

    corr_cols = [
        col
        for col in [
            "log_assets",
            "liabilities_to_assets",
            "debt_to_assets",
            "current_ratio",
            "roa",
            "profit_margin",
            "ebitda_margin",
            "gross_margin",
            "asset_turnover",
            "sales_growth_qoq",
            "asset_growth_qoq",
            TARGET_COL,
        ]
        if col in model_frame.columns
    ]
    corr = model_frame[corr_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Financial Feature Correlations")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()

    sample = model_frame.dropna(subset=["cluster_pca1", "cluster_pca2", "finance_cluster"])
    if len(sample) > 25000:
        sample = sample.sample(25000, random_state=RANDOM_STATE)
    clusters = sorted(sample["finance_cluster"].unique())
    color_map = {cluster: idx for idx, cluster in enumerate(clusters)}
    colors = sample["finance_cluster"].map(color_map)
    plt.figure(figsize=(9, 7))
    plt.scatter(sample["cluster_pca1"], sample["cluster_pca2"], c=colors, s=5, alpha=0.45, cmap="tab10")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("KMeans Clusters on Financial Ratios")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "pca_clusters.png", dpi=180)
    plt.close()


def write_report(
    profile: dict,
    missingness: pd.DataFrame,
    invalid_values: pd.DataFrame,
    split_summary: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    cluster_features: list[str],
    high_missing_columns: list[str],
    crsp_profile: dict | None = None,
    crsp_merge_profile: dict | None = None,
) -> None:
    duplicate_files = profile.get("raw_duplicate_files", [])
    duplicate_text = ", ".join(f"`{name}`" for name in duplicate_files) if duplicate_files else "None detected"
    lines = [
        "# Project 4 Data Cleaning and Preprocessing Report",
        "",
        "## Rubric Alignment",
        "",
        "The Project 4 rubric asks for an end-to-end machine learning workflow: data collection and cleaning, EDA, unsupervised learning, feature engineering, supervised modeling, model comparison, communication, and reproducibility. This preprocessing deliverable focuses on the data-preparation foundation and intentionally includes EDA outputs, unsupervised KMeans/PCA features, chronological splitting, and a reusable sklearn pipeline so the modeling stage can build directly on it.",
        "",
        "## Source Data",
        "",
        f"- Raw file: `{RAW_FILE}`",
        f"- Exact raw-data duplicate files: {duplicate_text}",
        f"- Raw shape: {profile['raw_rows']:,} rows x {profile['raw_columns']:,} columns",
        f"- Date range after parsing: {profile['min_date']} to {profile['max_date']}",
        f"- Unique companies (`gvkey`): {profile['unique_gvkeys']:,}",
        f"- Raw SHA-256: `{profile['raw_sha256']}`",
        "",
        "## Cleaning Decisions",
        "",
        f"- Removed {profile['exact_duplicate_rows']:,} exact duplicate rows.",
        f"- Resolved {profile['duplicate_key_rows_before_resolution']:,} duplicate `gvkey + datadate` rows by keeping the most complete record in each duplicate group.",
        f"- Dropped {profile['rows_dropped_bad_key_or_date']:,} rows with unusable keys or dates.",
        "- Standardized ticker, CUSIP, currency, format, consolidation, and status strings using trimming and uppercase conversion.",
        "- Converted `datadate` to datetime and numeric accounting fields to numeric dtypes.",
        "- Set impossible negative balance-sheet stock values (assets, liabilities, current assets/current liabilities, long-term debt, cash-like fields) to missing.",
        "- Set negative values in accounting flows that should be nonnegative for modeling ratios (sales, COGS, SG&A, capex) to missing. Negative income and operating cash flow were preserved because losses and negative cash flow are economically meaningful.",
        "- Fields with more than 90% missingness were excluded from model features but retained in raw-source documentation.",
        "",
        "High-missing columns excluded from model features:",
        "",
        ", ".join(high_missing_columns) if high_missing_columns else "None",
        "",
        "## Feature Engineering",
        "",
        "- Created date features: calendar year, quarter, month, and year-end indicator.",
        "- Created SIC industry features: two-digit SIC and broad SIC division.",
        "- Converted year-to-date `oancfy` and `capxy` into approximate quarterly `oancfq_approx` and `capxq_approx` by differencing within company-year.",
        "- Created financial ratios: log assets, leverage, debt-to-assets, equity-to-assets, current ratio, ROA, ROE proxy, margins, asset turnover, operating-cash-flow-to-assets, and capex-to-assets.",
        "- Created lagged growth features by company: sales growth, earnings growth, and asset growth.",
        "- Created the supervised target `target_next_niq_growth`, a next-quarter earnings-growth target. CRSP excess-return prediction still requires adding CRSP returns and a WRDS linking table.",
        "- Added missingness indicators for core accounting fields so missing-data patterns can remain predictive after imputation.",
        "- Fitted KMeans clusters and two PCA components on training data only, then assigned cluster/PCA features to all splits.",
        "",
        "## Preprocessing Pipeline",
        "",
        "- Numeric pipeline: median imputation, train-fitted 1st/99th percentile clipping, standard scaling.",
        "- Categorical pipeline: one-hot encoding with unknown-category handling.",
        "- Split strategy: chronological train/validation/test split to reduce future-data leakage.",
        f"- Numeric feature count: {len(numeric_features):,}",
        f"- Categorical feature count: {len(categorical_features):,}",
        f"- Clustering feature count: {len(cluster_features):,}",
        "",
        "## Split Summary",
        "",
        split_summary.to_markdown(index=False),
        "",
    ]

    if crsp_profile and crsp_merge_profile:
        crsp_split = pd.DataFrame(crsp_merge_profile["crsp_split_summary"])
        lines += [
            "## CRSP + Compustat Merge",
            "",
            f"- CRSP daily file: `{crsp_profile['crsp_file']}`",
            f"- CRSP raw shape: {crsp_profile['crsp_raw_rows']:,} rows x {crsp_profile['crsp_raw_columns']:,} columns",
            f"- CRSP date range: {crsp_profile['crsp_min_date']} to {crsp_profile['crsp_max_date']}",
            f"- Unique PERMNOs: {crsp_profile['crsp_unique_permnos']:,}",
            f"- CRSP monthly rows after aggregation and CUSIP-month dedupe: {crsp_profile['crsp_monthly_rows']:,}",
            "- Merge key used here: Compustat `cusip` first 8 characters + month-end date matched to CRSP `HdrCUSIP` + month-end date.",
            "- Important limitation: the official WRDS CRSP-Compustat linking table was not included in the folder, so this is a transparent CUSIP-based merge. Replace it with the linking table when available for production-quality entity resolution.",
            "- Created CRSP features: 1/3/6/12 month returns, one-month volatility, average volume, average dollar volume, last price, and last market capitalization.",
            "- Created return target: future 3-month CRSP return minus an equal-weight CRSP-universe future 3-month market proxy.",
            f"- Merged rows: {crsp_merge_profile['merge_rows']:,}",
            f"- Rows matched to CRSP: {crsp_merge_profile['merge_matched_rows']:,}",
            f"- Rows with nonmissing CRSP excess-return target: {crsp_merge_profile['merge_model_rows']:,}",
            "",
            "CRSP target split summary:",
            "",
            crsp_split.to_markdown(index=False),
            "",
        ]

    lines += [
        "## Output Files",
        "",
        "- `outputs/cleaned_compustat.csv.gz`: cleaned and feature-engineered dataset.",
        "- `outputs/train_clean.csv.gz`, `outputs/validation_clean.csv.gz`, `outputs/test_clean.csv.gz`: chronological modeling splits.",
        "- `outputs/preprocessing_pipeline.joblib`: sklearn transformer fitted on training data only.",
        "- `outputs/cluster_artifacts.joblib`: KMeans/PCA/clustering preprocessing artifacts fitted on training data only.",
        "- `outputs/data_profile.json`, `outputs/missingness.csv`, `outputs/invalid_values.csv`, `outputs/cluster_profiles.csv`: audit tables.",
        "- `outputs/figures/`: EDA and unsupervised-learning figures.",
        "- If CRSP data is present: `outputs/merged_crsp_compustat.csv.gz`, `outputs/crsp_train_clean.csv.gz`, `outputs/crsp_validation_clean.csv.gz`, `outputs/crsp_test_clean.csv.gz`, and `outputs/crsp_preprocessing_pipeline.joblib`.",
        "",
        "## Notes for Modeling",
        "",
        "Two modeling targets are now available when both datasets are present: `target_next_niq_growth` for Compustat earnings-growth regression/classification and `target_crsp_excess_return_3m` for CRSP-based future excess-return prediction. For the strongest final project version, use the official WRDS linking table instead of the CUSIP approximation and compare at least three supervised models using the chronological validation/test splits.",
        "",
    ]
    (OUTPUT_DIR / "preprocessing_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = read_raw_data(RAW_FILE)
    raw_sha = sha256(RAW_FILE)
    raw_duplicate_files = detect_raw_duplicates(RAW_FILE, raw_sha)
    crsp_path = detect_crsp_file()

    profile: dict[str, object] = {
        "raw_rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "raw_sha256": raw_sha,
        "raw_duplicate_files": raw_duplicate_files,
        "crsp_file_detected": crsp_path.name if crsp_path else None,
    }

    typed, type_notes = standardize_and_type(raw)
    profile.update(type_notes)
    deduped, duplicate_notes = remove_duplicates(typed)
    profile.update(duplicate_notes)
    cleaned, invalid_values = treat_invalid_values(deduped)
    engineered = engineer_features(cleaned)

    profile.update(
        {
            "cleaned_rows": int(engineered.shape[0]),
            "cleaned_columns": int(engineered.shape[1]),
            "min_date": engineered["datadate"].min().strftime("%Y-%m-%d"),
            "max_date": engineered["datadate"].max().strftime("%Y-%m-%d"),
            "unique_gvkeys": int(engineered["gvkey_str"].nunique()),
            "target_nonmissing_rows_before_split": int(engineered[TARGET_COL].notna().sum()),
        }
    )

    missingness = (
        pd.DataFrame(
            {
                "column": engineered.columns,
                "missing_count": engineered.isna().sum().values,
                "missing_pct": engineered.isna().mean().values,
                "dtype": [str(dtype) for dtype in engineered.dtypes],
            }
        )
        .sort_values(["missing_pct", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )
    high_missing_columns = missingness.loc[
        missingness["missing_pct"] > HIGH_MISSING_THRESHOLD, "column"
    ].tolist()

    model_df = engineered.loc[engineered[TARGET_COL].notna()].copy()
    split, split_notes = chronological_split(model_df)
    model_df["split"] = split
    profile.update(split_notes)

    _, _, initial_cluster_features = feature_columns(model_df)
    train_without_clusters = model_df.loc[model_df["split"] == "train"].copy()
    model_df, cluster_preprocessor, pca, kmeans, cluster_profile = fit_unsupervised_features(
        model_df,
        train_without_clusters,
        initial_cluster_features,
    )

    numeric_features, categorical_features, cluster_features = feature_columns(model_df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    train_df = model_df.loc[model_df["split"] == "train"].copy()
    validation_df = model_df.loc[model_df["split"] == "validation"].copy()
    test_df = model_df.loc[model_df["split"] == "test"].copy()

    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[TARGET_COL]
    preprocessor.fit(X_train, y_train)

    split_summary = (
        model_df.groupby("split")
        .agg(
            rows=("gvkey_str", "size"),
            companies=("gvkey_str", "nunique"),
            start_date=("datadate", "min"),
            end_date=("datadate", "max"),
            target_mean=(TARGET_COL, "mean"),
            target_median=(TARGET_COL, "median"),
        )
        .reset_index()
    )
    split_summary["start_date"] = split_summary["start_date"].dt.strftime("%Y-%m-%d")
    split_summary["end_date"] = split_summary["end_date"].dt.strftime("%Y-%m-%d")

    profile["model_rows"] = int(len(model_df))
    profile["train_rows"] = int(len(train_df))
    profile["validation_rows"] = int(len(validation_df))
    profile["test_rows"] = int(len(test_df))
    profile["numeric_features"] = numeric_features
    profile["categorical_features"] = categorical_features
    profile["cluster_features"] = cluster_features
    profile["high_missing_columns"] = high_missing_columns
    profile["pca_explained_variance_ratio"] = [float(x) for x in pca.explained_variance_ratio_]

    # Add the unsupervised features back to the full cleaned file where rows are modelable.
    engineered["split"] = pd.NA
    engineered["finance_cluster"] = pd.NA
    engineered["cluster_pca1"] = np.nan
    engineered["cluster_pca2"] = np.nan
    engineered.loc[model_df.index, ["split", "finance_cluster", "cluster_pca1", "cluster_pca2"]] = model_df[
        ["split", "finance_cluster", "cluster_pca1", "cluster_pca2"]
    ]

    engineered.to_csv(OUTPUT_DIR / "cleaned_compustat.csv.gz", index=False, compression="gzip")
    train_df.to_csv(OUTPUT_DIR / "train_clean.csv.gz", index=False, compression="gzip")
    validation_df.to_csv(OUTPUT_DIR / "validation_clean.csv.gz", index=False, compression="gzip")
    test_df.to_csv(OUTPUT_DIR / "test_clean.csv.gz", index=False, compression="gzip")

    missingness.to_csv(OUTPUT_DIR / "missingness.csv", index=False)
    invalid_values.to_csv(OUTPUT_DIR / "invalid_values.csv", index=False)
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)
    cluster_profile.to_csv(OUTPUT_DIR / "cluster_profiles.csv")
    with (OUTPUT_DIR / "data_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, default=str)

    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessing_pipeline.joblib")
    joblib.dump(
        {
            "cluster_preprocessor": cluster_preprocessor,
            "pca": pca,
            "kmeans": kmeans,
            "cluster_features": cluster_features,
        },
        OUTPUT_DIR / "cluster_artifacts.joblib",
    )

    crsp_profile = None
    crsp_merge_profile = None
    if crsp_path is not None:
        crsp_monthly, crsp_profile = preprocess_crsp_monthly(crsp_path)
        crsp_monthly.to_csv(OUTPUT_DIR / "crsp_monthly_features.csv.gz", index=False, compression="gzip")
        crsp_merge_profile = add_crsp_merge_outputs(
            engineered,
            numeric_features,
            categorical_features,
            crsp_monthly,
        )
        profile["crsp_profile"] = crsp_profile
        profile["crsp_merge_profile"] = {
            key: value
            for key, value in crsp_merge_profile.items()
            if key not in {"crsp_split_summary"}
        }

    with (OUTPUT_DIR / "data_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, default=str)

    write_figures(engineered, model_df, numeric_features)
    write_report(
        profile,
        missingness,
        invalid_values,
        split_summary,
        numeric_features,
        categorical_features,
        cluster_features,
        high_missing_columns,
        crsp_profile=crsp_profile,
        crsp_merge_profile=crsp_merge_profile,
    )

    print("Preprocessing complete.")
    print(f"Cleaned rows: {profile['cleaned_rows']:,}")
    print(f"Model rows: {profile['model_rows']:,}")
    print(f"Train/validation/test rows: {profile['train_rows']:,} / {profile['validation_rows']:,} / {profile['test_rows']:,}")
    if crsp_merge_profile:
        print(f"CRSP merged model rows: {crsp_merge_profile['merge_model_rows']:,}")
    print(f"Outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
