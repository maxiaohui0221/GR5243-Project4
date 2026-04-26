"""Exploratory data analysis and unsupervised learning for Project 4.

This script uses the cleaned Compustat-CRSP matched dataset and creates
reproducible figures/tables for the EDA section. The PCA + KMeans workflow is
exploratory only: monthly return is held out from clustering and used afterward
to describe the discovered groups.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


DATA_PATH = Path("compustat_crsp_merged_matched_only.csv")
FIG_DIR = Path("outputs/eda_figures")
TABLE_DIR = Path("outputs/eda_tables")
RANDOM_STATE = 42

IDENTIFIER_COLS = [
    "PERMNO",
    "LPERMNO",
    "gvkey",
    "cusip",
    "tic",
    "date",
    "datadate",
    "rdq",
    "rdq_month",
]

CATEGORICAL_COLS = ["costat", "curcdq", "datafmt", "indfmt", "consol"]

CLUSTER_FEATURES = [
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
    "roa",
    "asset_turnover",
]


def signed_log1p(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compress highly skewed numeric variables while preserving sign."""
    transformed = frame.copy()
    for col in columns:
        transformed[col] = np.sign(transformed[col]) * np.log1p(np.abs(transformed[col]))
    return transformed


def winsorize_frame(frame: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Limit extreme tails so PCA/KMeans are not driven by a few outliers."""
    clipped = frame.copy()
    quantiles = clipped.quantile([lower, upper], numeric_only=True)
    for col in clipped.columns:
        clipped[col] = clipped[col].clip(quantiles.loc[lower, col], quantiles.loc[upper, col])
    return clipped


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def write_summary_tables(df: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    overview = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "unique_permno",
                "unique_tickers",
                "first_month",
                "last_month",
                "first_rdq",
                "last_rdq",
            ],
            "value": [
                len(df),
                df.shape[1],
                df["PERMNO"].nunique(),
                df["tic"].nunique(),
                df["date"].min().date(),
                df["date"].max().date(),
                df["rdq"].min().date(),
                df["rdq"].max().date(),
            ],
        }
    )
    overview.to_csv(TABLE_DIR / "dataset_overview.csv", index=False)

    missing = (
        df.isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_rate", ascending=False)
    )
    missing.to_csv(TABLE_DIR / "missingness.csv", index=False)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    descriptive = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    descriptive.to_csv(TABLE_DIR / "numeric_descriptive_statistics.csv")


def make_eda_figures(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    missing = df.isna().mean().sort_values(ascending=False).head(18)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=missing.values, y=missing.index, color="#4C78A8")
    plt.xlabel("Missing rate")
    plt.ylabel("")
    plt.title("Highest Missingness by Column")
    savefig("missingness_top_columns.png")

    obs_by_month = df.groupby("date").size()
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=obs_by_month.index, y=obs_by_month.values, color="#2F6B3F")
    plt.xlabel("Month")
    plt.ylabel("Observations")
    plt.title("Matched Firm-Month Observations Over Time")
    savefig("observations_by_month.png")

    plt.figure(figsize=(8, 4))
    clipped_return = df["ret_m"].clip(df["ret_m"].quantile(0.01), df["ret_m"].quantile(0.99))
    sns.histplot(clipped_return, bins=70, color="#6B4E9B", kde=True)
    plt.xlabel("Monthly return, clipped at 1st/99th percentile")
    plt.ylabel("Count")
    plt.title("Distribution of Monthly Stock Returns")
    savefig("monthly_return_distribution.png")

    numeric_focus = [
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
        "roa",
        "asset_turnover",
    ]
    corr_data = signed_log1p(df[numeric_focus], [c for c in numeric_focus if c not in ["ret_m", "roa", "asset_turnover"]])
    corr = corr_data.corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.4, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Structure Among Market and Accounting Variables")
    savefig("correlation_heatmap.png")

    sector = df.dropna(subset=["sic"]).copy()
    sector["sic_sector"] = (sector["sic"] // 1000).astype(int)
    sector_summary = (
        sector.groupby("sic_sector")
        .agg(observations=("PERMNO", "size"), median_ret=("ret_m", "median"), median_roa=("roa", "median"))
        .query("observations >= 500")
        .sort_values("observations", ascending=False)
        .head(12)
    )
    sector_summary.to_csv(TABLE_DIR / "sic_sector_summary.csv")

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    sns.barplot(x=sector_summary.index.astype(str), y=sector_summary["observations"], color="#4C78A8", ax=ax1)
    ax1.set_xlabel("SIC first digit")
    ax1.set_ylabel("Observations")
    ax1.set_title("Most Represented SIC Sectors")
    savefig("sic_sector_counts.png")


def run_pca_kmeans(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    modeling = df[CLUSTER_FEATURES + ["ret_m", "PERMNO", "tic", "date"]].copy()

    skewed_cols = [
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
    ]
    modeling[CLUSTER_FEATURES] = signed_log1p(modeling[CLUSTER_FEATURES], skewed_cols)
    modeling[CLUSTER_FEATURES] = winsorize_frame(modeling[CLUSTER_FEATURES])

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("pca", PCA(n_components=8, random_state=RANDOM_STATE)),
        ]
    )
    pca_scores = pipeline.fit_transform(modeling[CLUSTER_FEATURES])
    pca = pipeline.named_steps["pca"]

    explained = pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, len(pca.explained_variance_ratio_) + 1)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(TABLE_DIR / "pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=CLUSTER_FEATURES,
        columns=[f"PC{i}" for i in range(1, len(pca.explained_variance_ratio_) + 1)],
    )
    loadings.to_csv(TABLE_DIR / "pca_loadings.csv")

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=explained, x="component", y="cumulative_explained_variance", marker="o", color="#2F6B3F")
    sns.barplot(data=explained, x="component", y="explained_variance_ratio", color="#92B7D5", alpha=0.7)
    plt.ylabel("Explained variance")
    plt.xlabel("")
    plt.title("PCA Explained Variance")
    savefig("pca_explained_variance.png")

    sample_size = min(10000, len(pca_scores))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(pca_scores), size=sample_size, replace=False)

    k_results = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE)
        labels = km.fit_predict(pca_scores[:, :5])
        sample_labels = labels[sample_idx]
        if len(np.unique(sample_labels)) < 2:
            sil = np.nan
        else:
            sil = silhouette_score(pca_scores[sample_idx, :5], sample_labels)
        k_results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    k_results_df = pd.DataFrame(k_results)
    k_results_df.to_csv(TABLE_DIR / "kmeans_k_selection.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(k_results_df["k"], k_results_df["inertia"], marker="o", color="#4C78A8")
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Inertia", color="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(k_results_df["k"], k_results_df["silhouette"], marker="s", color="#C44E52")
    ax2.set_ylabel("Silhouette score", color="#C44E52")
    plt.title("KMeans Cluster Selection on First Five PCs")
    savefig("kmeans_elbow_silhouette.png")

    selected_k = 4
    final_kmeans = KMeans(n_clusters=selected_k, n_init=50, random_state=RANDOM_STATE)
    labels = final_kmeans.fit_predict(pca_scores[:, :5])

    clusters = modeling[["PERMNO", "tic", "date", "ret_m"]].copy()
    clusters["cluster"] = labels
    for i in range(1, 6):
        clusters[f"PC{i}"] = pca_scores[:, i - 1]
    clusters.to_csv(TABLE_DIR / "pca_kmeans_assignments.csv", index=False)

    cluster_profile = (
        pd.concat([df[CLUSTER_FEATURES + ["ret_m"]], clusters["cluster"]], axis=1)
        .groupby("cluster")
        .agg(["count", "median", "mean"])
    )
    cluster_profile.to_csv(TABLE_DIR / "cluster_profile_full.csv")

    medians = pd.concat([df[CLUSTER_FEATURES + ["ret_m"]], clusters["cluster"]], axis=1).groupby("cluster").median()
    z_profile = (medians - df[CLUSTER_FEATURES + ["ret_m"]].median()) / df[CLUSTER_FEATURES + ["ret_m"]].std()
    z_profile.to_csv(TABLE_DIR / "cluster_profile_standardized_medians.csv")

    top_loadings = pd.DataFrame(
        {
            pc: [", ".join(loadings[pc].abs().sort_values(ascending=False).head(8).index)]
            for pc in ["PC1", "PC2", "PC3"]
        }
    ).T.rename(columns={0: "top_variables"})
    top_loadings.to_csv(TABLE_DIR / "top_pca_loading_variables.csv")

    plot_df = clusters.iloc[sample_idx].copy()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="tab10",
        s=10,
        alpha=0.45,
        linewidth=0,
    )
    plt.title("KMeans Clusters Projected onto First Two Principal Components")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    savefig("pca_kmeans_scatter.png")

    heatmap_vars = [
        "mv_end",
        "prc_end",
        "vol_avg",
        "vol_d",
        "atq",
        "saleq",
        "niq",
        "ltq",
        "ceqq",
        "dlttq",
        "roa",
        "asset_turnover",
        "ret_m",
    ]
    heatmap_data = z_profile[heatmap_vars]
    plt.figure(figsize=(10, 4.8))
    sns.heatmap(heatmap_data, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.4)
    plt.title("Cluster Profiles: Standardized Median Differences")
    plt.xlabel("")
    plt.ylabel("Cluster")
    savefig("cluster_profile_heatmap.png")

    cluster_summary = (
        pd.concat([df[["PERMNO"] + CLUSTER_FEATURES + ["ret_m"]], clusters["cluster"]], axis=1)
        .groupby("cluster")
        .agg(
            observations=("ret_m", "size"),
            unique_stocks=("PERMNO", "nunique"),
            median_return=("ret_m", "median"),
            mean_return=("ret_m", "mean"),
            median_market_value=("mv_end", "median"),
            median_assets=("atq", "median"),
            median_sales=("saleq", "median"),
            median_roa=("roa", "median"),
            median_asset_turnover=("asset_turnover", "median"),
            median_debt_long_term=("dlttq", "median"),
            median_volatility=("vol_d", "median"),
        )
    )
    cluster_summary["share_of_rows"] = cluster_summary["observations"] / len(df)
    cluster_summary.to_csv(TABLE_DIR / "cluster_summary.csv")

    return explained, cluster_summary


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date", "datadate", "rdq", "rdq_month"])
    write_summary_tables(df)
    make_eda_figures(df)
    explained, cluster_summary = run_pca_kmeans(df)

    print("EDA complete.")
    print(f"Rows: {len(df):,}; columns: {df.shape[1]:,}")
    print("PCA cumulative variance through PC5:", round(explained.loc[4, "cumulative_explained_variance"], 4))
    print("Cluster summary:")
    print(cluster_summary.round(4).to_string())
    print(f"Figures written to {FIG_DIR}")
    print(f"Tables written to {TABLE_DIR}")


if __name__ == "__main__":
    main()
