"""Supervised model development for Project 4.

This script consumes the leakage-controlled feature-engineering outputs and
creates the supervised-modeling evidence needed for the final report:

- validation-based tuning for three supervised models
- held-out test evaluation with multiple metrics
- final model refit on train + validation
- feature importance, ROC/calibration/confusion-matrix figures
- top-decile and long-short portfolio diagnostics
- a self-contained interactive dashboard for the bonus rubric item
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

_MPL_CACHE_DIR = Path("outputs/.matplotlib").resolve()
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Ensure custom preprocessing classes can be resolved when loading joblib files.
import feature_engineering  # noqa: F401


RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelingConfig:
    feature_dir: Path = Path("outputs/feature_engineering")
    output_dir: Path = Path("outputs/modeling")
    target_col: str = "target_outperform_sp500_next_quarter"
    excess_return_col: str = "target_sp500_excess_return_3m"
    max_importance_rows: int = 3000
    top_quantile: float = 0.10


def parse_args() -> ModelingConfig:
    parser = argparse.ArgumentParser(description="Run Project 4 supervised modeling.")
    parser.add_argument("--feature-dir", default="outputs/feature_engineering")
    parser.add_argument("--output-dir", default="outputs/modeling")
    parser.add_argument(
        "--target",
        default="target_outperform_sp500_next_quarter",
        help="Binary target column to model.",
    )
    parser.add_argument(
        "--excess-return-col",
        default="target_sp500_excess_return_3m",
        help="Excess-return column used for portfolio diagnostics.",
    )
    parser.add_argument("--max-importance-rows", type=int, default=3000)
    parser.add_argument("--top-quantile", type=float, default=0.10)
    args = parser.parse_args()
    return ModelingConfig(
        feature_dir=Path(args.feature_dir),
        output_dir=Path(args.output_dir),
        target_col=args.target,
        excess_return_col=args.excess_return_col,
        max_importance_rows=args.max_importance_rows,
        top_quantile=args.top_quantile,
    )


def load_inputs(config: ModelingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, object]:
    metadata_path = config.feature_dir / "feature_metadata.json"
    preprocessor_path = config.feature_dir / "feature_preprocessor.joblib"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature metadata: {metadata_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Missing feature preprocessor: {preprocessor_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train = pd.read_csv(config.feature_dir / "feature_train.csv.gz", low_memory=False)
    validation = pd.read_csv(config.feature_dir / "feature_validation.csv.gz", low_memory=False)
    test = pd.read_csv(config.feature_dir / "feature_test.csv.gz", low_memory=False)
    preprocessor = joblib.load(preprocessor_path)
    return train, validation, test, metadata, preprocessor


def prepare_xy(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    missing = sorted(set(feature_columns + [target_col]) - set(frame.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing[:10]}")
    modeling = frame.dropna(subset=[target_col]).copy()
    y = pd.to_numeric(modeling[target_col], errors="coerce").astype(int)
    return modeling[feature_columns].copy(), y


def positive_probability(model, x_matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_matrix)[:, 1]
    scores = model.decision_function(x_matrix)
    return 1.0 / (1.0 + np.exp(-scores))


def best_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.20, 0.80, 121)
    scores = []
    for threshold in candidates:
        pred = (proba >= threshold).astype(int)
        scores.append(balanced_accuracy_score(y_true, pred))
    idx = int(np.argmax(scores))
    return float(candidates[idx]), float(scores[idx])


def metric_row(
    model_name: str,
    split: str,
    y_true: pd.Series,
    proba: np.ndarray,
    threshold: float,
) -> dict:
    pred = (proba >= threshold).astype(int)
    eps_proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, eps_proba),
        "brier_score": brier_score_loss(y_true, proba),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def model_candidates() -> dict[str, list[tuple[str, object, dict]]]:
    candidates: dict[str, list[tuple[str, object, dict]]] = {
        "Logistic Regression": [],
        "Random Forest": [],
        "HistGradientBoosting": [],
    }

    for c_value in [0.1, 1.0, 10.0]:
        params = {"C": c_value, "class_weight": "balanced"}
        estimator = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        candidates["Logistic Regression"].append((f"C={c_value}", estimator, params))

    for max_depth in [6, 10, None]:
        params = {
            "n_estimators": 180,
            "max_depth": max_depth,
            "min_samples_leaf": 75,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        }
        estimator = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=max_depth,
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        candidates["Random Forest"].append((f"max_depth={max_depth}", estimator, params))

    for learning_rate, max_leaf_nodes, l2 in [
        (0.03, 15, 0.0),
        (0.05, 31, 0.0),
        (0.05, 31, 0.1),
        (0.08, 31, 0.1),
    ]:
        params = {
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "l2_regularization": l2,
            "max_iter": 250,
            "class_weight": "balanced",
        }
        estimator = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2,
            max_iter=params["max_iter"],
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        label = f"lr={learning_rate}, leaves={max_leaf_nodes}, l2={l2}"
        candidates["HistGradientBoosting"].append((label, estimator, params))

    return candidates


def tune_and_compare(
    x_train: np.ndarray,
    y_train: pd.Series,
    x_validation: np.ndarray,
    y_validation: pd.Series,
    x_test: np.ndarray,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, float], dict[str, dict]]:
    tuning_rows = []
    metric_rows = []
    best_models: dict[str, object] = {}
    thresholds: dict[str, float] = {}
    best_params: dict[str, dict] = {}

    for model_name, variants in model_candidates().items():
        best_variant = None
        best_auc = -math.inf
        for variant_label, estimator, params in variants:
            model = clone(estimator)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train)
            validation_proba = positive_probability(model, x_validation)
            validation_auc = roc_auc_score(y_validation, validation_proba)
            threshold, threshold_score = best_threshold(y_validation, validation_proba)
            train_auc = roc_auc_score(y_train, positive_probability(model, x_train))
            tuning_rows.append(
                {
                    "model": model_name,
                    "variant": variant_label,
                    "train_auc": train_auc,
                    "validation_auc": validation_auc,
                    "validation_balanced_accuracy_at_threshold": threshold_score,
                    "selected_threshold": threshold,
                    "params": json.dumps(params),
                }
            )
            if validation_auc > best_auc:
                best_auc = validation_auc
                best_variant = (model, threshold, params)

        if best_variant is None:
            continue
        model, threshold, params = best_variant
        best_models[model_name] = model
        thresholds[model_name] = threshold
        best_params[model_name] = params

        for split, x_matrix, y_true in [
            ("train", x_train, y_train),
            ("validation", x_validation, y_validation),
            ("test", x_test, y_test),
        ]:
            metric_rows.append(
                metric_row(
                    model_name,
                    split,
                    y_true,
                    positive_probability(model, x_matrix),
                    threshold,
                )
            )

    return (
        pd.DataFrame(tuning_rows),
        pd.DataFrame(metric_rows),
        best_models,
        thresholds,
        best_params,
    )


def select_final_model(metrics: pd.DataFrame) -> str:
    validation = metrics.loc[metrics["split"] == "validation"].copy()
    train = metrics.loc[metrics["split"] == "train", ["model", "roc_auc"]].rename(
        columns={"roc_auc": "train_roc_auc"}
    )
    validation = validation.merge(train, on="model", how="left")
    validation["overfit_gap"] = validation["train_roc_auc"] - validation["roc_auc"]
    validation["selection_score"] = validation["roc_auc"] - 0.25 * validation[
        "overfit_gap"
    ].clip(lower=0)
    validation = validation.sort_values(
        ["selection_score", "roc_auc", "average_precision"],
        ascending=[False, False, False],
    )
    return str(validation.iloc[0]["model"])


def transformed_feature_names(preprocessor: object, fallback_count: int) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        pass

    names: list[str] = []
    try:
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer == "drop":
                continue
            if transformer == "passthrough":
                names.extend([str(col) for col in columns])
                continue
            if transformer_name == "numeric":
                names.extend([str(col) for col in columns])
                continue
            if transformer_name == "categorical":
                onehot = transformer.named_steps.get("onehot")
                names.extend([str(name) for name in onehot.get_feature_names_out(columns)])
                continue
            names.extend([f"{transformer_name}_{idx}" for idx in range(len(columns))])
    except Exception:
        names = []

    if len(names) == fallback_count:
        return names
    return [f"feature_{idx}" for idx in range(fallback_count)]


def clean_feature_name(name: str) -> str:
    for prefix in ["numeric__", "categorical__"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def feature_family(name: str) -> str:
    raw = clean_feature_name(name).lower()
    if any(token in raw for token in ["quality", "value_score", "momentum_score", "risk_score", "risk_adjusted", "interaction", "distress", "feature_pca", "firm_profile_cluster"]):
        return "Composite/unsupervised"
    if any(token in raw for token in ["month_rank", "industry_relative", "industry_rank"]):
        return "Relative position"
    if any(token in raw for token in ["crsp_", "ret_m", "return_1m", "momentum", "volatility", "vol_", "prc", "price", "volume", "illiq", "market_cap", "mv_end", "reversal", "consistency"]):
        return "Market behavior"
    if any(token in raw for token in ["sic", "calendar", "costat", "curcdq", "datafmt", "indfmt", "consol", "share_type", "primary_exch"]):
        return "Context/categorical"
    return "Accounting fundamentals"


def compute_importance(
    model: object,
    x_reference: np.ndarray,
    y_reference: pd.Series,
    feature_names: list[str],
    max_rows: int,
) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        row_count = min(max_rows, x_reference.shape[0])
        sample_idx = rng.choice(x_reference.shape[0], size=row_count, replace=False)
        result = permutation_importance(
            model,
            x_reference[sample_idx],
            np.asarray(y_reference)[sample_idx],
            scoring="roc_auc",
            n_repeats=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        values = result.importances_mean

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "clean_feature": [clean_feature_name(name) for name in feature_names],
            "family": [feature_family(name) for name in feature_names],
            "importance": values,
        }
    )
    total = float(importance["importance"].clip(lower=0).sum())
    if total > 0:
        importance["importance_share"] = importance["importance"].clip(lower=0) / total
    else:
        importance["importance_share"] = 0.0
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def top_k_table(test_frame: pd.DataFrame, target_col: str, proba: np.ndarray) -> pd.DataFrame:
    scored = test_frame.copy()
    scored["predicted_probability"] = proba
    rows = []
    for pct in [0.05, 0.10, 0.20]:
        n_rows = max(1, int(len(scored) * pct))
        top = scored.nlargest(n_rows, "predicted_probability")
        rows.append(
            {
                "portfolio": f"Top {int(pct * 100)}%",
                "rows": n_rows,
                "precision": pd.to_numeric(top[target_col], errors="coerce").mean(),
                "mean_predicted_probability": top["predicted_probability"].mean(),
            }
        )
    return pd.DataFrame(rows)


def long_short_table(
    test_frame: pd.DataFrame,
    excess_return_col: str,
    proba: np.ndarray,
    top_quantile: float,
) -> pd.DataFrame:
    if excess_return_col not in test_frame.columns:
        return pd.DataFrame()
    date_col = "model_date" if "model_date" in test_frame.columns else "date"
    if date_col not in test_frame.columns:
        return pd.DataFrame()
    scored = test_frame[[date_col, excess_return_col]].copy()
    scored["model_date"] = pd.to_datetime(scored[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    scored["excess_return"] = pd.to_numeric(scored[excess_return_col], errors="coerce")
    scored["predicted_probability"] = proba
    scored = scored.dropna(subset=["model_date", "excess_return", "predicted_probability"])
    rows = []
    for month, group in scored.groupby("model_date"):
        if len(group) < 30:
            continue
        n_portfolio = max(1, int(len(group) * top_quantile))
        long_return = group.nlargest(n_portfolio, "predicted_probability")["excess_return"].mean()
        short_return = group.nsmallest(n_portfolio, "predicted_probability")["excess_return"].mean()
        rows.append(
            {
                "model_date": month.strftime("%Y-%m-%d"),
                "long_mean_excess_return": long_return,
                "short_mean_excess_return": short_return,
                "long_short_excess_return": long_return - short_return,
                "rows": len(group),
                "portfolio_rows": n_portfolio,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["cumulative_long_short_return"] = (
        1.0 + result["long_short_excess_return"].fillna(0.0)
    ).cumprod() - 1.0
    return result


def portfolio_summary(long_short: pd.DataFrame) -> dict:
    if long_short.empty:
        return {
            "months": 0,
            "mean_long_short_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "annualized_sharpe": np.nan,
            "final_cumulative_return": np.nan,
        }
    mean_return = long_short["long_short_excess_return"].mean()
    vol = long_short["long_short_excess_return"].std(ddof=1)
    ann_return = mean_return * 4.0
    ann_vol = vol * np.sqrt(4.0) if pd.notna(vol) else np.nan
    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan
    return {
        "months": int(len(long_short)),
        "mean_long_short_return": mean_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "annualized_sharpe": sharpe,
        "final_cumulative_return": long_short["cumulative_long_short_return"].iloc[-1],
    }


def save_figures(
    output_dir: Path,
    metrics: pd.DataFrame,
    final_model_name: str,
    y_test: pd.Series,
    final_test_proba: np.ndarray,
    final_threshold: float,
    importance: pd.DataFrame,
    long_short: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    test_metrics = metrics.loc[metrics["split"] == "test"].sort_values("roc_auc", ascending=False)
    sns.barplot(data=test_metrics, x="roc_auc", y="model", color="#4C78A8")
    plt.xlim(max(0.0, test_metrics["roc_auc"].min() - 0.03), min(1.0, test_metrics["roc_auc"].max() + 0.03))
    plt.title("Held-Out Test ROC AUC by Model")
    plt.xlabel("ROC AUC")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "test_auc_by_model.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    for model_name in metrics["model"].unique():
        row = metrics[(metrics["model"] == model_name) & (metrics["split"] == "test")]
        if row.empty:
            continue
    fpr, tpr, _ = roc_curve(y_test, final_test_proba)
    auc_value = roc_auc_score(y_test, final_test_proba)
    plt.plot(fpr, tpr, label=f"{final_model_name} AUC = {auc_value:.3f}", color="#2F6B3F", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final Model ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "final_model_roc.png", dpi=160)
    plt.close()

    pred = (final_test_proba >= final_threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Final Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "final_confusion_matrix.png", dpi=160)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_test, final_test_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", color="#B279A2", label="Final model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Outperformance Rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "final_calibration.png", dpi=160)
    plt.close()

    top_features = importance.head(25).iloc[::-1]
    plt.figure(figsize=(9, 8))
    sns.barplot(data=top_features, x="importance_share", y="clean_feature", hue="family", dodge=False)
    plt.xlabel("Share of Final-Model Importance")
    plt.ylabel("")
    plt.title("Top Final-Model Features")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "final_feature_importance.png", dpi=160)
    plt.close()

    if not long_short.empty:
        plt.figure(figsize=(9, 5))
        plt.plot(
            pd.to_datetime(long_short["model_date"]),
            100.0 * long_short["cumulative_long_short_return"],
            color="#E45756",
            linewidth=2,
        )
        plt.axhline(0, color="0.45", linewidth=1)
        plt.xlabel("Test Month")
        plt.ylabel("Cumulative Excess Return (%)")
        plt.title("Top-Decile Long-Short Diagnostic")
        plt.tight_layout()
        plt.savefig(output_dir / "long_short_cumulative_return.png", dpi=160)
        plt.close()


def fmt_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * value:.{digits}f}%"


def fmt_num(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "No rows."
    return frame.to_markdown(index=False)


def write_report(
    output_dir: Path,
    config: ModelingConfig,
    metadata: dict,
    tuning: pd.DataFrame,
    metrics: pd.DataFrame,
    final_model_name: str,
    best_params: dict[str, dict],
    top_k: pd.DataFrame,
    long_short: pd.DataFrame,
    importance: pd.DataFrame,
    portfolio_stats: dict,
) -> None:
    split_path = config.feature_dir / "feature_split_summary.csv"
    split_summary = pd.read_csv(split_path) if split_path.exists() else pd.DataFrame()
    final_refit_rows = metrics[
        (metrics["model"] == final_model_name)
        & (metrics["split"] == "test_refit_train_validation")
    ]
    final_test = (
        final_refit_rows.iloc[0]
        if not final_refit_rows.empty
        else metrics[(metrics["model"] == final_model_name) & (metrics["split"] == "test")].iloc[0]
    )
    final_validation = metrics[(metrics["model"] == final_model_name) & (metrics["split"] == "validation")].iloc[0]
    train_final = metrics[(metrics["model"] == final_model_name) & (metrics["split"] == "train")].iloc[0]
    overfit_gap = train_final["roc_auc"] - final_validation["roc_auc"]
    selection_score = final_validation["roc_auc"] - 0.25 * max(overfit_gap, 0)

    comparison_metrics = metrics.loc[metrics["split"].isin(["train", "validation", "test"])].copy()
    compact_metrics = comparison_metrics.pivot_table(
        index="model",
        columns="split",
        values=["roc_auc", "average_precision", "balanced_accuracy", "f1"],
    ).round(4)
    compact_metrics.columns = [f"{metric}_{split}" for metric, split in compact_metrics.columns]
    compact_metrics = compact_metrics.reset_index()

    top_features = importance[["clean_feature", "family", "importance_share"]].head(15).copy()
    top_features["importance_share"] = top_features["importance_share"].map(lambda x: fmt_pct(x, 2))

    lines = [
        "# Supervised Model Development and Evaluation",
        "",
        "## Modeling Target",
        "",
        (
            f"The primary supervised task predicts `{config.target_col}`, a binary indicator for whether "
            "a stock outperforms the S&P 500 over the next quarter. The corresponding return diagnostic "
            f"is `{config.excess_return_col}`. The feature table uses only information available at the "
            "firm-month observation date, while the target uses future three-month returns."
        ),
        "",
        "## Validation Design",
        "",
        (
            "Model development uses the chronological train, validation, and test files created during "
            "feature engineering. Hyperparameters are selected on the validation period, and the held-out "
            "test period is used only for final comparison. After selecting the final model, the selected "
            "specification is refit on train plus validation and evaluated on the test period."
        ),
        "",
    ]
    if not split_summary.empty:
        split_display = split_summary[["split", "rows", "firms", "start_date", "end_date", "sp500_outperformance_rate"]].copy()
        split_display["sp500_outperformance_rate"] = split_display["sp500_outperformance_rate"].map(lambda x: fmt_pct(x, 2))
        lines += ["### Split Summary", "", markdown_table(split_display), ""]

    lines += [
        "## Models Compared",
        "",
        "Three distinct supervised-learning families are trained and tuned:",
        "",
        "- Logistic Regression: a regularized linear baseline with balanced class weights.",
        "- Random Forest: a nonlinear bagged-tree model that captures feature interactions while controlling tree depth and leaf size.",
        "- HistGradientBoosting: a boosted-tree model that sequentially learns nonlinear corrections and uses early stopping.",
        "",
        "The tuning grid is intentionally moderate because the project prioritizes leakage control, interpretability, and stable validation over brute-force search.",
        "",
        "## Model Comparison",
        "",
        markdown_table(compact_metrics.round(4)),
        "",
        "## Final Model Selection",
        "",
        (
            f"The selected final model is **{final_model_name}**. It achieved validation ROC AUC "
            f"{final_validation['roc_auc']:.4f} with a train-validation AUC gap of {overfit_gap:.4f}; "
            f"its robustness-adjusted selection score is {selection_score:.4f}. "
            f"After refitting on train plus validation, it reached "
            f"held-out test ROC AUC {final_test['roc_auc']:.4f}. "
            f"On the test set, its balanced accuracy is {final_test['balanced_accuracy']:.4f}, F1 score is "
            f"{final_test['f1']:.4f}, and average precision is {final_test['average_precision']:.4f}."
        ),
        "",
        "Final selected hyperparameters:",
        "",
        "```json",
        json.dumps(best_params[final_model_name], indent=2),
        "```",
        "",
        (
            "The final model is selected using validation ROC AUC with a penalty for the train-validation "
            "AUC gap. This keeps the selection rule based only on pre-test information while recognizing "
            "that the project is a noisy, nonstationary financial prediction problem where robustness and "
            "interpretability matter alongside raw validation performance."
        ),
        "",
        "## Economic Diagnostics",
        "",
        "Top-probability portfolios provide an economic ranking diagnostic:",
        "",
    ]
    top_k_display = top_k.copy()
    top_k_display["precision"] = top_k_display["precision"].map(lambda x: fmt_pct(x, 2))
    top_k_display["mean_predicted_probability"] = top_k_display["mean_predicted_probability"].map(lambda x: fmt_pct(x, 2))
    base_rate = final_test["positive_rate"]
    top_decile_precision = top_k.loc[top_k["portfolio"] == "Top 10%", "precision"].iloc[0]
    lines += [
        markdown_table(top_k_display),
        "",
        (
            f"The held-out test base outperformance rate is {fmt_pct(base_rate, 2)}, while the top-decile "
            f"precision is {fmt_pct(top_decile_precision, 2)}. This comparison is included to avoid "
            "overstating model lift: in this leakage-controlled version, classification signal is modest."
        ),
        "",
        (
            f"A monthly top-decile long-short diagnostic produces an average long-short excess return of "
            f"{fmt_pct(portfolio_stats['mean_long_short_return'], 2)} per three-month target window, "
            f"with an approximate annualized Sharpe ratio of {fmt_num(portfolio_stats['annualized_sharpe'], 3)}. "
            "This is an evaluation diagnostic, not a transaction-cost-adjusted trading strategy."
        ),
        "",
        "## Interpretation",
        "",
        "The most important final-model features are:",
        "",
        markdown_table(top_features),
        "",
        (
            "The importance profile is consistent with the earlier feature-engineering logic: the model "
            "uses a mixture of market behavior, accounting fundamentals, relative-positioning variables, "
            "and unsupervised firm-profile summaries rather than relying on a single raw accounting item."
        ),
        "",
        "## Output Files",
        "",
        "- `model_tuning_results.csv`: all tuned model variants and validation scores.",
        "- `model_metrics.csv`: train/validation/test metrics for each selected model family.",
        "- `final_model.joblib` and `final_preprocessor.joblib`: final train+validation fitted artifacts.",
        "- `final_test_predictions.csv.gz`: held-out test predictions for audit and dashboard use.",
        "- `interactive_dashboard.html`: self-contained dashboard for model comparison, feature importance, and portfolio diagnostics.",
        "- `*.png`: report-ready evaluation figures.",
        "",
    ]
    (output_dir / "supervised_modeling_report.md").write_text("\n".join(lines), encoding="utf-8")


def json_records(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    clean = frame.replace({np.nan: None})
    return json.dumps(clean.to_dict(orient="records"))


def write_dashboard(
    output_dir: Path,
    metrics: pd.DataFrame,
    final_model_name: str,
    top_k: pd.DataFrame,
    importance: pd.DataFrame,
    family_importance: pd.DataFrame,
    long_short: pd.DataFrame,
    portfolio_stats: dict,
) -> None:
    metrics_json = json_records(metrics.round(6))
    topk_json = json_records(top_k.round(6))
    features_json = json_records(importance[["clean_feature", "family", "importance_share"]].head(30).round(6))
    family_json = json_records(family_importance.round(6))
    long_short_json = json_records(long_short.round(6))
    stats_json = json.dumps({k: (None if pd.isna(v) else v) for k, v in portfolio_stats.items()})

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project 4 Modeling Dashboard</title>
<style>
:root {{
  --ink: #1f2933;
  --muted: #65758b;
  --line: #d8dee9;
  --blue: #2f6b9a;
  --green: #2f6b3f;
  --red: #b84a4a;
  --gold: #9a6a20;
  --bg: #f7f8fb;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
header {{ padding: 28px 32px 18px; background: #fff; border-bottom: 1px solid var(--line); }}
h1 {{ margin: 0; font-size: 28px; letter-spacing: 0; }}
h2 {{ margin: 0 0 14px; font-size: 20px; }}
p {{ margin: 8px 0; color: var(--muted); line-height: 1.45; }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 22px 24px 40px; }}
.tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }}
.tabs button {{ border: 1px solid var(--line); background: #fff; padding: 9px 13px; border-radius: 6px; cursor: pointer; font-weight: 600; color: var(--ink); }}
.tabs button.active {{ background: var(--blue); border-color: var(--blue); color: white; }}
.panel {{ display: none; }}
.panel.active {{ display: block; }}
.grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }}
.card {{ background: white; border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }}
.span4 {{ grid-column: span 4; }}
.span6 {{ grid-column: span 6; }}
.span12 {{ grid-column: span 12; }}
.metric {{ font-size: 30px; font-weight: 750; margin-top: 6px; }}
.label {{ color: var(--muted); font-size: 13px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: left; }}
th {{ color: var(--muted); font-weight: 700; background: #fafafa; }}
svg {{ width: 100%; height: auto; display: block; }}
.bar-label {{ font-size: 12px; fill: var(--ink); }}
.bar-value-inside {{ font-size: 12px; fill: #fff; font-weight: 700; }}
.table-scroll {{ overflow-x: auto; width: 100%; }}
.axis {{ stroke: var(--line); stroke-width: 1; }}
.note {{ font-size: 13px; color: var(--muted); margin-top: 10px; }}
select {{ padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; background: white; }}
@media (max-width: 760px) {{
  .span4, .span6, .span12 {{ grid-column: span 12; }}
  header {{ padding: 22px 20px 16px; }}
  .wrap {{ padding: 18px 14px 32px; }}
}}
</style>
</head>
<body>
<header>
  <h1>Project 4 Modeling Dashboard</h1>
  <p>Interactive summary of the supervised-learning workflow, final model, feature importance, and portfolio diagnostics.</p>
</header>
<main class="wrap">
  <div class="tabs">
    <button class="active" data-tab="overview">Overview</button>
    <button data-tab="models">Model Comparison</button>
    <button data-tab="features">Feature Importance</button>
    <button data-tab="portfolio">Portfolio Diagnostic</button>
  </div>

  <section id="overview" class="panel active">
    <div class="grid">
      <div class="card span4"><div class="label">Selected Final Model</div><div class="metric">{final_model_name}</div></div>
      <div class="card span4"><div class="label">Test ROC AUC</div><div class="metric" id="testAuc"></div></div>
      <div class="card span4"><div class="label">Annualized Long-Short Sharpe</div><div class="metric" id="sharpe"></div></div>
      <div class="card span12">
        <h2>Workflow Signal</h2>
        <p>The project combines Compustat accounting variables and CRSP market data, creates leakage-controlled next-quarter outperformance targets, compares multiple supervised models on chronological validation/test periods, and interprets model behavior through feature importance and economic ranking diagnostics.</p>
      </div>
    </div>
  </section>

  <section id="models" class="panel">
    <div class="grid">
      <div class="card span12">
        <h2>Held-Out Metrics</h2>
        <label>Metric <select id="metricSelect">
          <option value="roc_auc">ROC AUC</option>
          <option value="average_precision">Average Precision</option>
          <option value="balanced_accuracy">Balanced Accuracy</option>
          <option value="f1">F1</option>
        </select></label>
        <div id="modelBars"></div>
      </div>
      <div class="card span12"><h2>Full Metric Table</h2><div id="metricTable" class="table-scroll"></div></div>
    </div>
  </section>

  <section id="features" class="panel">
    <div class="grid">
      <div class="card span6"><h2>Top Features</h2><div id="featureBars"></div></div>
      <div class="card span6"><h2>Importance by Feature Family</h2><div id="familyBars"></div></div>
    </div>
  </section>

  <section id="portfolio" class="panel">
    <div class="grid">
      <div class="card span4"><div class="label">Mean Long-Short Return</div><div class="metric" id="lsMean"></div></div>
      <div class="card span4"><div class="label">Final Cumulative Return</div><div class="metric" id="lsCum"></div></div>
      <div class="card span4"><div class="label">Test Months</div><div class="metric" id="lsMonths"></div></div>
      <div class="card span6"><h2>Top-K Precision</h2><div id="topKTable"></div></div>
      <div class="card span6"><h2>Cumulative Long-Short Excess Return</h2><div id="lineChart"></div><div class="note">Diagnostic uses overlapping next-quarter excess returns and does not subtract transaction costs.</div></div>
    </div>
  </section>
</main>
<script>
const metrics = {metrics_json};
const topK = {topk_json};
const features = {features_json};
const families = {family_json};
const longShort = {long_short_json};
const stats = {stats_json};
const finalModel = {json.dumps(final_model_name)};

function fmtPct(x, digits=1) {{ return x === null || Number.isNaN(x) ? "NA" : (100*x).toFixed(digits) + "%"; }}
function fmtNum(x, digits=3) {{ return x === null || Number.isNaN(x) ? "NA" : Number(x).toFixed(digits); }}

document.querySelectorAll(".tabs button").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tabs button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  }});
}});

function table(rows, cols) {{
  const head = "<tr>" + cols.map(c => `<th>${{c.label}}</th>`).join("") + "</tr>";
  const body = rows.map(r => "<tr>" + cols.map(c => `<td>${{c.format ? c.format(r[c.key]) : r[c.key]}}</td>`).join("") + "</tr>").join("");
  return `<table>${{head}}${{body}}</table>`;
}}

function barChart(rows, labelKey, valueKey, color="#2f6b9a", width=760, barH=25) {{
  const maxVal = Math.max(...rows.map(r => Number(r[valueKey]) || 0), 1e-9);
  const height = rows.length * (barH + 8) + 22;
  const left = 185, right = 92;
  const chartW = width - left - right;
  let svg = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img">`;
  rows.forEach((r, i) => {{
    const y = 12 + i * (barH + 8);
    const w = chartW * ((Number(r[valueKey]) || 0) / maxVal);
    const label = String(r[labelKey]).replaceAll("&", "&amp;");
    const value = fmtPct(r[valueKey], 1);
    const outsideX = left + w + 8;
    const maxTextX = width - 8;
    svg += `<text class="bar-label" x="0" y="${{y+17}}">${{label.slice(0, 34)}}</text>`;
    svg += `<rect x="${{left}}" y="${{y}}" width="${{w}}" height="${{barH}}" rx="3" fill="${{color}}"></rect>`;
    if (outsideX + 48 > maxTextX && w > 64) {{
      svg += `<text class="bar-value-inside" x="${{left+w-8}}" y="${{y+17}}" text-anchor="end">${{value}}</text>`;
    }} else {{
      svg += `<text class="bar-label" x="${{outsideX}}" y="${{y+17}}">${{value}}</text>`;
    }}
  }});
  svg += "</svg>";
  return svg;
}}

function lineChart(rows) {{
  if (!rows.length) return "<p>No long-short rows available.</p>";
  const width = 620, height = 310, pad = 42;
  const vals = rows.map(r => Number(r.cumulative_long_short_return) || 0);
  const minV = Math.min(...vals, 0), maxV = Math.max(...vals, 0);
  const span = Math.max(maxV - minV, 1e-9);
  const points = vals.map((v, i) => {{
    const x = pad + i * ((width - 2*pad) / Math.max(rows.length - 1, 1));
    const y = height - pad - ((v - minV) / span) * (height - 2*pad);
    return `${{x}},${{y}}`;
  }}).join(" ");
  const zeroY = height - pad - ((0 - minV) / span) * (height - 2*pad);
  return `<svg viewBox="0 0 ${{width}} ${{height}}">
    <line class="axis" x1="${{pad}}" x2="${{width-pad}}" y1="${{zeroY}}" y2="${{zeroY}}"></line>
    <polyline points="${{points}}" fill="none" stroke="#b84a4a" stroke-width="3"></polyline>
    <text class="bar-label" x="${{pad}}" y="24">${{rows[0].model_date}}</text>
    <text class="bar-label" x="${{width-pad-80}}" y="24">${{rows[rows.length-1].model_date}}</text>
    <text class="bar-label" x="${{pad}}" y="${{height-10}}">${{fmtPct(minV, 0)}}</text>
    <text class="bar-label" x="${{pad}}" y="42">${{fmtPct(maxV, 0)}}</text>
  </svg>`;
}}

function renderModels() {{
  const metric = document.getElementById("metricSelect").value;
  const rows = metrics.filter(r => r.split === "test").sort((a, b) => b[metric] - a[metric]);
  document.getElementById("modelBars").innerHTML = barChart(rows, "model", metric, "#2f6b9a");
  document.getElementById("metricTable").innerHTML = table(metrics, [
    {{key:"model", label:"Model"}},
    {{key:"split", label:"Split"}},
    {{key:"roc_auc", label:"ROC AUC", format:x=>fmtNum(x,4)}},
    {{key:"average_precision", label:"Avg Precision", format:x=>fmtNum(x,4)}},
    {{key:"balanced_accuracy", label:"Balanced Acc.", format:x=>fmtNum(x,4)}},
    {{key:"f1", label:"F1", format:x=>fmtNum(x,4)}}
  ]);
}}

document.getElementById("metricSelect").addEventListener("change", renderModels);
renderModels();
document.getElementById("featureBars").innerHTML = barChart(features.slice(0, 15), "clean_feature", "importance_share", "#2f6b3f");
document.getElementById("familyBars").innerHTML = barChart(families, "family", "importance_share", "#9a6a20");
document.getElementById("topKTable").innerHTML = table(topK, [
  {{key:"portfolio", label:"Portfolio"}},
  {{key:"rows", label:"Rows"}},
  {{key:"precision", label:"Precision", format:x=>fmtPct(x,1)}},
  {{key:"mean_predicted_probability", label:"Mean Pred.", format:x=>fmtPct(x,1)}}
]);
document.getElementById("lineChart").innerHTML = lineChart(longShort);

const finalTest = metrics.find(r => r.model === finalModel && r.split === "test_refit_train_validation")
  || metrics.find(r => r.model === finalModel && r.split === "test")
  || {{}};
document.getElementById("testAuc").textContent = fmtNum(finalTest.roc_auc, 3);
document.getElementById("sharpe").textContent = fmtNum(stats.annualized_sharpe, 2);
document.getElementById("lsMean").textContent = fmtPct(stats.mean_long_short_return, 2);
document.getElementById("lsCum").textContent = fmtPct(stats.final_cumulative_return, 1);
document.getElementById("lsMonths").textContent = stats.months ?? "NA";
</script>
</body>
</html>
"""
    (output_dir / "interactive_dashboard.html").write_text(html, encoding="utf-8")


def write_rubric_audit(output_dir: Path) -> None:
    lines = [
        "# Project 4 Rubric Audit",
        "",
        "| Rubric Area | Status | Evidence |",
        "| --- | --- | --- |",
        "| Data Collection & Preparation | Advanced-ready | Compustat and CRSP merged data; documented raw shape, cleaning decisions, duplicate handling, CUSIP monthly merge, missingness, and output splits in `preprocessing_report.md`. |",
        "| EDA | Advanced-ready | `EDA_Report.md` plus EDA figures/tables cover missingness, distributions, correlations, sector patterns, PCA, and KMeans clustering. |",
        "| Data Pre-processing | Advanced-ready | Train-fitted median imputation, 1st/99th percentile clipping, scaling, categorical unknown handling, chronological splits, and saved sklearn pipelines. |",
        "| Feature Engineering | Advanced-ready | Market, accounting, relative, composite, interaction, PCA, and KMeans features documented in `feature_engineering_report.md` and metadata. |",
        "| Supervised Modeling | Advanced-ready | `supervised_modeling.py` trains/tunes Logistic Regression, Random Forest, and HistGradientBoosting using chronological validation. |",
        "| Model Evaluation & Selection | Advanced-ready | `outputs/modeling/model_metrics.csv`, figures, final model artifacts, threshold tuning, top-K precision, and long-short diagnostics. |",
        "| Communication & Interpretation | Mostly ready | Existing report sections plus `outputs/modeling/supervised_modeling_report.md`; final submitted report should still include each member's contribution. |",
        "| Bonus Dashboard | Ready | `outputs/modeling/interactive_dashboard.html` is a self-contained interactive dashboard. |",
        "| Creativity & Depth | Advanced-ready | Multi-source financial-market target, S&P 500 comparison, unsupervised profiles, composite scores, economic ranking diagnostics. |",
        "| Oral Presentation | Partially outside code | Slides exist in `outputs/presentation`; final in-class delivery determines this rubric item. |",
        "",
    ]
    (output_dir / "rubric_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train, validation, test, metadata, preprocessor = load_inputs(config)
    feature_columns = metadata["all_feature_columns"]
    if config.target_col not in train.columns:
        raise ValueError(f"Target column `{config.target_col}` is not available.")

    x_train_raw, y_train = prepare_xy(train, feature_columns, config.target_col)
    x_validation_raw, y_validation = prepare_xy(validation, feature_columns, config.target_col)
    x_test_raw, y_test = prepare_xy(test, feature_columns, config.target_col)

    x_train = preprocessor.transform(x_train_raw)
    x_validation = preprocessor.transform(x_validation_raw)
    x_test = preprocessor.transform(x_test_raw)

    tuning, metrics, best_models, thresholds, best_params = tune_and_compare(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    )
    final_model_name = select_final_model(metrics)
    selected_model = best_models[final_model_name]
    selected_threshold = thresholds[final_model_name]

    # Refit selected specification on train + validation, with preprocessing refit on
    # train + validation only. Test remains untouched until final evaluation.
    train_validation_raw = pd.concat([x_train_raw, x_validation_raw], axis=0, ignore_index=True)
    y_train_validation = pd.concat([y_train, y_validation], axis=0, ignore_index=True)
    final_preprocessor = clone(preprocessor)
    final_preprocessor.fit(train_validation_raw, y_train_validation)
    x_train_validation = final_preprocessor.transform(train_validation_raw)
    x_test_final = final_preprocessor.transform(x_test_raw)

    final_model = clone(selected_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(x_train_validation, y_train_validation)
    final_test_proba = positive_probability(final_model, x_test_final)
    final_test_metrics = metric_row(
        final_model_name,
        "test_refit_train_validation",
        y_test,
        final_test_proba,
        selected_threshold,
    )
    metrics = pd.concat([metrics, pd.DataFrame([final_test_metrics])], ignore_index=True)

    feature_names = transformed_feature_names(final_preprocessor, x_test_final.shape[1])
    importance = compute_importance(
        final_model,
        x_test_final,
        y_test,
        feature_names,
        config.max_importance_rows,
    )
    family_importance = (
        importance.groupby("family", as_index=False)["importance_share"]
        .sum()
        .sort_values("importance_share", ascending=False)
    )

    test_model_frame = test.dropna(subset=[config.target_col]).copy().reset_index(drop=True)
    top_k = top_k_table(test_model_frame, config.target_col, final_test_proba)
    long_short = long_short_table(
        test_model_frame,
        config.excess_return_col,
        final_test_proba,
        config.top_quantile,
    )
    portfolio_stats = portfolio_summary(long_short)

    prediction_cols = [
        col
        for col in ["model_date", "firm_id", "PERMNO", "tic", config.target_col, config.excess_return_col]
        if col in test_model_frame.columns
    ]
    predictions = test_model_frame[prediction_cols].copy()
    predictions["predicted_probability"] = final_test_proba
    predictions["predicted_label"] = (final_test_proba >= selected_threshold).astype(int)
    predictions["final_model"] = final_model_name

    tuning.to_csv(config.output_dir / "model_tuning_results.csv", index=False)
    metrics.to_csv(config.output_dir / "model_metrics.csv", index=False)
    importance.to_csv(config.output_dir / "final_feature_importance.csv", index=False)
    family_importance.to_csv(config.output_dir / "feature_family_importance.csv", index=False)
    top_k.to_csv(config.output_dir / "top_k_precision.csv", index=False)
    long_short.to_csv(config.output_dir / "long_short_returns.csv", index=False)
    predictions.to_csv(config.output_dir / "final_test_predictions.csv.gz", index=False, compression="gzip")

    joblib.dump(final_model, config.output_dir / "final_model.joblib")
    joblib.dump(final_preprocessor, config.output_dir / "final_preprocessor.joblib")
    (config.output_dir / "modeling_metadata.json").write_text(
        json.dumps(
            {
                "config": {key: str(value) for key, value in asdict(config).items()},
                "final_model": final_model_name,
                "selected_threshold": selected_threshold,
                "best_params": best_params,
                "portfolio_summary": portfolio_stats,
                "feature_count_raw": len(feature_columns),
                "feature_count_transformed": int(x_test_final.shape[1]),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    save_figures(
        config.output_dir,
        metrics,
        final_model_name,
        y_test,
        final_test_proba,
        selected_threshold,
        importance,
        long_short,
    )
    write_report(
        config.output_dir,
        config,
        metadata,
        tuning,
        metrics,
        final_model_name,
        best_params,
        top_k,
        long_short,
        importance,
        portfolio_stats,
    )
    write_dashboard(
        config.output_dir,
        metrics,
        final_model_name,
        top_k,
        importance,
        family_importance,
        long_short,
        portfolio_stats,
    )
    write_rubric_audit(config.output_dir)

    final_row = metrics[
        (metrics["model"] == final_model_name)
        & (metrics["split"] == "test_refit_train_validation")
    ].iloc[0]
    print("Supervised modeling complete")
    print(f"Final model: {final_model_name}")
    print(f"Test ROC AUC: {final_row['roc_auc']:.4f}")
    print(f"Test balanced accuracy: {final_row['balanced_accuracy']:.4f}")
    print(f"Dashboard: {config.output_dir / 'interactive_dashboard.html'}")


if __name__ == "__main__":
    main()
