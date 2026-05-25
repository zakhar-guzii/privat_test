"""
Production-grade EDA utilities for the credit-risk panel dataset.

Designed for reuse in notebooks and downstream ML pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "neutral": "#6C757D",
    "positive": "#C73E1D",
    "negative": "#3A7D44",
}

DEFAULT_FIGSIZE = (10, 6)
MIN_VALID_VALUES = 50
CORR_THRESHOLD = 0.95
IQR_OUTLIER_THRESHOLD = 0.05
LOW_VARIANCE_THRESHOLD = 0.01


@dataclass(frozen=True)
class FeatureGroups:
    """Column groupings derived from naming conventions."""

    id_col: str
    target_col: str
    numeric: list[str]
    categorical: list[str]
    all_features: list[str]


@dataclass
class DatasetProfile:
    """High-level dataset summary."""

    n_rows: int
    n_cols: int
    n_ids: int
    rows_per_id: pd.Series
    target_rate: float
    target_counts: pd.Series
    imbalance_ratio: float


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def configure_plotting(style: str = "seaborn-v0_8-whitegrid") -> None:
    """Apply a cohesive matplotlib/seaborn style for stakeholder-ready charts."""
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette([PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]])
    plt.rcParams.update(
        {
            "figure.figsize": DEFAULT_FIGSIZE,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 100,
        }
    )


# ---------------------------------------------------------------------------
# Data loading & feature grouping
# ---------------------------------------------------------------------------


def load_data(
    path: str,
    sep: str = "\t",
    target_col: str = "gb",
    id_col: str = "id",
) -> pd.DataFrame:
    """Load training data with basic validation."""
    try:
        df = pd.read_csv(path, sep=sep)
    except FileNotFoundError:
        logger.exception("Data file not found: %s", path)
        raise
    except pd.errors.EmptyDataError:
        logger.exception("Data file is empty: %s", path)
        raise

    for col in (target_col, id_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset.")

    if not set(df[target_col].dropna().unique()).issubset({0, 1}):
        logger.warning("Target column contains values outside {{0, 1}}.")

    logger.info("Loaded %s rows x %s columns from %s", *df.shape, path)
    return df


def get_feature_groups(
    df: pd.DataFrame,
    id_col: str = "id",
    target_col: str = "gb",
) -> FeatureGroups:
    """Split columns into ID, target, numeric, and categorical feature lists."""
    numeric = sorted(c for c in df.columns if c.startswith("num_"))
    categorical = sorted(c for c in df.columns if c.startswith("cat_"))
    all_features = [c for c in df.columns if c not in (id_col, target_col)]
    return FeatureGroups(id_col, target_col, numeric, categorical, all_features)


def profile_dataset(
    df: pd.DataFrame,
    groups: FeatureGroups,
) -> DatasetProfile:
    """Compute dataset-level summary statistics."""
    target_counts = df[groups.target_col].value_counts().sort_index()
    neg, pos = target_counts.get(0, 0), target_counts.get(1, 0)
    imbalance_ratio = neg / pos if pos > 0 else np.inf
    rows_per_id = df.groupby(groups.id_col).size()

    return DatasetProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        n_ids=df[groups.id_col].nunique(),
        rows_per_id=rows_per_id,
        target_rate=df[groups.target_col].mean(),
        target_counts=target_counts,
        imbalance_ratio=imbalance_ratio,
    )


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def missingness_summary(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per-column missingness rates (percent), sorted descending."""
    cols = list(columns) if columns is not None else list(df.columns)
    pct = df[cols].isnull().mean().mul(100)
    return (
        pct.to_frame("missing_pct")
        .assign(n_missing=df[cols].isnull().sum())
        .sort_values("missing_pct", ascending=False)
    )


def plot_missingness_distribution(
    missing_df: pd.DataFrame,
    title: str = "Distribution of Missing Values Across Features",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of per-feature missing percentages."""
    ax = ax or plt.subplots(figsize=DEFAULT_FIGSIZE)[1]
    sns.histplot(
        missing_df["missing_pct"],
        bins=40,
        kde=True,
        color=PALETTE["primary"],
        ax=ax,
    )
    ax.axvline(50, color=PALETTE["accent"], linestyle="--", label="50% threshold")
    ax.axvline(100, color=PALETTE["positive"], linestyle="--", label="100% (drop)")
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("Number of features")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_top_missing_features(
    missing_df: pd.DataFrame,
    top_n: int = 20,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of features with highest missingness (excluding 100%)."""
    subset = missing_df[
        (missing_df["missing_pct"] > 0) & (missing_df["missing_pct"] < 100)
    ].head(top_n)
    ax = ax or plt.subplots(figsize=(10, 8))[1]
    sns.barplot(
        data=subset.reset_index().rename(columns={"index": "feature"}),
        y="feature",
        x="missing_pct",
        color=PALETTE["secondary"],
        ax=ax,
    )
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("")
    ax.set_title(f"Top {top_n} Features by Missingness (excluding 100% empty)")
    return ax


# ---------------------------------------------------------------------------
# Duplicates & redundancy
# ---------------------------------------------------------------------------


def detect_duplicate_column_groups(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    min_valid: int = MIN_VALID_VALUES,
) -> dict[str, list[str]]:
    """
    Find groups of columns with identical values (hash-based, vectorized).

    Only columns with at least `min_valid` non-null values are compared.
    """
    cols = [
        c
        for c in (columns or df.columns)
        if df[c].notna().sum() >= min_valid
    ]
    if not cols:
        return {}

    # Hash each column's values (NaN-aware via fillna sentinel)
    sentinel = "__NA__"
    hashes: dict[str, str] = {}
    for col in cols:
        hashes[col] = pd.util.hash_pandas_object(
            df[col].fillna(sentinel), index=False
        ).sum().__str__()

    hash_to_cols: dict[str, list[str]] = {}
    for col, h in hashes.items():
        hash_to_cols.setdefault(h, []).append(col)

    duplicates: dict[str, list[str]] = {}
    for group in hash_to_cols.values():
        if len(group) > 1:
            canonical = group[0]
            duplicates[canonical] = group[1:]

    return duplicates


def count_high_correlation_pairs(
    df: pd.DataFrame,
    numeric_cols: list[str],
    threshold: float = CORR_THRESHOLD,
    min_overlap: int = MIN_VALID_VALUES,
) -> tuple[int, pd.DataFrame]:
    """
    Count feature pairs with |r| > threshold on overlapping non-null rows.

    Returns pair count and a sample DataFrame of top pairs.
    """
    num_df = df[numeric_cols].dropna(how="all", axis=1)
    valid_cols = [c for c in num_df.columns if num_df[c].notna().sum() >= min_overlap]
    if len(valid_cols) < 2:
        return 0, pd.DataFrame()

    corr = num_df[valid_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "corr"})
    )
    pairs = pairs[pairs["corr"] > threshold].sort_values("corr", ascending=False)
    return len(pairs), pairs.head(20)


# ---------------------------------------------------------------------------
# Outliers (vectorized IQR)
# ---------------------------------------------------------------------------


def iqr_outlier_rates(num_df: pd.DataFrame, min_count: int = MIN_VALID_VALUES) -> pd.Series:
    """
    Per-column fraction of IQR outliers (vectorized per column).

    Columns with fewer than min_count non-null values return NaN.
    """
    rates = {}
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) < min_count:
            rates[col] = np.nan
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            rates[col] = 0.0
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rates[col] = ((s < lower) | (s > upper)).mean()
    return pd.Series(rates, name="outlier_rate")


def low_variance_columns(
    num_df: pd.DataFrame,
    threshold: float = LOW_VARIANCE_THRESHOLD,
    min_count: int = MIN_VALID_VALUES,
) -> list[str]:
    """Return columns with population variance below threshold."""
    variances = num_df.apply(
        lambda s: s.dropna().var(ddof=0) if s.notna().sum() >= min_count else np.nan
    )
    return variances[variances < threshold].index.tolist()


# ---------------------------------------------------------------------------
# Distributions & skewness
# ---------------------------------------------------------------------------


def numeric_skewness(num_df: pd.DataFrame, min_count: int = MIN_VALID_VALUES) -> pd.Series:
    """Skewness per numeric column (NaN if insufficient data)."""
    return num_df.apply(
        lambda s: s.dropna().skew() if s.notna().sum() >= min_count else np.nan
    )


def plot_target_distribution(
    profile: DatasetProfile,
    target_col: str = "gb",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Class balance bar chart with count labels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = profile.target_counts
    colors = [PALETTE["negative"], PALETTE["positive"]]
    bars = ax.bar(
        [f"{target_col}={idx}" for idx in counts.index],
        counts.values,
        color=colors[: len(counts)],
        edgecolor="white",
    )
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Number of records")
    ax.set_title(
        f"Target Class Distribution (positive rate: {profile.target_rate:.2%})"
    )
    plt.tight_layout()
    return fig


def plot_numeric_distribution_sample(
    df: pd.DataFrame,
    columns: list[str],
    target_col: str,
    n_cols: int = 3,
    max_features: int = 6,
) -> plt.Figure:
    """KDE overlays of top discriminative numeric features by class."""
    cols = columns[:max_features]
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, cols):
        for label, color in [(0, PALETTE["negative"]), (1, PALETTE["positive"])]:
            subset = df.loc[df[target_col] == label, col].dropna()
            if len(subset) > 10:
                sns.kdeplot(subset, ax=ax, label=f"gb={label}", color=color, fill=True, alpha=0.35)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.legend()

    for ax in axes[len(cols) :]:
        ax.set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Target Class", y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Statistical tests (vectorized where possible)
# ---------------------------------------------------------------------------


def chi2_missingness_vs_target(
    df: pd.DataFrame,
    columns: list[str],
    target_col: str,
    min_missing_pct: float = 10.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Test whether missingness is associated with the target (chi-square).

    Returns DataFrame with column, p_value, cramers_v.
    """
    results = []
    y = df[target_col]

    for col in columns:
        miss_pct = df[col].isnull().mean() * 100
        if miss_pct < min_missing_pct or miss_pct >= 100:
            continue

        table = pd.crosstab(df[col].isnull(), y)
        if table.shape != (2, 2):
            continue

        chi2, p_value, _, _ = chi2_contingency(table)
        n = table.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n > 0 else 0.0
        results.append({"feature": col, "p_value": p_value, "cramers_v": cramers_v})

    out = pd.DataFrame(results)
    if out.empty:
        return out
    return out.sort_values("p_value").reset_index(drop=True)


def mannwhitney_by_target(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target_col: str,
    min_per_class: int = 20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mann-Whitney U test per numeric feature between target classes."""
    results = []

    for col in numeric_cols:
        g0 = df.loc[df[target_col] == 0, col].dropna()
        g1 = df.loc[df[target_col] == 1, col].dropna()
        if len(g0) < min_per_class or len(g1) < min_per_class:
            continue
        try:
            _, p_value = mannwhitneyu(g0, g1, alternative="two-sided")
        except ValueError:
            continue
        results.append(
            {
                "feature": col,
                "p_value": p_value,
                "median_0": g0.median(),
                "median_1": g1.median(),
            }
        )

    out = pd.DataFrame(results)
    if out.empty:
        return out
    out["significant"] = out["p_value"] < alpha
    return out.sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Panel / longitudinal structure
# ---------------------------------------------------------------------------


def id_activity_risk_table(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Bad rate by entity activity (rows per ID), using quintiles."""
    id_stats = (
        df.groupby(id_col)
        .agg(row_count=(target_col, "count"), target_mean=(target_col, "mean"))
        .reset_index()
    )
    id_stats["activity_quantile"] = pd.qcut(
        id_stats["row_count"], q=n_quantiles, duplicates="drop"
    )
    return (
        id_stats.groupby("activity_quantile", observed=True)["target_mean"]
        .mean()
        .mul(100)
        .to_frame("bad_rate_pct")
    )


def target_stability_per_id(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
) -> tuple[int, int]:
    """Return (total_ids, ids_with_changing_target)."""
    nunique = df.groupby(id_col)[target_col].nunique()
    return len(nunique), int((nunique > 1).sum())


def monotonic_feature_scan(
    df: pd.DataFrame,
    id_col: str,
    numeric_cols: list[str],
    sample_ids: pd.Index | None = None,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Detect features that are monotonic within most entities (timeline proxies).

    Uses row order within each ID group.
    """
    if sample_ids is None:
        counts = df[id_col].value_counts()
        sample_ids = counts[counts >= 5].index[:50]

    sample = df[df[id_col].isin(sample_ids)]
    rows = []

    for col in numeric_cols:
        up = (
            sample.groupby(id_col)[col]
            .apply(lambda x: x.dropna().is_monotonic_increasing if len(x.dropna()) > 1 else False)
            .mean()
        )
        down = (
            sample.groupby(id_col)[col]
            .apply(lambda x: x.dropna().is_monotonic_decreasing if len(x.dropna()) > 1 else False)
            .mean()
        )
        if up >= threshold or down >= threshold:
            rows.append(
                {
                    "feature": col,
                    "pct_monotonic_increasing": up,
                    "pct_monotonic_decreasing": down,
                }
            )

    return pd.DataFrame(rows).sort_values(
        "pct_monotonic_increasing", ascending=False
    )


def static_vs_dynamic_features(
    df: pd.DataFrame,
    id_col: str,
    numeric_cols: list[str],
    min_count: int = 500,
    static_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Classify numeric features as static (constant within ID) vs dynamic.

    variance_ratio = mean(within-id variance) / global variance
    """
    valid = [c for c in numeric_cols if df[c].notna().sum() >= min_count]
    records = []

    for col in valid:
        total_var = df[col].var()
        if total_var == 0 or pd.isna(total_var):
            continue
        within_var = df.groupby(id_col)[col].var().mean()
        ratio = within_var / total_var if pd.notna(within_var) else 0.0
        records.append(
            {
                "feature": col,
                "variance_ratio": ratio,
                "type": "static" if ratio < static_threshold else "dynamic",
            }
        )

    return pd.DataFrame(records)


def global_position_drift(
    df: pd.DataFrame,
    target_col: str,
    n_chunks: int = 4,
) -> pd.DataFrame:
    """Target rate and mean missingness by file-order quartile."""
    work = df.copy()
    work["chunk"] = pd.qcut(work.index, q=n_chunks, duplicates="drop")
    target_by_chunk = work.groupby("chunk", observed=True)[target_col].mean().mul(100)
    missing_by_chunk = work.groupby("chunk", observed=True).apply(
        lambda x: x.isnull().mean().mean() * 100,
        include_groups=False,
    )
    return pd.DataFrame(
        {"bad_rate_pct": target_by_chunk, "avg_missing_pct": missing_by_chunk}
    )


# ---------------------------------------------------------------------------
# Categorical profiling
# ---------------------------------------------------------------------------


def categorical_cardinality(cat_df: pd.DataFrame) -> pd.Series:
    """Unique value counts per categorical column."""
    return cat_df.nunique().sort_values(ascending=False)


def plot_cardinality_distribution(
    cardinality: pd.Series,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of categorical cardinalities."""
    ax = ax or plt.subplots(figsize=DEFAULT_FIGSIZE)[1]
    sns.histplot(cardinality, bins=30, color=PALETTE["primary"], ax=ax)
    ax.axvline(50, color=PALETTE["accent"], linestyle="--", label="High-cardinality (>50)")
    ax.set_xlabel("Number of unique values")
    ax.set_ylabel("Number of categorical features")
    ax.set_title("Categorical Feature Cardinality Distribution")
    ax.legend()
    return ax


def cramers_v_correlation(cat_series: pd.Series, target: pd.Series) -> float:
    """Cramér's V association between a categorical feature and binary target."""
    table = pd.crosstab(cat_series, target)
    if table.size == 0:
        return 0.0
    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    min_dim = min(table.shape) - 1
    if n == 0 or min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))
