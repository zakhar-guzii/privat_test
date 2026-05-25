"""
Leakage-safe preprocessing: column pruning, MNAR indicators, correlation filtering.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

from eda_utils import detect_duplicate_column_groups, low_variance_columns
from ml_config import PipelineConfig

logger = logging.getLogger(__name__)


def identify_volatile_ids(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
) -> list:
    """Entity IDs whose target label changes across panel rows."""
    nunique = df.groupby(id_col)[target_col].nunique()
    return nunique[nunique > 1].index.tolist()


def identify_mnar_columns(
    df: pd.DataFrame,
    columns: list[str],
    target_col: str,
    min_missing_pct: float,
    alpha: float,
) -> list[str]:
    """
    Features where missingness is statistically associated with the target.

    Fit only on training data inside each CV fold to avoid leakage.
    """
    mnar_cols: list[str] = []
    y = df[target_col]

    for col in columns:
        miss_pct = df[col].isnull().mean() * 100
        if miss_pct < min_missing_pct or miss_pct >= 100:
            continue

        table = pd.crosstab(df[col].isnull(), y)
        if table.shape != (2, 2):
            continue

        _, p_value, _, _ = chi2_contingency(table)
        if p_value < alpha:
            mnar_cols.append(col)

    return sorted(mnar_cols)


def _mutual_info_scores(
    X: pd.DataFrame,
    y: pd.Series,
    columns: list[str],
    random_state: int,
) -> dict[str, float]:
    """Compute mutual information per column (numeric only for stability)."""
    if not columns:
        return {}

    X_sub = X[columns].copy()
    for col in columns:
        if X_sub[col].dtype == "object" or str(X_sub[col].dtype).startswith("category"):
            X_sub[col] = X_sub[col].astype("category").cat.codes
        X_sub[col] = X_sub[col].fillna(X_sub[col].median())

    mi = mutual_info_classif(
        X_sub.values,
        y.values,
        random_state=random_state,
        discrete_features=False,
    )
    return dict(zip(columns, mi))


def prune_correlated_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    y: pd.Series,
    threshold: float,
    min_valid: int,
    random_state: int,
) -> list[str]:
    """
    Deterministically drop one feature from each highly correlated pair.

    Keep the feature with higher (MI, variance, name) lexicographic rank.
    """
    valid = [c for c in numeric_cols if df[c].notna().sum() >= min_valid]
    if len(valid) < 2:
        return []

    variances = df[valid].var()
    mi_scores = _mutual_info_scores(df, y, valid, random_state)

    corr = df[valid].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "a", "level_1": "b", 0: "corr"})
    )
    pairs = pairs[pairs["corr"] > threshold].sort_values(
        ["corr", "a", "b"], ascending=[False, True, True]
    )

    def rank_key(col: str) -> tuple:
        return (
            mi_scores.get(col, 0.0),
            variances.get(col, 0.0) if pd.notna(variances.get(col)) else 0.0,
            col,
        )

    to_drop: set[str] = set()
    for _, row in pairs.iterrows():
        a, b = row["a"], row["b"]
        if a in to_drop or b in to_drop:
            continue
        keep = max((a, b), key=rank_key)
        drop = b if keep == a else a
        to_drop.add(drop)

    return sorted(to_drop)


class DataCleaner:
    """
    Fit on training fold only: drop empty, duplicate, quasi-constant, correlated.
    Add MNAR missingness indicator columns.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.columns_to_drop_: list[str] = []
        self.mnar_columns_: list[str] = []
        self.kept_columns_: list[str] = []
        self.indicator_columns_: list[str] = []

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "DataCleaner":
        id_col = self.config.id_col
        target_col = self.config.target_col

        feature_cols = [
            c for c in df.columns if c not in (id_col, target_col)
        ]
        num_cols = [c for c in feature_cols if c.startswith(self.config.num_feature_prefix)]
        cat_cols = [c for c in feature_cols if c.startswith(self.config.cat_feature_prefix)]

        drop_set: set[str] = set()

        # 1) 100% empty
        missing_pct = df[feature_cols].isnull().mean()
        empty = missing_pct[missing_pct >= 1.0].index.tolist()
        drop_set.update(empty)
        logger.info("Dropping %d entirely empty columns", len(empty))

        remaining = [c for c in feature_cols if c not in drop_set]

        # 2) Exact duplicate groups
        dup_groups = detect_duplicate_column_groups(
            df[remaining + [id_col, target_col]],
            remaining,
            min_valid=self.config.min_valid_values,
        )
        n_dup_dropped = 0
        for _canonical, twins in dup_groups.items():
            drop_set.update(twins)
            n_dup_dropped += len(twins)
        logger.info("Dropping %d duplicate columns from %d groups", n_dup_dropped, len(dup_groups))

        remaining = [c for c in remaining if c not in drop_set]

        # 3) Quasi-constant numerics
        low_var = low_variance_columns(
            df[remaining],
            threshold=self.config.low_variance_threshold,
            min_count=self.config.min_valid_values,
        )
        drop_set.update(low_var)
        logger.info("Dropping %d quasi-constant columns", len(low_var))

        remaining = [c for c in remaining if c not in drop_set]
        num_remaining = [c for c in remaining if c in num_cols]

        # 4) MNAR indicators (identify on train; do not drop those columns)
        if self.config.enable_missing_indicators:
            self.mnar_columns_ = identify_mnar_columns(
                df[[target_col] + num_remaining],
                num_remaining,
                target_col,
                self.config.mnar_min_missing_pct,
                self.config.mnar_pvalue_alpha,
            )
            logger.info("MNAR missingness indicators for %d columns", len(self.mnar_columns_))

        # 5) Correlation pruning on numerics
        corr_drop = prune_correlated_features(
            df,
            num_remaining,
            y,
            self.config.corr_threshold,
            self.config.min_valid_values,
            self.config.random_state,
        )
        drop_set.update(corr_drop)
        logger.info("Dropping %d highly correlated columns", len(corr_drop))

        self.columns_to_drop_ = sorted(drop_set)
        self.kept_columns_ = [c for c in feature_cols if c not in self.columns_to_drop_]
        self.indicator_columns_ = [
            f"{c}__is_missing" for c in self.mnar_columns_
        ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.config.id_col
        target_col = self.config.target_col

        out = df[[id_col, target_col] + self.kept_columns_].copy()

        if self.config.enable_missing_indicators and self.mnar_columns_:
            for col in self.mnar_columns_:
                if col in out.columns:
                    out[f"{col}__is_missing"] = out[col].isnull().astype(np.int8)

        return out

    def get_feature_columns(self) -> list[str]:
        """Model feature columns (excludes id and target)."""
        return self.kept_columns_ + self.indicator_columns_


class CategoricalEncoder:
    """Per-fold label encoding for LightGBM; CatBoost uses raw integers with cat indices."""

    def __init__(self, cat_columns: list[str]):
        self.cat_columns = cat_columns
        self.category_maps_: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        for col in self.cat_columns:
            if col not in df.columns:
                continue
            uniques = df[col].astype(str).fillna("__NA__").unique()
            self.category_maps_[col] = {v: i for i, v in enumerate(sorted(uniques))}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, mapping in self.category_maps_.items():
            if col not in out.columns:
                continue
            encoded = (
                out[col]
                .astype(str)
                .fillna("__NA__")
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )
            out[col] = encoded
        return out

    def cat_feature_indices(self, feature_columns: list[str]) -> list[int]:
        return [i for i, c in enumerate(feature_columns) if c in self.cat_columns]
