"""
Panel-aware feature engineering: per-ID aggregations and temporal deltas.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml_config import PipelineConfig

logger = logging.getLogger(__name__)


def _select_delta_columns(
    df: pd.DataFrame,
    numeric_cols: list[str],
    id_col: str,
    max_features: int,
) -> list[str]:
    """
    Prefer columns with meaningful within-ID variation (dynamic features).

    Static per-ID columns produce zero deltas and are skipped.
    """
    candidates: list[tuple[str, float]] = []

    for col in numeric_cols:
        if df[col].notna().sum() < 50:
            continue
        total_var = df[col].var()
        if total_var == 0 or pd.isna(total_var):
            continue
        within_var = df.groupby(id_col)[col].var().mean()
        ratio = within_var / total_var if pd.notna(within_var) else 0.0
        if ratio >= 0.05:
            candidates.append((col, ratio))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in candidates[:max_features]]


class PanelFeatureEngineer:
    """
    Add row-order temporal deltas and per-ID aggregations.

    Aggregations are computed within the provided dataframe only (fold-safe).
    No cross-entity statistics are merged from other folds.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.delta_columns_: list[str] = []
        self.agg_columns_: list[str] = []
        self.numeric_base_: list[str] = []

    def fit(self, df: pd.DataFrame, feature_columns: list[str]) -> "PanelFeatureEngineer":
        self.numeric_base_ = [
            c
            for c in feature_columns
            if c.startswith(self.config.num_feature_prefix)
            and not c.endswith("__is_missing")
        ]

        if self.config.enable_temporal_deltas:
            self.delta_columns_ = _select_delta_columns(
                df,
                self.numeric_base_,
                self.config.id_col,
                self.config.max_delta_features,
            )
            logger.info("Temporal deltas for %d dynamic numeric features", len(self.delta_columns_))

        if self.config.enable_id_aggregations:
            self.agg_columns_ = self.numeric_base_[: self.config.max_delta_features]
            logger.info("ID aggregations for %d numeric features", len(self.agg_columns_))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.config.id_col
        target_col = self.config.target_col

        meta = [id_col, target_col]
        feature_cols = [c for c in df.columns if c not in meta]
        out = df[meta + feature_cols].copy()

        # Preserve file order within each entity for causal deltas
        out["_row_order"] = np.arange(len(out))
        out = out.sort_values([id_col, "_row_order"])

        if self.config.enable_temporal_deltas and self.delta_columns_:
            for col in self.delta_columns_:
                if col in out.columns:
                    out[f"{col}__delta"] = out.groupby(id_col, sort=False)[col].diff()

        if self.config.enable_id_aggregations and self.agg_columns_:
            agg_frames = []
            for func in self.config.aggregation_funcs:
                grouped = out.groupby(id_col, sort=False)[self.agg_columns_].transform(func)
                grouped.columns = [f"{c}__id_{func}" for c in self.agg_columns_]
                agg_frames.append(grouped)
            out = pd.concat([out] + agg_frames, axis=1)

        out = out.drop(columns=["_row_order"])
        out = out.sort_index()
        return out

    def engineered_column_names(self) -> list[str]:
        names: list[str] = []
        if self.config.enable_temporal_deltas:
            names.extend(f"{c}__delta" for c in self.delta_columns_)
        if self.config.enable_id_aggregations:
            for col in self.agg_columns_:
                for func in self.config.aggregation_funcs:
                    names.append(f"{col}__id_{func}")
        return names
