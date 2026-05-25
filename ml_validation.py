"""
Group-aware cross-validation that isolates volatile entity IDs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


@dataclass
class FoldSplit:
    """Train/validation row indices for one CV fold."""

    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    n_volatile_in_train: int
    n_stable_val_groups: int


def entity_level_target(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
) -> pd.Series:
    """
    One label per entity for stratification.

    Uses the mode target per ID; volatile IDs should be excluded before splitting.
    """
    return df.groupby(id_col)[target_col].agg(lambda s: int(s.mode().iloc[0]))


def build_group_cv_splits(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    volatile_ids: list,
    n_splits: int,
    random_state: int,
    exclude_volatile_from_val: bool = True,
) -> list[FoldSplit]:
    """
    StratifiedGroupKFold on stable entities; volatile IDs always assigned to train.

    Prevents unstable labels from polluting validation metrics while retaining
    their rows for training (optional exclusion entirely via config).
    """
    volatile_set = set(volatile_ids)
    stable_mask = ~df[id_col].isin(volatile_set)
    stable_df = df.loc[stable_mask].copy()

    group_ids = stable_df[id_col].unique()
    entity_target = entity_level_target(stable_df, id_col, target_col)
    y_entity = entity_target.loc[group_ids].values

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # One sample per entity; group label equals entity id
    X_groups = np.zeros(len(group_ids))

    splits: list[FoldSplit] = []
    volatile_idx = df.index[df[id_col].isin(volatile_set)].to_numpy()

    for fold, (train_group_idx, val_group_idx) in enumerate(
        sgkf.split(X_groups, y_entity, groups=group_ids)
    ):
        train_groups = set(group_ids[train_group_idx])
        val_groups = set(group_ids[val_group_idx])

        stable_train_idx = stable_df.index[stable_df[id_col].isin(train_groups)].to_numpy()
        stable_val_idx = stable_df.index[stable_df[id_col].isin(val_groups)].to_numpy()

        if exclude_volatile_from_val:
            train_idx = np.concatenate([stable_train_idx, volatile_idx])
            val_idx = stable_val_idx
        else:
            train_idx = stable_train_idx
            val_idx = np.concatenate([stable_val_idx, volatile_idx])

        splits.append(
            FoldSplit(
                fold=fold,
                train_idx=train_idx,
                val_idx=val_idx,
                n_volatile_in_train=len(volatile_idx),
                n_stable_val_groups=len(val_groups),
            )
        )

        logger.info(
            "Fold %d: train=%d rows (%d volatile forced-in), val=%d rows (%d groups)",
            fold,
            len(train_idx),
            len(volatile_idx),
            len(val_idx),
            len(val_groups),
        )

    return splits
