"""
Gradient boosting model wrappers for LightGBM and CatBoost.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from ml_config import PipelineConfig

logger = logging.getLogger(__name__)


class BaseBoostingModel(ABC):
    """Common interface for fold training and inference."""

    name: str

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: list[int] | None = None,
    ) -> "BaseBoostingModel":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...

    @abstractmethod
    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        ...


class CatBoostTrainer(BaseBoostingModel):
    name = "catboost"

    def __init__(self, config: PipelineConfig):
        from catboost import CatBoostClassifier, Pool

        self._CatBoostClassifier = CatBoostClassifier
        self._Pool = Pool
        self.config = config
        self.model_: Any = None
        self.cat_features_: list[int] | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: list[int] | None = None,
    ) -> "CatBoostTrainer":
        self.cat_features_ = cat_features
        self.model_ = self._CatBoostClassifier(
            iterations=self.config.catboost_iterations,
            learning_rate=self.config.catboost_learning_rate,
            depth=self.config.catboost_depth,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=self.config.random_state,
            verbose=0,
            allow_writing_files=False,
            auto_class_weights="Balanced",
        )

        train_pool = self._Pool(
            X_train,
            y_train,
            cat_features=cat_features or [],
        )
        val_pool = self._Pool(
            X_val,
            y_val,
            cat_features=cat_features or [],
        )

        self.model_.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=50,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        scores = self.model_.get_feature_importance()
        return pd.Series(scores, index=feature_names, name="importance")


class LightGBMTrainer(BaseBoostingModel):
    name = "lightgbm"

    def __init__(self, config: PipelineConfig):
        import lightgbm as lgb

        self._lgb = lgb
        self.config = config
        self.model_: Any = None
        self.cat_features_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: list[int] | None = None,
    ) -> "LightGBMTrainer":
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        if cat_features:
            self.cat_features_ = [X_train.columns[i] for i in cat_features]

        self.model_ = self._lgb.LGBMClassifier(
            n_estimators=self.config.lightgbm_n_estimators,
            learning_rate=self.config.lightgbm_learning_rate,
            num_leaves=self.config.lightgbm_num_leaves,
            objective="binary",
            metric="average_precision",
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1,
        )

        self.model_.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            categorical_feature=self.cat_features_ if self.cat_features_ else "auto",
            callbacks=[
                self._lgb.early_stopping(stopping_rounds=50, verbose=False),
                self._lgb.log_evaluation(period=0),
            ],
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        scores = self.model_.feature_importances_
        return pd.Series(scores, index=feature_names, name="importance")


def build_model_trainer(name: str, config: PipelineConfig) -> BaseBoostingModel:
    if name == "catboost":
        return CatBoostTrainer(config)
    if name == "lightgbm":
        return LightGBMTrainer(config)
    raise ValueError(f"Unknown model: {name}")
