#!/usr/bin/env python3
"""
Production ML pipeline: cleaning, panel feature engineering, grouped CV,
LightGBM/CatBoost training, calibration, and reporting.

Usage:
    python train.py
    python train.py --iterations 400 --no-shap
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eda_utils import load_data
from ml_config import PipelineConfig
from ml_evaluation import (
    ProbabilityCalibrator,
    compute_metrics,
    plot_calibration_curve,
    plot_feature_importance,
    plot_roc_pr_curves,
    plot_shap_summary,
)
from ml_features import PanelFeatureEngineer
from ml_models import build_model_trainer
from ml_preprocess import (
    CategoricalEncoder,
    DataCleaner,
    identify_volatile_ids,
)
from ml_validation import build_group_cv_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def sanitize_feature_names(columns: list[str]) -> list[str]:
    """LightGBM rejects JSON-special characters in feature names."""
    clean = []
    for col in columns:
        name = re.sub(r"[^\w]", "_", col)
        clean.append(name)
    return clean


def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
    id_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df[feature_columns].copy()
    X.columns = sanitize_feature_names(feature_columns)
    y = df[target_col]
    groups = df[id_col]
    return X, y, groups


def run_fold_pipeline(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[int]]:
    """Clean + engineer features for one CV fold (fit on train only)."""
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    cleaner = DataCleaner(config)
    cleaner.fit(train_df, train_df[config.target_col])
    train_clean = cleaner.transform(train_df)
    val_clean = cleaner.transform(val_df)

    base_features = cleaner.get_feature_columns()
    cat_cols = [c for c in base_features if c.startswith(config.cat_feature_prefix)]

    panel = PanelFeatureEngineer(config)
    panel.fit(train_clean, base_features)
    train_eng = panel.transform(train_clean)
    val_eng = panel.transform(val_clean)

    all_features = base_features + panel.engineered_column_names()
    all_features = [c for c in all_features if c in train_eng.columns]

    encoder = CategoricalEncoder(cat_cols)
    encoder.fit(train_eng)
    train_enc = encoder.transform(train_eng)
    val_enc = encoder.transform(val_eng)

    cat_indices = encoder.cat_feature_indices(all_features)
    return train_enc, val_enc, all_features, cat_indices


def cross_validate(
    df: pd.DataFrame,
    config: PipelineConfig,
    model_names: list[str],
) -> tuple[pd.DataFrame, dict[str, ProbabilityCalibrator], dict[str, np.ndarray]]:
    """Run grouped stratified CV with per-fold preprocessing."""
    volatile_ids = identify_volatile_ids(df, config.id_col, config.target_col)
    logger.info(
        "Volatile entity IDs (changing target): %d — routed to train-only in CV",
        len(volatile_ids),
    )

    splits = build_group_cv_splits(
        df,
        config.id_col,
        config.target_col,
        volatile_ids,
        config.n_splits,
        config.random_state,
        config.exclude_volatile_ids_from_validation,
    )

    oof_preds: dict[str, np.ndarray] = {
        name: np.full(len(df), np.nan) for name in model_names
    }
    fold_rows: list[dict] = []

    for split in splits:
        train_enc, val_enc, features, cat_idx = run_fold_pipeline(
            df, split.train_idx, split.val_idx, config
        )

        X_train, y_train, _ = prepare_xy(
            train_enc, features, config.id_col, config.target_col
        )
        X_val, y_val, _ = prepare_xy(
            val_enc, features, config.id_col, config.target_col
        )

        for model_name in model_names:
            trainer = build_model_trainer(model_name, config)
            cat_arg = cat_idx if model_name == "catboost" else cat_idx

            trainer.fit(X_train, y_train, X_val, y_val, cat_features=cat_arg)
            val_prob = trainer.predict_proba(X_val)

            oof_preds[model_name][split.val_idx] = val_prob
            raw_metrics = compute_metrics(
                y_val.values, val_prob, config.precision_at_k_fraction
            )

            fold_rows.append(
                {
                    "fold": split.fold,
                    "model": model_name,
                    **raw_metrics,
                    "brier_raw": raw_metrics["brier"],
                }
            )

            logger.info(
                "Fold %d | %s | PR-AUC=%.4f ROC-AUC=%.4f P@%.0f%%=%.4f",
                split.fold,
                model_name,
                raw_metrics["pr_auc"],
                raw_metrics["roc_auc"],
                config.precision_at_k_fraction * 100,
                raw_metrics["precision_at_k"],
            )

    metrics_df = pd.DataFrame(fold_rows)
    summary = (
        metrics_df.groupby("model")
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            precision_at_k_mean=("precision_at_k", "mean"),
            brier_mean=("brier_raw", "mean"),
        )
        .reset_index()
    )
    logger.info("CV summary:\n%s", summary.to_string(index=False))

    calibrators: dict[str, ProbabilityCalibrator] = {}
    y_all = df[config.target_col].values

    for model_name in model_names:
        oof = oof_preds[model_name]
        valid = ~np.isnan(oof)
        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(y_all[valid], oof[valid])
        calibrators[model_name] = cal

        cal_oof = cal.predict(oof[valid])
        cal_metrics = compute_metrics(
            y_all[valid], cal_oof, config.precision_at_k_fraction
        )
        logger.info(
            "%s OOF calibrated | PR-AUC=%.4f Brier=%.4f",
            model_name,
            cal_metrics["pr_auc"],
            cal_metrics["brier"],
        )

    return metrics_df, calibrators, oof_preds


def train_final_models(
    df: pd.DataFrame,
    config: PipelineConfig,
    model_names: list[str],
    output_dir: Path,
) -> dict[str, object]:
    """Fit on full data with same preprocessing chain; save artifacts."""
    train_idx = np.arange(len(df))
    val_idx = np.array([], dtype=int)

    # Use 15% entity holdout for early stopping (group-safe)
    volatile_ids = identify_volatile_ids(df, config.id_col, config.target_col)
    stable = df[~df[config.id_col].isin(volatile_ids)]
    entity_y = stable.groupby(config.id_col)[config.target_col].first()
    rng = np.random.RandomState(config.random_state)
    val_entities = rng.choice(
        entity_y.index.values,
        size=max(1, int(0.15 * len(entity_y))),
        replace=False,
    )
    val_mask = df[config.id_col].isin(val_entities)
    train_mask = ~val_mask

    train_enc, val_enc, features, cat_idx = run_fold_pipeline(
        df,
        df.index[train_mask].to_numpy(),
        df.index[val_mask].to_numpy(),
        config,
    )

    X_train, y_train, _ = prepare_xy(train_enc, features, config.id_col, config.target_col)
    X_val, y_val, _ = prepare_xy(val_enc, features, config.id_col, config.target_col)

    models: dict[str, object] = {}
    for model_name in model_names:
        trainer = build_model_trainer(model_name, config)
        trainer.fit(X_train, y_train, X_val, y_val, cat_features=cat_idx)
        models[model_name] = trainer

        imp = trainer.feature_importance(features)
        plot_feature_importance(
            imp,
            model_name,
            output_dir / f"{model_name}_importance.png",
        )

        if config.run_shap and model_name == "catboost":
            _run_shap(trainer, X_val, features, model_name, output_dir, config)

        if config.save_models:
            _save_model(trainer, model_name, output_dir)

        y_prob = trainer.predict_proba(X_val)
        plot_roc_pr_curves(
            y_val.values,
            y_prob,
            model_name,
            output_dir,
        )

    return models


def _save_model(trainer: object, model_name: str, output_dir: Path) -> None:
    """Persist fitted booster to artifacts directory."""
    path = output_dir / f"{model_name}_model"
    if model_name == "catboost":
        trainer.model_.save_model(str(path) + ".cbm")
    else:
        trainer.model_.booster_.save_model(str(path) + ".lgb")
    logger.info("Saved %s model to %s", model_name, path)


def _run_shap(
    trainer: object,
    X_sample: pd.DataFrame,
    features: list[str],
    model_name: str,
    output_dir: Path,
    config: PipelineConfig,
) -> None:
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not available")
        return

    n = min(config.shap_sample_size, len(X_sample))
    X_sub = X_sample[features].sample(n=n, random_state=config.random_state)

    explainer = shap.TreeExplainer(trainer.model_)
    shap_values = explainer.shap_values(X_sub)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plot_shap_summary(
        shap_values,
        features,
        model_name,
        output_dir / f"{model_name}_shap.png",
    )


def save_artifacts(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    calibrators: dict[str, ProbabilityCalibrator],
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    config: PipelineConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)

    summary_rows = []
    for model_name, oof in oof_preds.items():
        valid = ~np.isnan(oof)
        cal = calibrators[model_name]
        cal_oof = cal.predict(oof[valid])
        raw_m = compute_metrics(y_true[valid], oof[valid], config.precision_at_k_fraction)
        cal_m = compute_metrics(y_true[valid], cal_oof, config.precision_at_k_fraction)

        plot_calibration_curve(
            y_true[valid],
            oof[valid],
            cal_oof,
            f"{model_name} — Calibration",
            output_dir / f"{model_name}_calibration.png",
        )
        plot_roc_pr_curves(y_true[valid], cal_oof, f"{model_name}_calibrated", output_dir)

        summary_rows.append({"model": model_name, "stage": "oof_raw", **raw_m})
        summary_rows.append({"model": model_name, "stage": "oof_calibrated", **cal_m})

    pd.DataFrame(summary_rows).to_csv(output_dir / "cv_summary.csv", index=False)

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()},
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit risk ML training pipeline")
    parser.add_argument("--data", type=str, default="train_df.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--models", nargs="+", default=["catboost", "lightgbm"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        run_shap=not args.no_shap,
    )
    if args.iterations:
        config.catboost_iterations = args.iterations
        config.lightgbm_n_estimators = args.iterations

    logger.info("Loading data from %s", config.data_path)
    df = load_data(str(config.data_path), sep=config.data_sep)

    metrics_df, calibrators, oof_preds = cross_validate(
        df, config, args.models
    )

    save_artifacts(
        config.output_dir,
        metrics_df,
        calibrators,
        oof_preds,
        df[config.target_col].values,
        config,
    )

    train_final_models(df, config, args.models, config.output_dir)
    logger.info("Pipeline complete. Artifacts saved to %s", config.output_dir.resolve())


if __name__ == "__main__":
    main()
