# ML Training Pipeline

Production pipeline for credit-risk panel data (`gb` target), built from EDA findings.

## Notebooks

| File | Description |
|------|-------------|
| `credit_risk_pipeline.ipynb` | **End-to-end notebook** — EDA, feature engineering, grouped CV, calibration, F1 threshold tuning |
| `main.ipynb` | Standalone EDA narrative |

**Run the pipeline notebook** with Python 3.11 (CatBoost/LightGBM):

```bash
pip install -r requirements.txt
jupyter notebook credit_risk_pipeline.ipynb
```

## Script quick start

```bash
pip install -r requirements.txt
python train.py                    # full run (800 iterations per model)
python train.py --iterations 300 --no-shap
python train.py --models catboost
```

## Architecture

| Module | Responsibility |
|--------|----------------|
| `ml_config.py` | Central hyperparameters and paths |
| `ml_preprocess.py` | Drop empty/duplicate/low-variance/correlated cols; MNAR indicators |
| `ml_features.py` | Per-ID aggregations + temporal deltas (fold-safe) |
| `ml_validation.py` | `StratifiedGroupKFold` with volatile IDs isolated from validation |
| `ml_models.py` | CatBoost & LightGBM trainers |
| `ml_evaluation.py` | PR-AUC, ROC-AUC, Precision@K, isotonic calibration, plots |
| `train.py` | Orchestration entry point |

## Validation design

- **5-fold `StratifiedGroupKFold`** on stable entity IDs (no row from an ID in both train and val).
- **13 volatile IDs** (target changes across rows) are **always assigned to training** folds, never validation.
- Preprocessing (MNAR detection, correlation pruning, MI) is **re-fit per fold** on training data only.

## Outputs (`artifacts/`)

- `cv_fold_metrics.csv` — per-fold metrics
- `cv_summary.csv` — OOF raw vs calibrated summary
- `{model}_calibration.png`, `{model}_roc_pr.png`, `{model}_importance.png`
- `{model}_model.cbm` / `.lgb` — serialized boosters
- `config.json` — run configuration

## Primary metrics

- **PR-AUC** (primary, imbalance-aware)
- **ROC-AUC**
- **Precision@10%** (top decile)
- **Brier score** (before/after isotonic calibration)
