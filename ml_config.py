"""
Central configuration for the credit-risk modeling pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Hyperparameters and paths for reproducible training."""

    # Data
    data_path: Path = Path("train_df.csv")
    data_sep: str = "\t"
    id_col: str = "id"
    target_col: str = "gb"

    # Cleaning (aligned with EDA)
    min_valid_values: int = 50
    low_variance_threshold: float = 0.01
    corr_threshold: float = 0.95
    mnar_min_missing_pct: float = 10.0
    mnar_pvalue_alpha: float = 0.05

    # Feature engineering
    enable_missing_indicators: bool = True
    enable_id_aggregations: bool = True
    enable_temporal_deltas: bool = True
    aggregation_funcs: tuple[str, ...] = ("mean", "std", "min", "max")
    max_delta_features: int = 80  # cap for memory/runtime on wide data

    # Validation
    n_splits: int = 5
    random_state: int = 42
    exclude_volatile_ids_from_validation: bool = True

    # Metrics
    precision_at_k_fraction: float = 0.10

    # Models
    catboost_iterations: int = 800
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 6
    lightgbm_n_estimators: int = 800
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 63

    # Outputs
    output_dir: Path = Path("artifacts")
    save_models: bool = True
    run_shap: bool = True
    shap_sample_size: int = 2000

    # Categorical handling
    cat_feature_prefix: str = "cat_"
    num_feature_prefix: str = "num_"

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
