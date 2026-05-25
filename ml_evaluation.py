"""
Evaluation metrics, probability calibration, and importance plotting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class FoldMetrics:
    """Scores for a single CV fold."""

    fold: int
    model_name: str
    pr_auc: float
    roc_auc: float
    precision_at_k: float
    brier_raw: float
    brier_calibrated: float


def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_fraction: float = 0.10,
) -> float:
    """Precision among the top k_fraction scored observations."""
    n = len(y_true)
    if n == 0:
        return 0.0
    top_n = max(1, int(n * k_fraction))
    order = np.argsort(y_score)
    top_idx = order[-top_n:]
    return float(np.mean(y_true[top_idx]))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_fraction: float,
) -> dict[str, float]:
    """Primary ranking and calibration metrics."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision_at_k": precision_at_k(y_true, y_prob, k_fraction),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


class ProbabilityCalibrator:
    """Wrap isotonic or Platt scaling for imbalanced binary scores."""

    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "platt"}:
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self._isotonic: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ProbabilityCalibrator":
        y_true = np.asarray(y_true)
        p = np.asarray(y_prob).reshape(-1, 1)

        if self.method == "isotonic":
            self._isotonic = IsotonicRegression(out_of_bounds="clip")
            self._isotonic.fit(y_prob, y_true)
        else:
            self._platt = LogisticRegression(max_iter=1000)
            self._platt.fit(p, y_true)

        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        y_prob = np.asarray(y_prob)
        if self.method == "isotonic" and self._isotonic is not None:
            return self._isotonic.predict(y_prob)
        if self._platt is not None:
            return self._platt.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        return y_prob


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Reliability diagram: raw vs calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for probs, label in [(y_prob_raw, "Raw"), (y_prob_cal, "Calibrated")]:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    """Save ROC and PR curve plots."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#2E86AB")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title(f"{model_name} — ROC (AUC={roc_auc_score(y_true, y_prob):.4f})")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")

    axes[1].plot(rec, prec, color="#A23B72")
    axes[1].set_title(
        f"{model_name} — PR (AP={average_precision_score(y_true, y_prob):.4f})"
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    plt.tight_layout()
    fig.savefig(output_dir / f"{model_name}_roc_pr.png", dpi=120)
    plt.close(fig)


def plot_feature_importance(
    importance: pd.Series,
    model_name: str,
    output_path: Path,
    top_n: int = 30,
) -> None:
    """Horizontal bar chart of native model gain importance."""
    top = importance.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.25)))
    top.plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_title(f"{model_name} — Top {top_n} Feature Importance (Gain)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    model_name: str,
    output_path: Path,
    max_display: int = 25,
) -> None:
    """SHAP beeswarm summary (optional dependency)."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed; skipping SHAP plot for %s", model_name)
        return

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        features=None,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title(f"{model_name} — SHAP Summary")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
