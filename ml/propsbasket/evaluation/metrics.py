"""Probabilistic evaluation metrics for prop predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute log loss, Brier score, and accuracy."""
    return {
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, (y_prob >= 0.5).astype(int)),
    }


def calibration_summary(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Return a table of predicted probability bucket vs actual hit rate."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"predicted_prob": prob_pred, "actual_hit_rate": prob_true})


def edge_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group predictions by edge bucket and show actual hit rates.

    Expects columns: model_probability, implied_probability, did_hit.
    """
    df = df.copy()
    df["edge"] = df["model_probability"] - df["implied_probability"]
    df["edge_bucket"] = pd.cut(df["edge"], bins=[-1, -0.05, 0, 0.05, 0.10, 1.0])
    return (
        df.groupby("edge_bucket", observed=True)
        .agg(count=("did_hit", "count"), hit_rate=("did_hit", "mean"))
        .reset_index()
    )
