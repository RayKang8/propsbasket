"""Train and persist prop prediction models."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from propsbasket.features.engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

MODELS: dict[str, object] = {
    "logistic": LogisticRegression(max_iter=1000),
    "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, eval_metric="logloss"),
    "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05),
}


def train(
    df: pd.DataFrame,
    model_name: str = "xgboost",
    output_dir: Path = Path("models/artifacts"),
) -> CalibratedClassifierCV:
    """Train, calibrate, and save a model.

    Args:
        df: Training dataframe containing FEATURE_COLS + 'did_hit' column
        model_name: One of 'logistic', 'xgboost', 'lightgbm'
        output_dir: Directory to write the .joblib artifact

    Returns:
        Fitted CalibratedClassifierCV
    """
    X = df[FEATURE_COLS].fillna(0)
    y = df["did_hit"]

    base_model = MODELS[model_name]
    model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
    model.fit(X, y)

    cv_scores = cross_val_score(base_model, X, y, cv=5, scoring="neg_log_loss")
    logger.info(
        "%s CV log-loss: %.4f ± %.4f", model_name, -cv_scores.mean(), cv_scores.std()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, artifact_path)
    logger.info("Saved model to %s", artifact_path)
    return model


def load(
    model_name: str = "xgboost",
    artifacts_dir: Path = Path("models/artifacts"),
) -> CalibratedClassifierCV:
    """Load a previously saved model artifact."""
    return joblib.load(artifacts_dir / f"{model_name}.joblib")
