#!/usr/bin/env python3
"""Train and evaluate prop prediction models."""
import argparse
import logging
import pandas as pd
from propsbasket.evaluation.metrics import evaluate
from propsbasket.features.engineering import FEATURE_COLS
from propsbasket.models.trainer import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main(model_name: str, features_path: str) -> None:
    df = pd.read_parquet(features_path).sort_values("GAME_DATE").reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(n * 0.70)]
    val_df   = df.iloc[int(n * 0.70):int(n * 0.85)]
    test_df  = df.iloc[int(n * 0.85):]
    logger.info(
        "Train: %d rows | Val: %d rows | Test: %d rows",
        len(train_df), len(val_df), len(test_df),
    )
    model = train(train_df, model_name=model_name)
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        X = split_df[FEATURE_COLS].fillna(0)
        y = split_df["did_hit"]
        metrics = evaluate(y.values, model.predict_proba(X)[:, 1])
        logger.info(
            "%s — log_loss: %.4f | brier: %.4f | accuracy: %.3f",
            split_name, metrics["log_loss"], metrics["brier_score"], metrics["accuracy"],
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train prop prediction model")
    parser.add_argument("--model", default="xgboost", choices=["logistic", "xgboost", "lightgbm"])
    parser.add_argument("--features", default="data/processed/features.parquet")
    args = parser.parse_args()
    main(args.model, args.features)
