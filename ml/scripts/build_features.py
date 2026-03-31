#!/usr/bin/env python3
"""Build training dataset from raw game logs and prop lines."""

import logging

import pandas as pd

from propsbasket.features.engineering import add_game_context, add_rolling_stats, add_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = "data/processed/features.parquet"


def main() -> None:
    logger.info("Building feature dataset...")

    # TODO: load from database once ingestion is complete
    # df_logs = pd.read_sql("SELECT * FROM player_game_logs JOIN games ...", con)
    # team_stats = pd.read_sql("SELECT * FROM team_stats", con)
    # prop_lines = pd.read_sql("SELECT * FROM prop_lines", con)

    # df = add_rolling_stats(df_logs)
    # df = add_game_context(df, team_stats)
    # df = df.merge(prop_lines[["game_id", "player_id", "line_value", "odds"]], ...)
    # df = add_target(df)
    # df.to_parquet(OUTPUT_PATH, index=False)

    logger.info("Done. Feature dataset saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
