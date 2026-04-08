#!/usr/bin/env python3
"""Ingest NBA player game logs and team stats for one or more seasons."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from propsbasket.ingestion.nba_stats import get_all_active_players, get_player_game_logs, get_team_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def ingest_season(season: str) -> None:
    """Fetch all player game logs and team stats for one season and save to parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    players = get_all_active_players()
    logger.info("Ingesting game logs for %d players, season %s", len(players), season)

    all_logs: list[pd.DataFrame] = []
    for player in players:
        try:
            df = get_player_game_logs(player["id"], season=season)
            if df.empty:
                continue
            df["player_name"] = player["full_name"]
            df["season"] = int(season.split("-")[0])  # e.g. "2024-25" → 2024
            all_logs.append(df)
            logger.info("  %s: %d games", player["full_name"], len(df))
        except Exception as exc:
            logger.warning("  Failed for %s: %s", player["full_name"], exc)

    if not all_logs:
        logger.warning("No game logs collected for %s — skipping.", season)
        return

    logs_path = RAW_DIR / f"game_logs_{season}.parquet"
    pd.concat(all_logs, ignore_index=True).to_parquet(logs_path, index=False)
    logger.info("Saved %d game log rows → %s", sum(len(d) for d in all_logs), logs_path)

    logger.info("Fetching team stats for %s...", season)
    try:
        team_stats = get_team_stats(season=season)
        team_path = RAW_DIR / f"team_stats_{season}.parquet"
        team_stats.to_parquet(team_path, index=False)
        logger.info("Saved team stats → %s", team_path)
    except Exception as exc:
        logger.warning("Failed to fetch team stats for %s: %s", season, exc)


def main(seasons: list[str]) -> None:
    for season in seasons:
        ingest_season(season)
    logger.info("Ingestion complete. Run build_features.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NBA game logs")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2024-25"],
        metavar="SEASON",
        help="One or more season strings, e.g. --seasons 2022-23 2023-24 2024-25",
    )
    args = parser.parse_args()
    main(args.seasons)
