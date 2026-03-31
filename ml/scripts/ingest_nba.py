#!/usr/bin/env python3
"""Ingest NBA player game logs for all active players into the database."""

import argparse
import logging
import sys

from propsbasket.ingestion.nba_stats import get_all_active_players, get_player_game_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(season: str) -> None:
    players = get_all_active_players()
    logger.info("Ingesting game logs for %d players, season %s", len(players), season)

    for player in players:
        try:
            df = get_player_game_logs(player["id"], season=season)
            logger.info("  %s: %d games", player["full_name"], len(df))
            # TODO: upsert rows into player_game_logs table
        except Exception as exc:
            logger.warning("  Failed for %s: %s", player["full_name"], exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NBA game logs")
    parser.add_argument("--season", default="2024-25", help="Season string, e.g. 2024-25")
    args = parser.parse_args()
    main(args.season)
