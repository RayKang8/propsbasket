#!/usr/bin/env python3
"""Ingest NBA player game logs and team stats for one or more seasons."""
import argparse
import logging
import random
import time
from pathlib import Path
import pandas as pd
from propsbasket.ingestion.nba_stats import get_all_active_players, get_player_game_logs, get_team_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
RAW_DIR = Path("data/raw")
DELAY_MIN = 3.0
DELAY_MAX = 6.0
MAX_RETRIES = 5
COOLDOWN_EVERY = 50   # pause for longer every N players to avoid IP throttle
COOLDOWN_SECS = 60


def fetch_with_retry(player, season):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = get_player_game_logs(player["id"], season=season)
            return df
        except Exception as exc:
            if attempt < MAX_RETRIES:
                wait = DELAY_MAX * attempt * 2
                logger.warning("  Retry %d/%d for %s (waiting %.0fs): %s", attempt, MAX_RETRIES, player["full_name"], wait, exc)
                time.sleep(wait)
            else:
                logger.warning("  Failed for %s after %d attempts: %s", player["full_name"], MAX_RETRIES, exc)
                return None


def ingest_season(season: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    players = get_all_active_players()
    logger.info("Ingesting game logs for %d players, season %s", len(players), season)
    all_logs = []
    for i, player in enumerate(players):
        df = fetch_with_retry(player, season)
        if df is not None and not df.empty:
            df["player_name"] = player["full_name"]
            df["season"] = int(season.split("-")[0])
            all_logs.append(df)
            logger.info("  [%d/%d] %s: %d games", i+1, len(players), player["full_name"], len(df))

        # Random jitter between requests to avoid looking like a bot
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        # Longer cooldown every N players to let the server recover
        if (i + 1) % COOLDOWN_EVERY == 0:
            logger.info("  Cooldown pause (%ds) after %d players...", COOLDOWN_SECS, i + 1)
            time.sleep(COOLDOWN_SECS)
    if not all_logs:
        logger.warning("No game logs collected for %s — skipping.", season)
        return
    logs_path = RAW_DIR / f"game_logs_{season}.parquet"
    pd.concat(all_logs, ignore_index=True).to_parquet(logs_path, index=False)
    logger.info("Saved %d rows → %s", sum(len(d) for d in all_logs), logs_path)
    try:
        team_stats = get_team_stats(season=season)
        team_path = RAW_DIR / f"team_stats_{season}.parquet"
        team_stats.to_parquet(team_path, index=False)
        logger.info("Saved team stats → %s", team_path)
    except Exception as exc:
        logger.warning("Failed to fetch team stats for %s: %s", season, exc)

def main(seasons):
    for season in seasons:
        ingest_season(season)
    logger.info("Ingestion complete. Run build_features.py next.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=["2024-25"], metavar="SEASON")
    args = parser.parse_args()
    main(args.seasons)
