"""Ingest NBA player game logs and team stats via nba_api."""

from __future__ import annotations

import logging
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, playergamelog
from nba_api.stats.static import players

logger = logging.getLogger(__name__)

# nba_api is rate-limited; stay under ~1 req/sec
_REQUEST_DELAY = 0.6


def get_all_active_players() -> list[dict]:
    """Return list of all active NBA players."""
    return players.get_active_players()


def get_player_game_logs(player_id: int, season: str = "2024-25") -> pd.DataFrame:
    """Fetch game-by-game stats for a player in a given season.

    Args:
        player_id: NBA player ID from nba_api
        season: Season string, e.g. "2024-25"

    Returns:
        DataFrame with one row per game played
    """
    time.sleep(_REQUEST_DELAY)
    log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = log.get_data_frames()[0]
    logger.info("Fetched %d game logs for player %d", len(df), player_id)
    return df


def get_team_stats(season: str = "2024-25") -> pd.DataFrame:
    """Fetch league-wide team stats including defensive rating and pace."""
    time.sleep(_REQUEST_DELAY)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
    )
    return stats.get_data_frames()[0]
