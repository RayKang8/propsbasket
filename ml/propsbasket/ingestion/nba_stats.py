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


def _normalize_name(name: str) -> str:
    """Normalize a player name for nba_api lookup.

    Handles common formatting differences between data sources:
      - Removes periods from initials: "R.J." → "RJ"
      - Strips trailing punctuation from suffixes: "Jr." → "Jr"
    """
    # Remove periods that are part of initials or suffixes (e.g. R.J., Jr.)
    return name.replace(".", "").strip()


def _find_player(name: str) -> dict | None:
    """Look up a player by name, falling back to a normalized version."""
    matches = players.find_players_by_full_name(name)
    if matches:
        return matches[0]
    normalized = _normalize_name(name)
    if normalized != name:
        matches = players.find_players_by_full_name(normalized)
        if matches:
            logger.debug("Matched '%s' via normalized name '%s'", name, normalized)
            return matches[0]
    return None


def get_game_logs_for_players(
    player_names: list[str],
    season: str = "2024-25",
) -> pd.DataFrame:
    """Fetch game logs for a specific list of player names.

    Looks up each name in nba_api's static player list (with name
    normalization fallback), skips any that can't be matched, and
    concatenates results into one DataFrame.

    Args:
        player_names: Display names, e.g. ["LeBron James", "R.J. Barrett"]
        season: Season string, e.g. "2024-25"

    Returns:
        DataFrame with one row per game played across all matched players.
    """
    all_logs: list[pd.DataFrame] = []
    not_found: list[str] = []

    for name in player_names:
        player = _find_player(name)
        if not player:
            not_found.append(name)
            continue
        df = get_player_game_logs(player["id"], season=season)
        if df.empty:
            continue
        # Use the original Odds API name so downstream joins work correctly
        df["player_name"] = name
        df["season"] = int(season.split("-")[0])
        all_logs.append(df)

    if not_found:
        logger.warning("Could not find nba_api IDs for: %s", not_found)

    if not all_logs:
        return pd.DataFrame()

    result = pd.concat(all_logs, ignore_index=True)
    logger.info(
        "Fetched %d game log rows for %d/%d players",
        len(result),
        len(all_logs),
        len(player_names),
    )
    return result


def get_team_stats(season: str = "2024-25") -> pd.DataFrame:
    """Fetch league-wide team stats including defensive rating and pace."""
    time.sleep(_REQUEST_DELAY)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
    )
    return stats.get_data_frames()[0]
