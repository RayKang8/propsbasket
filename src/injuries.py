"""Fetch NBA injury reports via nba_api LeagueInjuries."""

from __future__ import annotations

import time

import pandas as pd

_DELAY = 1.0


def get_injuries() -> pd.DataFrame:
    """Return current NBA injury report as a DataFrame.

    Columns vary by nba_api version; we normalize to:
      player_name, team_abbrev, status, comment
    """
    try:
        from nba_api.stats.endpoints import leagueinjuries
        time.sleep(_DELAY)
        inj = leagueinjuries.LeagueInjuries()
        df = inj.get_data_frames()[0]
    except Exception:
        return pd.DataFrame(columns=["player_name", "team_abbrev", "status", "comment"])

    # Normalize column names (nba_api versions differ)
    df.columns = [c.upper() for c in df.columns]

    col_map = {
        "PLAYER_NAME": "player_name",
        "TEAM_ABV": "team_abbrev",
        "STATUS": "status",
        "COMMENT": "comment",
        "RETURN": "return_date",
    }
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename)

    keep = [v for v in col_map.values() if v in df.columns]
    return df[keep].copy()


def get_team_injuries(team_abbrev: str, injury_df: pd.DataFrame) -> pd.DataFrame:
    """Filter injury report to a specific team."""
    if injury_df.empty or "team_abbrev" not in injury_df.columns:
        return pd.DataFrame()
    return injury_df[injury_df["team_abbrev"].str.upper() == team_abbrev.upper()].copy()


def get_player_injury_status(player_name: str, injury_df: pd.DataFrame) -> str | None:
    """Return injury status for a specific player, or None if not on report."""
    if injury_df.empty or "player_name" not in injury_df.columns:
        return None
    mask = injury_df["player_name"].str.lower() == player_name.lower()
    rows = injury_df[mask]
    if rows.empty:
        return None
    return rows.iloc[0].get("status", "UNKNOWN")
