"""Fetch live NBA injury reports via ESPN's public API."""

from __future__ import annotations

import requests
import pandas as pd

_ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
_TIMEOUT = 10


def get_injuries() -> pd.DataFrame:
    """Fetch current NBA injury report from ESPN.

    Returns DataFrame with columns:
      player_name, team_abbrev, team_name, status, comment
    """
    try:
        resp = requests.get(_ESPN_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Warning: Could not fetch ESPN injury data ({e})")
        return pd.DataFrame(columns=["player_name", "team_abbrev", "team_name", "status", "comment"])

    rows = []
    for team_entry in data.get("injuries", []):
        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            player_name = athlete.get("displayName", "")

            # Team info lives inside athlete.team
            team = athlete.get("team", {})
            team_abbrev = team.get("abbreviation", "")
            team_name = team.get("displayName", "")

            # Injury status ("Out", "Questionable", "Doubtful", "Day-To-Day")
            status = inj.get("status", "")

            # Build comment from details + shortComment
            details = inj.get("details", {})
            parts = [
                details.get("type", ""),       # e.g. "Ankle"
                details.get("side", ""),        # e.g. "Right"
                details.get("detail", ""),      # e.g. "Sprain"
            ]
            comment = " ".join(p for p in parts if p)
            if not comment:
                comment = inj.get("shortComment", "")

            if player_name and status:
                rows.append({
                    "player_name": player_name,
                    "team_abbrev": team_abbrev,
                    "team_name": team_name,
                    "status": status,
                    "comment": comment,
                })

    return pd.DataFrame(rows)


def get_team_injuries(team_abbrev: str, injury_df: pd.DataFrame) -> pd.DataFrame:
    """Filter injury report to a specific team by abbreviation."""
    if injury_df.empty or "team_abbrev" not in injury_df.columns:
        return pd.DataFrame()
    return injury_df[injury_df["team_abbrev"].str.upper() == team_abbrev.upper()].copy()


def get_player_injury_status(player_name: str, injury_df: pd.DataFrame) -> str | None:
    """Return injury status for a player, or None if not on report (i.e. active)."""
    if injury_df.empty or "player_name" not in injury_df.columns:
        return None
    mask = injury_df["player_name"].str.lower() == player_name.lower()
    rows = injury_df[mask]
    if rows.empty:
        return None
    return rows.iloc[0].get("status", "Unknown")
