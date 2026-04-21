"""NBA data fetching via nba_api — game logs, team stats, player lookup."""

from __future__ import annotations

import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, playergamelog
from nba_api.stats.static import players, teams

_DELAY = 1.5  # nba_api rate limit

STAT_COLUMN_MAP = {
    "points": "PTS",
    "point": "PTS",
    "pts": "PTS",
    "rebounds": "REB",
    "rebound": "REB",
    "reb": "REB",
    "assists": "AST",
    "assist": "AST",
    "ast": "AST",
    "threes": "FG3M",
    "three": "FG3M",
    "3s": "FG3M",
    "3-pointers": "FG3M",
    "3 pointers": "FG3M",
    "fg3m": "FG3M",
    "steals": "STL",
    "steal": "STL",
    "stl": "STL",
    "blocks": "BLK",
    "block": "BLK",
    "blk": "BLK",
    "turnovers": "TOV",
    "turnover": "TOV",
    "tov": "TOV",
    "pra": "PRA",
    "pts+reb+ast": "PRA",
    "points+rebounds+assists": "PRA",
}

TEAM_ABBREV_MAP: dict[str, str] = {
    t["full_name"].lower(): t["abbreviation"] for t in teams.get_teams()
}
# also index by abbreviation and common nicknames
for _t in teams.get_teams():
    TEAM_ABBREV_MAP[_t["abbreviation"].lower()] = _t["abbreviation"]
    TEAM_ABBREV_MAP[_t["nickname"].lower()] = _t["abbreviation"]
    TEAM_ABBREV_MAP[_t["city"].lower()] = _t["abbreviation"]


def resolve_team_abbrev(name: str) -> str | None:
    """Return NBA team abbreviation for a fuzzy name/city/nickname."""
    key = name.strip().lower()
    if key in TEAM_ABBREV_MAP:
        return TEAM_ABBREV_MAP[key]
    # partial match fallback
    for k, v in TEAM_ABBREV_MAP.items():
        if key in k or k in key:
            return v
    return None


def find_player(name: str) -> dict | None:
    """Find player by full name with period-stripping fallback."""
    matches = players.find_players_by_full_name(name)
    if matches:
        return matches[0]
    normalized = name.replace(".", "").strip()
    if normalized != name:
        matches = players.find_players_by_full_name(normalized)
        if matches:
            return matches[0]
    return None


def get_game_logs(player_id: int, season: str = "2025-26") -> pd.DataFrame:
    """Fetch regular season + playoff game logs for a player, combined.

    Playoff games are fetched separately and tagged with a GAME_TYPE column
    so the formula can show how many games are regular season vs playoffs.
    """
    frames = []

    # Regular season
    time.sleep(_DELAY)
    reg = playergamelog.PlayerGameLog(
        player_id=player_id, season=season, season_type_all_star="Regular Season"
    )
    reg_df = reg.get_data_frames()[0]
    if not reg_df.empty:
        reg_df["GAME_TYPE"] = "Regular Season"
        frames.append(reg_df)

    # Play-In tournament
    time.sleep(_DELAY)
    try:
        pi = playergamelog.PlayerGameLog(
            player_id=player_id, season=season, season_type_all_star="PlayIn"
        )
        pi_df = pi.get_data_frames()[0]
        if not pi_df.empty:
            pi_df["GAME_TYPE"] = "Play-In"
            frames.append(pi_df)
    except Exception:
        pass

    # Playoffs
    time.sleep(_DELAY)
    try:
        po = playergamelog.PlayerGameLog(
            player_id=player_id, season=season, season_type_all_star="Playoffs"
        )
        po_df = po.get_data_frames()[0]
        if not po_df.empty:
            po_df["GAME_TYPE"] = "Playoffs"
            frames.append(po_df)
    except Exception:
        pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Sort by actual date descending so playoffs appear first when most recent
    df["_date_parsed"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    df = df.sort_values("_date_parsed", ascending=False).drop(columns=["_date_parsed"]).reset_index(drop=True)

    if {"PTS", "REB", "AST"}.issubset(df.columns):
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df


def get_team_def_ratings(season: str = "2025-26") -> pd.DataFrame:
    """Fetch all team advanced stats including DEF_RATING."""
    time.sleep(_DELAY)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
    )
    df = stats.get_data_frames()[0]
    return df[["TEAM_ID", "TEAM_NAME", "DEF_RATING"]].copy()


def parse_prop_line(line: str) -> tuple[float, str] | None:
    """Parse '20+ points' → (20.0, 'PTS'). Returns None if unrecognized."""
    import re
    line = line.strip().lower()
    m = re.match(r"([0-9]+(?:\.[0-9]+)?)\s*\+?\s*(.*)", line)
    if not m:
        return None
    threshold = float(m.group(1))
    stat_raw = m.group(2).strip().rstrip("+").strip()
    col = STAT_COLUMN_MAP.get(stat_raw)
    if col is None:
        # try partial match
        for k, v in STAT_COLUMN_MAP.items():
            if k in stat_raw or stat_raw in k:
                col = v
                break
    if col is None:
        return None
    return threshold, col
