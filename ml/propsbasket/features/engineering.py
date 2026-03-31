"""Feature engineering for player prop prediction."""

from __future__ import annotations

import pandas as pd

FEATURE_COLS = [
    "pts_avg_5g",
    "pts_avg_10g",
    "pts_std_5g",
    "pts_std_10g",
    "min_avg_5g",
    "min_avg_10g",
    "opp_def_rating",
    "opp_pace",
    "is_home",
    "rest_days",
    "is_back_to_back",
    "line_value",
    "implied_probability",
    "line_movement",
]


def add_rolling_stats(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling averages and std devs for points and minutes.

    Expects df sorted by date ascending with columns: PLAYER_ID, PTS, MIN.
    All rolling windows use shift(1) to avoid data leakage.
    """
    if windows is None:
        windows = [5, 10]
    df = df.sort_values("GAME_DATE").copy()
    for w in windows:
        df[f"pts_avg_{w}g"] = (
            df.groupby("PLAYER_ID")["PTS"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"pts_std_{w}g"] = (
            df.groupby("PLAYER_ID")["PTS"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        )
        df[f"min_avg_{w}g"] = (
            df.groupby("PLAYER_ID")["MIN"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
    return df


def add_game_context(df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge opponent defensive rating and pace into each game log row."""
    opp_stats = team_stats[["TEAM_ABBREVIATION", "DEF_RATING", "PACE"]].rename(
        columns={
            "TEAM_ABBREVIATION": "opponent",
            "DEF_RATING": "opp_def_rating",
            "PACE": "opp_pace",
        }
    )
    return df.merge(opp_stats, on="opponent", how="left")


def add_target(df: pd.DataFrame, line_col: str = "line_value") -> pd.DataFrame:
    """Add binary target: 1 if player scored more than the prop line."""
    df = df.copy()
    df["did_hit"] = (df["PTS"] > df[line_col]).astype(int)
    return df
