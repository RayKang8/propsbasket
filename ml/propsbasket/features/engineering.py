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
    "fga_avg_5g",
    "fga_avg_10g",
    "pts_trend",
    "pts_avg_vs_opp",
    "games_vs_opp",
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
    if windows is None:
        windows = [5, 10]
    df = df.sort_values("GAME_DATE").copy()
    for w in windows:
        df[f"pts_avg_{w}g"] = (
            df.groupby("Player_ID")["PTS"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"pts_std_{w}g"] = (
            df.groupby("Player_ID")["PTS"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        )
        df[f"min_avg_{w}g"] = (
            df.groupby("Player_ID")["MIN"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"fga_avg_{w}g"] = (
            df.groupby("Player_ID")["FGA"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
    # Trend: positive = player scoring more recently than their 10-game baseline (hot)
    #        negative = player scoring less recently (cold)
    df["pts_trend"] = df["pts_avg_5g"] - df["pts_avg_10g"]
    return df

def add_game_context(df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge opponent defensive rating and pace into each game log row."""
    df = df.copy()
    df["opp_abbrev"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s*(\w+)")
    df["is_home"] = (~df["MATCHUP"].str.contains("@")).astype(int)
    abbrev_map = {name.split()[-1]: name for name in team_stats["TEAM_NAME"]}
    df["opp_team_name"] = df["opp_abbrev"].map(abbrev_map)
    opp_stats = team_stats[["TEAM_NAME", "DEF_RATING", "PACE"]].rename(
        columns={
            "TEAM_NAME": "opp_team_name",
            "DEF_RATING": "opp_def_rating",
            "PACE": "opp_pace",
        }
    )
    return df.merge(opp_stats, on="opp_team_name", how="left")

def add_vs_opponent_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add player's historical average points against each specific opponent.

    Uses expanding mean with shift(1) to avoid leakage — only uses games
    against that opponent that occurred *before* the current game.

    Requires 'opponent' column (team abbreviation, e.g. 'GSW') and 'Player_ID'.
    Falls back to the player's overall pts_avg_10g when no prior history exists.
    """
    df = df.sort_values("GAME_DATE").copy()

    df["pts_avg_vs_opp"] = (
        df.groupby(["Player_ID", "opponent"])["PTS"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    df["games_vs_opp"] = (
        df.groupby(["Player_ID", "opponent"])["PTS"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).count())
    )

    # For first-ever game vs an opponent, fall back to overall 10-game average
    df["pts_avg_vs_opp"] = df["pts_avg_vs_opp"].fillna(df["pts_avg_10g"])
    df["games_vs_opp"] = df["games_vs_opp"].fillna(0)

    return df


def add_target(df: pd.DataFrame, line_col: str = "line_value") -> pd.DataFrame:
    """Add binary target: 1 if player scored more than the prop line."""
    df = df.copy()
    df["did_hit"] = (df["PTS"] > df[line_col]).astype(int)
    return df
