#!/usr/bin/env python3
"""Build training dataset from raw game logs."""

import logging
from pathlib import Path

import pandas as pd

from propsbasket.features.engineering import add_game_context, add_rolling_stats, add_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/features.parquet")


# ---------------------------------------------------------------------------
# Parsing helpers for nba_api raw output
# ---------------------------------------------------------------------------

def _parse_minutes(min_str: object) -> float:
    """Convert nba_api MIN string ('36:12') to decimal minutes (36.2)."""
    try:
        parts = str(min_str).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except Exception:
        return 0.0


def _parse_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """Extract is_home (1/0) and opponent abbreviation from MATCHUP column.

    MATCHUP examples:
      'LAL vs. GSW'  → home game, opponent = 'GSW'
      'LAL @ GSW'    → away game, opponent = 'GSW'
    """
    df = df.copy()
    df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)
    df["opponent"] = df["MATCHUP"].str.split().str[-1]
    return df


def _compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest_days and is_back_to_back per player, sorted by date."""
    df = df.copy()
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    df["rest_days"] = (
        df.groupby("PLAYER_ID")["GAME_DATE"]
        .diff()
        .dt.days
        .fillna(3)  # assume 3 days rest for season opener
    )
    df["is_back_to_back"] = (df["rest_days"] <= 1).astype(int)
    return df


def _simulate_prop_line(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate prop line features for training data (no real odds available).

    line_value       — rolling 10-game points average (shifted to avoid leakage)
    implied_probability — 0.5 (fair odds; real values come from The Odds API)
    line_movement    — 0.0 (unknown without historical odds)

    These columns are replaced with real odds data when running inference
    against live prop lines.
    """
    df = df.copy()
    df["line_value"] = (
        df.groupby("PLAYER_ID")["PTS"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .round(1)
    )
    df["implied_probability"] = 0.5
    df["line_movement"] = 0.0
    return df


def _load_team_stats() -> pd.DataFrame | None:
    """Load the most recent team stats parquet available."""
    paths = sorted(RAW_DIR.glob("team_stats_*.parquet"))
    if not paths:
        logger.warning("No team stats files found — opp_def_rating and opp_pace will be NaN.")
        return None
    # Use the last (most recent) season's team stats
    return pd.read_parquet(paths[-1])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    log_paths = sorted(RAW_DIR.glob("game_logs_*.parquet"))
    if not log_paths:
        logger.error(
            "No game log files found in %s. Run ingest_nba.py first.", RAW_DIR
        )
        raise SystemExit(1)

    logger.info("Loading %d game log file(s)...", len(log_paths))
    df = pd.concat([pd.read_parquet(p) for p in log_paths], ignore_index=True)
    logger.info("Loaded %d total game log rows across %d file(s)", len(df), len(log_paths))

    # Parse nba_api types
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
    df["MIN"] = df["MIN"].apply(_parse_minutes)
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce").fillna(0).astype(int)

    # Game context columns
    df = _parse_matchup(df)
    df = _compute_rest_days(df)

    # Rolling stats (no leakage — shift(1) inside add_rolling_stats)
    df = add_rolling_stats(df)

    # Opponent defensive context
    team_stats = _load_team_stats()
    if team_stats is not None:
        df = add_game_context(df, team_stats)
    else:
        df["opp_def_rating"] = float("nan")
        df["opp_pace"] = float("nan")

    # Simulated prop line features (for training only)
    df = _simulate_prop_line(df)

    # Binary target: did player outscore the (simulated) line?
    df = add_target(df)

    # Drop rows where we can't compute rolling features (first game per player)
    before = len(df)
    df = df.dropna(subset=["pts_avg_5g", "line_value", "did_hit"])
    logger.info("Dropped %d rows with NaN features (%d remain)", before - len(df), len(df))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Feature dataset saved → %s (%d rows, %d cols)", OUTPUT_PATH, len(df), len(df.columns))


if __name__ == "__main__":
    main()
