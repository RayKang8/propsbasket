"""Live prediction pipeline: Odds API props + recent game logs → edge scores."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from nba_api.stats.static import teams as nba_teams

from propsbasket.features.engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("models/artifacts")
RAW_DIR = Path("data/raw")


def _build_abbrev_to_fullname() -> dict[str, str]:
    """Map nba_api team abbreviation → full name matching Odds API. e.g. 'LAL' → 'Los Angeles Lakers'."""
    return {t["abbreviation"]: t["full_name"] for t in nba_teams.get_teams()}


def _parse_minutes(min_str: object) -> float:
    try:
        parts = str(min_str).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except Exception:
        return 0.0


def _rolling_stats(games: pd.DataFrame) -> dict:
    """Compute rolling form features from a player's sorted game log.

    Uses actual last-N games with no shift — appropriate for inference
    where we're predicting the *next* (unseen) game.
    """
    def tail_mean(col: str, n: int) -> float:
        return float(games[col].iloc[-n:].mean()) if not games.empty else 0.0

    def tail_std(col: str, n: int) -> float:
        vals = games[col].iloc[-n:]
        return float(vals.std()) if len(vals) > 1 else 0.0

    return {
        "pts_avg_5g": tail_mean("PTS", 5),
        "pts_avg_10g": tail_mean("PTS", 10),
        "pts_std_5g": tail_std("PTS", 5),
        "pts_std_10g": tail_std("PTS", 10),
        "min_avg_5g": tail_mean("MIN", 5),
        "min_avg_10g": tail_mean("MIN", 10),
    }


def _rest_days(games: pd.DataFrame, today: date) -> tuple[int, int]:
    last_game = games["GAME_DATE"].max()
    if pd.isna(last_game):
        return 3, 0
    delta = (today - last_game.date()).days
    return delta, int(delta <= 1)


def _load_team_stats() -> pd.DataFrame | None:
    paths = sorted(RAW_DIR.glob("team_stats_*.parquet"))
    if not paths:
        logger.warning("No team stats parquet found — opp features will be 0")
        return None
    return pd.read_parquet(paths[-1])


def build_prediction_rows(
    props: list[dict],
    game_logs: pd.DataFrame,
    today: date | None = None,
) -> pd.DataFrame:
    """Assemble one feature row per prop record, ready for model scoring.

    Args:
        props: Flat prop dicts from parse_props() — one per player/market/Over outcome.
        game_logs: Recent game logs from get_game_logs_for_players().
        today: Reference date for rest_days calculation (defaults to date.today()).

    Returns:
        DataFrame with FEATURE_COLS + metadata columns (player_name, market, line, implied_prob, etc.).
    """
    if today is None:
        today = date.today()

    team_stats = _load_team_stats()
    abbrev_map = _build_abbrev_to_fullname()

    logs = game_logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], format="mixed")
    logs["MIN"] = logs["MIN"].apply(_parse_minutes)
    logs["PTS"] = pd.to_numeric(logs["PTS"], errors="coerce").fillna(0)

    # Build full_name → {def_rating, pace} lookup from team_stats
    opp_lookup: dict[str, dict] = {}
    if team_stats is not None:
        for _, row in team_stats.iterrows():
            opp_lookup[row["TEAM_NAME"]] = {
                "opp_def_rating": float(row.get("DEF_RATING", 0.0)),
                "opp_pace": float(row.get("PACE", 0.0)),
            }

    rows = []
    for prop in props:
        player_name = prop["player_name"]
        player_games = logs[logs["player_name"] == player_name].sort_values("GAME_DATE")

        if player_games.empty:
            logger.warning("No game logs for '%s' — skipping prop", player_name)
            continue

        rolling = _rolling_stats(player_games)
        rest, b2b = _rest_days(player_games, today)

        # Determine is_home and opponent using player's team abbreviation + Odds API game
        last_matchup = player_games["MATCHUP"].iloc[-1]
        player_abbrev = last_matchup.split()[0]  # e.g. "LAL" from "LAL vs. GSW"
        player_fullname = abbrev_map.get(player_abbrev, "")

        home_team = prop.get("home_team", "")
        away_team = prop.get("away_team", "")

        if player_fullname and player_fullname == home_team:
            is_home = 1
            opp_fullname = away_team
        elif player_fullname and player_fullname == away_team:
            is_home = 0
            opp_fullname = home_team
        else:
            # Fallback: use last game's home/away indicator; opponent unknown
            is_home = int("vs." in last_matchup)
            opp_fullname = ""
            logger.debug(
                "Could not match '%s' (%s) to Odds API teams '%s'/'%s'",
                player_name, player_abbrev, home_team, away_team,
            )

        opp_ctx = opp_lookup.get(opp_fullname, {"opp_def_rating": 0.0, "opp_pace": 0.0})

        rows.append({
            # Metadata (not used by model, kept for output)
            "player_name": player_name,
            "market": prop["market"],
            "line": prop["line"],
            "implied_prob": prop["implied_prob"],
            "home_team": home_team,
            "away_team": away_team,
            # Model features
            **rolling,
            "opp_def_rating": opp_ctx["opp_def_rating"],
            "opp_pace": opp_ctx["opp_pace"],
            "is_home": is_home,
            "rest_days": rest,
            "is_back_to_back": b2b,
            "line_value": prop["line"],
            "implied_probability": prop["implied_prob"],
            "line_movement": 0.0,
        })

    logger.info("Built %d prediction rows from %d props", len(rows), len(props))
    return pd.DataFrame(rows)


def score_props(
    feature_rows: pd.DataFrame,
    model_name: str = "xgboost",
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> pd.DataFrame:
    """Score feature rows with the trained model and compute edges.

    Args:
        feature_rows: Output of build_prediction_rows().
        model_name: Which .joblib artifact to load.
        artifacts_dir: Directory containing model artifacts.

    Returns:
        DataFrame with model_prob and edge columns added, sorted by edge descending.
    """
    artifact_path = artifacts_dir / f"{model_name}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No model artifact at {artifact_path}. Run scripts/train.py first."
        )

    model = joblib.load(artifact_path)
    X = feature_rows[FEATURE_COLS].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    result = feature_rows.copy()
    result["model_prob"] = probs
    result["edge"] = result["model_prob"] - result["implied_prob"]
    return result.sort_values("edge", ascending=False).reset_index(drop=True)
