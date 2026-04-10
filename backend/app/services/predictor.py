"""Prediction service: fetch live odds, run ML model, write results to DB."""
from __future__ import annotations

import asyncio
import datetime
import logging
from pathlib import Path
from typing import TypedDict

import time

import joblib
import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.orm import Game, ModelPrediction, Player, PropLine, Sportsbook

logger = logging.getLogger(__name__)

# Must match FEATURE_COLS in ml/propsbasket/features/engineering.py
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
    "opp_def_rating",
    "opp_pace",
    "is_home",
    "rest_days",
    "is_back_to_back",
    "line_value",
    "implied_probability",
    "line_movement",
]

_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"


# ---------------------------------------------------------------------------
# Result type returned from the sync compute step
# ---------------------------------------------------------------------------


class PropPrediction(TypedDict):
    player_name: str
    player_team_abbrev: str | None
    home_team_full: str
    away_team_full: str
    line_value: float
    odds_val: int
    implied_prob: float
    model_prob: float
    edge: float


# ---------------------------------------------------------------------------
# Pure sync helpers (safe to run in a thread)
# ---------------------------------------------------------------------------


def _american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _parse_minutes(min_str: object) -> float:
    """Convert nba_api MIN string ('36:12') to decimal minutes."""
    try:
        parts = str(min_str).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except Exception:
        return 0.0


def _current_season() -> str:
    """Return the current NBA season string, e.g. '2025-26'."""
    today = datetime.date.today()
    # NBA season starts in October; before October we're still in the prior season
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _normalize_name(name: str) -> str:
    """Strip periods from initials/suffixes: 'R.J.' → 'RJ', 'Jr.' → 'Jr'."""
    return name.replace(".", "").strip()


def _find_nba_player(name: str) -> dict | None:
    matches = nba_players.find_players_by_full_name(name)
    if matches:
        return matches[0]
    normalized = _normalize_name(name)
    if normalized != name:
        matches = nba_players.find_players_by_full_name(normalized)
        if matches:
            return matches[0]
    return None


def _fetch_game_logs_for_players(player_names: list[str], season: str) -> pd.DataFrame:
    """Fetch fresh game logs from nba_api for only the named players."""
    all_logs: list[pd.DataFrame] = []
    not_found: list[str] = []

    for name in player_names:
        player = _find_nba_player(name)
        if not player:
            not_found.append(name)
            continue
        time.sleep(0.6)
        try:
            log = playergamelog.PlayerGameLog(player_id=player["id"], season=season)
            df = log.get_data_frames()[0]
        except Exception as exc:
            logger.warning("Failed to fetch logs for %s: %s", name, exc)
            continue
        if df.empty:
            continue
        df["player_name"] = name  # keep Odds API name for downstream joins
        all_logs.append(df)

    if not_found:
        logger.warning("Could not find nba_api IDs for: %s", not_found)

    if not all_logs:
        return pd.DataFrame()

    result = pd.concat(all_logs, ignore_index=True)
    result["GAME_DATE"] = pd.to_datetime(result["GAME_DATE"], format="mixed")
    result["MIN"] = result["MIN"].apply(_parse_minutes)
    result["PTS"] = pd.to_numeric(result["PTS"], errors="coerce").fillna(0)
    logger.info("Fetched fresh game logs for %d/%d players", len(all_logs), len(player_names))
    return result


def _load_team_stats(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Team stats not found at {p}. Run ml/scripts/ingest_nba.py first."
        )
    return pd.read_parquet(p)


# nba_api abbreviation → full name, e.g. "LAL" → "Los Angeles Lakers"
_ABBREV_TO_FULLNAME: dict[str, str] = {
    t["abbreviation"]: t["full_name"] for t in nba_teams.get_teams()
}


def _rolling_stats(logs: pd.DataFrame, windows: tuple[int, ...] = (5, 10)) -> dict:
    logs = logs.sort_values("GAME_DATE")
    result: dict[str, float] = {}
    for w in windows:
        recent = logs.tail(w)
        pts = recent["PTS"].values
        mins = recent["MIN"].values
        fga = recent["FGA"].values if "FGA" in recent.columns else np.array([])
        result[f"pts_avg_{w}g"] = float(np.mean(pts)) if len(pts) > 0 else 0.0
        result[f"pts_std_{w}g"] = float(np.std(pts, ddof=1)) if len(pts) >= 2 else 0.0
        result[f"min_avg_{w}g"] = float(np.mean(mins)) if len(mins) > 0 else 0.0
        result[f"fga_avg_{w}g"] = float(np.mean(fga)) if len(fga) > 0 else 0.0
    result["pts_trend"] = result["pts_avg_5g"] - result["pts_avg_10g"]
    return result


def _compute_prop_predictions() -> list[PropPrediction]:
    """Blocking: fetch live odds, ingest fresh game logs, run inference.

    Two-pass approach:
      Pass 1 — collect all player names from today's props (fast, API only)
      Pass 2 — fetch fresh nba_api game logs for only those players, then score

    Runs in a thread via asyncio.to_thread — must not touch the event loop.
    Returns a flat list of PropPrediction dicts ready for DB writes.
    """
    model_path = Path(settings.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run ml/scripts/train.py first."
        )
    model = joblib.load(model_path)
    team_stats = _load_team_stats(settings.team_stats_path)

    # Pass 1: fetch all events + props, collect into flat list
    events_resp = requests.get(
        f"{_ODDS_API_BASE}/events",
        params={"apiKey": settings.odds_api_key},
        timeout=15,
    )
    events_resp.raise_for_status()
    events: list[dict] = events_resp.json()
    logger.info("Fetched %d upcoming NBA events", len(events))

    # Collect raw prop records: {player_name, line, odds, implied_prob, home, away}
    raw_props: list[dict] = []
    for event in events:
        time.sleep(0.5)
        props_resp = requests.get(
            f"{_ODDS_API_BASE}/events/{event['id']}/odds",
            params={
                "apiKey": settings.odds_api_key,
                "regions": "us",
                "markets": "player_points",
                "bookmakers": "fanduel",
                "oddsFormat": "american",
            },
            timeout=15,
        )
        if props_resp.status_code == 404:
            continue
        if props_resp.status_code == 429:
            logger.warning("Odds API rate limit hit — stopping early")
            break
        props_resp.raise_for_status()
        event_odds = props_resp.json()

        for bookmaker in event_odds.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "player_points":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") != "Over":
                        continue
                    raw_props.append({
                        "player_name": outcome["description"],
                        "line_value": float(outcome.get("point", 0)),
                        "odds_val": int(outcome.get("price", -110)),
                        "home_team_full": event["home_team"],
                        "away_team_full": event["away_team"],
                    })

    if not raw_props:
        return []

    # Pass 2: fetch fresh game logs for only today's players
    player_names = list({p["player_name"] for p in raw_props})
    season = _current_season()
    logger.info("Fetching fresh game logs for %d players (season %s)...", len(player_names), season)
    game_logs = _fetch_game_logs_for_players(player_names, season)

    today = datetime.date.today()
    results: list[PropPrediction] = []

    for prop in raw_props:
        player_name = prop["player_name"]
        implied_prob = _american_to_implied(prop["odds_val"])

        player_logs = (
            game_logs[game_logs["player_name"] == player_name].sort_values("GAME_DATE")
            if not game_logs.empty
            else pd.DataFrame()
        )

        # Determine is_home and opponent via nba_api abbreviation map
        is_home = 0
        player_team_abbrev: str | None = None
        opp_full = prop["away_team_full"]

        if not player_logs.empty:
            last_matchup = player_logs.iloc[-1]["MATCHUP"]
            player_team_abbrev = last_matchup.split()[0]
            player_fullname = _ABBREV_TO_FULLNAME.get(player_team_abbrev, "")
            if player_fullname == prop["home_team_full"]:
                is_home = 1
                opp_full = prop["away_team_full"]
            else:
                is_home = 0
                opp_full = prop["home_team_full"]

        opp_row = team_stats[team_stats["TEAM_NAME"] == opp_full]
        opp_def_rating = float(opp_row["DEF_RATING"].values[0]) if len(opp_row) > 0 else 0.0
        opp_pace = float(opp_row["PACE"].values[0]) if len(opp_row) > 0 else 0.0

        if not player_logs.empty:
            rest_days = (pd.Timestamp(today) - player_logs.iloc[-1]["GAME_DATE"]).days
        else:
            rest_days = 3

        rolling = (
            _rolling_stats(player_logs)
            if not player_logs.empty
            else {k: 0.0 for k in ["pts_avg_5g", "pts_avg_10g", "pts_std_5g", "pts_std_10g", "min_avg_5g", "min_avg_10g"]}
        )

        features = {
            **rolling,
            "opp_def_rating": opp_def_rating,
            "opp_pace": opp_pace,
            "is_home": float(is_home),
            "rest_days": float(rest_days),
            "is_back_to_back": 1.0 if rest_days == 1 else 0.0,
            "line_value": prop["line_value"],
            "implied_probability": implied_prob,
            "line_movement": 0.0,
        }

        X = pd.DataFrame([features])[FEATURE_COLS].fillna(0)
        model_prob = float(model.predict_proba(X)[0, 1])

        results.append(PropPrediction(
            player_name=player_name,
            player_team_abbrev=player_team_abbrev,
            home_team_full=prop["home_team_full"],
            away_team_full=prop["away_team_full"],
            line_value=prop["line_value"],
            odds_val=prop["odds_val"],
            implied_prob=implied_prob,
            model_prob=model_prob,
            edge=model_prob - implied_prob,
        ))

    logger.info("Computed %d predictions", len(results))
    return results


# ---------------------------------------------------------------------------
# Async DB helpers
# ---------------------------------------------------------------------------


async def _get_or_create_player(db: AsyncSession, name: str, team: str | None) -> Player:
    result = await db.execute(select(Player).where(Player.name == name))
    player = result.scalar_one_or_none()
    if player is None:
        player = Player(name=name, team=team)
        db.add(player)
        await db.flush()
    return player


async def _get_or_create_sportsbook(db: AsyncSession, name: str) -> Sportsbook:
    result = await db.execute(select(Sportsbook).where(Sportsbook.name == name))
    sb = result.scalar_one_or_none()
    if sb is None:
        sb = Sportsbook(name=name)
        db.add(sb)
        await db.flush()
    return sb


async def _get_or_create_game(
    db: AsyncSession, date: datetime.date, home_team: str, away_team: str
) -> Game:
    result = await db.execute(
        select(Game).where(
            Game.date == date,
            Game.home_team == home_team,
            Game.away_team == away_team,
        )
    )
    game = result.scalar_one_or_none()
    if game is None:
        game = Game(date=date, home_team=home_team, away_team=away_team)
        db.add(game)
        await db.flush()
    return game


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_predictions(db: AsyncSession) -> int:
    """Fetch live odds, run model in a thread, persist predictions asynchronously.

    Returns the number of new predictions written to the DB.

    Notes:
    - Only player_points props are predicted — the model was trained on PTS.
    - Player name matching is case-insensitive exact match against nba_api player_name.
    - line_movement defaults to 0.0 (no historical odds tracking yet).
    """
    # All blocking I/O and CPU work runs in a thread — event loop stays free
    prop_predictions = await asyncio.to_thread(_compute_prop_predictions)

    if not prop_predictions:
        logger.info("No live NBA props — nothing to persist.")
        return 0

    # Clear today's predictions before inserting fresh ones so refresh is idempotent
    today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
    await db.execute(
        delete(ModelPrediction).where(ModelPrediction.prediction_time >= today_start)
    )
    await db.execute(
        delete(PropLine).where(PropLine.timestamp >= today_start)
    )
    await db.flush()

    sportsbook = await _get_or_create_sportsbook(db, "fanduel")
    today = datetime.date.today()
    now = datetime.datetime.utcnow()

    # Cache games looked up this run to avoid redundant DB queries
    game_cache: dict[tuple[str, str], Game] = {}

    for p in prop_predictions:
        game_key = (p["home_team_full"], p["away_team_full"])
        if game_key not in game_cache:
            game_cache[game_key] = await _get_or_create_game(
                db, today, p["home_team_full"], p["away_team_full"]
            )
        game = game_cache[game_key]

        player_obj = await _get_or_create_player(
            db, p["player_name"], p["player_team_abbrev"]
        )

        prop_line = PropLine(
            player_id=player_obj.id,
            game_id=game.id,
            sportsbook_id=sportsbook.id,
            line_value=p["line_value"],
            odds=p["odds_val"],
            market_type="points",
            timestamp=now,
        )
        db.add(prop_line)
        await db.flush()

        db.add(
            ModelPrediction(
                prop_line_id=prop_line.id,
                model_probability=p["model_prob"],
                implied_probability=p["implied_prob"],
                edge=p["edge"],
                prediction_time=now,
            )
        )

    await db.commit()
    logger.info("Persisted %d predictions", len(prop_predictions))
    return len(prop_predictions)
