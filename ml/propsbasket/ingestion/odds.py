"""Fetch NBA player prop lines from The Odds API."""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"


def get_player_props(
    api_key: str | None = None,
    markets: str = "player_points",
    bookmakers: str = "fanduel",
) -> list[dict]:
    """Fetch live player prop lines from The Odds API.

    Args:
        api_key: API key (defaults to ODDS_API_KEY env var)
        markets: Comma-separated prop markets, e.g. "player_points,player_rebounds"
        bookmakers: Comma-separated sportsbooks to include

    Returns:
        List of event dicts with nested odds/outcomes data
    """
    key = api_key or os.environ["ODDS_API_KEY"]
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": markets,
        "bookmakers": bookmakers,
        "oddsFormat": "american",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    logger.info("Fetched %d events from The Odds API", len(data))
    return data


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability (vig not removed)."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def parse_props(events: list[dict], bookmaker: str = "fanduel") -> list[dict]:
    """Parse raw Odds API event list into flat prop records.

    Each record contains:
      player_name, line, implied_prob, market,
      home_team, away_team, game_id, commence_time

    Args:
        events: Raw list returned by get_player_props()
        bookmaker: Which bookmaker's lines to use

    Returns:
        List of prop dicts, one per player/market/over-under outcome.
    """
    records = []
    for event in events:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        game_id = event.get("id", "")
        commence_time = event.get("commence_time", "")

        for bm in event.get("bookmakers", []):
            if bm["key"] != bookmaker:
                continue
            for market in bm.get("markets", []):
                market_key = market["key"]
                for outcome in market.get("outcomes", []):
                    # Outcomes come in Over/Under pairs; we want the Over
                    if outcome.get("name") == "Over":
                        records.append({
                            "player_name": outcome.get("description", ""),
                            "line": outcome.get("point"),
                            "implied_prob": american_to_implied_prob(outcome["price"]),
                            "market": market_key,
                            "home_team": home_team,
                            "away_team": away_team,
                            "game_id": game_id,
                            "commence_time": commence_time,
                        })
    logger.info("Parsed %d prop records from %d events", len(records), len(events))
    return records
