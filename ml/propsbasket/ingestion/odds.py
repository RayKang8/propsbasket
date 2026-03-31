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
