"""Fetch NBA player prop lines from The Odds API."""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"


def get_events(api_key: str) -> list[dict]:
    """Fetch today's NBA event list (no odds, just game metadata).

    Returns:
        List of event dicts with id, home_team, away_team, commence_time.
    """
    url = f"{BASE_URL}/sports/{SPORT}/events"
    response = requests.get(url, params={"apiKey": api_key}, timeout=10)
    response.raise_for_status()
    events = response.json()
    logger.info("Found %d NBA events today", len(events))
    return events


def get_event_props(
    event_id: str,
    api_key: str,
    markets: str = "player_points",
    bookmakers: str = "fanduel",
) -> dict:
    """Fetch player prop odds for a single event.

    Player props must be fetched per-event via the event-level endpoint.
    The bulk /odds endpoint only supports game markets (h2h, spreads, totals).

    Returns:
        Single event dict with bookmakers/markets/outcomes nested inside.
    """
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "bookmakers": bookmakers,
        "oddsFormat": "american",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def get_player_props(
    api_key: str | None = None,
    markets: str = "player_points",
    bookmakers: str = "fanduel",
) -> list[dict]:
    """Fetch live player prop lines for all of today's NBA games.

    Fetches the event list first, then queries props per event (required
    by The Odds API v4 — player props are not available on the bulk endpoint).

    Args:
        api_key: API key (defaults to ODDS_API_KEY env var)
        markets: Comma-separated prop markets, e.g. "player_points,player_rebounds"
        bookmakers: Comma-separated sportsbooks to include

    Returns:
        List of event dicts with nested odds/outcomes data, one per game.
    """
    key = api_key or os.environ["ODDS_API_KEY"]
    events = get_events(key)
    if not events:
        return []

    results = []
    for event in events:
        try:
            data = get_event_props(event["id"], key, markets=markets, bookmakers=bookmakers)
            results.append(data)
        except Exception as exc:
            logger.warning("Could not fetch props for event %s: %s", event.get("id"), exc)

    logger.info("Fetched props for %d/%d events", len(results), len(events))
    return results


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
