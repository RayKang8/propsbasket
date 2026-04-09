#!/usr/bin/env python3
"""Score today's live props and print top edges.

Usage:
    python scripts/predict_live.py
    python scripts/predict_live.py --markets player_points,player_rebounds
    python scripts/predict_live.py --top-n 20 --min-edge 0.05
    python scripts/predict_live.py --model lightgbm
"""
import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from propsbasket.ingestion.nba_stats import get_game_logs_for_players, get_team_stats
from propsbasket.ingestion.odds import get_player_props, parse_props
from propsbasket.prediction.live import build_prediction_rows, score_props

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def _refresh_team_stats(season: str) -> None:
    """Re-fetch and overwrite team stats parquet (fast, single request)."""
    try:
        logger.info("Refreshing team stats for %s...", season)
        df = get_team_stats(season=season)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RAW_DIR / f"team_stats_{season}.parquet", index=False)
        logger.info("Team stats saved.")
    except Exception as exc:
        logger.warning("Could not refresh team stats: %s", exc)


def main(
    markets: str,
    model: str,
    top_n: int,
    min_edge: float,
    season: str,
) -> None:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY environment variable is not set.")

    # 1. Fetch live odds
    logger.info("Fetching live props from The Odds API (markets: %s)...", markets)
    events = get_player_props(api_key=api_key, markets=markets)
    if not events:
        logger.warning("No events returned from The Odds API — no games today?")
        return

    props = parse_props(events)
    if not props:
        logger.warning("No prop records parsed. Check that FanDuel has lines posted.")
        return

    # 2. Scoped ingestion — only players with active props
    player_names = sorted({p["player_name"] for p in props})
    logger.info("Fetching game logs for %d players...", len(player_names))
    game_logs = get_game_logs_for_players(player_names, season=season)

    if game_logs.empty:
        logger.error("No game logs returned. Cannot score props.")
        return

    # 3. Refresh team stats (single fast request)
    _refresh_team_stats(season)

    # 4. Build feature rows + score
    feature_rows = build_prediction_rows(props, game_logs)
    if feature_rows.empty:
        logger.error("No feature rows built — all players skipped.")
        return

    results = score_props(feature_rows, model_name=model)

    # 5. Filter and display
    if min_edge > 0:
        results = results[results["edge"] >= min_edge]

    top = results.head(top_n)

    if top.empty:
        print(f"\nNo props found with edge >= {min_edge:.2f}")
        return

    print(f"\n{'='*70}")
    print(f"  TOP {len(top)} EDGES  |  model={model}  |  markets={markets}")
    print(f"{'='*70}")
    print(
        f"{'Player':<25} {'Market':<20} {'Line':>6} "
        f"{'Implied':>8} {'Model':>7} {'Edge':>7}"
    )
    print("-" * 70)
    for _, row in top.iterrows():
        print(
            f"{row['player_name']:<25} {row['market']:<20} {row['line']:>6.1f} "
            f"{row['implied_prob']:>7.1%} {row['model_prob']:>6.1%} {row['edge']:>+7.1%}"
        )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score live NBA player props.")
    parser.add_argument(
        "--markets",
        default="player_points",
        help="Comma-separated Odds API market keys (default: player_points)",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "lightgbm", "logistic"],
        help="Model artifact to use (default: xgboost)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top edges to display (default: 10)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.0,
        help="Minimum edge threshold to include (default: 0.0)",
    )
    parser.add_argument(
        "--season",
        default="2024-25",
        help="NBA season string for game log fetching (default: 2024-25)",
    )
    args = parser.parse_args()
    main(
        markets=args.markets,
        model=args.model,
        top_n=args.top_n,
        min_edge=args.min_edge,
        season=args.season,
    )
