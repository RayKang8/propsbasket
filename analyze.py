#!/usr/bin/env python3
"""NBA Player Prop Analyzer — terminal CLI.

Usage (interactive):
    python analyze.py

Usage (arguments):
    python analyze.py --player "LeBron James" --line "20+ points" --odds -166 --opponent "Golden State Warriors"
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.formula import analyze_prop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NBA Player Prop Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py
  python analyze.py --player "LeBron James" --line "20+ points" --odds -166 --opponent "Golden State Warriors"
  python analyze.py --player "Nikola Jokic" --line "10+ rebounds" --odds -200 --opponent "Lakers"
  python analyze.py --player "Stephen Curry" --line "4+ threes" --odds -130 --opponent "Boston Celtics"
  python analyze.py --player "Giannis Antetokounmpo" --line "30+ pra" --odds -250 --opponent "Heat"
        """,
    )
    parser.add_argument("--player", help="Player full name")
    parser.add_argument("--line", help="Prop line, e.g. '20+ points'")
    parser.add_argument("--odds", type=int, help="American odds, e.g. -166 or +120")
    parser.add_argument("--opponent", help="Tonight's opponent (team name, city, or abbreviation)")
    parser.add_argument("--season", default="2025-26", help="NBA season (default: 2025-26)")
    args = parser.parse_args()

    if not all([args.player, args.line, args.odds is not None, args.opponent]):
        print("\n=== NBA Player Prop Analyzer ===")
        print("Enter prop details (or Ctrl+C to exit)\n")
        try:
            if not args.player:
                args.player = input("Player name: ").strip()
            if not args.line:
                args.line = input("Prop line (e.g. '20+ points'): ").strip()
            if args.odds is None:
                args.odds = int(input("American odds (e.g. -166 or +120): ").strip().replace("+", ""))
            if not args.opponent:
                args.opponent = input("Tonight's opponent: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

    analyze_prop(
        player_name=args.player,
        prop_line=args.line,
        odds=args.odds,
        opponent=args.opponent,
        season=args.season,
    )


if __name__ == "__main__":
    main()
