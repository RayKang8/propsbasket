#!/usr/bin/env python3
"""NBA Player Prop Analyzer — terminal CLI.

Usage (interactive):
    python analyze.py

Usage (arguments):
    python analyze.py --player "LeBron James" --line "20+ points" --odds -166 --opponent "Golden State Warriors"
    python analyze.py --player "LeBron James" --line "20+ points" --odds -166 --line "8+ rebounds" --odds -115 --opponent "GSW"
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.formula import analyze_prop


def _parse_odds(raw: str) -> int:
    return int(raw.strip().replace("+", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NBA Player Prop Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py
  python analyze.py --player "LeBron James" --line "20+ points" --odds -166 --opponent "Golden State Warriors"
  python analyze.py --player "LeBron James" --line "20+ points" --odds -166 --line "8+ rebounds" --odds -115 --opponent "GSW"
  python analyze.py --player "Nikola Jokic" --line "10+ rebounds" --odds -200 --opponent "Lakers"
  python analyze.py --player "Stephen Curry" --line "4+ threes" --odds -130 --opponent "Boston Celtics"
        """,
    )
    parser.add_argument("--player", help="Player full name")
    parser.add_argument("--line", action="append", dest="lines", help="Prop line (repeatable), e.g. '20+ points'")
    parser.add_argument("--odds", action="append", dest="odds_list", help="American odds (repeatable), e.g. -166 or +120")
    parser.add_argument("--opponent", help="Tonight's opponent (team name, city, or abbreviation)")
    parser.add_argument("--season", default="2025-26", help="NBA season (default: 2025-26)")
    args = parser.parse_args()

    # ── Interactive mode ──────────────────────────────────────────────────────
    if not all([args.player, args.lines, args.odds_list, args.opponent]):
        print("\n=== NBA Player Prop Analyzer ===")
        print("Enter prop details (or Ctrl+C to exit)\n")
        try:
            if not args.player:
                args.player = input("Player name: ").strip()
            if not args.opponent:
                args.opponent = input("Tonight's opponent: ").strip()

            props: list[tuple[str, int]] = []
            # Collect first prop (required)
            first_line = input("Prop line (e.g. '20+ points'): ").strip()
            first_odds = _parse_odds(input("American odds (e.g. -166 or +120): "))
            props.append((first_line, first_odds))

            # Offer additional props
            while True:
                more = input("\nAdd another prop for this player? (e.g. '8+ rebounds -115', or Enter to skip): ").strip()
                if not more:
                    break
                parts = more.rsplit(None, 1)
                if len(parts) == 2:
                    try:
                        props.append((parts[0].strip(), _parse_odds(parts[1])))
                        continue
                    except ValueError:
                        pass
                # If they didn't include odds inline, ask separately
                extra_odds = _parse_odds(input(f"  Odds for '{more}': "))
                props.append((more, extra_odds))

        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)
    else:
        lines = args.lines or []
        odds_list = [_parse_odds(o) for o in (args.odds_list or [])]
        if len(lines) != len(odds_list):
            print("Error: number of --line and --odds arguments must match.")
            sys.exit(1)
        props = list(zip(lines, odds_list))

    # ── Run analysis for each prop ────────────────────────────────────────────
    for prop_line, odds in props:
        analyze_prop(
            player_name=args.player,
            prop_line=prop_line,
            odds=odds,
            opponent=args.opponent,
            season=args.season,
        )


if __name__ == "__main__":
    main()
