"""Core formula engine — runs all 5 steps and prints the analysis."""

from __future__ import annotations

import sys

import pandas as pd

from src.nba import (
    find_player,
    get_game_logs,
    get_team_def_ratings,
    parse_prop_line,
    resolve_team_abbrev,
)
from src.injuries import get_injuries, get_player_injury_status, get_team_injuries


# ── helpers ──────────────────────────────────────────────────────────────────

def _american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _hit_rate(df: pd.DataFrame, col: str, threshold: float) -> tuple[int, int, float]:
    """Return (hits, games, rate) for games where col >= threshold."""
    valid = df.dropna(subset=[col])
    hits = int((valid[col] >= threshold).sum())
    games = len(valid)
    rate = hits / games if games > 0 else 0.0
    return hits, games, rate


def _tier_label(rank: int, total: int) -> str:
    third = total // 3
    if rank <= third:
        return "top third (toughest defense)"
    elif rank <= third * 2:
        return "middle third"
    else:
        return "bottom third (weakest defense)"


# ── main entry ────────────────────────────────────────────────────────────────

def analyze_prop(
    player_name: str,
    prop_line: str,
    odds: int,
    opponent: str,
    season: str = "2024-25",
) -> None:
    sep = "=" * 66

    # ── parse prop line ───────────────────────────────────────────────────────
    parsed = parse_prop_line(prop_line)
    if parsed is None:
        print(f"Could not parse prop line: '{prop_line}'")
        print("Examples: '20+ points', '8+ assists', '10+ rebounds', '2+ threes', '30+ pra'")
        sys.exit(1)
    threshold, stat_col = parsed

    # ── resolve opponent abbreviation ─────────────────────────────────────────
    opp_abbrev = resolve_team_abbrev(opponent)
    if opp_abbrev is None:
        print(f"Could not resolve team: '{opponent}'")
        sys.exit(1)

    # ── resolve player ────────────────────────────────────────────────────────
    player = find_player(player_name)
    if player is None:
        print(f"Could not find player: '{player_name}'")
        sys.exit(1)

    print(f"\n{sep}")
    print(f"  PROP ANALYSIS: {player['full_name']} — {prop_line.upper()} vs. {opponent}")
    print(f"  Season: {season}   Odds: {odds:+d}")
    print(sep)

    # ── fetch data ────────────────────────────────────────────────────────────
    print("\nFetching data...")
    injury_df = get_injuries()
    game_logs = get_game_logs(player["id"], season=season)
    def_ratings = get_team_def_ratings(season=season)

    if game_logs.empty:
        print("No game logs found for this player/season.")
        sys.exit(1)

    if stat_col not in game_logs.columns:
        print(f"Stat column '{stat_col}' not found in game logs.")
        print(f"Available columns: {list(game_logs.columns)}")
        sys.exit(1)

    # ── STEP 0: Injury report ─────────────────────────────────────────────────
    print(f"\n{'─'*66}")
    print("  STEP 0 — INJURY & AVAILABILITY CHECK")
    print(f"{'─'*66}")

    target_status = get_player_injury_status(player["full_name"], injury_df)
    if target_status:
        flag = "  ⚠  WARNING:" if target_status.upper() in ("QUESTIONABLE", "DOUBTFUL", "OUT") else "  "
        print(f"{flag} {player['full_name']}: {target_status.upper()}")
    else:
        print(f"  {player['full_name']}: ACTIVE (not on injury report)")

    # Find opponent's team abbreviation in the team stats to get full name
    opp_team_row = def_ratings[def_ratings["TEAM_NAME"].str.contains(opponent, case=False, na=False)]
    if opp_team_row.empty:
        # fallback: try matching by abbreviation via game logs MATCHUP column
        opp_display_name = opponent
    else:
        opp_display_name = opp_team_row.iloc[0]["TEAM_NAME"]

    # Get player's team abbreviation from their most recent game
    player_team_abbrev = None
    if "MATCHUP" in game_logs.columns and len(game_logs) > 0:
        # MATCHUP looks like "LAL vs. GSW" or "LAL @ GSW"
        matchup = game_logs.iloc[0]["MATCHUP"]
        player_team_abbrev = matchup.split()[0]

    opp_injuries = get_team_injuries(opp_abbrev, injury_df)
    player_team_injuries = get_team_injuries(player_team_abbrev, injury_df) if player_team_abbrev else pd.DataFrame()

    if not opp_injuries.empty:
        print(f"\n  {opp_display_name} injury report:")
        for _, row in opp_injuries.iterrows():
            status = row.get("status", "")
            comment = row.get("comment", "")
            comment_str = f" — {comment}" if comment and str(comment) != "nan" else ""
            print(f"    • {row['player_name']}: {status.upper()}{comment_str}")
    else:
        print(f"\n  {opp_display_name}: No players on injury report")

    if not player_team_injuries.empty and player_team_abbrev:
        # Exclude the target player themselves (already shown above)
        others = player_team_injuries[
            player_team_injuries["player_name"].str.lower() != player["full_name"].lower()
        ]
        if not others.empty:
            print(f"\n  {player_team_abbrev} injury report (teammates):")
            for _, row in others.iterrows():
                status = row.get("status", "")
                comment = row.get("comment", "")
                comment_str = f" — {comment}" if comment and str(comment) != "nan" else ""
                print(f"    • {row['player_name']}: {status.upper()}{comment_str}")

    # ── STEP 1: Season hit rate ───────────────────────────────────────────────
    print(f"\n{'─'*66}")
    print("  STEP 1 — SEASON HIT RATE")
    print(f"{'─'*66}")

    s1_hits, s1_games, s1_rate = _hit_rate(game_logs, stat_col, threshold)
    print(f"  {player['full_name']} has hit {prop_line} in {s1_hits}/{s1_games} games this season")
    print(f"  Season hit rate: {s1_rate * 100:.1f}%")

    # ── STEP 2: Similar defense hit rate ─────────────────────────────────────
    print(f"\n{'─'*66}")
    print("  STEP 2 — SIMILAR DEFENSE HIT RATE")
    print(f"{'─'*66}")

    # Get opponent's defensive rating
    opp_def_row = def_ratings[def_ratings["TEAM_NAME"].str.contains(opponent, case=False, na=False)]
    if opp_def_row.empty:
        print(f"  Could not find defensive rating for '{opponent}' — skipping Step 2")
        s2_rate = s1_rate
        direct_rate = None
        tier_rate = None
    else:
        opp_def_rtg = float(opp_def_row.iloc[0]["DEF_RATING"])

        # Rank all teams by DefRtg (ascending = better defense)
        def_sorted = def_ratings.sort_values("DEF_RATING").reset_index(drop=True)
        total_teams = len(def_sorted)
        opp_rank = int(def_sorted[def_sorted["TEAM_NAME"].str.contains(opponent, case=False, na=False)].index[0]) + 1
        tier = _tier_label(opp_rank, total_teams)

        third = total_teams // 3
        if opp_rank <= third:
            tier_teams = def_sorted.iloc[:third]
        elif opp_rank <= third * 2:
            tier_teams = def_sorted.iloc[third:third * 2]
        else:
            tier_teams = def_sorted.iloc[third * 2:]

        tier_abbrevs = set()
        for _, row in tier_teams.iterrows():
            abbrev = resolve_team_abbrev(row["TEAM_NAME"])
            if abbrev:
                tier_abbrevs.add(abbrev)

        print(f"  {opp_display_name} defensive rating: {opp_def_rtg:.1f} (rank #{opp_rank} of {total_teams})")
        print(f"  Defensive tier: {tier}")

        # Direct history vs opponent
        direct_games = pd.DataFrame()
        if "MATCHUP" in game_logs.columns:
            direct_games = game_logs[game_logs["MATCHUP"].str.contains(opp_abbrev, case=False, na=False)]

        if len(direct_games) > 0:
            d_hits, d_games, direct_rate = _hit_rate(direct_games, stat_col, threshold)
            print(f"\n  vs. {opp_display_name} (direct history): {d_hits}/{d_games} games = {direct_rate*100:.1f}%")
        else:
            direct_rate = None
            print(f"\n  vs. {opp_display_name} (direct history): No games played this season — using tier rate only")

        # Hit rate vs similar-tier teams
        tier_games = pd.DataFrame()
        if "MATCHUP" in game_logs.columns and tier_abbrevs:
            pattern = "|".join(tier_abbrevs)
            tier_games = game_logs[game_logs["MATCHUP"].str.contains(pattern, case=False, na=False)]
            # Exclude direct opponent games from tier (already counted separately)
            tier_games = tier_games[~tier_games["MATCHUP"].str.contains(opp_abbrev, case=False, na=False)]

        if len(tier_games) > 0:
            t_hits, t_games, tier_rate = _hit_rate(tier_games, stat_col, threshold)
            print(f"  vs. similar-tier teams ({tier}): {t_hits}/{t_games} games = {tier_rate*100:.1f}%")
        else:
            tier_rate = None
            print(f"  vs. similar-tier teams: No games found in this tier")

        # Average the two
        rates_to_avg = [r for r in [direct_rate, tier_rate] if r is not None]
        if rates_to_avg:
            s2_rate = sum(rates_to_avg) / len(rates_to_avg)
            components = " + ".join(
                [f"{r*100:.1f}%" for r in rates_to_avg if r is not None]
            )
            divisor = len(rates_to_avg)
            print(f"\n  Similar defense hit rate: ({components}) / {divisor} = {s2_rate*100:.1f}%")
        else:
            s2_rate = s1_rate
            print(f"\n  No similar defense data — falling back to season hit rate: {s2_rate*100:.1f}%")

    # ── STEP 3: True probability ──────────────────────────────────────────────
    print(f"\n{'─'*66}")
    print("  STEP 3 — TRUE PROBABILITY")
    print(f"{'─'*66}")
    true_prob = (s1_rate + s2_rate) / 2
    print(f"  ({s1_rate*100:.1f}% + {s2_rate*100:.1f}%) / 2 = {true_prob*100:.1f}%")

    # ── STEP 4: Implied probability ───────────────────────────────────────────
    print(f"\n{'─'*66}")
    print("  STEP 4 — IMPLIED PROBABILITY")
    print(f"{'─'*66}")
    implied_prob = _american_to_implied(odds)
    direction = "negative" if odds < 0 else "positive"
    print(f"  Odds: {odds:+d} ({direction}) → {implied_prob*100:.1f}%")

    # ── STEP 5: Edge ─────────────────────────────────────────────────────────
    print(f"\n{'─'*66}")
    print("  STEP 5 — EDGE")
    print(f"{'─'*66}")
    edge = true_prob - implied_prob
    edge_str = f"{edge*100:+.1f}%"
    print(f"  {true_prob*100:.1f}% − {implied_prob*100:.1f}% = {edge_str}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  VERDICT")
    print(sep)

    if target_status and target_status.upper() in ("OUT", "DOUBTFUL"):
        print(f"  ⚠  SKIP — {player['full_name']} is {target_status.upper()}, prop invalid")
    elif edge > 0:
        print(f"  BET  ✓  Edge: {edge_str} — Positive edge detected")
        if odds < -400:
            print(f"  Note: Odds ({odds:+d}) are very heavy juice — high implied probability eats into edge")
        elif odds > -100:
            print(f"  Note: Odds ({odds:+d}) are plus-money/light-minus — lower confidence per the formula rules")
        else:
            print(f"  Odds ({odds:+d}) are within target range")
    else:
        print(f"  SKIP  ✗  Edge: {edge_str} — No positive edge")

    if target_status and target_status.upper() == "QUESTIONABLE":
        print(f"  ⚠  Flag: {player['full_name']} is QUESTIONABLE — confirm availability before betting")

    print(sep + "\n")
