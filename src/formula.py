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

_SEP = "=" * 68
_DIV = "─" * 68


# ── helpers ───────────────────────────────────────────────────────────────────

def _american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _hit_rate(df: pd.DataFrame, col: str, threshold: float) -> tuple[int, int, float]:
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
    return "bottom third (weakest defense)"


def _extract_team_from_matchup(matchup: str) -> str:
    """'LAL vs. GSW' or 'LAL @ GSW' → 'LAL'"""
    return matchup.strip().split()[0].upper()


def _filter_to_current_team(game_logs: pd.DataFrame) -> tuple[pd.DataFrame, str | None, bool]:
    """If a player changed teams mid-season, return only current team games.

    Returns (filtered_df, current_team_abbrev, was_traded).
    """
    if "MATCHUP" not in game_logs.columns or game_logs.empty:
        return game_logs, None, False

    teams_seen = game_logs["MATCHUP"].apply(_extract_team_from_matchup)
    unique_teams = teams_seen.unique()

    if len(unique_teams) == 1:
        return game_logs, unique_teams[0], False

    # Rows are newest-first in nba_api; current team is from the most recent game
    current_team = _extract_team_from_matchup(game_logs.iloc[0]["MATCHUP"])
    filtered = game_logs[teams_seen == current_team].copy()
    return filtered, current_team, True


def _format_game_row(row: pd.Series, col: str, threshold: float) -> str:
    """Format a single game log row for display."""
    date = str(row.get("GAME_DATE", ""))[:10]
    matchup = str(row.get("MATCHUP", ""))
    # Extract opponent abbreviation from matchup ("LAL vs. GSW" or "LAL @ GSW")
    parts = matchup.split()
    opp = parts[-1] if len(parts) >= 3 else matchup
    val = row.get(col)
    if pd.isna(val):
        return f"    {date} vs {opp}: DNP"
    hit = "HIT " if val >= threshold else "MISS"
    return f"    {date} vs {opp}: {val:.0f}  [{hit}]"


# ── main entry ────────────────────────────────────────────────────────────────

def analyze_prop(
    player_name: str,
    prop_line: str,
    odds: int,
    opponent: str,
    season: str = "2024-25",
) -> None:

    # ── parse prop line ───────────────────────────────────────────────────────
    parsed = parse_prop_line(prop_line)
    if parsed is None:
        print(f"Could not parse prop line: '{prop_line}'")
        print("Examples: '20+ points', '8+ assists', '10+ rebounds', '2+ threes', '30+ pra'")
        sys.exit(1)
    threshold, stat_col = parsed

    # ── resolve opponent ──────────────────────────────────────────────────────
    opp_abbrev = resolve_team_abbrev(opponent)
    if opp_abbrev is None:
        print(f"Could not resolve team: '{opponent}'")
        sys.exit(1)

    # ── resolve player ────────────────────────────────────────────────────────
    player = find_player(player_name)
    if player is None:
        print(f"Could not find player: '{player_name}'")
        sys.exit(1)

    print(f"\n{_SEP}")
    print(f"  PROP ANALYSIS: {player['full_name']} — {prop_line.upper()} vs. {opponent}")
    print(f"  Season: {season}   Odds: {odds:+d}")
    print(_SEP)

    # ── fetch data ────────────────────────────────────────────────────────────
    print("\nFetching data (this takes ~10 seconds due to NBA API rate limits)...")
    injury_df = get_injuries()
    raw_logs = get_game_logs(player["id"], season=season)
    def_ratings = get_team_def_ratings(season=season)

    if raw_logs.empty:
        print("No game logs found for this player/season.")
        sys.exit(1)

    if stat_col not in raw_logs.columns:
        print(f"Stat column '{stat_col}' not found. Available: {list(raw_logs.columns)}")
        sys.exit(1)

    # Mid-season trade check — filter to current team only
    game_logs, player_team_abbrev, was_traded = _filter_to_current_team(raw_logs)

    # ── STEP 0: Injury report ─────────────────────────────────────────────────
    print(f"\n{_DIV}")
    print("  STEP 0 — INJURY & AVAILABILITY CHECK")
    print(_DIV)

    target_status = get_player_injury_status(player["full_name"], injury_df)
    if target_status:
        flag = "  ⚠  WARNING:" if target_status.upper() in ("QUESTIONABLE", "DOUBTFUL", "OUT") else " "
        print(f"{flag} {player['full_name']}: {target_status.upper()}")
    else:
        print(f"  {player['full_name']}: ACTIVE (not on injury report)")

    # Opponent display name
    opp_def_row = def_ratings[def_ratings["TEAM_NAME"].str.contains(opponent, case=False, na=False)]
    opp_display_name = opp_def_row.iloc[0]["TEAM_NAME"] if not opp_def_row.empty else opponent

    def _print_team_injuries(abbrev: str, label: str) -> list[str]:
        """Print injury report for a team, return list of OUT players."""
        inj = get_team_injuries(abbrev, injury_df)
        out_players = []
        if inj.empty:
            print(f"\n  {label}: No players on injury report")
        else:
            print(f"\n  {label} injury report:")
            for _, row in inj.iterrows():
                status = str(row.get("status", "")).upper()
                comment = row.get("comment", "")
                comment_str = f" — {comment}" if comment and str(comment) not in ("nan", "") else ""
                marker = "  ⚠ " if status in ("OUT", "DOUBTFUL") else "    "
                print(f"{marker}  {row['player_name']}: {status}{comment_str}")
                if status == "OUT":
                    out_players.append(row["player_name"])
        return out_players

    opp_out = _print_team_injuries(opp_abbrev, opp_display_name)
    player_team_out: list[str] = []
    if player_team_abbrev:
        # Show teammates' injuries (excluding the target player)
        team_inj = get_team_injuries(player_team_abbrev, injury_df)
        if not team_inj.empty:
            others = team_inj[team_inj["player_name"].str.lower() != player["full_name"].lower()]
            if not others.empty:
                print(f"\n  {player_team_abbrev} teammates injury report:")
                for _, row in others.iterrows():
                    status = str(row.get("status", "")).upper()
                    comment = row.get("comment", "")
                    comment_str = f" — {comment}" if comment and str(comment) not in ("nan", "") else ""
                    marker = "  ⚠ " if status in ("OUT", "DOUBTFUL") else "    "
                    print(f"{marker}  {row['player_name']}: {status}{comment_str}")
                    if status == "OUT":
                        player_team_out.append(row["player_name"])

    # Usage context note
    if player_team_out:
        print(f"\n  Usage note: {', '.join(player_team_out)} are OUT — potential usage increase for {player['full_name']}")
    if opp_out:
        print(f"  Matchup note: {', '.join(opp_out)} are OUT for {opp_display_name}")

    # Recent minutes trend (last 5 games)
    if "MIN" in game_logs.columns and len(game_logs) >= 3:
        recent_mins = game_logs.head(5)["MIN"].mean()
        season_mins = game_logs["MIN"].mean()
        print(f"\n  Minutes — last 5 games avg: {recent_mins:.1f} min  |  season avg: {season_mins:.1f} min")

    # ── STEP 1: Season hit rate ───────────────────────────────────────────────
    print(f"\n{_DIV}")
    print("  STEP 1 — SEASON HIT RATE")
    print(_DIV)

    if was_traded:
        discarded = len(raw_logs) - len(game_logs)
        print(f"  ⚠  Mid-season trade detected — discarding {discarded} games from previous team(s)")
        print(f"  Using only {player_team_abbrev} games ({len(game_logs)} games)\n")

    s1_hits, s1_games, s1_rate = _hit_rate(game_logs, stat_col, threshold)

    # Regular season vs playoff breakdown
    if "GAME_TYPE" in game_logs.columns:
        reg = game_logs[game_logs["GAME_TYPE"] == "Regular Season"]
        po = game_logs[game_logs["GAME_TYPE"] == "Playoffs"]
        if not reg.empty:
            rh, rg, rr = _hit_rate(reg, stat_col, threshold)
            print(f"  Regular season: {rh}/{rg} games = {rr*100:.1f}%")
        if not po.empty:
            ph, pg, pr = _hit_rate(po, stat_col, threshold)
            print(f"  Playoffs:       {ph}/{pg} games = {pr*100:.1f}%")
        print()

    print(f"  Season hit rate: {s1_hits}/{s1_games} games = {s1_rate*100:.1f}%")

    # Recent form — last 10 games
    recent_10 = game_logs.head(10)
    r10_hits, r10_games, r10_rate = _hit_rate(recent_10, stat_col, threshold)
    trend_note = ""
    if r10_rate > s1_rate + 0.10:
        trend_note = " ↑ trending UP vs season average"
    elif r10_rate < s1_rate - 0.10:
        trend_note = " ↓ trending DOWN vs season average"
    print(f"  Last 10 games:   {r10_hits}/{r10_games} games = {r10_rate*100:.1f}%{trend_note}")

    # Recent form — last 5 games
    if stat_col in game_logs.columns:
        recent_5 = game_logs.head(5)
        print(f"\n  Last 5 games:")
        for _, row in recent_5.iterrows():
            print(_format_game_row(row, stat_col, threshold))

    # ── STEP 2: Similar defense hit rate ─────────────────────────────────────
    print(f"\n{_DIV}")
    print("  STEP 2 — SIMILAR DEFENSE HIT RATE")
    print(_DIV)

    if opp_def_row.empty:
        print(f"  Could not find defensive rating for '{opponent}' — skipping Step 2")
        s2_rate = s1_rate
        direct_rate = None
        tier_rate = None
    else:
        opp_def_rtg = float(opp_def_row.iloc[0]["DEF_RATING"])

        # Rank all teams by DefRtg ascending (lower = better defense)
        def_sorted = def_ratings.sort_values("DEF_RATING").reset_index(drop=True)
        total_teams = len(def_sorted)
        opp_idx = def_sorted[def_sorted["TEAM_NAME"].str.contains(opponent, case=False, na=False)].index
        opp_rank = int(opp_idx[0]) + 1
        tier = _tier_label(opp_rank, total_teams)

        third = total_teams // 3
        if opp_rank <= third:
            tier_teams_df = def_sorted.iloc[:third]
        elif opp_rank <= third * 2:
            tier_teams_df = def_sorted.iloc[third:third * 2]
        else:
            tier_teams_df = def_sorted.iloc[third * 2:]

        tier_abbrevs = set()
        for _, row in tier_teams_df.iterrows():
            abbrev = resolve_team_abbrev(row["TEAM_NAME"])
            if abbrev:
                tier_abbrevs.add(abbrev)

        print(f"  {opp_display_name} defensive rating: {opp_def_rtg:.1f}")
        print(f"  League rank: #{opp_rank} of {total_teams}  |  Tier: {tier}")

        # Direct history vs opponent
        direct_games = pd.DataFrame()
        if "MATCHUP" in game_logs.columns:
            direct_games = game_logs[game_logs["MATCHUP"].str.contains(opp_abbrev, case=False, na=False)]

        print(f"\n  vs. {opp_display_name} — direct history this season:")
        if len(direct_games) > 0:
            for _, row in direct_games.iterrows():
                print(_format_game_row(row, stat_col, threshold))
            d_hits, d_games, direct_rate = _hit_rate(direct_games, stat_col, threshold)
            print(f"  Direct hit rate: {d_hits}/{d_games} = {direct_rate*100:.1f}%")
        else:
            direct_rate = None
            print(f"  No games played vs {opp_display_name} this season")

        # Hit rate vs similar-tier teams (excluding direct opponent)
        tier_games = pd.DataFrame()
        if "MATCHUP" in game_logs.columns and tier_abbrevs:
            pattern = "|".join(tier_abbrevs)
            tier_games = game_logs[game_logs["MATCHUP"].str.contains(pattern, case=False, na=False)]
            tier_games = tier_games[~tier_games["MATCHUP"].str.contains(opp_abbrev, case=False, na=False)]

        print(f"\n  vs. comparable {tier} teams:")
        if len(tier_games) > 0:
            for _, row in tier_games.iterrows():
                print(_format_game_row(row, stat_col, threshold))
            t_hits, t_games, tier_rate = _hit_rate(tier_games, stat_col, threshold)
            print(f"  Comparable teams hit rate: {t_hits}/{t_games} = {tier_rate*100:.1f}%")
        else:
            tier_rate = None
            print(f"  No games vs comparable teams found")

        rates_to_avg = [r for r in [direct_rate, tier_rate] if r is not None]
        if rates_to_avg:
            s2_rate = sum(rates_to_avg) / len(rates_to_avg)
            parts_str = " + ".join(f"{r*100:.1f}%" for r in rates_to_avg)
            divisor = len(rates_to_avg)
            print(f"\n  Similar defense hit rate: ({parts_str}) / {divisor} = {s2_rate*100:.1f}%")
        else:
            s2_rate = s1_rate
            print(f"\n  No similar defense data — using season hit rate: {s2_rate*100:.1f}%")

    # ── STEP 3: True probability ──────────────────────────────────────────────
    print(f"\n{_DIV}")
    print("  STEP 3 — TRUE PROBABILITY")
    print(_DIV)
    true_prob = (s1_rate + s2_rate) / 2
    print(f"  ({s1_rate*100:.1f}% + {s2_rate*100:.1f}%) / 2 = {true_prob*100:.1f}%")

    # ── STEP 4: Implied probability ───────────────────────────────────────────
    print(f"\n{_DIV}")
    print("  STEP 4 — IMPLIED PROBABILITY")
    print(_DIV)
    implied_prob = _american_to_implied(odds)
    if odds < 0:
        print(f"  {abs(odds)} / ({abs(odds)} + 100) = {implied_prob*100:.1f}%")
    else:
        print(f"  100 / ({odds} + 100) = {implied_prob*100:.1f}%")

    # ── STEP 5: Edge ─────────────────────────────────────────────────────────
    print(f"\n{_DIV}")
    print("  STEP 5 — EDGE")
    print(_DIV)
    edge = true_prob - implied_prob
    edge_str = f"{edge*100:+.1f}%"
    print(f"  {true_prob*100:.1f}% − {implied_prob*100:.1f}% = {edge_str}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  VERDICT")
    print(_SEP)

    if target_status and target_status.upper() in ("OUT", "DOUBTFUL"):
        print(f"  SKIP — {player['full_name']} is {target_status.upper()}, prop is invalid")
    elif edge > 0:
        print(f"  BET  ✓  Edge: {edge_str} — Positive edge")
        if odds < -400:
            print(f"  Note: Juice is very heavy ({odds:+d}) — high implied probability eats into value")
        elif odds > -100:
            print(f"  Note: Odds ({odds:+d}) are plus-money or light-minus — lower confidence line per formula rules")
        if r10_rate > s1_rate + 0.10:
            print(f"  Momentum: Player is trending UP over last 10 games ({r10_rate*100:.1f}%) — edge may be understated")
        elif r10_rate < s1_rate - 0.10:
            print(f"  Caution: Player is trending DOWN over last 10 games ({r10_rate*100:.1f}%)")
        if player_team_out:
            print(f"  Usage boost: {', '.join(player_team_out)} OUT — increased role expected")
    else:
        print(f"  SKIP  ✗  Edge: {edge_str} — No positive edge")
        if implied_prob > 0.70:
            print(f"  The market ({odds:+d}) has already priced this heavily — no mathematical value at this line")
        if r10_rate < s1_rate - 0.10:
            print(f"  Recent form is declining ({r10_rate*100:.1f}% last 10 vs {s1_rate*100:.1f}% season avg)")

    if target_status and target_status.upper() == "QUESTIONABLE":
        print(f"  ⚠  Flag: {player['full_name']} is QUESTIONABLE — confirm active before betting")

    print(f"\n  Parlay note: Only combine with other props that have positive edge.")
    print(f"  Run 'python analyze.py' on other tonight's props to find parlay combinations.")
    print(_SEP + "\n")
