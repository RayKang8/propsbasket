# CLAUDE.md

## Project Overview

**PropsBasket** is a terminal-based NBA player prop analyzer. You input a player, prop line, odds, and opponent — it runs a 5-step formula using live NBA data to calculate true probability vs. implied probability and surface an edge.

This is strictly an analytics tool — it does not place bets or guarantee profits.

## Usage

```bash
# Setup (first time)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Interactive mode
python analyze.py

# Argument mode
python analyze.py --player "LeBron James" --line "20+ points" --odds -166 --opponent "Golden State Warriors"
python analyze.py --player "Nikola Jokic" --line "10+ rebounds" --odds -200 --opponent "Lakers"
python analyze.py --player "Stephen Curry" --line "4+ threes" --odds -130 --opponent "Boston Celtics"
python analyze.py --player "Giannis Antetokounmpo" --line "30+ pra" --odds -250 --opponent "Heat"
```

Supported stat types: `points`, `rebounds`, `assists`, `threes`, `steals`, `blocks`, `turnovers`, `pra`

## The Formula

| Step | What it calculates |
|---|---|
| Step 0 | Injury report for both teams from NBA API |
| Step 1 | Season hit rate — games hitting the line / total games |
| Step 2 | Similar defense hit rate — direct history vs opponent + hit rate vs same defensive tier (thirds by DefRtg) |
| Step 3 | True probability = (Step 1 + Step 2) / 2 |
| Step 4 | Implied probability from American odds |
| Step 5 | Edge = True probability − Implied probability |

## Structure

```
analyze.py       # CLI entry point
src/
  nba.py         # nba_api wrappers: game logs, team def ratings, player/team lookup
  injuries.py    # NBA injury report via nba_api LeagueInjuries
  formula.py     # Formula engine + terminal output
requirements.txt
.env             # ODDS_API_KEY (not currently used in analysis flow)
```

## Data Sources

- **nba_api** — player game logs, team advanced stats (DefRtg), injury reports
- Odds are entered manually by the user
