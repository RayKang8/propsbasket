# PropsBasket AI

**NBA player prop probability modeling and market analysis platform**

> An experimental ML system that models NBA player prop outcomes and compares predicted probabilities against sportsbook-implied probabilities. This is strictly an analytics platform — it does not place bets or guarantee profits.

---

## What It Does

1. **Fetches** today's live FanDuel prop lines via The Odds API
2. **Ingests** fresh game logs for only the players with active props (scoped, ~30-50 players vs 530)
3. **Engineers** rolling performance features, opponent context, and rest/schedule signals
4. **Scores** props with a calibrated XGBoost model trained on 2024-25 and 2025-26 NBA seasons
5. **Computes edges** — `edge = model_probability − implied_probability`
6. **Displays** predictions ranked by model probability or edge on a Next.js dashboard

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, React, TypeScript, TailwindCSS |
| Backend API | Python FastAPI, SQLAlchemy 2.0 (async), Alembic |
| Database | PostgreSQL 16 |
| ML | scikit-learn, XGBoost, CalibratedClassifierCV |
| NBA Data | `nba_api` |
| Odds Data | The Odds API (FanDuel) |

---

## Architecture

```
Click "Refresh Predictions"
        │
        ▼
Odds API → today's FanDuel player_points props
        │
        ▼
nba_api → fresh game logs for only today's players (scoped ingestion)
        │
        ▼
Feature Engineering → rolling stats, opponent def rating, rest days, line/implied prob
        │
        ▼
XGBoost Model → model_probability per prop
        │
        ▼
PostgreSQL → predictions written, today-scoped
        │
        ▼
Next.js Dashboard → ranked by model probability or edge
```

**Key design decisions:**
- **Scoped ingestion** — on each refresh, only fetches game logs for players with active props (~1-2 min vs ~15 min for full league)
- **Refresh is idempotent** — clears today's predictions before inserting fresh ones, no duplicates
- **Today-only** — all dashboard queries filter to `prediction_time >= today midnight`
- **Calibrated model** — XGBoost wrapped in `CalibratedClassifierCV` so probabilities reflect real hit rates
- **No data leakage** — `add_rolling_stats()` uses `shift(1)` before rolling windows during training

---

## Daily Usage (No Retraining Needed)

```bash
# Terminal 1 — Backend:
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --reload-dir app

# Terminal 2 — Frontend:
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and click **Refresh Predictions**.

The backend handles everything: fetches live odds, ingests fresh game logs, scores with the model, saves to DB. Takes ~2-3 minutes on first click (nba_api rate limits).

---

## Model Retraining

Only needed when:
- A new NBA season starts (October)
- You want to add more historical seasons
- You add new features to `FEATURE_COLS`

```bash
cd ml
source .venv/bin/activate
pip install -e ".[dev]"

# Ingest seasons (2024-25 already exists, only fetches new ones)
python scripts/ingest_nba.py --seasons 2024-25 2025-26

# Rebuild feature dataset (combines all seasons automatically)
python scripts/build_features.py

# Retrain — artifact saved to models/artifacts/xgboost.joblib
python scripts/train.py --model xgboost
```

Backend picks up the new artifact automatically on next request — no restart needed.

---

## ML Approach

**Target:** Binary `did_hit` — did the player score more than their points line?

**Features (14 total):**
- Rolling 5/10-game averages and standard deviations for points and minutes
- Opponent defensive rating and pace
- Home/away, rest days, back-to-back indicator
- Prop line value, implied probability from sportsbook odds, line movement

**Training data:** 2024-25 + 2025-26 NBA seasons (~48k rows)

**Train/val/test split:** Chronological 70/15/15 by game date — no lookahead

**Evaluation:** Log loss, Brier score, accuracy
- Val accuracy: ~53.4% | Test accuracy: ~53.8%

**Current scope:** Points props (Over) only. Rebounds and assists would require separate model targets.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predictions/refresh` | Fetch live odds, run model, persist today's predictions |
| `GET` | `/api/predictions/top-edges` | Today's props ranked by edge |
| `GET` | `/api/predictions` | Today's predictions, most recent first |
| `GET` | `/api/predictions/summary` | Aggregate stats for today |
| `GET` | `/api/props` | Most recent prop lines |

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Repository Structure

```
backend/
  app/
    routers/predictions.py   # REST endpoints (today-scoped)
    services/predictor.py    # Live ingestion + model scoring + DB writes
    models/orm.py            # SQLAlchemy ORM (7 tables)
    schemas/props.py         # Pydantic response models
  migrations/                # Alembic migrations

ml/
  propsbasket/
    ingestion/
      nba_stats.py           # nba_api wrappers + scoped player ingestion
      odds.py                # Odds API wrapper + parse_props()
    features/engineering.py  # Rolling stats, game context, FEATURE_COLS
    models/trainer.py        # train() + load() for calibrated models
    prediction/live.py       # Inference feature assembly + scoring
  scripts/
    ingest_nba.py            # CLI: fetch all active player game logs
    build_features.py        # CLI: assemble training dataset
    train.py                 # CLI: train + save model artifact
    predict_live.py          # CLI: standalone live prediction (testing)

frontend/
  app/
    page.tsx                 # Dashboard layout
    components/
      TopEdgesTable.tsx      # Predictions table (sort by model prob / edge / recent)
      RefreshButton.tsx      # Triggers POST /api/predictions/refresh
      SummaryCards.tsx       # Aggregate stats

data/
  raw/                       # gitignored — nba_api parquet files
  processed/                 # gitignored — feature parquet
```

---

## Notes

- **Team badges can be stale** for recently traded players — derived from last game log, not current roster. Does not affect predictions.
- **All edges currently negative** — model was trained on simulated prop lines (rolling avg), not real historical lines. Model probability ranking is more reliable than edge ranking for now.
- The `.env` file must contain `ODDS_API_KEY` — copy from `.env.example`.
