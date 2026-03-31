# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PropsBasket AI** is an NBA player prop probability modeling and market analysis platform. It collects live sportsbook odds and historical NBA performance data, engineers predictive features, trains ML models to estimate prop outcome probabilities, and compares model probabilities against sportsbook implied probabilities to surface pricing edges.

This is strictly an analytics/probability modeling platform — it does not place bets or guarantee profits.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, React, TypeScript, TailwindCSS, App Router |
| Backend API | Python FastAPI, SQLAlchemy 2.0 (async), Alembic |
| Database | PostgreSQL 16 |
| ML | Python, scikit-learn, XGBoost, LightGBM |
| NBA Data | `nba_api` library |
| Odds Data | The Odds API (FanDuel focus) |
| Deployment | Docker + docker-compose |

## Development Commands

### Full stack (Docker)
```bash
cp .env.example .env          # then fill in ODDS_API_KEY
docker compose up --build
```
Services: Postgres → `5432`, Backend → `8000`, Frontend → `3000`

### Backend (FastAPI)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload          # dev server
pytest                                  # tests
ruff check . && ruff format .          # lint + format

# Database migrations (Alembic)
alembic revision --autogenerate -m "description"
alembic upgrade head
```
Interactive API docs: http://localhost:8000/docs

### ML pipeline
```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/ingest_nba.py --season 2024-25
python scripts/build_features.py
python scripts/train.py --model xgboost
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev        # dev server on :3000
npm run build      # production build
npm run lint       # ESLint
```

## Repository Structure

```
backend/
  app/
    main.py          # FastAPI app, lifespan, CORS, router registration
    config.py        # Pydantic Settings (reads .env)
    database.py      # Async SQLAlchemy engine + Base + get_db dependency
    models/orm.py    # SQLAlchemy ORM models (all 7 tables)
    schemas/props.py # Pydantic response models
    routers/
      props.py       # GET /api/props
      predictions.py # GET /api/predictions, /api/predictions/top-edges
  migrations/        # Alembic; env.py swaps asyncpg→psycopg2 for migration runs
  alembic.ini
  pyproject.toml     # deps + ruff config

ml/
  propsbasket/
    ingestion/
      nba_stats.py   # nba_api wrappers: player game logs, team stats
      odds.py        # The Odds API wrapper + american_to_implied_prob()
    features/
      engineering.py # add_rolling_stats(), add_game_context(), add_target(); FEATURE_COLS list
    models/
      trainer.py     # train() wraps model in CalibratedClassifierCV, saves .joblib; load()
    evaluation/
      metrics.py     # evaluate(), calibration_summary(), edge_summary()
  scripts/
    ingest_nba.py    # CLI: fetch all active player game logs
    build_features.py# CLI: assemble training dataset → data/processed/features.parquet
    train.py         # CLI: chronological train/val/test split, train model, log metrics
  pyproject.toml

frontend/            # Next.js 15 App Router, TypeScript, TailwindCSS
  next.config.ts     # output: "standalone" for Docker
  Dockerfile         # multi-stage: deps → builder → runner

data/
  raw/               # gitignored — raw API dumps
  processed/         # gitignored — feature parquet files

docker-compose.yml   # postgres + backend + frontend; postgres healthcheck gates backend start
.env.example         # copy to .env and fill in ODDS_API_KEY
```

## Architecture Notes

- **edge = model_probability − implied_probability** is the core signal; `top-edges` endpoint ranks by this
- **Calibration matters more than accuracy**: models are wrapped in `CalibratedClassifierCV` so predicted probabilities match real hit rates
- **No data leakage**: `add_rolling_stats()` uses `shift(1)` before rolling; train/val/test split is chronological by season
- **Alembic migrations**: `migrations/env.py` replaces `asyncpg` with `psycopg2` at runtime since Alembic runs synchronously
- **ML artifacts** (`.joblib` files) are gitignored — regenerate by running `scripts/train.py`
