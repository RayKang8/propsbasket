# PropsBasket AI

**NBA player prop probability modeling and market analysis platform**

> An experimental ML system that models NBA player prop outcomes and compares predicted probabilities against sportsbook-implied probabilities to quantify market edges.
> This is strictly an analytics platform — it does not place bets or guarantee profits.

---

## What It Does

1. **Ingests** live sportsbook odds (FanDuel via The Odds API) and historical NBA game logs (`nba_api`)
2. **Engineers** rolling performance features, game context signals, and market features
3. **Trains** calibrated ML models (Logistic Regression, XGBoost, LightGBM) to estimate the probability a player hits their points prop
4. **Computes edges** — `edge = model_probability − implied_probability` — and ranks props by largest discrepancy
5. **Displays** predictions, calibration curves, and model metrics on a Next.js dashboard

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, React, TypeScript, TailwindCSS |
| Backend API | Python FastAPI, SQLAlchemy 2.0 (async), Alembic |
| Database | PostgreSQL 16 |
| ML | scikit-learn, XGBoost, LightGBM, CalibratedClassifierCV |
| NBA Data | `nba_api` |
| Odds Data | The Odds API (FanDuel) |
| Deployment | Docker + docker-compose |

---

## Architecture

```
nba_api + Odds API
        │
        ▼
  Ingestion Layer          (player game logs, sportsbook odds)
        │
        ▼
  Feature Engineering      (rolling stats, game context, market signals)
        │
        ▼
  ML Training Pipeline     (calibrated classifiers, chronological splits)
        │
        ▼
  Prediction Engine        (edge = model_prob − implied_prob)
        │
        ▼
  FastAPI Backend          (REST API serving predictions and props)
        │
        ▼
  Next.js Dashboard        (props table, edge rankings, calibration charts)
```

**Key design decisions:**
- Models are wrapped in `CalibratedClassifierCV` — calibration quality matters more than raw accuracy
- `add_rolling_stats()` uses `shift(1)` before rolling windows to prevent data leakage
- Train/val/test split is strictly chronological: train 2018–2022, val 2023, test 2024

---

## Getting Started

```bash
cp .env.example .env   # add your ODDS_API_KEY
docker compose up --build
```

Services:
- Frontend → [http://localhost:3000](http://localhost:3000)
- Backend API → [http://localhost:8000](http://localhost:8000)
- Interactive API docs → [http://localhost:8000/docs](http://localhost:8000/docs)

### Run the ML pipeline

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/ingest_nba.py --season 2024-25
python scripts/build_features.py
python scripts/train.py --model xgboost
```

---

## Repository Structure

```
backend/       FastAPI app, ORM models, Alembic migrations, REST endpoints
ml/            Ingestion, feature engineering, model training, evaluation
frontend/      Next.js 15 App Router dashboard
data/          Raw + processed data (gitignored)
docker-compose.yml
```

---

## ML Approach

**Target:** Binary `did_hit` — did the player exceed their points line?

**Features (14 total):**
- Rolling 5/10-game averages and standard deviations for points and minutes
- Usage rate trend, shot attempt rate
- Opponent defensive rating and pace
- Home/away, rest days, back-to-back indicator
- Implied probability from sportsbook odds

**Evaluation metrics:** Log loss, Brier score, calibration curves

**Backtesting:** Chronological train/val/test split by season — no lookahead

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/props` | Most recent prop lines |
| `GET` | `/api/props/{id}` | Single prop detail |
| `GET` | `/api/predictions` | Latest model predictions |
| `GET` | `/api/predictions/top-edges` | Props ranked by model edge |

---

## Status

Active development. Core infrastructure (schema, API, ML pipeline) is scaffolded; data ingestion and frontend dashboard are in progress.

---

## start up
Terminal 1 (backend):                                                                                                                                           
cd /Users/raykang/Desktop/propsbasket/backend
source .venv/bin/activate                                                                                                                                         
uvicorn app.main:app --reload --reload-dir app
                                                                                                                                                                    
Terminal 2 (frontend):                                                                                                                                          
cd /Users/raykang/Desktop/propsbasket/frontend
npm run dev                                                                                                                                                       
                                                            
Terminal 3 (ML - live predictions)
:

cd /Users/raykang/Desktop/propsbasket/ml
source .venv/bin/activate                                                                                                                                  
python scripts/predict_live.py
                                                                                                                                                                    
Then open http://localhost:3000  
                                                                                                                                    