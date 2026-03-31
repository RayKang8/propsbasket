# PropsBasket AI
### NBA Player Prop Probability Modeling & Market Analysis Platform

Author: Ray Kang  
Version: 1.0  
Status: Draft

---

# 1. Overview

PropsBasket AI is a machine learning platform designed to analyze NBA player prop markets using historical player statistics and sportsbook odds data.

The system collects live sportsbook odds and historical NBA performance data, engineers predictive features, and trains machine learning models to estimate the probability that a player prop outcome will occur.

PropsBasket AI compares model-generated probabilities with sportsbook implied probabilities to identify potential pricing discrepancies and evaluate how predictive models perform relative to market expectations.

The project is designed as an experimental machine learning system to explore the capabilities and limitations of AI when applied to real-world sports markets.

---

# 2. Goals

Primary Goals

• Build a full machine learning pipeline using real NBA data  
• Train models that estimate probabilities for NBA player prop outcomes  
• Compare model predictions against sportsbook implied probabilities  
• Evaluate prediction quality using probabilistic metrics  
• Run historical simulations to measure model performance over time  

Secondary Goals

• Gain experience with real-world data engineering pipelines  
• Explore sports market efficiency and prediction accuracy  
• Build an analytics dashboard for model outputs and evaluation  

---

# 3. Non-Goals

PropsBasket AI will NOT:

• Guarantee profitable betting strategies  
• Automatically place bets  
• Execute or optimize gambling strategies  

The system is strictly an **analytics and probability modeling platform**.

---

# 4. Scope

Initial Version (MVP)

League  
NBA

Market Type  
Player Points Props

Sportsbook Data  
FanDuel odds via sportsbook odds API

Prediction Target

Binary outcome:

did_player_go_over_points_line

Example

Player Points Line: 24.5

Outcome  
1 → player scored 25+  
0 → player scored 24 or fewer

---

# 5. System Architecture

Frontend  
Next.js  
React  
TailwindCSS  

Backend API  
Python FastAPI  

Machine Learning  
Python  
scikit-learn  
XGBoost / LightGBM  

Database  
PostgreSQL  

Data Processing  
Pandas  
NumPy  

Scheduling  
Cron jobs / background workers  

Deployment  
Docker  
Cloud VM or container hosting  

---

# 6. Data Sources

NBA Statistics

Source  
nba_api

Data collected

• player game logs  
• team stats  
• advanced metrics  
• pace  
• minutes played  
• usage rate  

Sportsbook Odds

Source  
Sportsbook odds API (example: The Odds API)

Data collected

• sportsbook name  
• player prop line  
• odds  
• timestamp  
• line movement  
• market type  

Game Context

• opponent defensive rating  
• rest days  
• home vs away  
• back-to-back games  
• injuries (future expansion)  

---

# 7. Database Schema

players

id  
name  
team  
position  

games

id  
date  
home_team  
away_team  

player_game_logs

id  
player_id  
game_id  
minutes  
points  
rebounds  
assists  
usage_rate  
shots_attempted  
shots_made  

sportsbooks

id  
name  

prop_lines

id  
player_id  
game_id  
sportsbook_id  
line_value  
odds  
market_type  
timestamp  

prop_outcomes

id  
prop_line_id  
actual_stat  
did_hit  

model_predictions

id  
prop_line_id  
model_probability  
implied_probability  
edge  
prediction_time  

evaluation_metrics

id  
date  
log_loss  
brier_score  
accuracy  

---

# 8. Feature Engineering

Player Performance Features

• last 5 game average points  
• last 10 game average points  
• rolling standard deviation  
• minutes trend  
• usage rate trend  
• shot attempt rate  

Game Context Features

• opponent defensive rating  
• opponent pace  
• home vs away  
• rest days  
• back-to-back indicator  

Market Features

• opening line  
• current line  
• line movement  
• consensus line across sportsbooks  
• implied probability from odds  

---

# 9. Machine Learning Models

Baseline Model

Logistic Regression

Advanced Models

Random Forest  
XGBoost  
LightGBM  

Target Variable

did_player_go_over_line

Model Output

probability_of_over

Example

model_probability = 0.58

---

# 10. Market Comparison Layer

Sportsbook odds are converted to implied probability.

Example

Odds = -110

implied_probability ≈ 52.4%

Edge calculation

edge = model_probability - implied_probability

Example

model_probability = 0.58  
implied_probability = 0.524  

edge = 0.056

PropsBasket AI ranks props based on largest edge values.

---

# 11. Evaluation Metrics

The model will be evaluated using probabilistic scoring metrics.

Log Loss

Measures probability prediction quality.

Brier Score

Measures mean squared error of probability predictions.

Calibration

Checks whether predicted probabilities match real frequencies.

Example

Predictions of 60% should occur roughly 60% of the time.

Backtesting

Historical predictions are simulated using past seasons.

Example

Training Data

2018–2022

Validation

2023

Test

2024

---

# 12. Backtesting System

A simulation engine will evaluate model predictions historically.

Simulation workflow

1. Load historical prop lines  
2. Generate model predictions  
3. Compare predictions to actual outcomes  
4. Evaluate model performance metrics  

Simulation outputs

• prediction accuracy  
• calibration curves  
• expected value estimates  
• performance over time  

---

# 13. Dashboard

Frontend dashboard will display:

Daily Props

• player name  
• sportsbook line  
• model probability  
• implied probability  
• edge score  

Model Metrics

• calibration chart  
• log loss history  
• prediction accuracy  

Historical Analysis

• line movement visualization  
• player performance trends  

---

# 14. Build Phases

Phase 1

Data Infrastructure

• build database schema  
• ingest NBA stats  
• ingest sportsbook odds  
• store historical props  

Phase 2

Feature Engineering

• build player features  
• build game context features  
• generate training dataset  

Phase 3

Model Training

• train baseline models  
• evaluate models  
• tune hyperparameters  

Phase 4

Prediction Engine

• generate daily predictions  
• compute edges vs market  
• store predictions  

Phase 5

Dashboard

• build frontend dashboard  
• visualize predictions  
• display model metrics  

---

# 15. Future Improvements

Additional Sports

NFL  
MLB  
NHL  

Additional Prop Markets

rebounds  
assists  
three pointers  

Advanced Modeling

• neural networks  
• player matchup models  
• injury-aware models  

Market Microstructure

• line movement prediction  
• sportsbook consensus modeling  

---

# 16. Risks & Challenges

Data Quality

Sportsbook APIs may have limited historical data.

Market Efficiency

Sportsbooks already incorporate advanced statistical models.

Feature Complexity

Player props depend on many contextual factors.

Overfitting

Models may appear strong in backtests but fail on new seasons.

---

# 17. Success Criteria

PropsBasket AI will be considered successful if:

• the system can generate probability predictions for NBA props  
• predictions are well calibrated  
• model performance can be evaluated using probabilistic metrics  
• the full pipeline runs from data ingestion to prediction output  
• the platform processes real sportsbook and NBA data end-to-end