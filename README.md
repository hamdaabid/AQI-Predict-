# AQI-Predict-


> Predict the Air Quality Index (AQI) for your city 72 hours into the future using a fully serverless, automated ML pipeline.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Quickstart](#quickstart)
5. [Configuration & Secrets](#configuration--secrets)
6. [Pipeline Details](#pipeline-details)
   - [Feature Pipeline](#1-feature-pipeline)
   - [Historical Backfill](#2-historical-backfill)
   - [Training Pipeline](#3-training-pipeline)
   - [Web Dashboard](#4-web-dashboard)
7. [CI/CD with GitHub Actions](#cicd-with-github-actions)
8. [Models & Evaluation](#models--evaluation)
9. [SHAP Explainability](#shap-explainability)
10. [AQI Alert System](#aqi-alert-system)
11. [Running Tests](#running-tests)
12. [Deployment Options](#deployment-options)
13. [API Keys & Free Tiers](#api-keys--free-tiers)
14. [Contributing](#contributing)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  External APIs                                          │
│  AQICN · OpenWeatherMap Air Pollution History           │
└────────────────┬────────────────────────────────────────┘
                 │ raw data (hourly, via GitHub Actions)
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Feature Pipeline  (pipelines/feature_pipeline.py)      │
│  • Parse pollutants (PM2.5, PM10, O3, NO2, SO2, CO)     │
│  • Time features (hour, day, month, cyclical encoding)  │
│  • Lag features (1h, 3h, 6h, 12h, 24h)                 │
│  • Rolling statistics (mean, std)                       │
│  • AQI change-rate features                             │
└────────────────┬────────────────────────────────────────┘
                 │ engineered features
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Feature Store  (Hopsworks / local CSV fallback)        │
│  • Versioned feature groups                             │
│  • Serves both training and inference                   │
└──────┬──────────────────────────────────┬───────────────┘
       │ features + targets               │ latest features
       ▼                                  ▼
┌──────────────────────┐      ┌───────────────────────────┐
│  Training Pipeline   │      │  Web Dashboard            │
│  • Ridge Regression  │      │  (app/dashboard.py)       │
│  • Random Forest     │──────│  • Streamlit UI           │
│  • Gradient Boosting │model │  • 72-h forecast chart    │
│  • XGBoost           │      │  • SHAP importance        │
│  • LSTM (TF)         │      │  • Hazardous AQI alerts   │
│  • RMSE / MAE / R²   │      │  • Pollutant breakdown    │
└──────────────────────┘      └───────────────────────────┘
       │
       ▼
  Model Registry
  (saved_models/model_registry.json)
```

**Automation:** GitHub Actions runs the feature pipeline **every hour** and the training pipeline **every day at 02:00 UTC**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML models | scikit-learn, XGBoost, TensorFlow/Keras |
| Explainability | SHAP |
| Feature store | Hopsworks (free tier) · local CSV fallback |
| Dashboard | Streamlit + Plotly |
| CI/CD | GitHub Actions |
| Data APIs | AQICN, OpenWeatherMap Air Pollution History |

---

## Project Structure

```
aqi_predictor/
├── config/
│   └── config.py               # Centralised configuration dataclasses
├── pipelines/
│   ├── feature_pipeline.py     # Fetch → engineer → store (runs hourly)
│   ├── backfill_pipeline.py    # Populate historical training data
│   └── training_pipeline.py   # Train → evaluate → register (runs daily)
├── app/
│   └── dashboard.py            # Streamlit dashboard (predictions + EDA)
├── utils/
│   ├── helpers.py              # Logging, AQI helpers, alert system
│   └── feature_store.py       # Hopsworks / local CSV abstraction
├── tests/
│   └── test_pipelines.py      # pytest unit tests
├── .github/
│   └── workflows/
│       └── pipelines.yml       # GitHub Actions CI/CD
├── quickstart.py               # One-command end-to-end demo
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/your-org/aqi-predictor.git
cd aqi-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the full pipeline locally (no API keys needed)

The quickstart uses **synthetic data** so you can see every component working before wiring up real APIs:

```bash
python quickstart.py --days 60 --city karachi
```

Expected output:

```
============================================================
  AQI Predictor — Quick-start Demo
============================================================

[1/3] Backfilling 60 days of data for 'karachi' ...
      ✓  1,393 feature rows ready

[2/3] Training model ...
      ✓  Best model : GradientBoosting
         RMSE       : 14.32
         MAE        : 10.87
         R²         : 0.891

[3/3] Generating 72-hour forecast ...

      Time                     AQI  Category
      ──────────────────────────────────────────────────
      Fri May 30  00:00          82  Moderate
      Fri May 30  08:00          94  Moderate
      Fri May 30  16:00         118  Unhealthy for Sensitive Groups
      ...

============================================================
  ✓ Pipeline complete!
  Launch dashboard:  streamlit run app/dashboard.py
============================================================
```

### 3. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

---

## Configuration & Secrets

All secrets are read from **environment variables** (never hard-coded).

| Variable | Description | Required |
|---|---|---|
| `AQICN_TOKEN` | AQICN API token ([get free](https://aqicn.org/api/)) | For live data |
| `OPENWEATHER_API_KEY` | OpenWeatherMap key ([free tier](https://openweathermap.org/api)) | For weather features |
| `HOPSWORKS_API_KEY` | Hopsworks project key | Optional |
| `HOPSWORKS_PROJECT` | Hopsworks project name | Optional |
| `TARGET_CITY` | City slug for AQICN (e.g. `karachi`) | Recommended |
| `CITY_LAT` / `CITY_LON` | Coordinates for OpenWeatherMap | Recommended |
| `ALERT_EMAIL` | Email address to receive AQI alerts | Optional |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` | SMTP credentials for email alerts | Optional |

For local development, create a `.env` file (never commit it):

```dotenv
AQICN_TOKEN=your_token_here
OPENWEATHER_API_KEY=your_key_here
TARGET_CITY=karachi
CITY_LAT=24.8607
CITY_LON=67.0011
```

Load it before running:

```bash
export $(cat .env | xargs)
python pipelines/feature_pipeline.py
```

For GitHub Actions, add these as **repository secrets** (`Settings → Secrets → Actions`).

---

## Pipeline Details

### 1. Feature Pipeline

**File:** `pipelines/feature_pipeline.py`  
**Runs:** Every hour via GitHub Actions (`0 * * * *`)

```bash
python pipelines/feature_pipeline.py [--city karachi] [--lat 24.86] [--lon 67.00]
```

**What it does:**
1. Calls the AQICN API for current AQI and pollutant readings (PM2.5, PM10, O₃, NO₂, SO₂, CO).
2. Calls the OpenWeatherMap current-weather endpoint for temperature, humidity, wind speed/direction, and pressure.
3. Loads recent history from the Feature Store to compute lag and rolling features.
4. Engineers 28+ features (see table below).
5. Upserts the new row into the Feature Store.

**Engineered features:**

| Group | Features |
|---|---|
| Raw pollutants | `pm25`, `pm10`, `o3`, `no2`, `so2`, `co` |
| Weather | `temperature`, `humidity`, `wind_speed`, `wind_direction`, `pressure` |
| Time | `hour`, `day_of_week`, `month`, `is_weekend` |
| Cyclical time | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Lag | `aqi_lag_1h`, `_3h`, `_6h`, `_12h`, `_24h` |
| Rolling mean | `aqi_rolling_mean_3h`, `_6h`, `_24h` |
| Rolling std | `aqi_rolling_std_3h`, `_24h` |
| Change rate | `aqi_change_rate_1h`, `_3h`, `_6h` |
| **Target** | `aqi_next_24h` ← supervised learning label |

---

### 2. Historical Backfill

**File:** `pipelines/backfill_pipeline.py`

Run once to seed the feature store with enough historical data to train:

```bash
python pipelines/backfill_pipeline.py --days 90 --city karachi
```

Data sources (tried in order):
1. **OpenWeatherMap Air Pollution History API** (free, unlimited history)
2. **Synthetic data** — realistic random-walk + diurnal patterns (fallback when no API key is set)

---

### 3. Training Pipeline

**File:** `pipelines/training_pipeline.py`  
**Runs:** Daily at 02:00 UTC

```bash
python pipelines/training_pipeline.py [--city karachi]
```

**What it does:**
1. Loads all historical features from the Feature Store.
2. Performs a **chronological train/test split** (80/20) to prevent look-ahead bias.
3. Trains and evaluates all models (see [Models & Evaluation](#models--evaluation)).
4. Selects the best model by lowest RMSE.
5. Computes SHAP feature importances on the test set.
6. Saves the model pickle and registers metadata in `saved_models/model_registry.json`.

---

### 4. Web Dashboard

**File:** `app/dashboard.py`

```bash
streamlit run app/dashboard.py
```

Features:
- **Current AQI gauge** with EPA colour coding and category badge
- **72-hour forecast chart** with coloured AQI band shading
- **Metric cards**: avg AQI next 24h, peak AQI, avg over 72h
- **Pollutant breakdown table** (PM2.5, PM10, O₃, NO₂, SO₂, CO)
- **SHAP feature importance** horizontal bar chart
- **Hourly forecast table** with health guidance per row
- **Hazardous AQI alert banners** (AQI ≥ 150)
- **Live data refresh** button
- **Model information** expander (RMSE, MAE, R², training size)
- **AQI health guide** table

---

## CI/CD with GitHub Actions

**File:** `.github/workflows/pipelines.yml`

| Trigger | Job | Schedule |
|---|---|---|
| `0 * * * *` | Feature Pipeline | Every hour |
| `0 2 * * *` | Training Pipeline | Daily 02:00 UTC |
| Manual (`workflow_dispatch`) | Feature / Training / Backfill | On demand |

**Required GitHub secrets:**

```
AQICN_TOKEN
OPENWEATHER_API_KEY
HOPSWORKS_API_KEY      (optional)
HOPSWORKS_PROJECT      (optional)
```

**Required GitHub variables:**

```
TARGET_CITY   (e.g. karachi)
CITY_LAT      (e.g. 24.8607)
CITY_LON      (e.g. 67.0011)
```

---

## Models & Evaluation

All models are evaluated on the held-out **chronological** test set.

| Model | Notes |
|---|---|
| **Ridge Regression** | Baseline linear model with StandardScaler |
| **Random Forest** | 200 trees, max_depth=12, parallelised |
| **Gradient Boosting** | 200 estimators, learning_rate=0.05 |
| **XGBoost** | 300 rounds, colsample_bytree=0.8 (if installed) |
| **LSTM** | 2-layer Keras LSTM (64→32 units, dropout 0.2) |

**Metrics reported:** RMSE · MAE · R²

The model with the lowest RMSE on the test set is automatically selected and registered.

---

## SHAP Explainability

After training, SHAP values are computed on a 200-row random sample of the test set using:

- `TreeExplainer` for tree-based models (fast, exact)
- `KernelExplainer` for linear/unknown models (approximate)

Mean absolute SHAP values are stored in the model registry and displayed as a horizontal bar chart in the dashboard.  Typical top features:

1. `aqi_lag_1h` — the single most predictive feature
2. `aqi_rolling_mean_24h`
3. `pm25`
4. `aqi_change_rate_1h`
5. `hour_sin` / `hour_cos`

---

## AQI Alert System

The alert system fires whenever any predicted AQI value exceeds `alert_threshold` (default: **150 — Unhealthy**).

Alerts are:
1. **Logged** at WARNING level to `logs/alerts.log`
2. **Displayed as banner cards** on the dashboard
3. **Emailed** if `SMTP_*` and `ALERT_EMAIL` environment variables are set

Alert categories and EPA guidance are included in each alert dict:

```json
{
  "city": "karachi",
  "timestamp": "2024-06-01T14:00:00+00:00",
  "predicted_aqi": 183,
  "category": "Unhealthy",
  "message": "Everyone may begin to experience health effects...",
  "alert_time": "2024-06-01T13:02:11"
}
```

---

## Running Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ -v --cov=. --cov-report=term-missing
```

Test suite covers:
- Feature engineering (time features, lags, rolling stats, change rates, target)
- AQI category and colour mapping
- Health message completeness
- Alert threshold logic
- PM2.5 → AQI conversion
- Feature store save / load round-trip

---

## Deployment Options

### Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Point at `app/dashboard.py`.
4. Add your secrets under **Advanced settings → Secrets**.

### Railway / Render (free tier)

```bash
# Procfile
web: streamlit run app/dashboard.py --server.port=$PORT --server.address=0.0.0.0
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t aqi-predictor .
docker run -p 8501:8501 --env-file .env aqi-predictor
```

---

## API Keys & Free Tiers

| Service | Free tier | Sign-up |
|---|---|---|
| **AQICN** | Unlimited real-time AQI | [aqicn.org/api](https://aqicn.org/api/) |
| **OpenWeatherMap** | 60 calls/min, unlimited air-pollution history | [openweathermap.org](https://home.openweathermap.org/users/sign_up) |
| **Hopsworks** | 10 GB storage, 3 feature groups | [app.hopsworks.ai](https://app.hopsworks.ai/) |
| **GitHub Actions** | 2,000 min/month on public repos | [github.com](https://github.com/) |
| **Streamlit Cloud** | Free hosting for public repos | [share.streamlit.io](https://share.streamlit.io) |

The project runs **entirely within free tiers**.

*Built for the Pearls AQI Predictor project — serverless ML pipeline for air quality forecasting.*
