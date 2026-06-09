"""
Feature Pipeline
================
Fetches raw weather & pollutant data from AQICN and OpenWeatherMap,
engineers features, and stores them in the Feature Store.

Run directly:
    python pipelines/feature_pipeline.py

Or import and call:
    from pipelines.feature_pipeline import run_feature_pipeline
    run_feature_pipeline()
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.config import api_config, model_config, pipeline_config
from utils.feature_store import FeatureStoreClient
from utils.helpers import get_logger

logger = get_logger("feature_pipeline")


# ---------------------------------------------------------------------------
# 1. Data Fetching
# ---------------------------------------------------------------------------

def fetch_aqicn_data(city: str, token: str) -> Optional[dict]:
    """
    Fetch real-time AQI and pollutant data from the AQICN API.
    Returns a raw dict or None on failure.
    """
    url = f"{api_config.aqicn_base_url}/feed/{city}/?token={token}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            logger.warning("AQICN returned non-ok status for %s: %s", city, data.get("data"))
            return None
        logger.info("AQICN data fetched for %s — AQI=%s", city, data["data"].get("aqi"))
        return data["data"]
    except requests.RequestException as exc:
        logger.error("AQICN request failed: %s", exc)
        return None


def fetch_openweather_data(lat: float, lon: float, api_key: str) -> Optional[dict]:
    """
    Fetch current weather (temperature, humidity, wind, pressure) from
    OpenWeatherMap.  Returns a raw dict or None on failure.
    """
    if not api_key:
        logger.warning("OPENWEATHER_API_KEY not set — skipping weather fetch")
        return None

    url = (
        f"{api_config.openweather_base_url}/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "OpenWeather data fetched — temp=%.1f°C, humidity=%d%%",
            data["main"]["temp"],
            data["main"]["humidity"],
        )
        return data
    except requests.RequestException as exc:
        logger.error("OpenWeather request failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

def parse_aqicn_row(raw: dict, city: str, ts: Optional[datetime] = None) -> dict:
    """Extract a flat dict of pollutant readings from a raw AQICN response."""
    iaqi = raw.get("iaqi", {})
    row = {
        "city": city,
        "timestamp": ts or datetime.now(timezone.utc).replace(microsecond=0),
        "aqi": float(raw.get("aqi", np.nan)),
        "pm25": float(iaqi.get("pm25", {}).get("v", np.nan)),
        "pm10": float(iaqi.get("pm10", {}).get("v", np.nan)),
        "o3": float(iaqi.get("o3", {}).get("v", np.nan)),
        "no2": float(iaqi.get("no2", {}).get("v", np.nan)),
        "so2": float(iaqi.get("so2", {}).get("v", np.nan)),
        "co": float(iaqi.get("co", {}).get("v", np.nan)),
        "temperature": float(iaqi.get("t", {}).get("v", np.nan)),
        "humidity": float(iaqi.get("h", {}).get("v", np.nan)),
        "wind_speed": float(iaqi.get("w", {}).get("v", np.nan)),
        "pressure": float(iaqi.get("p", {}).get("v", np.nan)),
    }
    return row


def parse_openweather_row(raw: dict) -> dict:
    """Extract weather fields from an OpenWeatherMap current-weather response."""
    wind = raw.get("wind", {})
    return {
        "temperature": raw["main"].get("temp", np.nan),
        "humidity": raw["main"].get("humidity", np.nan),
        "pressure": raw["main"].get("pressure", np.nan),
        "wind_speed": wind.get("speed", np.nan),
        "wind_direction": wind.get("deg", np.nan),
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features from a sorted, time-indexed DataFrame of
    raw pollutant + weather readings.

    Adds
    ----
    - Time-based features: hour, day_of_week, month, is_weekend, hour_sin/cos
    - Lag features: aqi_lag_{1,3,6,12,24}h
    - Rolling statistics: mean and std over 3 h, 6 h, 24 h windows
    - AQI change rates over 1 h, 3 h, 6 h
    - Forward target (aqi_next_24h) for supervised learning
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Time-based features ---
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Cyclical encoding so midnight and 23:00 are close
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Lag features (per city) ---
    aqi = df.groupby("city")["aqi"]
    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = aqi.shift(lag)

    # --- Rolling statistics ---
    for window in [3, 6, 24]:
        rolled = aqi.transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"aqi_rolling_mean_{window}h"] = rolled
        rolled_std = aqi.transform(lambda s: s.rolling(window, min_periods=1).std())
        df[f"aqi_rolling_std_{window}h"] = rolled_std.fillna(0)

    # --- AQI change rate (absolute Δ per hour) ---
    for lag in [1, 3, 6]:
        lag_col = f"aqi_lag_{lag}h"
        df[f"aqi_change_rate_{lag}h"] = (df["aqi"] - df[lag_col]) / lag

    # --- Target: AQI 24 hours into the future ---
    df["aqi_next_24h"] = aqi.shift(-24)

    # Fill forward/backward to reduce NaN pollution
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    logger.info("Feature engineering complete — %d rows, %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# 3. Main pipeline entry point
# ---------------------------------------------------------------------------

def run_feature_pipeline(
    city: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    historical_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Fetch → parse → engineer → store one round of features.

    Parameters
    ----------
    city            : City slug recognised by AQICN (default from config)
    lat, lon        : Coordinates for OpenWeatherMap (default from config)
    historical_df   : If provided, append to this instead of fetching live data
                      (used by the backfill pipeline)

    Returns
    -------
    Engineered feature DataFrame that was written to the feature store.
    """
    city = city or api_config.city
    lat = lat or api_config.city_lat
    lon = lon or api_config.city_lon

    logger.info("=== Feature pipeline START  city=%s ===", city)

    # Fetch
    if historical_df is not None:
        base_df = historical_df.copy()
    else:
        aqicn_raw = fetch_aqicn_data(city, api_config.aqicn_token)
        if aqicn_raw is None:
            logger.error("Could not fetch AQI data — aborting")
            return pd.DataFrame()

        row = parse_aqicn_row(aqicn_raw, city)

        # Overlay OpenWeatherMap weather if available
        weather_raw = fetch_openweather_data(lat, lon, api_config.openweather_key)
        if weather_raw:
            weather = parse_openweather_row(weather_raw)
            for k, v in weather.items():
                if np.isnan(row.get(k, np.nan)):
                    row[k] = v

        # Wind direction not in AQICN — default 0 if still missing
        row.setdefault("wind_direction", 0.0)

        base_df = pd.DataFrame([row])

    # Load recent history from feature store to compute lags properly
    fs = FeatureStoreClient()
    history = fs.load_features(city=city)

    if not history.empty:
        combined = pd.concat([history, base_df], ignore_index=True)
    else:
        combined = base_df.copy()

    # Engineer
    featured_df = engineer_features(combined)

    # Only persist the new rows (those in base_df timestamps)
    new_timestamps = pd.to_datetime(base_df["timestamp"])
    to_store = featured_df[featured_df["timestamp"].isin(new_timestamps)].copy()

    # Store
    fs.save_features(to_store)

    logger.info("=== Feature pipeline END — %d rows stored ===", len(to_store))
    return to_store


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the AQI feature pipeline once")
    parser.add_argument("--city", default=None, help="City slug for AQICN")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    args = parser.parse_args()

    result = run_feature_pipeline(city=args.city, lat=args.lat, lon=args.lon)
    if result.empty:
        sys.exit(1)
    print(result.tail(3).to_string())
