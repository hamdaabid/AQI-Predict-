"""
Central configuration for the AQI Predictor project.
All secrets should be set as environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class APIConfig:
    # AQICN API (https://aqicn.org/api/)
    aqicn_token: str = os.getenv("AQICN_TOKEN", "demo")
    # OpenWeatherMap API (https://openweathermap.org/api)
    openweather_key: str = os.getenv("OPENWEATHER_API_KEY", "")

    # Default city for predictions
    city: str = os.getenv("TARGET_CITY", "karachi")
    city_lat: float = float(os.getenv("CITY_LAT", "24.8607"))
    city_lon: float = float(os.getenv("CITY_LON", "67.0011"))

    aqicn_base_url: str = "https://api.waqi.info"
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5"


@dataclass
class HopsworksConfig:
    api_key: str = os.getenv("HOPSWORKS_API_KEY", "")
    project_name: str = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor")
    feature_group_name: str = "aqi_features"
    feature_group_version: int = 1
    model_name: str = "aqi_forecaster"
    model_version: int = 1


@dataclass
class ModelConfig:
    # Forecasting horizon: predict next 3 days (72 hours)
    forecast_horizon: int = 72
    # How many past hours to use as input features
    lookback_window: int = 168  # 7 days
    # Train/test split ratio
    test_size: float = 0.2
    random_state: int = 42

    # Feature columns used for training
    feature_cols: List[str] = field(default_factory=lambda: [
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temperature", "humidity", "wind_speed", "wind_direction", "pressure",
        "hour", "day_of_week", "month", "is_weekend",
        "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
        "aqi_rolling_mean_3h", "aqi_rolling_mean_6h", "aqi_rolling_mean_24h",
        "aqi_rolling_std_3h", "aqi_rolling_std_24h",
        "aqi_change_rate_1h", "aqi_change_rate_3h", "aqi_change_rate_6h",
    ])
    target_col: str = "aqi_next_24h"

    # AQI thresholds (US EPA scale)
    aqi_thresholds: dict = field(default_factory=lambda: {
        "Good": (0, 50),
        "Moderate": (51, 100),
        "Unhealthy for Sensitive Groups": (101, 150),
        "Unhealthy": (151, 200),
        "Very Unhealthy": (201, 300),
        "Hazardous": (301, 500),
    })
    alert_threshold: int = 150  # Alert when AQI exceeds this


@dataclass
class PipelineConfig:
    # How many days of historical data to backfill
    backfill_days: int = int(os.getenv("BACKFILL_DAYS", "90"))
    # Feature pipeline runs every hour; training pipeline runs every day
    feature_cron: str = "0 * * * *"
    training_cron: str = "0 2 * * *"

    # Local paths for offline/fallback storage
    data_dir: str = os.getenv("DATA_DIR", "./data")
    models_dir: str = os.getenv("MODELS_DIR", "./saved_models")
    logs_dir: str = os.getenv("LOGS_DIR", "./logs")


# Singleton instances
api_config = APIConfig()
hopsworks_config = HopsworksConfig()
model_config = ModelConfig()
pipeline_config = PipelineConfig()
