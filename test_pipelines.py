"""
Unit tests
==========
Run with:
    pytest tests/ -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Tests for pipelines/feature_pipeline.py::engineer_features"""

    @pytest.fixture
    def sample_df(self):
        """48 hourly rows of minimal raw data."""
        from pipelines.feature_pipeline import engineer_features  # noqa

        hours = pd.date_range(
            start=datetime(2024, 6, 1, 0, tzinfo=timezone.utc),
            periods=48,
            freq="h",
        )
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "city": "testcity",
            "timestamp": hours,
            "aqi": rng.uniform(30, 150, 48),
            "pm25": rng.uniform(5, 80, 48),
            "pm10": rng.uniform(10, 120, 48),
            "o3": rng.uniform(10, 60, 48),
            "no2": rng.uniform(5, 50, 48),
            "so2": rng.uniform(1, 20, 48),
            "co": rng.uniform(0.1, 1, 48),
            "temperature": rng.uniform(20, 35, 48),
            "humidity": rng.uniform(40, 80, 48),
            "wind_speed": rng.uniform(0, 10, 48),
            "wind_direction": rng.uniform(0, 360, 48),
            "pressure": rng.uniform(1005, 1020, 48),
        })

    def test_time_features_created(self, sample_df):
        from pipelines.feature_pipeline import engineer_features
        result = engineer_features(sample_df)
        for col in ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_lag_features_created(self, sample_df):
        from pipelines.feature_pipeline import engineer_features
        result = engineer_features(sample_df)
        for lag in [1, 3, 6, 12, 24]:
            assert f"aqi_lag_{lag}h" in result.columns

    def test_rolling_features_created(self, sample_df):
        from pipelines.feature_pipeline import engineer_features
        result = engineer_features(sample_df)
        for w in [3, 6, 24]:
            assert f"aqi_rolling_mean_{w}h" in result.columns
            assert f"aqi_rolling_std_{w}h" in result.columns

    def test_change_rate_features(self, sample_df):
        from pipelines.feature_pipeline import engineer_features
        result = engineer_features(sample_df)
        for lag in [1, 3, 6]:
            assert f"aqi_change_rate_{lag}h" in result.columns

    def test_target_column_created(self, sample_df):
        from pipelines.feature_pipeline import engineer_features
        result = engineer_features(sample_df)
        assert "aqi_next_24h" in result.columns

    def test_output_row_count(self, sample_df):
        from pipelines.feature_pipeline import engineer_features
        result = engineer_features(sample_df)
        assert len(result) == len(sample_df)

    def test_is_weekend_values(self, sample_df):
        from pipelines.feature_pipeline import engineer_features
        result = engineer_features(sample_df)
        assert result["is_weekend"].isin([0, 1]).all()

    def test_hour_cyclical_range(self, sample_df):
        from pipelines.feature_pipeline import engineer_features
        result = engineer_features(sample_df)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()


# ---------------------------------------------------------------------------
# Helper / utility tests
# ---------------------------------------------------------------------------

class TestAQIHelpers:
    """Tests for utils/helpers.py"""

    def test_good_category(self):
        from utils.helpers import get_aqi_category
        assert get_aqi_category(25) == "Good"

    def test_moderate_category(self):
        from utils.helpers import get_aqi_category
        assert get_aqi_category(75) == "Moderate"

    def test_hazardous_category(self):
        from utils.helpers import get_aqi_category
        assert get_aqi_category(400) == "Hazardous"

    def test_boundary_values(self):
        from utils.helpers import get_aqi_category
        assert get_aqi_category(50) == "Good"
        assert get_aqi_category(51) == "Moderate"
        assert get_aqi_category(100) == "Moderate"
        assert get_aqi_category(101) == "Unhealthy for Sensitive Groups"

    def test_aqi_color_returns_hex(self):
        from utils.helpers import get_aqi_color
        for aqi in [25, 75, 125, 175, 250, 400]:
            color = get_aqi_color(aqi)
            assert color.startswith("#"), f"Not a hex colour for AQI {aqi}"
            assert len(color) == 7

    def test_health_message_not_empty(self):
        from utils.helpers import aqi_health_message
        for aqi in [25, 75, 125, 175, 250, 400]:
            msg = aqi_health_message(aqi)
            assert isinstance(msg, str) and len(msg) > 10

    def test_alert_triggers_above_threshold(self):
        from utils.helpers import check_and_alert
        from config.config import model_config

        preds = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "predicted_aqi": [50, 100, 160, 220, 350],
        })
        alerts = check_and_alert(preds, "testcity")
        hazardous = [a for a in alerts if a["predicted_aqi"] >= model_config.alert_threshold]
        assert len(hazardous) == 3  # 160, 220, 350 exceed threshold of 150

    def test_no_alert_below_threshold(self):
        from utils.helpers import check_and_alert

        preds = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "predicted_aqi": [40, 80, 120],
        })
        alerts = check_and_alert(preds, "testcity")
        assert alerts == []


# ---------------------------------------------------------------------------
# PM2.5 → AQI conversion test
# ---------------------------------------------------------------------------

class TestAQIConversion:
    def test_pm25_low(self):
        from pipelines.backfill_pipeline import _pm25_to_aqi
        assert _pm25_to_aqi(5) <= 50          # Good

    def test_pm25_moderate(self):
        from pipelines.backfill_pipeline import _pm25_to_aqi
        aqi = _pm25_to_aqi(20)
        assert 51 <= aqi <= 100               # Moderate

    def test_pm25_unhealthy(self):
        from pipelines.backfill_pipeline import _pm25_to_aqi
        aqi = _pm25_to_aqi(60)
        assert 151 <= aqi <= 200              # Unhealthy


# ---------------------------------------------------------------------------
# Feature store local fallback
# ---------------------------------------------------------------------------

class TestFeatureStoreLocal:
    def test_save_and_load(self, tmp_path):
        import os
        os.environ["DATA_DIR"] = str(tmp_path)

        # Re-import with patched config
        import importlib
        import config.config as cfg
        cfg.pipeline_config.data_dir = str(tmp_path)

        from utils.feature_store import FeatureStoreClient
        fs = FeatureStoreClient()

        df = pd.DataFrame({
            "city": ["testcity"],
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "aqi": [85.0],
            "pm25": [30.0],
        })
        fs.save_features(df)
        loaded = fs.load_features(city="testcity")

        assert not loaded.empty
        assert abs(loaded.iloc[0]["aqi"] - 85.0) < 1e-3
