"""
Feature Store client — wraps Hopsworks with a local CSV fallback so the
project runs even without Hopsworks credentials.

Usage
-----
    from utils.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()

    fs.save_features(df)                      # upsert features
    df = fs.load_features(start, end)         # fetch a date range
    fs.save_model_metadata(meta)              # register a trained model
    meta = fs.load_latest_model_metadata()    # fetch latest model record
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.config import hopsworks_config, pipeline_config
from utils.helpers import get_logger

logger = get_logger("feature_store")


class FeatureStoreClient:
    """
    Thin wrapper around Hopsworks.  Falls back to CSV/pickle on disk when
    Hopsworks is not configured (HOPSWORKS_API_KEY not set).
    """

    def __init__(self):
        self._hopsworks_available = False
        self._project = None
        self._fs = None
        self._fg = None

        Path(pipeline_config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(pipeline_config.models_dir).mkdir(parents=True, exist_ok=True)

        if hopsworks_config.api_key:
            self._init_hopsworks()
        else:
            logger.info(
                "HOPSWORKS_API_KEY not set — using local CSV fallback at %s",
                pipeline_config.data_dir,
            )

    # ------------------------------------------------------------------
    # Hopsworks initialisation
    # ------------------------------------------------------------------

    def _init_hopsworks(self):
        try:
            import hopsworks  # type: ignore

            self._project = hopsworks.login(
                api_key_value=hopsworks_config.api_key,
                project=hopsworks_config.project_name,
            )
            self._fs = self._project.get_feature_store()
            self._fg = self._fs.get_or_create_feature_group(
                name=hopsworks_config.feature_group_name,
                version=hopsworks_config.feature_group_version,
                description="Hourly AQI features and pollutant readings",
                primary_key=["city", "timestamp"],
                event_time="timestamp",
            )
            self._hopsworks_available = True
            logger.info("Connected to Hopsworks project '%s'", hopsworks_config.project_name)
        except Exception as exc:
            logger.warning("Hopsworks init failed (%s) — falling back to local CSV", exc)
            self._hopsworks_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_features(self, df: pd.DataFrame) -> None:
        """Upsert a DataFrame of features into the feature store."""
        if df.empty:
            logger.warning("save_features called with empty DataFrame — skipping")
            return

        if self._hopsworks_available:
            self._fg.insert(df, write_options={"wait_for_job": False})
            logger.info("Inserted %d rows into Hopsworks feature group", len(df))
        else:
            self._save_local(df)

    def load_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        city: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve features, optionally filtered by date range and city."""
        if self._hopsworks_available:
            return self._load_hopsworks(start_date, end_date, city)
        else:
            return self._load_local(start_date, end_date, city)

    def save_model_metadata(self, metadata: dict) -> None:
        """Persist model metadata (metrics, path, timestamp) to the registry."""
        path = Path(pipeline_config.models_dir) / "model_registry.json"

        registry: list = []
        if path.exists():
            with open(path) as f:
                registry = json.load(f)

        metadata["registered_at"] = datetime.utcnow().isoformat()
        registry.append(metadata)

        with open(path, "w") as f:
            json.dump(registry, f, indent=2, default=str)

        logger.info("Model metadata saved to registry: %s", path)

    def load_latest_model_metadata(self) -> Optional[dict]:
        """Return the most recently registered model's metadata dict."""
        path = Path(pipeline_config.models_dir) / "model_registry.json"
        if not path.exists():
            logger.warning("Model registry not found at %s", path)
            return None

        with open(path) as f:
            registry = json.load(f)

        if not registry:
            return None

        return registry[-1]

    def load_model(self, model_path: Optional[str] = None):
        """Load a pickled model from disk (or from path stored in registry)."""
        if model_path is None:
            meta = self.load_latest_model_metadata()
            if meta is None:
                raise FileNotFoundError("No model found in registry")
            model_path = meta.get("model_path")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded from %s", model_path)
        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_local(self, df: pd.DataFrame) -> None:
        path = Path(pipeline_config.data_dir) / "features.csv"

        if path.exists():
            existing = pd.read_csv(path, parse_dates=["timestamp"])
            combined = pd.concat([existing, df], ignore_index=True)
            combined.drop_duplicates(subset=["city", "timestamp"], keep="last", inplace=True)
            combined.sort_values("timestamp", inplace=True)
            combined.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)

        logger.info("Saved %d rows to local CSV at %s", len(df), path)

    def _load_local(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        city: Optional[str],
    ) -> pd.DataFrame:
        path = Path(pipeline_config.data_dir) / "features.csv"
        if not path.exists():
            logger.warning("Local features CSV not found at %s", path)
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["timestamp"])

        if city:
            df = df[df["city"] == city]
        if start_date:
            df = df[df["timestamp"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.Timestamp(end_date)]

        logger.info("Loaded %d rows from local CSV", len(df))
        return df.reset_index(drop=True)

    def _load_hopsworks(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        city: Optional[str],
    ) -> pd.DataFrame:
        try:
            fv = self._fs.get_or_create_feature_view(
                name=f"{hopsworks_config.feature_group_name}_view",
                version=1,
                query=self._fg.select_all(),
            )
            df = fv.get_batch_data()

            if city:
                df = df[df["city"] == city]
            if start_date:
                df = df[df["timestamp"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["timestamp"] <= pd.Timestamp(end_date)]

            logger.info("Loaded %d rows from Hopsworks", len(df))
            return df.reset_index(drop=True)
        except Exception as exc:
            logger.error("Hopsworks read failed: %s — returning empty DataFrame", exc)
            return pd.DataFrame()
