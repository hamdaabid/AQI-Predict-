"""
Training Pipeline
=================
Loads features from the Feature Store, trains and evaluates multiple
ML models (Ridge, RandomForest, XGBoost, LSTM), selects the best one,
saves it with SHAP explainability metadata, and registers it.

Run directly:
    python pipelines/training_pipeline.py

Or import:
    from pipelines.training_pipeline import run_training_pipeline
    metrics = run_training_pipeline()
"""

import os
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

from config.config import model_config, pipeline_config
from utils.feature_store import FeatureStoreClient
from utils.helpers import get_logger

logger = get_logger("training_pipeline")


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------

def prepare_training_data(df: pd.DataFrame):
    """
    Split a feature DataFrame into X (inputs) and y (target), then into
    chronological train/test sets.  Returns (X_train, X_test, y_train, y_test).
    """
    df = df.dropna(subset=[model_config.target_col]).copy()

    available_features = [
        c for c in model_config.feature_cols if c in df.columns
    ]
    missing = set(model_config.feature_cols) - set(available_features)
    if missing:
        logger.warning("Missing feature columns (will be ignored): %s", missing)

    X = df[available_features].copy()
    y = df[model_config.target_col].copy()

    # Chronological split — no shuffle to avoid leakage
    split_idx = int(len(df) * (1 - model_config.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        "Training data: %d train rows, %d test rows, %d features",
        len(X_train), len(X_test), X.shape[1],
    )
    return X_train, X_test, y_train, y_test, available_features


# ---------------------------------------------------------------------------
# 2. Model definitions
# ---------------------------------------------------------------------------

def _build_models():
    """Return a dict of {name: unfitted_model} to try."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=model_config.random_state,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=model_config.random_state,
        ),
    }

    # XGBoost (optional)
    try:
        from xgboost import XGBRegressor  # type: ignore
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=model_config.random_state,
            verbosity=0,
        )
    except ImportError:
        logger.info("XGBoost not installed — skipping")

    return models


def _build_lstm(input_shape: tuple):
    """Return a compiled Keras LSTM model for sequence regression."""
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf  # type: ignore
        from tensorflow.keras import layers  # type: ignore

        model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model
    except Exception as exc:
        logger.warning("TensorFlow not available for LSTM: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 3. Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, model_name: str) -> dict:
    """Compute RMSE, MAE, and R² for a fitted model."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    preds = model.predict(X_test)
    if hasattr(preds, "flatten"):
        preds = preds.flatten()

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    r2   = float(r2_score(y_test, preds))

    logger.info("%-20s  RMSE=%6.2f  MAE=%6.2f  R²=%5.3f", model_name, rmse, mae, r2)
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# 4. SHAP explainability
# ---------------------------------------------------------------------------

def compute_shap_values(model, X_sample: pd.DataFrame, model_name: str) -> Optional[dict]:
    """
    Compute SHAP feature importances.  Returns a dict mapping feature name
    to mean absolute SHAP value, sorted descending.
    """
    try:
        import shap  # type: ignore

        sample = X_sample.sample(min(200, len(X_sample)), random_state=42)

        # Tree-based models: TreeExplainer (fast)
        if model_name in ("RandomForest", "GradientBoosting", "XGBoost"):
            inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
            explainer = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(sample)
        else:
            # Linear / unknown: use KernelExplainer (slower but general)
            explainer = shap.KernelExplainer(model.predict, shap.sample(sample, 50))
            shap_values = explainer.shap_values(sample)

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(sample.columns, mean_abs.tolist()))
        importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

        top5 = list(importance.items())[:5]
        logger.info("SHAP top-5 features: %s", top5)
        return importance
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 5. Main training entry point
# ---------------------------------------------------------------------------

def run_training_pipeline(city: Optional[str] = None) -> dict:
    """
    Full training pipeline: load → prepare → train → evaluate → save.

    Returns a dict with the winning model's metrics.
    """
    logger.info("=== Training pipeline START ===")

    # --- Load features ---
    fs = FeatureStoreClient()
    df = fs.load_features(city=city)

    if df.empty or len(df) < 200:
        logger.error(
            "Not enough data (%d rows) — run backfill_pipeline.py first", len(df)
        )
        return {}

    # --- Prepare ---
    X_train, X_test, y_train, y_test, feature_cols = prepare_training_data(df)

    # --- Train & evaluate all models ---
    best_model = None
    best_metrics: dict = {"rmse": float("inf")}
    all_results = []

    for name, model in _build_models().items():
        try:
            logger.info("Training %s ...", name)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test, name)
            all_results.append(metrics)
            if metrics["rmse"] < best_metrics["rmse"]:
                best_metrics = metrics
                best_model = model
        except Exception as exc:
            logger.error("Model %s failed: %s", name, exc)

    if best_model is None:
        logger.error("All models failed — aborting")
        return {}

    logger.info(
        "Best model: %s  (RMSE=%.2f, MAE=%.2f, R²=%.3f)",
        best_metrics["model"], best_metrics["rmse"],
        best_metrics["mae"], best_metrics["r2"],
    )

    # --- SHAP ---
    shap_importance = compute_shap_values(best_model, X_test, best_metrics["model"])

    # --- Save model ---
    Path(pipeline_config.models_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(pipeline_config.models_dir, f"aqi_model_{ts}.pkl")
    feature_path = os.path.join(pipeline_config.models_dir, "feature_cols.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(feature_path, "wb") as f:
        pickle.dump(feature_cols, f)

    logger.info("Model saved to %s", model_path)

    # --- Register ---
    metadata = {
        "model_name": best_metrics["model"],
        "model_path": model_path,
        "feature_cols_path": feature_path,
        "city": city or "all",
        "metrics": best_metrics,
        "all_model_results": all_results,
        "shap_importance": shap_importance,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": len(feature_cols),
    }
    fs.save_model_metadata(metadata)

    logger.info("=== Training pipeline END ===")
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the AQI forecasting model")
    parser.add_argument("--city", default=None)
    args = parser.parse_args()

    result = run_training_pipeline(city=args.city)
    if result:
        m = result["metrics"]
        print(
            f"\n✓ Best model : {m['model']}\n"
            f"  RMSE       : {m['rmse']:.2f}\n"
            f"  MAE        : {m['mae']:.2f}\n"
            f"  R²         : {m['r2']:.3f}\n"
        )
    else:
        sys.exit(1)
