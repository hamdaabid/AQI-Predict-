#!/usr/bin/env python3
"""
quickstart.py
=============
End-to-end demonstration that runs the full pipeline locally without any
external API keys.  Generates synthetic data, trains a model, prints metrics,
and shows sample predictions.

Usage:
    python quickstart.py
    python quickstart.py --days 30 --city karachi
"""

import argparse
import os
import sys

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from config.config import pipeline_config
from utils.helpers import ensure_dirs, get_aqi_category, get_aqi_color


def main(days: int = 60, city: str = "karachi"):
    print("=" * 60)
    print("  AQI Predictor — Quick-start Demo")
    print("=" * 60)

    ensure_dirs()

    # ── Step 1: Backfill historical data ──────────────────────────────────
    print(f"\n[1/3] Backfilling {days} days of data for '{city}' ...")
    from pipelines.backfill_pipeline import run_backfill
    df = run_backfill(days=days, city=city)
    print(f"      ✓  {len(df):,} feature rows ready")

    # ── Step 2: Train model ────────────────────────────────────────────────
    print("\n[2/3] Training model ...")
    from pipelines.training_pipeline import run_training_pipeline
    meta = run_training_pipeline(city=city)
    if not meta:
        print("      ✗  Training failed — check logs/training_pipeline.log")
        sys.exit(1)

    m = meta["metrics"]
    print(f"      ✓  Best model : {m['model']}")
    print(f"         RMSE       : {m['rmse']:.2f}")
    print(f"         MAE        : {m['mae']:.2f}")
    print(f"         R²         : {m['r2']:.3f}")

    # ── Step 3: Sample predictions ─────────────────────────────────────────
    print("\n[3/3] Generating 72-hour forecast ...")
    import pickle
    from pathlib import Path
    import pandas as pd
    from datetime import timedelta

    model_path = meta["model_path"]
    feature_path = meta.get("feature_cols_path")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feature_path, "rb") as f:
        feature_cols = pickle.load(f)

    from utils.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    feature_df = fs.load_features(city=city)

    from app.dashboard import generate_predictions
    preds = generate_predictions(feature_df, model, feature_cols, 72)

    print(f"\n      {'Time':<22} {'AQI':>6}  Category")
    print("      " + "─" * 50)
    for _, row in preds.iloc[::8].iterrows():     # every 8 hours
        ts = row["timestamp"].strftime("%a %b %d  %H:00")
        cat = get_aqi_category(row["predicted_aqi"])
        print(f"      {ts:<22} {row['predicted_aqi']:>6.0f}  {cat}")

    print("\n" + "=" * 60)
    print("  ✓ Pipeline complete!")
    print("  Launch dashboard:  streamlit run app/dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60, help="Days of history to generate")
    parser.add_argument("--city", default="karachi")
    args = parser.parse_args()
    main(days=args.days, city=args.city)
