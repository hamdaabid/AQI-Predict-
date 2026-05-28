"""
Streamlit Dashboard — AQI Forecaster
=====================================
Loads the trained model and latest features from the Feature Store, generates
3-day (72-hour) predictions, and renders an interactive dashboard with:
  • Current AQI gauge with health-category colouring
  • 72-hour forecast chart
  • Pollutant breakdown table
  • SHAP feature importance bar chart
  • Hazardous AQI alerts banner

Run:
    streamlit run app/dashboard.py
"""

import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.config import api_config, model_config, pipeline_config
from pipelines.feature_pipeline import run_feature_pipeline
from utils.feature_store import FeatureStoreClient
from utils.helpers import (
    check_and_alert,
    get_aqi_category,
    get_aqi_color,
    aqi_health_message,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252b3b);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        border: 1px solid #2d3550;
    }
    .aqi-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.05rem;
        color: #fff;
        margin-top: 6px;
    }
    .alert-box {
        background: #3d1a1a;
        border: 1px solid #e53935;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Air_quality_icon.svg/240px-Air_quality_icon.svg.png",
        width=80,
    )
    st.title("AQI Predictor")
    st.caption("Pearls — End-to-End ML Pipeline")
    st.divider()

    city = st.text_input("City slug (AQICN)", value=api_config.city)
    forecast_hours = st.slider("Forecast horizon (hours)", 24, 72, 72, step=24)
    auto_refresh = st.checkbox("Auto-refresh every 60 min", value=False)
    run_live = st.button("🔄 Fetch live data now")
    st.divider()
    st.caption("Model registry path: " + pipeline_config.models_dir)


# ---------------------------------------------------------------------------
# Load model & feature columns
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_model_and_features():
    fs = FeatureStoreClient()
    meta = fs.load_latest_model_metadata()
    if meta is None:
        return None, None, None

    model_path = meta.get("model_path")
    feature_path = meta.get("feature_cols_path")

    if not model_path or not Path(model_path).exists():
        return None, None, meta

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    feature_cols = None
    if feature_path and Path(feature_path).exists():
        with open(feature_path, "rb") as f:
            feature_cols = pickle.load(f)

    return model, feature_cols, meta


model, feature_cols, model_meta = load_model_and_features()


# ---------------------------------------------------------------------------
# Load or refresh features
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading features…")
def load_features(city_slug: str):
    fs = FeatureStoreClient()
    df = fs.load_features(city=city_slug)
    return df


if run_live:
    with st.spinner("Fetching live AQI data…"):
        run_feature_pipeline(city=city)
    st.cache_data.clear()
    st.success("Live data refreshed!")

feature_df = load_features(city)


# ---------------------------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------------------------

def generate_predictions(
    df: pd.DataFrame,
    mdl,
    feat_cols: list,
    horizon: int,
) -> pd.DataFrame:
    """
    Roll forward the latest feature row ``horizon`` times to produce hourly
    predictions.  Each step updates lag & rolling features with the
    previously predicted AQI.
    """
    if df.empty or mdl is None:
        return pd.DataFrame()

    df = df.sort_values("timestamp").reset_index(drop=True)
    latest = df.iloc[-1:].copy()

    predictions = []
    current = latest.copy()

    for h in range(1, horizon + 1):
        future_ts = pd.Timestamp(latest["timestamp"].values[0]) + timedelta(hours=h)

        # Update time features
        current = current.copy()
        current["timestamp"] = future_ts
        current["hour"] = future_ts.hour
        current["day_of_week"] = future_ts.dayofweek
        current["month"] = future_ts.month
        current["is_weekend"] = int(future_ts.dayofweek >= 5)
        current["hour_sin"] = np.sin(2 * np.pi * future_ts.hour / 24)
        current["hour_cos"] = np.cos(2 * np.pi * future_ts.hour / 24)

        # Build feature vector
        X = current[[c for c in feat_cols if c in current.columns]]
        # Fill any missing feature columns with 0
        for c in feat_cols:
            if c not in X.columns:
                X[c] = 0.0

        pred_aqi = float(mdl.predict(X[feat_cols])[0])
        pred_aqi = max(0.0, min(500.0, pred_aqi))

        predictions.append({
            "timestamp": future_ts,
            "predicted_aqi": round(pred_aqi, 1),
            "category": get_aqi_category(pred_aqi),
            "color": get_aqi_color(pred_aqi),
        })

        # Roll lag features forward
        for lag in [1, 3, 6, 12, 24]:
            lag_col = f"aqi_lag_{lag}h"
            if lag == 1:
                current[lag_col] = current.get("aqi", pred_aqi)
            else:
                prev_lag = f"aqi_lag_{lag-1}h"
                current[lag_col] = current.get(prev_lag, pred_aqi)
        current["aqi"] = pred_aqi

    return pd.DataFrame(predictions)


predictions_df = generate_predictions(
    feature_df, model, feature_cols or model_config.feature_cols, forecast_hours
)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

alerts = []
if not predictions_df.empty:
    alerts = check_and_alert(predictions_df, city)


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

st.title(f"🌬️ Air Quality Forecast — {city.title()}")
st.caption(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ── Alert banner ────────────────────────────────────────────────────────────
if alerts:
    st.markdown("### ⚠️ Hazardous AQI Alert")
    for a in alerts[:3]:
        st.markdown(
            f'<div class="alert-box">⚠️ <b>{a["timestamp"].strftime("%a %b %d %H:00")}</b> — '
            f'AQI <b>{a["predicted_aqi"]:.0f}</b> ({a["category"]}). '
            f'{a["message"]}</div>',
            unsafe_allow_html=True,
        )

# ── Top metrics row ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

if not feature_df.empty:
    latest_row = feature_df.sort_values("timestamp").iloc[-1]
    current_aqi = latest_row.get("aqi", 0)
    current_cat = get_aqi_category(current_aqi)
    current_color = get_aqi_color(current_aqi)
else:
    current_aqi, current_cat, current_color = 0, "Unknown", "#888"

with col1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div style="color:#aaa;font-size:.85rem">Current AQI</div>'
        f'<div style="font-size:2.4rem;font-weight:800;color:{current_color}">'
        f'{current_aqi:.0f}</div>'
        f'<span class="aqi-badge" style="background:{current_color}">'
        f'{current_cat}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

if not predictions_df.empty:
    avg_24h = predictions_df.iloc[:24]["predicted_aqi"].mean()
    max_24h = predictions_df.iloc[:24]["predicted_aqi"].max()
    avg_72h = predictions_df["predicted_aqi"].mean()
else:
    avg_24h = max_24h = avg_72h = 0

with col2:
    st.metric("Avg AQI (next 24 h)", f"{avg_24h:.0f}", delta=f"{avg_24h - current_aqi:+.0f}")

with col3:
    st.metric("Peak AQI (next 24 h)", f"{max_24h:.0f}")

with col4:
    st.metric("Avg AQI (next 72 h)", f"{avg_72h:.0f}")

st.divider()

# ── Forecast chart ──────────────────────────────────────────────────────────
st.subheader("📈 72-Hour AQI Forecast")

if not predictions_df.empty:
    import plotly.graph_objects as go  # type: ignore

    # Shade background by AQI category bands
    bands = [
        (0,   50,  "rgba(0,228,0,0.07)",    "Good"),
        (51,  100, "rgba(255,255,0,0.07)",   "Moderate"),
        (101, 150, "rgba(255,126,0,0.07)",   "USG"),
        (151, 200, "rgba(255,0,0,0.07)",     "Unhealthy"),
        (201, 300, "rgba(143,63,151,0.07)",  "Very Unhealthy"),
        (301, 500, "rgba(126,0,35,0.07)",    "Hazardous"),
    ]

    fig = go.Figure()

    for lo, hi, fill, label in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=fill, line_width=0, annotation_text=label,
                      annotation_position="right", annotation_font_size=10)

    fig.add_trace(go.Scatter(
        x=predictions_df["timestamp"],
        y=predictions_df["predicted_aqi"],
        mode="lines+markers",
        name="Predicted AQI",
        line=dict(color="#4fc3f7", width=2.5),
        marker=dict(
            color=predictions_df["color"],
            size=7,
            line=dict(width=1, color="#fff"),
        ),
        hovertemplate="<b>%{x|%a %b %d %H:00}</b><br>AQI: %{y:.0f}<extra></extra>",
    ))

    # Add current AQI as reference point
    if not feature_df.empty:
        fig.add_trace(go.Scatter(
            x=[latest_row["timestamp"]],
            y=[current_aqi],
            mode="markers",
            name="Current",
            marker=dict(color="#ff9800", size=12, symbol="star"),
        ))

    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        font=dict(color="#cfd8dc"),
        xaxis=dict(gridcolor="#2d3550", showgrid=True),
        yaxis=dict(gridcolor="#2d3550", showgrid=True, range=[0, 350],
                   title="AQI"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10, b=10, l=10, r=80),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No predictions available. Run backfill + training pipelines first.")

st.divider()

# ── Pollutant breakdown & SHAP ───────────────────────────────────────────────
left, right = st.columns([1, 1])

with left:
    st.subheader("🧪 Current Pollutant Readings")
    if not feature_df.empty:
        pollutants = {
            "PM2.5 (µg/m³)": "pm25",
            "PM10 (µg/m³)": "pm10",
            "Ozone O₃ (ppb)": "o3",
            "NO₂ (ppb)": "no2",
            "SO₂ (ppb)": "so2",
            "CO (ppm)": "co",
        }
        rows = []
        for label, col in pollutants.items():
            val = latest_row.get(col, np.nan)
            rows.append({"Pollutant": label, "Value": f"{val:.1f}" if not np.isnan(val) else "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No pollutant data available.")

with right:
    st.subheader("📊 Feature Importance (SHAP)")
    if model_meta and model_meta.get("shap_importance"):
        import plotly.express as px  # type: ignore

        shap_data = model_meta["shap_importance"]
        top_n = 12
        features = list(shap_data.keys())[:top_n]
        values = [shap_data[f] for f in features]

        fig_shap = px.bar(
            x=values[::-1],
            y=features[::-1],
            orientation="h",
            color=values[::-1],
            color_continuous_scale="Blues",
            labels={"x": "Mean |SHAP value|", "y": "Feature"},
        )
        fig_shap.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.8)",
            font=dict(color="#cfd8dc"),
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("Train a model with SHAP enabled to see feature importance.")

st.divider()

# ── Forecast table ───────────────────────────────────────────────────────────
st.subheader("🗓️ Hourly Forecast Table")
if not predictions_df.empty:
    display_df = predictions_df.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%a %b %d  %H:00")
    display_df["health advice"] = display_df["predicted_aqi"].apply(aqi_health_message)
    display_df = display_df.rename(columns={
        "timestamp": "Date / Time",
        "predicted_aqi": "AQI",
        "category": "Category",
    }).drop(columns=["color"])
    st.dataframe(display_df, hide_index=True, use_container_width=True, height=300)
else:
    st.info("No forecast data available.")

# ── Model info ───────────────────────────────────────────────────────────────
with st.expander("ℹ️ Model Information"):
    if model_meta:
        m = model_meta.get("metrics", {})
        st.markdown(f"""
| Field | Value |
|---|---|
| Model type | `{m.get('model', '—')}` |
| RMSE | `{m.get('rmse', '—'):.2f}` |
| MAE | `{m.get('mae', '—'):.2f}` |
| R² | `{m.get('r2', '—'):.3f}` |
| Training rows | `{model_meta.get('train_rows', '—'):,}` |
| Features used | `{model_meta.get('n_features', '—')}` |
| Registered at | `{model_meta.get('registered_at', '—')}` |
        """)
    else:
        st.warning("No model registered yet. Run the training pipeline first.")

# ── Health guidance ──────────────────────────────────────────────────────────
with st.expander("💡 AQI Health Guide"):
    rows = [
        {"Range": f"{lo}–{hi}", "Category": cat, "Colour": color, "Guidance": aqi_health_message((lo + hi) / 2)}
        for cat, (lo, hi) in model_config.aqi_thresholds.items()
        for color in [get_aqi_color((lo + hi) / 2)]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
