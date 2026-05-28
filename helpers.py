"""
Shared utility functions: logging setup, AQI helpers, alerting.
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

from config.config import model_config, pipeline_config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console and a rotating file."""
    Path(pipeline_config.logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(pipeline_config.logs_dir, f"{name}.log")

    logger = logging.getLogger(name)
    if logger.handlers:          # Avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# AQI helpers
# ---------------------------------------------------------------------------

def get_aqi_category(aqi: float) -> str:
    """Map a numeric AQI value to its EPA category label."""
    for category, (lo, hi) in model_config.aqi_thresholds.items():
        if lo <= aqi <= hi:
            return category
    return "Hazardous"


def get_aqi_color(aqi: float) -> str:
    """Return a hex colour for the AQI value (EPA colour scheme)."""
    if aqi <= 50:
        return "#00E400"   # Green
    elif aqi <= 100:
        return "#FFFF00"   # Yellow
    elif aqi <= 150:
        return "#FF7E00"   # Orange
    elif aqi <= 200:
        return "#FF0000"   # Red
    elif aqi <= 300:
        return "#8F3F97"   # Purple
    else:
        return "#7E0023"   # Maroon


def aqi_health_message(aqi: float) -> str:
    """Return a health guidance message for the given AQI level."""
    category = get_aqi_category(aqi)
    messages = {
        "Good": "Air quality is satisfactory. No precautions needed.",
        "Moderate": "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
        "Unhealthy": "Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exertion.",
        "Very Unhealthy": "Health alert: everyone may experience more serious health effects. Avoid outdoor activities.",
        "Hazardous": "Health emergency — everyone is likely to be affected. Remain indoors and keep activity levels low.",
    }
    return messages.get(category, "Check local guidelines.")


# ---------------------------------------------------------------------------
# Alert system
# ---------------------------------------------------------------------------

def check_and_alert(predictions: pd.DataFrame, city: str) -> list[dict]:
    """
    Scan predictions for hazardous AQI levels and return a list of alert
    dictionaries.  Optionally sends an e-mail if ALERT_EMAIL is configured.

    Parameters
    ----------
    predictions : DataFrame with columns ``timestamp`` and ``predicted_aqi``
    city        : City name for the alert message

    Returns
    -------
    List of alert dicts (empty if no threshold breaches).
    """
    logger = get_logger("alerts")
    alerts = []

    hazardous = predictions[
        predictions["predicted_aqi"] >= model_config.alert_threshold
    ]

    for _, row in hazardous.iterrows():
        alert = {
            "city": city,
            "timestamp": row["timestamp"],
            "predicted_aqi": row["predicted_aqi"],
            "category": get_aqi_category(row["predicted_aqi"]),
            "message": aqi_health_message(row["predicted_aqi"]),
            "alert_time": datetime.utcnow().isoformat(),
        }
        alerts.append(alert)
        logger.warning(
            "HAZARDOUS AQI ALERT — %s | %s | AQI=%.0f (%s)",
            city,
            row["timestamp"],
            row["predicted_aqi"],
            alert["category"],
        )

    if alerts:
        _send_email_alerts(alerts, city)

    return alerts


def _send_email_alerts(alerts: list[dict], city: str) -> None:
    """Send alert e-mail if SMTP credentials are configured."""
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    alert_email = os.getenv("ALERT_EMAIL")

    if not all([smtp_host, smtp_user, smtp_pass, alert_email]):
        return  # E-mail not configured — skip silently

    logger = get_logger("alerts")
    try:
        lines = [f"⚠️  AQI Alert for {city}\n"]
        for a in alerts:
            lines.append(
                f"  {a['timestamp']}  →  AQI {a['predicted_aqi']:.0f}"
                f"  ({a['category']})\n  {a['message']}\n"
            )
        body = "\n".join(lines)

        msg = MIMEText(body)
        msg["Subject"] = f"⚠️ Hazardous AQI Forecast — {city}"
        msg["From"] = smtp_user
        msg["To"] = alert_email

        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, alert_email, msg.as_string())

        logger.info("Alert e-mail sent to %s", alert_email)
    except Exception as exc:
        logger.error("Failed to send alert e-mail: %s", exc)


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create all required local directories if they don't exist."""
    for d in [pipeline_config.data_dir, pipeline_config.models_dir, pipeline_config.logs_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
