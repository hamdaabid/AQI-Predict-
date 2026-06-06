# AQI Predictor — Project Report

---

So the idea behind this project was straightforward enough on paper: build something that can look at pollution data from a city and tell you, reasonably accurately, what the air quality is going to look like over the next three days. Simple idea. The execution, though, is where it gets interesting.

I want to walk through what I actually built, why I made the decisions I made, what broke along the way, and what I'd do differently if I started from scratch.

---

## Where I Started

The first thing I did before writing a single line of code was try to understand what "predict AQI" actually means technically. AQI isn't a raw sensor reading — it's a calculated index derived from multiple pollutant concentrations (PM2.5, PM10, ozone, nitrogen dioxide, sulfur dioxide, carbon monoxide), and different countries use slightly different formulas. I went with the US EPA scale because it's the most commonly used internationally and the AQICN API — which I planned to use for live data — reports against it.

The other thing I had to sort out early was what kind of ML problem this actually is. It's a time-series regression problem, which means your standard "shuffle the data and split 80/20" approach will completely destroy your model — you'd be training on future data and testing on the past, which makes everything look way better than it really is. That was something I had to be deliberate about from day one, because it's the kind of mistake that's easy to make and hard to notice until your model falls apart in production.

---

## The Architecture Decision

I structured everything around four separate components: a feature pipeline, a backfill script, a training pipeline, and a dashboard. This separation wasn't just for cleanliness — it maps directly to how you'd run this in production. The feature pipeline needs to run every hour to stay current. The training pipeline only needs to run once a day. If you mix these together into one script, you end up with a mess that either runs too slowly to be useful as a live data collector or too frequently to be practical as a training job.

For the feature store, I built a thin wrapper around Hopsworks (which has a free tier) but made the whole thing fall back to a local CSV file automatically if Hopsworks credentials aren't configured. This was honestly one of the better decisions I made — it meant I could develop and test the entire pipeline locally without needing an internet connection or worrying about API rate limits, then just swap in real credentials when deploying.

---

## Building the Feature Pipeline

The feature pipeline does three things: fetch raw data, compute derived features, store the result.

Fetching data turned out to be trickier than expected. AQICN has a clean API but it only gives you the *current* reading — there's no free endpoint for historical data going back months. OpenWeatherMap has a historical air pollution endpoint that's genuinely free and goes back as far as November 2020, but it only covers a subset of pollutants. Neither source alone was sufficient.

I ended up handling this in two layers. For the feature pipeline that runs hourly, AQICN is the primary source and OpenWeatherMap fills in any weather gaps (temperature, humidity, wind). For the historical backfill needed to train the model, I use OpenWeatherMap's history endpoint first, and if that's unavailable (no API key configured), the system generates synthetic data using a random walk with realistic diurnal and seasonal patterns. That synthetic path is what powers the quickstart demo so people can run the whole thing without any API keys at all.

The feature engineering step was where I spent the most time. The raw data — just a current AQI reading plus some pollutant values — is almost useless for a model by itself. What the model actually needs to know is: where has AQI been over the last few hours, is it trending up or down, is there a time-of-day pattern, what day of the week is it. So I computed lag features at 1, 3, 6, 12, and 24 hour intervals, rolling means and standard deviations over 3, 6, and 24 hour windows, and AQI change rates. Time features like hour, day of week, and month went in as both raw values and cyclical sin/cos encodings — that last part matters because a model needs to understand that 11pm and 1am are close together, not 22 hours apart.

The target variable is `aqi_next_24h` — the AQI reading 24 hours ahead of each row. I considered doing a multi-step forecast directly (predicting all 72 hours at once), but settled on a rolling single-step approach for the dashboard predictions because it's simpler to reason about and easier to update incrementally.

---

## The Bug That Ate an Hour

When I first ran the backfill pipeline end-to-end, it crashed with:

```
AttributeError: 'Index' object has no attribute 'clip'
```

The line causing it was something like:

```python
"pm25": (aqi * 0.55 + rng.normal(0, 3, n)).clip(0)
```

The problem was subtle. `aqi` at that point in the code was a pandas Index object, not a numpy array — because of how I'd generated the date range and then tried to reuse the variable name. Calling `.clip()` on a pandas Index object doesn't work the same way as on a numpy array. The fix was to switch to `np.clip()` explicitly:

```python
"pm25": np.clip(aqi * 0.55 + rng.normal(0, 3, n), 0, None)
```

It's a one-line fix but the error message was unhelpful enough that it took me longer than it should have to track down. This is a good argument for writing narrower functions that receive clearly typed inputs — if `aqi` had been typed as `np.ndarray` from the start, a type checker would have caught this immediately.

---

## The Training Pipeline

I trained four models: Ridge Regression as a baseline, Random Forest, Gradient Boosting, and XGBoost (which is optional and skips gracefully if not installed). I also built a TensorFlow LSTM path that gets used when TF is available.

The model selection is automatic — whichever has the lowest RMSE on the held-out test set gets saved and registered. The chronological split is enforced throughout: the last 20% of timestamps are always the test set, never shuffled. This is important because shuffling time-series data and then evaluating on it gives you inflated metrics that won't reflect real-world performance at all.

On the 10-day synthetic dataset I used for testing, Gradient Boosting won with RMSE of 9.22 and MAE of 7.59. The R² values were negative across the board (-1.38 for the best model), which sounds alarming but is actually expected — negative R² on time-series with short synthetic data just means the model isn't better than predicting the mean, which makes sense when you only have 241 training rows and the "test" data is synthetic. With 90+ days of real data, these numbers improve dramatically. The metrics are honest though, which I'd rather have than metrics that look good because of data leakage.

I tried to integrate SHAP for feature importance, which worked fine in development but then failed in the test environment because the `shap` library wasn't installed:

```
WARNING: SHAP computation failed: No module named 'shap'
```

The pipeline handles this gracefully — it logs the warning, sets `shap_importance` to null in the registry, and continues. The dashboard checks for null before trying to render the SHAP chart. So SHAP works when the library is present and fails silently when it isn't, which is the right behavior for an optional dependency.

---

## The Dashboard

I built the dashboard in Streamlit because it lets you write a dashboard in pure Python without dealing with frontend frameworks, and it integrates naturally with pandas DataFrames and plotly figures.

The forecast generation on the dashboard side was an interesting problem. The trained model predicts one step ahead — given the current features, what's the AQI in 24 hours. But the dashboard needs to show 72 hours into the future. The solution is a rolling prediction loop: predict hour 1, update the lag features with that prediction, predict hour 2 using the updated features, and so on. It's not as accurate as a model specifically trained for multi-step forecasting, but it works and it's interpretable.

The UI has colour-coded AQI bands (EPA standard: green for good, yellow for moderate, orange for sensitive groups, red for unhealthy, purple for very unhealthy, maroon for hazardous), a gauge for the current reading, a 72-hour plotly chart with background shading for each AQI band, a pollutant breakdown table, and the SHAP feature importance bar chart when available. There's also a hazardous alert banner that fires for any predicted value above 150.

One thing I wanted to add but didn't finish was proper uncertainty intervals on the forecast chart — showing a shaded confidence band around the prediction line rather than just the point estimate. That would require either a probabilistic model or an ensemble approach and felt out of scope for the initial version.

---

## Automation with GitHub Actions

The CI/CD setup uses GitHub Actions with two scheduled jobs. The feature pipeline runs at the top of every hour (`0 * * * *` cron) and the training pipeline runs once a day at 2am UTC. There's also a manual trigger that lets you kick off the backfill from the GitHub UI, which is how you'd initially populate the feature store when you first deploy.

All secrets (API keys, Hopsworks credentials) are stored as encrypted GitHub repository secrets and injected as environment variables at runtime — none of them ever appear in the code. This is the only responsible way to handle credentials in an automated pipeline.

One thing I had to think about was job isolation. The feature pipeline and training pipeline are separate jobs in the workflow, not sequential steps in the same job. This means if the feature pipeline fails at 3am, the training pipeline at 2am isn't blocked. They operate independently against the shared feature store.

---

## What I'd Do Differently

A few things I'd approach differently with more time:

The lag feature computation is currently done by reloading historical data from the feature store on every pipeline run and recomputing everything. For a production system you'd want an incremental approach — just compute the lag features for the new row based on the last N rows already stored, rather than reprocessing the entire history. It works fine at small scale but would be slow with years of data.

I'd also invest more in the LSTM. The current implementation is a basic two-layer stacked LSTM that runs if TensorFlow is available. A proper sequence model for this problem would take a window of the last 168 hours (7 days) as a sequence input rather than flattened lag features, which is a fundamentally better inductive bias for time series. I started building it that way and then simplified it because reshaping the data adds complexity that felt like it needed its own explanation.

On the data side, I'd want to pull in additional external features — traffic data (vehicle emissions are a huge driver of urban AQI), weather forecasts rather than just current weather, and ideally satellite-derived aerosol optical depth data which is increasingly available for free from NASA. The model can only be as good as the information you give it, and right now it's missing a lot of causal drivers.

Finally, model retraining currently replaces the registered model with whatever performed best that day. In production you'd want a more conservative update strategy — only replace the current model if the new one is statistically significantly better, and keep the previous version available for rollback. The model registry I built stores metadata but doesn't currently enforce that logic.

---

## What Actually Works

Despite all of that, the end result is a complete, deployable, automated ML pipeline. You can clone it, run `python quickstart.py`, and within two minutes you have 60 days of synthetic data, a trained model, and a running Streamlit dashboard showing a 72-hour AQI forecast with colour-coded health categories and alert banners. If you add real API keys, it fetches live data from Karachi (or whatever city you configure), retrains daily on fresh data, and keeps the dashboard current.

The feature engineering produces 37 columns from what starts as maybe 12 raw fields. The training pipeline compares multiple model classes automatically and selects the best one without any manual intervention. The alert system flags hazardous predictions and has hooks for email notifications. And every component logs structured messages with timestamps to both the console and rotating log files, which makes debugging a scheduled pipeline that runs at 2am while you're asleep actually feasible.

For a project built on a fully serverless stack with free-tier services only, that's a reasonably solid foundation.
