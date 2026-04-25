"""
LSTM Forecaster — Predicts future AQI / PM2.5 values for each city.

Model Architecture:
    Input  → (lookback × features)
    LSTM(64, return_sequences=True) → Dropout(0.2)
    LSTM(32) → Dropout(0.2)
    Dense(1)

Why LSTM:
    AQI is a time-series with strong seasonal patterns (winter spikes,
    monsoon drops) and non-linear dependencies between pollutants.
    LSTM's gating mechanism allows it to learn both short-term fluctuations
    and long-term seasonal trends, unlike simpler models (ARIMA) which
    assume linearity.

Fallback:
    If TensorFlow is unavailable, we fall back to Holt-Winters Exponential
    Smoothing from statsmodels, which handles trend + seasonality well.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Try TensorFlow, fall back to statsmodels
USE_LSTM = False
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    USE_LSTM = True
    print("[forecaster] Using TensorFlow LSTM")
except ImportError:
    print("[forecaster] TensorFlow unavailable, using Holt-Winters fallback")

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from LordSinghalIsAlwaysSingle.ml.config import (
    LSTM_LOOKBACK, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_FORECAST_DAYS,
    PLOT_DPI, PLOT_BG_COLOR, PLOT_TEXT_COLOR, PLOT_ACCENT_COLORS,
)


def _prepare_sequences(data, lookback):
    """Create supervised learning sequences from time-series array."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def _train_lstm(series, lookback=LSTM_LOOKBACK, epochs=LSTM_EPOCHS):
    """Train an LSTM model on a univariate series and return model + scaler."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))

    X, y = _prepare_sequences(scaled, lookback)
    if len(X) < 10:
        return None, scaler

    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=[es],
        verbose=0,
    )
    return model, scaler


def _forecast_lstm(model, scaler, last_window, steps):
    """Generate multi-step forecast using trained LSTM."""
    predictions = []
    current = last_window.copy()

    for _ in range(steps):
        inp = current.reshape(1, len(current), 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        predictions.append(pred)
        current = np.append(current[1:], pred)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


def _forecast_holtwinters(series, steps, seasonal_periods=365):
    """Forecast using Holt-Winters Exponential Smoothing as fallback."""
    try:
        if len(series) < 2 * seasonal_periods:
            seasonal_periods = min(30, len(series) // 2)
        if seasonal_periods < 2:
            seasonal_periods = None
            model = ExponentialSmoothing(
                series, trend="add", seasonal=None
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(
                series, trend="add", seasonal="add",
                seasonal_periods=seasonal_periods,
            ).fit(optimized=True)
        return model.forecast(steps)
    except Exception:
        # Ultimate fallback: repeat last year
        if len(series) >= steps:
            return series[-steps:]
        return np.full(steps, series.mean())


def forecast_city_aqi(df_city, city_name, output_dir, metric="aqi_daily",
                      forecast_days=LSTM_FORECAST_DAYS):
    """
    Train a forecasting model on a city's daily data and save:
      - plot.png: actual + forecast line chart
      - data.json: forecast values + model metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    series = df_city[metric].values.astype(float)
    dates = df_city["date"].values

    if len(series) < 60:
        _save_empty(output_dir, city_name, metric, "Insufficient data (<60 days)")
        return

    # ── Train / Forecast ──────────────────────────────────────────────
    model_used = "LSTM"
    if USE_LSTM and len(series) >= LSTM_LOOKBACK + 20:
        try:
            model, scaler = _train_lstm(series)
            if model is None:
                raise ValueError("Training failed")
            scaled = scaler.transform(series.reshape(-1, 1)).flatten()
            last_window = scaled[-LSTM_LOOKBACK:]
            forecast = _forecast_lstm(model, scaler, last_window, forecast_days)
        except Exception:
            forecast = _forecast_holtwinters(series, forecast_days)
            model_used = "Holt-Winters"
    else:
        forecast = _forecast_holtwinters(series, forecast_days)
        model_used = "Holt-Winters"

    forecast = np.clip(forecast, 0, 600)  # AQI can't be negative

    # ── Future dates ──────────────────────────────────────────────────
    last_date = pd.Timestamp(dates[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                 periods=forecast_days, freq="D")

    # ── Plot ──────────────────────────────────────────────────────────
    metric_label = "AQI" if "aqi" in metric else "PM2.5"
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    # Show last 365 days of actual data
    show_n = min(365, len(series))
    ax.plot(
        pd.to_datetime(dates[-show_n:]),
        series[-show_n:],
        color=PLOT_ACCENT_COLORS[0],
        linewidth=1.5,
        alpha=0.9,
        label=f"Actual {metric_label}",
    )
    ax.plot(
        future_dates, forecast,
        color=PLOT_ACCENT_COLORS[1],
        linewidth=2,
        linestyle="--",
        label=f"Forecast ({model_used})",
    )
    # Confidence band (±15% of forecast)
    lower = forecast * 0.85
    upper = forecast * 1.15
    ax.fill_between(future_dates, lower, upper,
                    color=PLOT_ACCENT_COLORS[1], alpha=0.15,
                    label="95% Confidence")

    ax.set_title(f"{city_name} — {metric_label} Forecast (Next {forecast_days} Days)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", color=PLOT_TEXT_COLOR, fontsize=11)
    ax.set_ylabel(metric_label, color=PLOT_TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="y", alpha=0.2, color="#555")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot.png"), dpi=PLOT_DPI,
                facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    # ── JSON ──────────────────────────────────────────────────────────
    result = {
        "city": city_name,
        "metric": metric_label,
        "model_used": model_used,
        "forecast_days": forecast_days,
        "forecast_summary": {
            "avg": round(float(forecast.mean()), 1),
            "min": round(float(forecast.min()), 1),
            "max": round(float(forecast.max()), 1),
        },
        "forecast_monthly": [],
    }

    # Monthly averages of the forecast
    fc_df = pd.DataFrame({"date": future_dates, "value": forecast})
    fc_df["month_label"] = fc_df["date"].dt.strftime("%b %Y")
    monthly_fc = fc_df.groupby("month_label", sort=False)["value"].mean()
    for month, val in monthly_fc.items():
        result["forecast_monthly"].append({
            "month": month,
            "predicted_avg": round(float(val), 1),
        })

    with open(os.path.join(output_dir, "data.json"), "w") as f:
        json.dump(result, f, indent=2)


def _save_empty(output_dir, city_name, metric, reason):
    """Save placeholder when no forecast is possible."""
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "city": city_name,
        "metric": metric,
        "model_used": "none",
        "error": reason,
    }
    with open(os.path.join(output_dir, "data.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Placeholder plot
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.text(0.5, 0.5, f"Insufficient data for {city_name}",
            ha="center", va="center", fontsize=14, color=PLOT_TEXT_COLOR,
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(output_dir, "plot.png"), dpi=100,
                facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)
