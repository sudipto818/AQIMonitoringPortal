"""
Anomaly Detector — Identifies unusual AQI spikes using Isolation Forest.

Model: sklearn.ensemble.IsolationForest

Why Isolation Forest:
    AQI anomalies are rare events in multi-dimensional pollutant space
    (AQI + PM2.5 + PM10 + CO + NO2 + O3). Isolation Forest works by
    randomly partitioning the feature space — anomalies, being "few and
    different," get isolated in fewer partitions (shorter path length).
    Unlike threshold-based SQL (mean + k×stddev), it detects anomalies
    across ALL pollutant dimensions simultaneously, catching hidden
    multi-pollutant anomalies that single-variable methods miss.

No training labels needed (unsupervised), fast, and works well on
tabular data with mixed scales.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from LordSinghalIsAlwaysSingle.ml.config import (
    ISOLATION_FOREST_CONTAMINATION,
    PLOT_DPI, PLOT_BG_COLOR, PLOT_TEXT_COLOR, PLOT_ACCENT_COLORS,
)


def detect_anomalies(df, city_name, output_dir, contamination=ISOLATION_FOREST_CONTAMINATION):
    """
    Run Isolation Forest on a city's multi-pollutant data.
    Saves:
      - anomaly_scatter.png: time-series with anomalies highlighted
      - data.json: list of anomaly dates + scores
    """
    os.makedirs(output_dir, exist_ok=True)

    features = ["aqi_daily", "pm25", "pm10", "co", "no2", "o3"]
    available = [f for f in features if f in df.columns and df[f].notna().sum() > 10]

    if len(available) < 2 or len(df) < 30:
        _save_empty(output_dir, city_name)
        return

    X = df[available].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    labels = iso.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X_scaled)    # lower = more anomalous

    df = df.copy()
    df["anomaly"] = labels
    df["anomaly_score"] = scores

    normal = df[df["anomaly"] == 1]
    anomalous = df[df["anomaly"] == -1]

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    ax.scatter(normal["date"], normal["aqi_daily"],
               color=PLOT_ACCENT_COLORS[0], s=8, alpha=0.4, label="Normal")
    ax.scatter(anomalous["date"], anomalous["aqi_daily"],
               color="#ff4444", s=30, alpha=0.9, zorder=5,
               edgecolors="white", linewidths=0.5, label="Anomaly (ML)")

    # SQL threshold line (mean + 2*std)
    mean_aqi = df["aqi_daily"].mean()
    std_aqi = df["aqi_daily"].std()
    threshold = mean_aqi + 2 * std_aqi
    ax.axhline(y=threshold, color=PLOT_ACCENT_COLORS[3], linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"SQL Threshold ({threshold:.0f})")

    ax.set_title(f"{city_name} — AQI Anomaly Detection (Isolation Forest)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", color=PLOT_TEXT_COLOR, fontsize=11)
    ax.set_ylabel("AQI", color=PLOT_TEXT_COLOR, fontsize=11)
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
    anomaly_list = []
    for _, row in anomalous.iterrows():
        anomaly_list.append({
            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10],
            "aqi": round(float(row["aqi_daily"]), 1),
            "pm25": round(float(row.get("pm25", 0)), 1),
            "pm10": round(float(row.get("pm10", 0)), 1),
            "score": round(float(row["anomaly_score"]), 4),
        })

    result = {
        "city": city_name,
        "model": "Isolation Forest",
        "contamination": contamination,
        "features_used": available,
        "total_points": len(df),
        "anomalies_detected": len(anomalous),
        "anomaly_rate": round(len(anomalous) / len(df) * 100, 2),
        "sql_threshold": round(threshold, 1),
        "ml_extra_detections": int((anomalous["aqi_daily"] <= threshold).sum()),
        "anomalies": anomaly_list[:50],  # keep top 50
    }

    with open(os.path.join(output_dir, "data.json"), "w") as f:
        json.dump(result, f, indent=2)


def detect_anomalies_all_cities(df, output_dir):
    """
    Run anomaly detection across all cities and produce a combined
    summary plot + per-city results.
    """
    os.makedirs(output_dir, exist_ok=True)
    cities = sorted(df["city"].unique())
    city_stats = []

    for city in cities:
        df_city = df[df["city"] == city].reset_index(drop=True)
        city_dir = os.path.join(output_dir, city.replace(" ", "_").lower())
        detect_anomalies(df_city, city, city_dir)

        # Collect stats
        json_path = os.path.join(city_dir, "data.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                stats = json.load(f)
                if "anomaly_rate" in stats:
                    city_stats.append({
                        "city": city,
                        "anomaly_rate": stats["anomaly_rate"],
                        "anomalies_detected": stats["anomalies_detected"],
                    })

    # ── Summary bar chart ─────────────────────────────────────────────
    if city_stats:
        city_stats.sort(key=lambda x: x["anomaly_rate"], reverse=True)
        fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
        ax.set_facecolor(PLOT_BG_COLOR)

        names = [s["city"] for s in city_stats[:15]]
        rates = [s["anomaly_rate"] for s in city_stats[:15]]
        colors = [PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)] for i in range(len(names))]

        bars = ax.barh(names[::-1], rates[::-1], color=colors[::-1], edgecolor="#333", height=0.6)
        for bar, rate in zip(bars, rates[::-1]):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{rate}%", va="center", color=PLOT_TEXT_COLOR, fontsize=9)

        ax.set_title("Anomaly Rate by City (Isolation Forest)",
                     color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Anomaly Rate (%)", color=PLOT_TEXT_COLOR, fontsize=11)
        ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#444")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "summary.png"), dpi=PLOT_DPI,
                    facecolor=PLOT_BG_COLOR, bbox_inches="tight")
        plt.close(fig)

        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump({"cities": city_stats}, f, indent=2)


def _save_empty(output_dir, city_name):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "data.json"), "w") as f:
        json.dump({"city": city_name, "error": "Insufficient data"}, f, indent=2)
