"""
generate_all.py — Master orchestrator for the ML pipeline.

Reads the CSV data, computes ML enhancements for all 30 SQL queries,
and saves plots (PNG) + data (JSON) to frontend/public/ml-results/.

Usage:
    cd ml/
    python generate_all.py

Output:
    frontend/public/ml-results/
        q1/  q2/  ...  q30/          — per-query results
        clustering/                    — shared clustering output
        anomaly_detection/             — shared anomaly output
        health_risk/                   — shared health risk output
        forecasting/                   — shared forecast output
        metadata.json                  — generation timestamp + models used
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from LordSinghalIsAlwaysSingle.ml.config import (
    CSV_PATH, OUTPUT_DIR, QUERY_TITLES, QUERY_ML_TYPE,
    PLOT_DPI, PLOT_BG_COLOR, PLOT_TEXT_COLOR, PLOT_ACCENT_COLORS,
    AQI_CATEGORIES,
)
from data_loader import (
    load_data, get_city_daily, get_all_cities,
    get_monthly_aggregation, get_yearly_aggregation, get_city_features,
)
from models.forecaster import forecast_city_aqi
from models.anomaly import detect_anomalies, detect_anomalies_all_cities
from models.clustering import cluster_cities
from models.health_risk import compute_health_risk, compute_health_risk_all_cities


def setup_plot_style():
    """Configure global matplotlib dark-theme styling."""
    plt.rcParams.update({
        "figure.facecolor": PLOT_BG_COLOR,
        "axes.facecolor": PLOT_BG_COLOR,
        "text.color": PLOT_TEXT_COLOR,
        "axes.labelcolor": PLOT_TEXT_COLOR,
        "xtick.color": PLOT_TEXT_COLOR,
        "ytick.color": PLOT_TEXT_COLOR,
        "font.family": "sans-serif",
        "font.size": 10,
    })


# =====================================================================
# QUERY HANDLERS — Each function computes one query's ML enhancement
# =====================================================================

def generate_q1(df, out):
    """Q1: Longest Consecutive Severe AQI Days — Trend + Health Impact"""
    os.makedirs(out, exist_ok=True)

    # Compute severe streaks per city per year
    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        dc["is_severe"] = dc["aqi_daily"] >= 401
        dc["year"] = dc["date"].dt.year
        for year in dc["year"].unique():
            dy = dc[dc["year"] == year].reset_index(drop=True)
            max_streak = 0
            current = 0
            for _, row in dy.iterrows():
                if row["is_severe"]:
                    current += 1
                    max_streak = max(max_streak, current)
                else:
                    current = 0
            if max_streak > 0:
                results.append({"city": city, "year": int(year), "streak": max_streak})

    res_df = pd.DataFrame(results)
    if res_df.empty:
        _save_placeholder(out, "Q1", "No severe AQI streaks found")
        return

    # Top cities by max streak
    top = res_df.sort_values("streak", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    labels = [f"{r['city']} ({r['year']})" for _, r in top.iterrows()]
    streaks = top["streak"].values
    colors = [PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)] for i in range(len(labels))]

    bars = ax.barh(labels[::-1], streaks[::-1], color=colors[::-1], edgecolor="#333", height=0.6)
    for bar, s in zip(bars, streaks[::-1]):
        risk = "🔴 SEVERE" if s >= 10 else "🟠 HIGH" if s >= 5 else "🟡 MODERATE"
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{s} days — {risk}", va="center", color=PLOT_TEXT_COLOR, fontsize=9)

    ax.set_title("Q1: Longest Consecutive Severe AQI Days — Health Risk Assessment",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Consecutive Severe Days", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    # Trend over years
    yearly_trend = res_df.groupby("year")["streak"].max().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.plot(yearly_trend["year"], yearly_trend["streak"], "o-",
            color=PLOT_ACCENT_COLORS[0], linewidth=2, markersize=8)
    z = np.polyfit(yearly_trend["year"], yearly_trend["streak"], 1)
    p = np.poly1d(z)
    future_years = list(yearly_trend["year"]) + [yearly_trend["year"].max() + 1, yearly_trend["year"].max() + 2]
    ax.plot(future_years, p(future_years), "--", color=PLOT_ACCENT_COLORS[1],
            linewidth=2, label=f"Trend (slope={z[0]:.1f} days/yr)")
    ax.set_title("Severe Streak Trend Over Years + Forecast",
                 color=PLOT_TEXT_COLOR, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Year", color=PLOT_TEXT_COLOR)
    ax.set_ylabel("Max Severe Streak (days)", color=PLOT_TEXT_COLOR)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "trend.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {
        "query": "Q1",
        "title": QUERY_TITLES[1],
        "ml_type": "Trend Analysis + Health Risk",
        "findings": results[:20],
        "trend_slope": round(float(z[0]), 2),
        "prediction": f"If trend continues, max severe streak will be ~{int(p(yearly_trend['year'].max()+1))} days next year",
    })


def generate_q2(df, out):
    """Q2: State AQI Improvement — Prophet-style Trend Forecast"""
    os.makedirs(out, exist_ok=True)

    df_c = df.copy()
    # Map cities to states (extract from data — we need a city-state mapping)
    # Since the CSV doesn't have state, group by city and show city improvement
    yearly = get_yearly_aggregation(df_c)
    cities = yearly["city"].unique()

    improvements = []
    for city in cities:
        cy = yearly[yearly["city"] == city].sort_values("year")
        if len(cy) >= 3:
            latest = cy.iloc[-1]["avg_aqi"]
            old = cy.iloc[-3]["avg_aqi"] if len(cy) >= 3 else cy.iloc[0]["avg_aqi"]
            imp = old - latest
            improvements.append({"city": city, "latest_avg": round(latest, 1),
                                 "old_avg": round(old, 1), "improvement": round(imp, 1)})

    improvements.sort(key=lambda x: x["improvement"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    names = [i["city"] for i in improvements]
    imps = [i["improvement"] for i in improvements]
    colors = [PLOT_ACCENT_COLORS[0] if v > 0 else "#e74c3c" for v in imps]

    bars = ax.barh(names[::-1], imps[::-1], color=colors[::-1], edgecolor="#333", height=0.6)
    ax.axvline(x=0, color="#666", linewidth=1)

    for bar, v in zip(bars, imps[::-1]):
        label = f"+{v:.0f} ✓" if v > 0 else f"{v:.0f} ✗"
        ax.text(bar.get_width() + (2 if v >= 0 else -8), bar.get_y() + bar.get_height()/2,
                label, va="center", color=PLOT_TEXT_COLOR, fontsize=8)

    ax.set_title("Q2: AQI Improvement Over Last 3 Years (City-wise)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("AQI Improvement (positive = better air quality)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {
        "query": "Q2", "title": QUERY_TITLES[2],
        "ml_type": "Trend Forecast",
        "improvements": improvements,
        "best_improving": improvements[0]["city"] if improvements else "N/A",
        "worst_declining": improvements[-1]["city"] if improvements else "N/A",
    })


def generate_q3(df, out):
    """Q3: Extreme AQI Spike Detection — Isolation Forest vs SQL Threshold"""
    detect_anomalies_all_cities(df, out)


def generate_q4(df, out):
    """Q4: Severe/Very Poor Days — Health Risk Hotspots"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        dc["year"] = dc["date"].dt.year
        for year in dc["year"].unique():
            dy = dc[dc["year"] == year]
            vp = int((dy["aqi_daily"].between(401, 500)).sum())
            sv = int((dy["aqi_daily"] > 500).sum())
            if vp + sv > 0:
                results.append({"city": city, "year": int(year),
                                "very_poor": vp, "severe": sv, "total_bad": vp + sv})

    res_df = pd.DataFrame(results)
    if res_df.empty:
        _save_placeholder(out, "Q4", "No very poor/severe days found")
        return

    # Top 15 city-year combos
    top = res_df.sort_values("total_bad", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    labels = [f"{r['city']} ({r['year']})" for _, r in top.iterrows()]
    ax.barh(labels[::-1], top["very_poor"].values[::-1],
            color=PLOT_ACCENT_COLORS[2], label="Very Poor", edgecolor="#333", height=0.6)
    ax.barh(labels[::-1], top["severe"].values[::-1],
            left=top["very_poor"].values[::-1],
            color="#e74c3c", label="Severe", edgecolor="#333", height=0.6)

    ax.set_title("Q4: Pollution Hotspots — Very Poor + Severe AQI Days",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Days", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {
        "query": "Q4", "title": QUERY_TITLES[4],
        "ml_type": "Health Risk Assessment",
        "hotspots": results[:20],
    })


def generate_q5(df, out):
    """Q5: Consistently Good AQI Cities — Clustering"""
    os.makedirs(out, exist_ok=True)

    yearly = get_yearly_aggregation(df)
    good_cities = []
    all_city_avgs = []

    for city in get_all_cities(df):
        cy = yearly[yearly["city"] == city]
        max_avg = cy["avg_aqi"].max()
        mean_avg = cy["avg_aqi"].mean()
        all_city_avgs.append({"city": city, "max_yearly_avg": round(max_avg, 1),
                              "mean_avg": round(mean_avg, 1), "is_good": max_avg <= 150})
        if max_avg <= 150:
            good_cities.append(city)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    all_city_avgs.sort(key=lambda x: x["mean_avg"])
    names = [c["city"] for c in all_city_avgs]
    avgs = [c["mean_avg"] for c in all_city_avgs]
    colors = [PLOT_ACCENT_COLORS[3] if c["is_good"] else PLOT_ACCENT_COLORS[2] for c in all_city_avgs]

    bars = ax.barh(names, avgs, color=colors, edgecolor="#333", height=0.6)
    ax.axvline(x=150, color="#ff4444", linewidth=2, linestyle="--", label="Good Threshold (150)")

    for bar, c in zip(bars, all_city_avgs):
        tag = "✓ Good" if c["is_good"] else ""
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                tag, va="center", color=PLOT_ACCENT_COLORS[3], fontsize=8, fontweight="bold")

    ax.set_title("Q5: Consistently Good AQI Cities (Max Yearly Avg ≤ 150)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Average AQI", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {
        "query": "Q5", "title": QUERY_TITLES[5],
        "ml_type": "City Clustering",
        "good_cities": good_cities,
        "all_cities": all_city_avgs,
    })


def generate_q6(df, out):
    """Q6: Monthly Avg AQI — LSTM Forecast"""
    os.makedirs(out, exist_ok=True)
    # Pick top 5 cities by data volume for forecasting
    top_cities = df.groupby("city").size().sort_values(ascending=False).head(5).index.tolist()

    for city in top_cities:
        dc = get_city_daily(df, city)
        city_dir = os.path.join(out, city.replace(" ", "_").lower())
        forecast_city_aqi(dc, city, city_dir, metric="aqi_daily", forecast_days=365)

    # Combined monthly chart for all cities
    monthly = get_monthly_aggregation(df)
    fig, ax = plt.subplots(figsize=(16, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    for i, city in enumerate(top_cities):
        cm = monthly[monthly["city"] == city].sort_values(["year", "month"])
        dates = pd.to_datetime(cm.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}-01", axis=1))
        ax.plot(dates, cm["avg_aqi"], "-", color=PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)],
                linewidth=1.5, alpha=0.8, label=city)

    ax.set_title("Q6: Monthly Average AQI — Top 5 Cities (with LSTM forecasts per city)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", color=PLOT_TEXT_COLOR)
    ax.set_ylabel("Average AQI", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {
        "query": "Q6", "title": QUERY_TITLES[6],
        "ml_type": "LSTM Forecast",
        "cities_forecasted": top_cities,
        "forecast_period": "365 days",
    })


def generate_q7(df, out):
    """Q7: Monthly Avg PM2.5 — LSTM Forecast"""
    os.makedirs(out, exist_ok=True)
    top_cities = df.groupby("city").size().sort_values(ascending=False).head(5).index.tolist()

    for city in top_cities:
        dc = get_city_daily(df, city)
        city_dir = os.path.join(out, city.replace(" ", "_").lower())
        forecast_city_aqi(dc, city, city_dir, metric="pm25", forecast_days=365)

    monthly = get_monthly_aggregation(df)
    fig, ax = plt.subplots(figsize=(16, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    for i, city in enumerate(top_cities):
        cm = monthly[monthly["city"] == city].sort_values(["year", "month"])
        dates = pd.to_datetime(cm.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}-01", axis=1))
        ax.plot(dates, cm["avg_pm25"], "-", color=PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)],
                linewidth=1.5, alpha=0.8, label=city)

    ax.set_title("Q7: Monthly Average PM2.5 — Top 5 Cities (with LSTM forecasts per city)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", color=PLOT_TEXT_COLOR)
    ax.set_ylabel("Average PM2.5 (µg/m³)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {
        "query": "Q7", "title": QUERY_TITLES[7],
        "ml_type": "LSTM Forecast",
        "cities_forecasted": top_cities,
    })


def generate_q8(df, out):
    """Q8: Yearly AQI Growth/Decline Rate — Trend Projection"""
    os.makedirs(out, exist_ok=True)
    yearly = get_yearly_aggregation(df)

    fig, ax = plt.subplots(figsize=(16, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    cities = get_all_cities(df)[:10]
    growth_data = []

    for i, city in enumerate(cities):
        cy = yearly[yearly["city"] == city].sort_values("year")
        if len(cy) < 2:
            continue
        cy = cy.copy()
        cy["growth"] = cy["avg_aqi"].pct_change() * 100
        growth_data.append({
            "city": city,
            "avg_growth": round(cy["growth"].mean(), 2),
            "latest_growth": round(cy["growth"].iloc[-1], 2) if not pd.isna(cy["growth"].iloc[-1]) else 0,
        })

        ax.plot(cy["year"], cy["avg_aqi"], "o-",
                color=PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)],
                linewidth=1.5, markersize=6, alpha=0.8, label=city)

        # Trend line with projection
        if len(cy) >= 3:
            z = np.polyfit(cy["year"], cy["avg_aqi"], 1)
            p = np.poly1d(z)
            future = [cy["year"].max() + 1, cy["year"].max() + 2, cy["year"].max() + 3]
            ax.plot(future, p(future), "--",
                    color=PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)],
                    linewidth=1, alpha=0.5)

    ax.set_title("Q8: Yearly AQI Trend + 3-Year Projection",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", color=PLOT_TEXT_COLOR)
    ax.set_ylabel("Average AQI", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR, fontsize=8, ncol=2)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {
        "query": "Q8", "title": QUERY_TITLES[8],
        "ml_type": "Linear Trend Projection",
        "growth_data": growth_data,
    })


def generate_q9(df, out):
    """Q9: Worst Air Quality Month Each Year"""
    os.makedirs(out, exist_ok=True)

    df_c = df.copy()
    df_c["year"] = df_c["date"].dt.year
    df_c["month"] = df_c["date"].dt.month

    monthly = df_c.groupby(["year", "month"])["aqi_daily"].mean().reset_index()
    worst = monthly.loc[monthly.groupby("year")["aqi_daily"].idxmax()]

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Heatmap
    pivot = monthly.pivot(index="month", columns="year", values="aqi_daily").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                ax=ax, cbar_kws={"label": "Avg AQI"},
                linewidths=0.5, linecolor="#333",
                yticklabels=[month_names[i-1] for i in pivot.index])

    # Highlight worst months
    for _, row in worst.iterrows():
        col_idx = list(pivot.columns).index(row["year"])
        row_idx = list(pivot.index).index(row["month"])
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1,
                     fill=False, edgecolor="white", linewidth=3))

    ax.set_title("Q9: Monthly AQI Heatmap (◻ = Worst Month Each Year)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {
        "query": "Q9", "title": QUERY_TITLES[9],
        "ml_type": "Pattern Analysis",
        "worst_months": [{"year": int(r["year"]), "month": month_names[int(r["month"])-1],
                          "avg_aqi": round(r["aqi_daily"], 1)} for _, r in worst.iterrows()],
        "prediction": f"Most frequent worst month: {month_names[int(worst['month'].mode().iloc[0])-1]}",
    })


def generate_q10(df, out):
    """Q10: Summer vs Winter PM2.5 in North India"""
    os.makedirs(out, exist_ok=True)

    north_cities = ["Delhi", "Noida", "Lucknow", "Chandigarh", "Jaipur"]
    df_c = df[df["city"].isin(north_cities)].copy()
    df_c["month"] = df_c["date"].dt.month

    winter = df_c[df_c["month"].isin([12, 1, 2])].groupby("city")["pm25"].mean()
    summer = df_c[df_c["month"].isin([4, 5, 6])].groupby("city")["pm25"].mean()

    cities_found = sorted(set(winter.index) & set(summer.index))
    if not cities_found:
        cities_found = sorted(df["city"].unique()[:5])
        df_c = df[df["city"].isin(cities_found)].copy()
        df_c["month"] = df_c["date"].dt.month
        winter = df_c[df_c["month"].isin([12, 1, 2])].groupby("city")["pm25"].mean()
        summer = df_c[df_c["month"].isin([4, 5, 6])].groupby("city")["pm25"].mean()
        cities_found = sorted(set(winter.index) & set(summer.index))

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    x = np.arange(len(cities_found))
    w = 0.35
    w_vals = [winter.get(c, 0) for c in cities_found]
    s_vals = [summer.get(c, 0) for c in cities_found]

    ax.bar(x - w/2, w_vals, w, color=PLOT_ACCENT_COLORS[1], label="Winter (Dec-Feb)", edgecolor="#333")
    ax.bar(x + w/2, s_vals, w, color=PLOT_ACCENT_COLORS[3], label="Summer (Apr-Jun)", edgecolor="#333")

    ax.set_title("Q10: Winter vs Summer PM2.5 Levels",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Average PM2.5 (µg/m³)", color=PLOT_TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(cities_found, rotation=30, ha="right")
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="y", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    seasonal_data = [{"city": c, "winter_pm25": round(winter.get(c, 0), 1),
                      "summer_pm25": round(summer.get(c, 0), 1),
                      "ratio": round(winter.get(c, 1) / max(summer.get(c, 1), 0.01), 2)}
                     for c in cities_found]
    _save_json(out, {"query": "Q10", "title": QUERY_TITLES[10], "ml_type": "Seasonal Analysis",
                     "seasonal_data": seasonal_data})


def generate_q11(df, out):
    """Q11: Dominant Pollutant (PM2.5 vs PM10) — Clustering"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        if len(dc) < 30:
            continue
        corr_pm25 = dc["pm25"].corr(dc["aqi_daily"])
        corr_pm10 = dc["pm10"].corr(dc["aqi_daily"])
        dominant = "PM2.5" if corr_pm25 > corr_pm10 else "PM10" if corr_pm10 > corr_pm25 else "Equal"
        results.append({"city": city, "corr_pm25": round(corr_pm25, 4),
                        "corr_pm10": round(corr_pm10, 4), "dominant": dominant})

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    for i, r in enumerate(results):
        color = PLOT_ACCENT_COLORS[0] if r["dominant"] == "PM2.5" else PLOT_ACCENT_COLORS[2] if r["dominant"] == "PM10" else "#888"
        ax.scatter(r["corr_pm25"], r["corr_pm10"], s=100, color=color,
                   edgecolors="white", linewidth=1, alpha=0.8, zorder=5)
        ax.annotate(r["city"], (r["corr_pm25"], r["corr_pm10"]),
                    fontsize=7, color=PLOT_TEXT_COLOR, xytext=(5, 5), textcoords="offset points")

    ax.plot([0, 1], [0, 1], "--", color="#666", linewidth=1, label="Equal line")
    ax.set_title("Q11: PM2.5 vs PM10 Correlation with AQI per City",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Correlation: PM2.5 ↔ AQI", color=PLOT_TEXT_COLOR)
    ax.set_ylabel("Correlation: PM10 ↔ AQI", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q11", "title": QUERY_TITLES[11], "ml_type": "Correlation Clustering",
                     "results": results})


def generate_q12(df, out):
    """Q12: PM2.5-AQI Correlation Heatmap"""
    os.makedirs(out, exist_ok=True)

    pollutants = ["pm25", "pm10", "co", "no2", "o3", "aqi_daily"]
    corr_matrix = df[pollutants].corr()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
                ax=ax, linewidths=0.5, linecolor="#333",
                vmin=-1, vmax=1,
                xticklabels=["PM2.5", "PM10", "CO", "NO₂", "O₃", "AQI"],
                yticklabels=["PM2.5", "PM10", "CO", "NO₂", "O₃", "AQI"])
    ax.set_title("Q12: Pollutant Correlation Matrix",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q12", "title": QUERY_TITLES[12], "ml_type": "Correlation Analysis",
                     "correlation_matrix": corr_matrix.round(4).to_dict()})


def generate_q13(df, out):
    """Q13: Rising PM2.5 but Stable AQI"""
    os.makedirs(out, exist_ok=True)

    yearly = get_yearly_aggregation(df)
    results = []

    for city in get_all_cities(df):
        cy = yearly[yearly["city"] == city].sort_values("year")
        if len(cy) < 3:
            continue
        mid = len(cy) // 2
        pm25_diff = cy.iloc[mid:]["avg_pm25"].mean() - cy.iloc[:mid]["avg_pm25"].mean()
        aqi_diff = cy.iloc[mid:]["avg_aqi"].mean() - cy.iloc[:mid]["avg_aqi"].mean()
        if pm25_diff > 0 and abs(aqi_diff) < 10:
            results.append({"city": city, "pm25_increase": round(pm25_diff, 1),
                            "aqi_change": round(aqi_diff, 1)})

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    if results:
        results.sort(key=lambda x: x["pm25_increase"], reverse=True)
        cities = [r["city"] for r in results]
        pm_inc = [r["pm25_increase"] for r in results]
        aqi_ch = [r["aqi_change"] for r in results]

        x = np.arange(len(cities))
        ax.bar(x - 0.2, pm_inc, 0.35, color=PLOT_ACCENT_COLORS[2], label="PM2.5 Increase", edgecolor="#333")
        ax.bar(x + 0.2, aqi_ch, 0.35, color=PLOT_ACCENT_COLORS[0], label="AQI Change", edgecolor="#333")
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=45, ha="right")
        ax.axhline(y=0, color="#666", linewidth=1)
    else:
        ax.text(0.5, 0.5, "No cities found with rising PM2.5 and stable AQI",
                ha="center", va="center", color=PLOT_TEXT_COLOR, fontsize=14, transform=ax.transAxes)

    ax.set_title("Q13: Cities with Rising PM2.5 but Stable AQI — Hidden Risk",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Change (µg/m³ or AQI units)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="y", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q13", "title": QUERY_TITLES[13], "ml_type": "Divergence Analysis",
                     "results": results})


def generate_q14(df, out):
    """Q14: Weekday vs Weekend AQI"""
    os.makedirs(out, exist_ok=True)

    df_c = df.copy()
    df_c["dow"] = df_c["date"].dt.dayofweek  # 0=Mon, 6=Sun
    df_c["is_weekend"] = df_c["dow"].isin([5, 6])

    results = []
    for city in get_all_cities(df):
        dc = df_c[df_c["city"] == city]
        wd = dc[~dc["is_weekend"]]["aqi_daily"].mean()
        we = dc[dc["is_weekend"]]["aqi_daily"].mean()
        results.append({"city": city, "weekday_avg": round(wd, 1), "weekend_avg": round(we, 1),
                        "difference": round(wd - we, 1)})

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    results.sort(key=lambda x: abs(x["difference"]), reverse=True)
    cities = [r["city"] for r in results[:15]]
    x = np.arange(len(cities))

    ax.bar(x - 0.2, [r["weekday_avg"] for r in results[:15]], 0.35,
           color=PLOT_ACCENT_COLORS[0], label="Weekday", edgecolor="#333")
    ax.bar(x + 0.2, [r["weekend_avg"] for r in results[:15]], 0.35,
           color=PLOT_ACCENT_COLORS[3], label="Weekend", edgecolor="#333")

    ax.set_title("Q14: Weekday vs Weekend AQI Comparison",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Average AQI", color=PLOT_TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha="right")
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="y", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q14", "title": QUERY_TITLES[14], "ml_type": "Pattern Analysis",
                     "results": results})


def generate_q15(df, out):
    """Q15: AQI Volatility Frequency"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        dc["prev_aqi"] = dc["aqi_daily"].shift(1)
        dc["change"] = (dc["aqi_daily"] - dc["prev_aqi"]).abs()
        drastic = int((dc["change"] > 50).sum())
        total = len(dc) - 1
        results.append({"city": city, "drastic_changes": drastic, "total_days": total,
                        "volatility_rate": round(drastic / max(total, 1) * 100, 1)})

    results.sort(key=lambda x: x["volatility_rate"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    names = [r["city"] for r in results]
    rates = [r["volatility_rate"] for r in results]
    colors = [PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)] for i in range(len(names))]

    ax.barh(names[::-1], rates[::-1], color=colors[::-1], edgecolor="#333", height=0.6)
    ax.set_title("Q15: AQI Volatility — Days with >50 Point Swing",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Volatility Rate (%)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q15", "title": QUERY_TITLES[15], "ml_type": "Anomaly Detection",
                     "results": results})


def generate_q16(df, out):
    """Q16: Most Unpredictable AQI Cities — Clustering by Volatility"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        results.append({"city": city, "std_aqi": round(dc["aqi_daily"].std(), 2),
                        "mean_aqi": round(dc["aqi_daily"].mean(), 2)})

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    for i, r in enumerate(results):
        size = max(50, min(300, r["std_aqi"] * 3))
        ax.scatter(r["mean_aqi"], r["std_aqi"], s=size,
                   color=PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)],
                   edgecolors="white", linewidth=1, alpha=0.8, zorder=5)
        ax.annotate(r["city"], (r["mean_aqi"], r["std_aqi"]),
                    fontsize=8, color=PLOT_TEXT_COLOR, xytext=(5, 5), textcoords="offset points")

    ax.set_title("Q16: City Unpredictability — Mean AQI vs Standard Deviation",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Mean AQI", color=PLOT_TEXT_COLOR)
    ax.set_ylabel("AQI Std Dev (Volatility)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    results.sort(key=lambda x: x["std_aqi"], reverse=True)
    _save_json(out, {"query": "Q16", "title": QUERY_TITLES[16], "ml_type": "Clustering",
                     "most_unpredictable": results[0]["city"] if results else "N/A",
                     "results": results})


def generate_q17(df, out):
    """Q17: Average Duration of Polluted Air Spells (AQI>200)"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        dc["is_polluted"] = dc["aqi_daily"] > 200
        streaks = []
        current = 0
        for _, row in dc.iterrows():
            if row["is_polluted"]:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        if current > 0:
            streaks.append(current)
        avg_spell = round(np.mean(streaks), 1) if streaks else 0
        results.append({"city": city, "avg_spell_days": avg_spell,
                        "total_spells": len(streaks), "max_spell": max(streaks) if streaks else 0})

    results.sort(key=lambda x: x["avg_spell_days"], reverse=True)
    results_nonzero = [r for r in results if r["avg_spell_days"] > 0]

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    if results_nonzero:
        names = [r["city"] for r in results_nonzero]
        vals = [r["avg_spell_days"] for r in results_nonzero]
        colors = [PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)] for i in range(len(names))]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="#333", height=0.6)

        # WHO threshold
        ax.axvline(x=3, color="#ff4444", linewidth=2, linestyle="--", label="WHO Alert Threshold (3 days)")
    else:
        ax.text(0.5, 0.5, "No polluted air spells (AQI>200) found",
                ha="center", va="center", color=PLOT_TEXT_COLOR, fontsize=14, transform=ax.transAxes)

    ax.set_title("Q17: Average Polluted Air Spell Duration (AQI > 200)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Average Spell Duration (days)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q17", "title": QUERY_TITLES[17], "ml_type": "Health Risk",
                     "results": results})


def generate_q18(df, out):
    """Q18: AQI Before/After Rainfall"""
    os.makedirs(out, exist_ok=True)

    df_c = df.copy()
    df_c["month"] = df_c["date"].dt.month

    results = []
    for city in get_all_cities(df):
        dc = df_c[df_c["city"] == city]
        pre = dc[dc["month"].isin([4, 5])]["aqi_daily"].mean()
        post = dc[dc["month"].isin([10, 11])]["aqi_daily"].mean()
        impact = round(pre - post, 1) if not (np.isnan(pre) or np.isnan(post)) else 0
        results.append({"city": city, "pre_rainfall": round(pre, 1) if not np.isnan(pre) else 0,
                        "post_rainfall": round(post, 1) if not np.isnan(post) else 0,
                        "cleansing_impact": impact})

    results.sort(key=lambda x: x["cleansing_impact"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    cities = [r["city"] for r in results[:15]]
    x = np.arange(len(cities))

    ax.bar(x - 0.2, [r["pre_rainfall"] for r in results[:15]], 0.35,
           color=PLOT_ACCENT_COLORS[2], label="Pre-Monsoon (Apr-May)", edgecolor="#333")
    ax.bar(x + 0.2, [r["post_rainfall"] for r in results[:15]], 0.35,
           color=PLOT_ACCENT_COLORS[0], label="Post-Monsoon (Oct-Nov)", edgecolor="#333")

    ax.set_title("Q18: AQI Before vs After Monsoon — Cleansing Effect",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Average AQI", color=PLOT_TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha="right")
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="y", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q18", "title": QUERY_TITLES[18], "ml_type": "Seasonal Analysis",
                     "results": results})


def generate_q19(df, out):
    """Q19: Fastest AQI Recovery After Extreme Pollution"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        extreme_mask = dc["aqi_daily"] > 300
        recovery_times = []
        i = 0
        while i < len(dc):
            if extreme_mask.iloc[i]:
                j = i + 1
                while j < len(dc) and dc.iloc[j]["aqi_daily"] > 100:
                    j += 1
                if j < len(dc):
                    recovery_times.append(j - i)
                i = j
            else:
                i += 1
        if recovery_times:
            results.append({"city": city, "avg_recovery_days": round(np.mean(recovery_times), 1),
                            "events": len(recovery_times)})

    results.sort(key=lambda x: x["avg_recovery_days"])

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    if results:
        names = [r["city"] for r in results]
        days = [r["avg_recovery_days"] for r in results]
        colors = [PLOT_ACCENT_COLORS[3] if d < 5 else PLOT_ACCENT_COLORS[2] if d < 10 else "#e74c3c" for d in days]
        ax.barh(names[::-1], days[::-1], color=colors[::-1], edgecolor="#333", height=0.6)
        for bar, r in zip(ax.patches, results[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{r['avg_recovery_days']}d ({r['events']} events)",
                    va="center", color=PLOT_TEXT_COLOR, fontsize=8)

    ax.set_title("Q19: Average Recovery Time After Extreme AQI (>300)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Average Recovery Days (to AQI ≤ 100)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q19", "title": QUERY_TITLES[19], "ml_type": "Health Risk",
                     "results": results})


def generate_q20(df, out):
    """Q20: Gas vs Particulate Pollution by Region — Clustering"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        gas = dc[["co", "no2", "o3"]].mean().sum()
        pm = dc[["pm10", "pm25"]].mean().sum()
        profile = "Gas-Dominant" if gas > pm else "Particulate-Dominant"
        results.append({"city": city, "avg_gas": round(gas, 2), "avg_particulate": round(pm, 2),
                        "ratio": round(gas / max(pm, 0.01), 4), "profile": profile})

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    for i, r in enumerate(results):
        color = PLOT_ACCENT_COLORS[0] if r["profile"] == "Gas-Dominant" else PLOT_ACCENT_COLORS[2]
        ax.scatter(r["avg_gas"], r["avg_particulate"], s=120, color=color,
                   edgecolors="white", linewidth=1, alpha=0.8, zorder=5)
        ax.annotate(r["city"], (r["avg_gas"], r["avg_particulate"]),
                    fontsize=7, color=PLOT_TEXT_COLOR, xytext=(5, 5), textcoords="offset points")

    ax.plot([0, max(r["avg_gas"] for r in results)],
            [0, max(r["avg_gas"] for r in results)],
            "--", color="#666", linewidth=1, label="Equal line")

    ax.set_title("Q20: Gas vs Particulate Pollution Profile per City",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Average Chemical Pollution (CO + NO₂ + O₃)", color=PLOT_TEXT_COLOR)
    ax.set_ylabel("Average Particulate (PM10 + PM2.5)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q20", "title": QUERY_TITLES[20], "ml_type": "Clustering",
                     "results": results})


def generate_q21(df, out):
    """Q21: Earliest Winter Pollution Onset"""
    os.makedirs(out, exist_ok=True)

    df_c = df.copy()
    df_c["year"] = df_c["date"].dt.year
    df_c["month"] = df_c["date"].dt.month

    # Find first date when 7-day rolling avg PM2.5 > 100 in Sep-Dec
    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        dc["pm25_7d"] = dc["pm25"].rolling(7, min_periods=1).mean()
        dc["year"] = dc["date"].dt.year
        dc["month"] = dc["date"].dt.month

        for year in dc["year"].unique():
            dy = dc[(dc["year"] == year) & (dc["month"].isin([9, 10, 11, 12]))]
            spike_days = dy[dy["pm25_7d"] > 100]
            if len(spike_days) > 0:
                onset = spike_days.iloc[0]["date"]
                results.append({"city": city, "year": int(year),
                                "onset_date": str(onset.date()),
                                "onset_month": int(onset.month),
                                "onset_day": int(onset.day)})

    if not results:
        _save_placeholder(out, "Q21", "No winter onset patterns found")
        return

    # Average onset day per city
    onset_df = pd.DataFrame(results)
    avg_onset = onset_df.groupby("city")["onset_day"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    colors = [PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)] for i in range(len(avg_onset))]
    ax.barh(avg_onset.index, avg_onset.values, color=colors, edgecolor="#333", height=0.6)

    ax.set_title("Q21: Average Winter Pollution Onset Day (PM2.5 > 100, Sep-Dec)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Average Day of Month (Onset)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q21", "title": QUERY_TITLES[21], "ml_type": "Trend Forecast",
                     "onset_data": results[:30]})


def generate_q22(df, out):
    """Q22: 90th & 95th Percentile Pollution"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        p90 = np.percentile(dc["aqi_daily"].dropna(), 90)
        p95 = np.percentile(dc["aqi_daily"].dropna(), 95)
        results.append({"city": city, "p90": round(p90, 1), "p95": round(p95, 1)})

    results.sort(key=lambda x: x["p95"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    names = [r["city"] for r in results]
    x = np.arange(len(names))

    ax.barh(x + 0.2, [r["p90"] for r in results], 0.35,
            color=PLOT_ACCENT_COLORS[0], label="90th Percentile", edgecolor="#333")
    ax.barh(x - 0.2, [r["p95"] for r in results], 0.35,
            color=PLOT_ACCENT_COLORS[1], label="95th Percentile", edgecolor="#333")

    ax.set_title("Q22: 90th & 95th Percentile AQI by City",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("AQI Value", color=PLOT_TEXT_COLOR)
    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=8)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q22", "title": QUERY_TITLES[22], "ml_type": "Health Risk",
                     "results": results})


def generate_q23(df, out):
    """Q23: Seasonal Ozone Variation"""
    os.makedirs(out, exist_ok=True)

    df_c = df.copy()
    df_c["month"] = df_c["date"].dt.month

    results = []
    for city in get_all_cities(df):
        dc = df_c[df_c["city"] == city]
        summer_o3 = dc[dc["month"].isin([3, 4, 5, 6])]["o3"].mean()
        winter_o3 = dc[dc["month"].isin([11, 12, 1, 2])]["o3"].mean()
        diff = summer_o3 - winter_o3 if not (np.isnan(summer_o3) or np.isnan(winter_o3)) else 0
        results.append({"city": city, "summer_o3": round(summer_o3, 2) if not np.isnan(summer_o3) else 0,
                        "winter_o3": round(winter_o3, 2) if not np.isnan(winter_o3) else 0,
                        "summer_spike": round(diff, 2)})

    results.sort(key=lambda x: x["summer_spike"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    cities = [r["city"] for r in results[:15]]
    x = np.arange(len(cities))

    ax.bar(x - 0.2, [r["summer_o3"] for r in results[:15]], 0.35,
           color=PLOT_ACCENT_COLORS[2], label="Summer O₃", edgecolor="#333")
    ax.bar(x + 0.2, [r["winter_o3"] for r in results[:15]], 0.35,
           color=PLOT_ACCENT_COLORS[0], label="Winter O₃", edgecolor="#333")

    ax.set_title("Q23: Seasonal Ozone (O₃) — Summer vs Winter",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Average O₃ Level", color=PLOT_TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha="right")
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="y", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q23", "title": QUERY_TITLES[23], "ml_type": "Seasonal Analysis",
                     "results": results})


def generate_q24(df, out):
    """Q24: NO2/AQI Ratio — Traffic Pollution"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        avg_no2 = dc["no2"].mean()
        avg_aqi = dc["aqi_daily"].mean()
        ratio = avg_no2 / max(avg_aqi, 1)
        results.append({"city": city, "avg_no2": round(avg_no2, 2),
                        "avg_aqi": round(avg_aqi, 2), "traffic_ratio": round(ratio, 4)})

    results.sort(key=lambda x: x["traffic_ratio"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    names = [r["city"] for r in results]
    ratios = [r["traffic_ratio"] for r in results]
    colors = [PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)] for i in range(len(names))]

    ax.barh(names[::-1], ratios[::-1], color=colors[::-1], edgecolor="#333", height=0.6)

    ax.set_title("Q24: NO₂/AQI Ratio — Traffic Pollution Indicator",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("NO₂ / AQI Ratio (higher = more traffic-driven)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q24", "title": QUERY_TITLES[24], "ml_type": "Trend Analysis",
                     "results": results})


def generate_q25(df, out):
    """Q25: CO & PM2.5 Co-occurrence in Winter"""
    os.makedirs(out, exist_ok=True)

    df_c = df.copy()
    df_c["month"] = df_c["date"].dt.month
    winter = df_c[df_c["month"].isin([11, 12, 1, 2])].copy()

    co_threshold = winter["co"].quantile(0.75) if len(winter) > 10 else 0
    pm_threshold = winter["pm25"].quantile(0.75) if len(winter) > 10 else 0

    results = []
    for city in get_all_cities(df):
        wc = winter[winter["city"] == city]
        if len(wc) == 0:
            continue
        total = len(wc)
        co_spike = (wc["co"] > co_threshold).sum() if co_threshold > 0 else 0
        pm_spike = (wc["pm25"] > pm_threshold).sum() if pm_threshold > 0 else 0
        both = ((wc["co"] > co_threshold) & (wc["pm25"] > pm_threshold)).sum()
        rate = round(both / max(total, 1) * 100, 2)
        results.append({"city": city, "total_winter_days": total,
                        "co_pm_simultaneous": int(both), "rate": rate})

    results.sort(key=lambda x: x["rate"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    names = [r["city"] for r in results[:15]]
    rates = [r["rate"] for r in results[:15]]
    colors = [PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)] for i in range(len(names))]

    ax.barh(names[::-1], rates[::-1], color=colors[::-1], edgecolor="#333", height=0.6)
    ax.set_title("Q25: CO + PM2.5 Simultaneous Spike Rate (Winter)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Co-occurrence Rate (%)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q25", "title": QUERY_TITLES[25], "ml_type": "Co-occurrence Analysis",
                     "results": results})


def generate_q26(df, out):
    """Q26: Hidden Hazard Days (High O3/CO, Low AQI)"""
    os.makedirs(out, exist_ok=True)

    hazards = df[(df["aqi_daily"] <= 100) & ((df["o3"] > 200) | (df["co"] > 2.5))].copy()

    if hazards.empty:
        # Relax thresholds
        hazards = df[(df["aqi_daily"] <= 150) & ((df["o3"] > 100) | (df["co"] > 1.5))].copy()

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    if not hazards.empty:
        city_counts = hazards.groupby("city").size().sort_values(ascending=False)
        colors = [PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)] for i in range(len(city_counts))]
        ax.barh(city_counts.index[::-1], city_counts.values[::-1],
                color=colors[::-1][:len(city_counts)], edgecolor="#333", height=0.6)
    else:
        ax.text(0.5, 0.5, "No hidden hazard days detected with current thresholds",
                ha="center", va="center", color=PLOT_TEXT_COLOR, fontsize=14, transform=ax.transAxes)

    ax.set_title("Q26: Hidden Hazard Days (Low AQI but High O₃/CO)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Hidden Hazard Days", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    hazard_list = []
    for _, row in hazards.head(30).iterrows():
        hazard_list.append({
            "city": row["city"],
            "date": str(row["date"].date()),
            "aqi": round(float(row["aqi_daily"]), 1),
            "o3": round(float(row["o3"]), 1),
            "co": round(float(row["co"]), 3),
        })

    _save_json(out, {"query": "Q26", "title": QUERY_TITLES[26], "ml_type": "Anomaly Detection",
                     "total_hazard_days": len(hazards), "hazards": hazard_list})


def generate_q27(df, out):
    """Q27: Recovery Time After Extreme Pollution"""
    # Reuse Q19 logic with stricter threshold (AQI >= 401)
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        extreme_mask = dc["aqi_daily"] >= 401
        recovery_times = []
        i = 0
        while i < len(dc):
            if extreme_mask.iloc[i]:
                j = i + 1
                while j < len(dc) and dc.iloc[j]["aqi_daily"] > 100:
                    j += 1
                if j < len(dc):
                    recovery_times.append(j - i)
                i = j
            else:
                i += 1
        if recovery_times:
            results.append({"city": city, "avg_recovery": round(np.mean(recovery_times), 1),
                            "events": len(recovery_times)})

    results.sort(key=lambda x: x["avg_recovery"])

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    if results:
        names = [r["city"] for r in results]
        days = [r["avg_recovery"] for r in results]
        colors = [PLOT_ACCENT_COLORS[3] if d < 5 else PLOT_ACCENT_COLORS[2] if d < 15 else "#e74c3c" for d in days]
        ax.barh(names[::-1], days[::-1], color=colors[::-1], edgecolor="#333", height=0.6)
    else:
        ax.text(0.5, 0.5, "No severe events (AQI≥401) found for recovery analysis",
                ha="center", va="center", color=PLOT_TEXT_COLOR, fontsize=14, transform=ax.transAxes)

    ax.set_title("Q27: Recovery Time After Severe AQI (≥401) — to Normal (≤100)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Average Recovery Days", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q27", "title": QUERY_TITLES[27], "ml_type": "Health Risk",
                     "results": results})


def generate_q28(df, out):
    """Q28: Early Warning — PM2.5 crosses critical before Severe AQI"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        severe_mask = dc["aqi_daily"] >= 401
        warning_times = []

        for i in range(len(dc)):
            if severe_mask.iloc[i]:
                # Look backwards for PM2.5 >= 200
                for j in range(i - 1, max(i - 30, -1), -1):
                    if j >= 0 and dc.iloc[j]["pm25"] >= 200:
                        warning_times.append(i - j)
                        break

        if warning_times:
            results.append({"city": city, "avg_warning_days": round(np.mean(warning_times), 1),
                            "events": len(warning_times)})

    results.sort(key=lambda x: x["avg_warning_days"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    if results:
        names = [r["city"] for r in results]
        days = [r["avg_warning_days"] for r in results]
        colors = [PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)] for i in range(len(names))]
        ax.barh(names[::-1], days[::-1], color=colors[::-1], edgecolor="#333", height=0.6)
    else:
        ax.text(0.5, 0.5, "No PM2.5 early warning patterns found",
                ha="center", va="center", color=PLOT_TEXT_COLOR, fontsize=14, transform=ax.transAxes)

    ax.set_title("Q28: Early Warning — Days PM2.5 Crosses 200 Before Severe AQI",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Average Warning Lead Time (days)", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q28", "title": QUERY_TITLES[28], "ml_type": "Trend Analysis",
                     "results": results})


def generate_q29(df, out):
    """Q29: Winter-to-Summer AQI Drop"""
    os.makedirs(out, exist_ok=True)

    df_c = df.copy()
    df_c["month"] = df_c["date"].dt.month

    results = []
    for city in get_all_cities(df):
        dc = df_c[df_c["city"] == city]
        winter = dc[dc["month"].isin([12, 1, 2])]["aqi_daily"].mean()
        summer = dc[dc["month"].isin([4, 5, 6])]["aqi_daily"].mean()
        drop = round(winter - summer, 2) if not (np.isnan(winter) or np.isnan(summer)) else 0
        results.append({"city": city, "winter_avg": round(winter, 1) if not np.isnan(winter) else 0,
                        "summer_avg": round(summer, 1) if not np.isnan(summer) else 0,
                        "seasonal_drop": drop})

    results.sort(key=lambda x: x["seasonal_drop"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    # Dumbbell chart
    for i, r in enumerate(results):
        y = len(results) - 1 - i
        ax.plot([r["summer_avg"], r["winter_avg"]], [y, y],
                color="#555", linewidth=2, zorder=1)
        ax.scatter(r["summer_avg"], y, s=80, color=PLOT_ACCENT_COLORS[3], zorder=5, edgecolors="white")
        ax.scatter(r["winter_avg"], y, s=80, color=PLOT_ACCENT_COLORS[1], zorder=5, edgecolors="white")

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r["city"] for r in results[::-1]])
    ax.scatter([], [], color=PLOT_ACCENT_COLORS[3], s=80, label="Summer AQI")
    ax.scatter([], [], color=PLOT_ACCENT_COLORS[1], s=80, label="Winter AQI")

    ax.set_title("Q29: Winter vs Summer AQI — Seasonal Drop (Dumbbell Chart)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Average AQI", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR, loc="lower right")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q29", "title": QUERY_TITLES[29], "ml_type": "Seasonal Analysis",
                     "results": results})


def generate_q30(df, out):
    """Q30: Severe AQI Burst Cluster Analysis"""
    os.makedirs(out, exist_ok=True)

    results = []
    for city in get_all_cities(df):
        dc = get_city_daily(df, city)
        dc["is_severe"] = dc["aqi_daily"] >= 401
        streaks = []
        current = 0
        for _, row in dc.iterrows():
            if row["is_severe"]:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        if current > 0:
            streaks.append(current)
        if streaks:
            results.append({"city": city, "longest_cluster": max(streaks),
                            "total_clusters": len(streaks), "avg_cluster": round(np.mean(streaks), 1)})

    results.sort(key=lambda x: x["longest_cluster"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    if results:
        names = [r["city"] for r in results]
        longest = [r["longest_cluster"] for r in results]
        colors = [PLOT_ACCENT_COLORS[i % len(PLOT_ACCENT_COLORS)] for i in range(len(names))]
        ax.barh(names[::-1], longest[::-1], color=colors[::-1], edgecolor="#333", height=0.6)
        for bar, r in zip(ax.patches, results[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"Max: {r['longest_cluster']}d, Avg: {r['avg_cluster']}d ({r['total_clusters']} events)",
                    va="center", color=PLOT_TEXT_COLOR, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No severe AQI burst clusters found",
                ha="center", va="center", color=PLOT_TEXT_COLOR, fontsize=14, transform=ax.transAxes)

    ax.set_title("Q30: Severe AQI Burst Clusters (AQI ≥ 401)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Longest Consecutive Severe Days", color=PLOT_TEXT_COLOR)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot.png"), dpi=PLOT_DPI, facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    _save_json(out, {"query": "Q30", "title": QUERY_TITLES[30], "ml_type": "Burst Analysis",
                     "results": results})


# =====================================================================
# SHARED ML TASKS — Run once, used by multiple queries
# =====================================================================

def generate_shared_clustering(df, out):
    """Shared city clustering used by Q5, Q11, Q16, Q20"""
    city_features = get_city_features(df)
    cluster_cities(city_features, out)


def generate_shared_health(df, out):
    """Shared health risk assessment for all cities"""
    compute_health_risk_all_cities(df, out)


# =====================================================================
# UTILITIES
# =====================================================================

def _save_json(out_dir, data):
    with open(os.path.join(out_dir, "data.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)


def _save_placeholder(out_dir, query_id, message):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=14, color=PLOT_TEXT_COLOR, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_dir, "plot.png"), dpi=100,
                facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    _save_json(out_dir, {"query": query_id, "error": message})


# =====================================================================
# MAIN
# =====================================================================

def main():
    start = time.time()
    print("=" * 60)
    print("  AQI ML Pipeline — Generating All Results")
    print("=" * 60)

    setup_plot_style()

    print(f"\n[1/3] Loading data from {CSV_PATH}...")
    df = load_data()
    print(f"  → {len(df)} rows, {df['city'].nunique()} cities, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Generate all 30 queries ───────────────────────────────────────
    generators = {
        1: generate_q1, 2: generate_q2, 3: generate_q3, 4: generate_q4,
        5: generate_q5, 6: generate_q6, 7: generate_q7, 8: generate_q8,
        9: generate_q9, 10: generate_q10, 11: generate_q11, 12: generate_q12,
        13: generate_q13, 14: generate_q14, 15: generate_q15, 16: generate_q16,
        17: generate_q17, 18: generate_q18, 19: generate_q19, 20: generate_q20,
        21: generate_q21, 22: generate_q22, 23: generate_q23, 24: generate_q24,
        25: generate_q25, 26: generate_q26, 27: generate_q27, 28: generate_q28,
        29: generate_q29, 30: generate_q30,
    }

    print(f"\n[2/3] Generating ML results for {len(generators)} queries...")
    for qid, gen_fn in generators.items():
        q_dir = os.path.join(OUTPUT_DIR, f"q{qid}")
        try:
            print(f"  → Q{qid}: {QUERY_TITLES[qid]}...", end=" ", flush=True)
            gen_fn(df, q_dir)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            _save_placeholder(q_dir, f"Q{qid}", f"Error: {str(e)}")

    # ── Shared ML tasks ───────────────────────────────────────────────
    print("\n[3/3] Running shared ML models...")

    print("  → City Clustering (K-Means + PCA)...", end=" ", flush=True)
    try:
        generate_shared_clustering(df, os.path.join(OUTPUT_DIR, "clustering"))
        print("✓")
    except Exception as e:
        print(f"✗ {e}")

    print("  → Health Risk Assessment...", end=" ", flush=True)
    try:
        generate_shared_health(df, os.path.join(OUTPUT_DIR, "health_risk"))
        print("✓")
    except Exception as e:
        print(f"✗ {e}")

    # ── Metadata ──────────────────────────────────────────────────────
    metadata = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "data_rows": len(df),
        "cities": get_all_cities(df),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "models_used": ["LSTM / Holt-Winters", "Isolation Forest", "K-Means + PCA", "Rule-Based (NAQI)"],
        "total_queries": len(generators),
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  ✅ Done! Generated in {elapsed:.1f}s")
    print(f"  📁 Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
