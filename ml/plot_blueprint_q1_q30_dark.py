"""
Visualization-only renderer for AQI query outputs Q1-Q30.

Reads precomputed JSON artifacts from:
  frontend/public/ml-results/

Writes plot images only (PNG). No ML model training, inference, or data pipeline reruns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
ML_RESULTS = ROOT / "frontend" / "public" / "ml-results"
RNG = np.random.default_rng(42)

PALETTE = {
    "bg": "#0e1117",
    "ax": "#151a22",
    "grid": "#2a3240",
    "text": "#e8ecf1",
    "muted": "#9aa4b2",
    "primary": "#4ea1ff",
    "secondary": "#36d399",
    "accent": "#f59e0b",
    "warn": "#f97316",
    "danger": "#ef4444",
    "pink": "#e879f9",
}


def _setup_dark_style() -> None:
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid")
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["ax"],
            "savefig.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["muted"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "text.color": PALETTE["text"],
            "axes.titleweight": "bold",
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )


def _qdir(qid: int) -> Path:
    return ML_RESULTS / f"q{qid}"


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _city_slug(city: str) -> str:
    return city.strip().lower().replace(" ", "_")


def _jitter(values: Sequence[float], scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr + RNG.normal(0.0, scale, size=len(arr))


def cleanup_plots(dry_run: bool = False) -> int:
    """Delete existing PNG plots only under q1..q30 folders."""
    count = 0
    for q in ML_RESULTS.glob("q*"):
        if not q.is_dir() or not q.name[1:].isdigit():
            continue
        for png in q.rglob("*.png"):
            count += 1
            if not dry_run:
                png.unlink(missing_ok=True)
    return count


def _to_df(qid: int, list_key: str = "results") -> pd.DataFrame:
    data = _load_json(_qdir(qid) / "data.json")
    return pd.DataFrame(data.get(list_key, []))


def plot_q1() -> None:
    qdir = _qdir(1)
    df = pd.DataFrame(_load_json(qdir / "data.json").get("findings", []))
    if df.empty:
        return

    by_city = df.groupby("city", as_index=False)["streak"].max().sort_values("streak", ascending=False)
    heat = df.pivot_table(index="city", columns="year", values="streak", aggfunc="max", fill_value=0)
    heat = heat.loc[by_city["city"].tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    sns.heatmap(
        heat,
        cmap="magma",
        annot=True,
        fmt=".0f",
        linewidths=0.6,
        linecolor=PALETTE["grid"],
        cbar_kws={"label": "Severe streak (days)"},
        ax=axes[0],
    )
    axes[0].set_title("Q1 Heatmap: Severe Streak Concentration")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("City")

    y = np.arange(len(by_city))
    axes[1].hlines(y=y, xmin=0, xmax=by_city["streak"], color=PALETTE["primary"], lw=2.2)
    axes[1].scatter(by_city["streak"], y, s=95, c=PALETTE["accent"], alpha=0.95, zorder=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(by_city["city"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Top severe streak (days)")
    axes[1].set_title("Q1 Lollipop: Worst Cities")

    top = by_city.iloc[0]
    axes[1].annotate(
        f"Takeaway: {top['city']} highest ({int(top['streak'])} days)",
        xy=(float(top["streak"]), 0),
        xytext=(float(top["streak"]) + 0.4, 0.8),
        arrowprops={"arrowstyle": "->", "color": PALETTE["muted"]},
        fontsize=9,
    )

    fig.suptitle("Q1: City-Year Severe Streak Heatmap + Top-Streak Lollipop", y=1.02)
    _save(fig, qdir / "plot.png")


def plot_q2() -> None:
    qdir = _qdir(2)
    df = pd.DataFrame(_load_json(qdir / "data.json").get("improvements", []))
    if df.empty:
        return

    df = df.sort_values("improvement", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, row in df.iterrows():
        old_avg = float(row["old_avg"])
        latest = float(row["latest_avg"])
        improved = latest < old_avg
        color = PALETTE["secondary"] if improved else PALETTE["danger"]

        ax.plot([old_avg, latest], [i, i], color=color, lw=2.3, alpha=0.9)
        ax.scatter(old_avg, i, c=PALETTE["muted"], s=40, zorder=3)
        ax.scatter(latest, i, c=color, s=72, zorder=3)
        ax.text(max(old_avg, latest) + 1.5, i, f"{row['improvement']:+.1f}", va="center", fontsize=8, color=color)

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis()
    ax.set_xlabel("AQI average (lower is better)")
    ax.set_title("Q2: Sorted Diverging Dumbbell (Old vs Latest AQI)")
    _save(fig, qdir / "plot.png")


def plot_q3() -> None:
    qdir = _qdir(3)
    city_frames: List[pd.DataFrame] = []

    for city_dir in sorted(qdir.iterdir()):
        if not city_dir.is_dir():
            continue
        path = city_dir / "data.json"
        if not path.exists():
            continue
        payload = _load_json(path)
        cdf = pd.DataFrame(payload.get("anomalies", []))
        if cdf.empty:
            continue
        cdf["city"] = payload.get("city", city_dir.name.title())
        city_frames.append(cdf)

    if not city_frames:
        return

    df = pd.concat(city_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "aqi", "score", "city"])
    if df.empty:
        return

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    y_jit = _jitter(df["aqi"], scale=1.25)
    sc = ax1.scatter(
        df["date"],
        y_jit,
        c=df["score"],
        cmap="coolwarm",
        s=24,
        alpha=0.55,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("Anomaly score")
    ax1.set_title("Q3 Panel A: Anomaly Timeline Scatter")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("AQI")

    worst = df.loc[df["score"].idxmin()]
    ax1.annotate(
        f"Most anomalous\n{worst['city']} ({worst['aqi']:.0f})",
        xy=(worst["date"], worst["aqi"]),
        xytext=(12, 12),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": PALETTE["muted"]},
        fontsize=8,
    )

    ax2 = fig.add_subplot(gs[0, 1])
    order = df.groupby("city")["score"].median().sort_values().index.tolist()
    sns.violinplot(data=df, x="score", y="city", order=order, inner="quartile", cut=0, color=PALETTE["primary"], ax=ax2)
    ax2.set_title("Q3 Panel B: Anomaly Score Distribution")
    ax2.set_xlabel("Anomaly score")
    ax2.set_ylabel("City")

    fig.suptitle("Q3: Timeline + Violin Distribution", y=1.02)
    _save(fig, qdir / "plot.png")


def plot_q4() -> None:
    qdir = _qdir(4)
    df = pd.DataFrame(_load_json(qdir / "data.json").get("hotspots", []))
    if df.empty:
        return

    df["label"] = df["city"] + " (" + df["year"].astype(str) + ")"
    df = df.sort_values("total_bad", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(df["label"], df["very_poor"], color=PALETTE["pink"], label="Very Poor")
    ax.barh(df["label"], df["severe"], left=df["very_poor"], color=PALETTE["danger"], label="Severe")
    ax.set_title("Q4: Stacked Bad AQI Days by City-Year")
    ax.set_xlabel("Days")
    ax.set_ylabel("City-Year")
    ax.legend(frameon=False)

    top = df.iloc[-1]
    ax.annotate(f"Peak hotspot: {top['label']} ({int(top['total_bad'])} days)", xy=(float(top["total_bad"]), len(df)-1), xytext=(8, 8), textcoords="offset points", fontsize=8)
    _save(fig, qdir / "plot.png")


def plot_q5() -> None:
    qdir = _qdir(5)
    df = pd.DataFrame(_load_json(qdir / "data.json").get("all_cities", []))
    if df.empty:
        return

    is_good = df["is_good"].astype(str).str.lower().isin(["true", "1"])
    df = df.assign(is_good_bool=is_good).sort_values("max_yearly_avg", ascending=True).reset_index(drop=True)

    x_max = max(180.0, float(df["max_yearly_avg"].max()) * 1.15)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.barh(y, np.full(len(df), x_max), color="#252c3a", height=0.64)
    colors = [PALETTE["secondary"] if v else PALETTE["warn"] for v in df["is_good_bool"]]
    ax.barh(y, df["max_yearly_avg"], color=colors, height=0.64)
    ax.axvline(150, color=PALETTE["text"], lw=1.5, ls="--", label="Threshold 150")

    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.set_xlabel("Max yearly AQI")
    ax.set_title("Q5: Threshold Bullet Chart (Max Yearly AQI vs 150)")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, qdir / "plot.png")


def _plot_forecast_small_multiples(qid: int, metric_label: str) -> None:
    qdir = _qdir(qid)
    root = _load_json(qdir / "data.json")
    cities = root.get("cities_forecasted", [])
    if not cities:
        return

    n = len(cities)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows), sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    for i, city in enumerate(cities):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        city_path = qdir / _city_slug(city) / "data.json"
        if not city_path.exists():
            ax.set_axis_off()
            continue

        mdf = pd.DataFrame(_load_json(city_path).get("forecast_monthly", []))
        if mdf.empty:
            ax.set_axis_off()
            continue

        mdf["month_dt"] = pd.to_datetime(mdf["month"], format="%b %Y", errors="coerce")
        mdf = mdf.dropna(subset=["month_dt"]).sort_values("month_dt")

        ax.plot(mdf["month_dt"], mdf["predicted_avg"], color=PALETTE["primary"], lw=2)
        ax.scatter(mdf["month_dt"], mdf["predicted_avg"], color=PALETTE["primary"], s=18, alpha=0.85)
        peak = mdf.loc[mdf["predicted_avg"].idxmax()]
        ax.annotate(f"peak {peak['predicted_avg']:.1f}", xy=(peak["month_dt"], peak["predicted_avg"]), xytext=(4, 6), textcoords="offset points", fontsize=8, color=PALETTE["accent"])

        ax.set_title(city)
        ax.set_ylabel(metric_label)
        ax.tick_params(axis="x", rotation=30)

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].set_axis_off()

    fig.suptitle(f"Q{qid}: Small-Multiple Seasonal Forecast Lines", y=1.02)
    fig.tight_layout()
    _save(fig, qdir / "plot.png")


def plot_q6() -> None:
    _plot_forecast_small_multiples(6, "Forecast AQI")


def plot_q7() -> None:
    _plot_forecast_small_multiples(7, "Forecast PM2.5")


def plot_q8() -> None:
    qdir = _qdir(8)
    df = _to_df(8, "growth_data")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    x = _jitter(df["avg_growth"], 0.12)
    y = _jitter(df["latest_growth"], 0.18)
    ax.scatter(x, y, s=110, color=PALETTE["primary"], alpha=0.55, edgecolor="none")

    ax.axhline(0, color=PALETTE["muted"], lw=1)
    ax.axvline(0, color=PALETTE["muted"], lw=1)
    for _, row in df.iterrows():
        ax.text(row["avg_growth"] + 0.2, row["latest_growth"] + 0.2, row["city"], fontsize=8)

    ax.set_title("Q8: Quadrant Scatter (avg_growth vs latest_growth)")
    ax.set_xlabel("Average growth rate (%)")
    ax.set_ylabel("Latest growth rate (%)")
    ax.text(0.98, 0.02, "Takeaway: top-right cities show persistent growth pressure", transform=ax.transAxes, ha="right", fontsize=8)
    _save(fig, qdir / "plot.png")


def plot_q9() -> None:
    qdir = _qdir(9)
    df = pd.DataFrame(_load_json(qdir / "data.json").get("worst_months", []))
    if df.empty:
        return

    month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    df["month_num"] = df["month"].map(month_map)
    df = df.dropna(subset=["month_num"]).copy()

    vmin, vmax = float(df["avg_aqi"].min()), float(df["avg_aqi"].max())
    if vmax > vmin:
        df["r"] = np.interp(df["avg_aqi"], (vmin, vmax), (35, 100))
    else:
        df["r"] = 70.0

    base = 2 * np.pi * (df["month_num"] - 1) / 12.0
    month_counts = df.groupby("month_num").cumcount() - (df.groupby("month_num")["month_num"].transform("count") - 1) / 2
    angle_jitter = month_counts * (2 * np.pi / 12.0) * 0.09
    theta = base + angle_jitter

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    norm = plt.Normalize(df["year"].min(), df["year"].max())
    cmap = plt.cm.plasma

    size = np.interp(df["avg_aqi"], (vmin, vmax), (70, 190)) if vmax > vmin else np.full(len(df), 110.0)
    for i, row in df.iterrows():
        color = cmap(norm(float(row["year"])))
        ax.scatter(theta.iloc[i], row["r"], s=size[i], c=[color], alpha=0.85, edgecolors="none")
        ax.text(theta.iloc[i], row["r"] + 3, str(int(row["year"])), fontsize=8, ha="center")

    ax.set_xticks(2 * np.pi * np.arange(12) / 12)
    ax.set_xticklabels(list(month_map.keys()))
    ax.set_ylim(30, 110)
    ax.set_title("Q9: Polar Month Wheel (scaled radius + anti-clustering jitter)", pad=18)

    ticks = np.linspace(35, 100, 4)
    labels = np.linspace(vmin, vmax, 4)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.0f}" for v in labels])
    ax.text(0.02, 0.02, "Radius scaled to avg AQI", transform=ax.transAxes, fontsize=8)
    _save(fig, qdir / "plot.png")


def plot_q10() -> None:
    qdir = _qdir(10)
    df = pd.DataFrame(_load_json(qdir / "data.json").get("seasonal_data", []))
    if df.empty:
        return

    df["delta"] = df["winter_pm25"] - df["summer_pm25"]
    df = df.sort_values("delta", ascending=False).reset_index(drop=True)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, row in df.iterrows():
        ax.plot([row["summer_pm25"], row["winter_pm25"]], [i, i], color=PALETTE["muted"], lw=2)

    ax.scatter(df["summer_pm25"], y, c=PALETTE["secondary"], s=72, label="Summer PM2.5")
    ax.scatter(df["winter_pm25"], y, c=PALETTE["danger"], s=72, label="Winter PM2.5")
    for i, row in df.iterrows():
        ax.text(float(row["winter_pm25"]) + 1.5, i, f"ratio {row['ratio']:.2f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis()
    ax.set_xlabel("PM2.5 (ug/m3)")
    ax.set_title("Q10: Sorted Seasonal PM2.5 Dumbbell")
    ax.legend(frameon=False)
    _save(fig, qdir / "plot.png")


def plot_q11() -> None:
    qdir = _qdir(11)
    df = _to_df(11)
    if df.empty:
        return

    # FIX: Create a temporary column to sort by the absolute difference, then drop it
    df["corr_diff"] = (df["corr_pm25"] - df["corr_pm10"]).abs()
    df = df.sort_values("corr_diff", ascending=False).drop(columns=["corr_diff"]).reset_index(drop=True)

    y = np.arange(len(df))
    pm10_mag = -np.abs(df["corr_pm10"].astype(float))
    pm25_mag = np.abs(df["corr_pm25"].astype(float))

    fig, ax = plt.subplots(figsize=(13, 8))
    
    # Map colors based on which pollutant is dominant
    c25 = [PALETTE["primary"] if "pm2.5" in str(d).lower() else "#3a4a64" for d in df["dominant"]]
    c10 = [PALETTE["accent"] if "pm10" in str(d).lower() else "#4a3a2a" for d in df["dominant"]]

    # Plot the mirrored horizontal bars
    ax.barh(y, pm10_mag, color=c10, alpha=0.9, label="|corr_pm10| (left)")
    ax.barh(y, pm25_mag, color=c25, alpha=0.9, label="|corr_pm25| (right)")
    
    # Draw the center zero line
    ax.axvline(0, color=PALETTE["muted"], lw=1)

    # Format axes and labels
    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis() # Put the highest difference at the top
    ax.set_xlabel("Correlation magnitude (mirrored)")
    ax.set_title("Q11: Mirrored Correlation Bars + Dominant Pollutant")
    ax.legend(frameon=False, fontsize=8)
    
    # Save output
    _save(fig, qdir / "plot.png")

def plot_q12() -> None:
    qdir = _qdir(12)
    data = _load_json(qdir / "data.json")
    cm = data.get("correlation_matrix", {})
    if not cm:
        return

    mat = pd.DataFrame(cm)
    mat = mat.loc[mat.columns, mat.columns]
    mask = np.tril(np.ones_like(mat, dtype=bool), k=-1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(mat, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.6, linecolor=PALETTE["grid"], cbar_kws={"label": "Correlation"}, ax=ax)
    ax.set_title("Q12: Correlation Matrix Heatmap (Upper Triangle)")
    _save(fig, qdir / "plot.png")


def plot_q13() -> None:
    qdir = _qdir(13)
    df = _to_df(13)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    x = _jitter(df["pm25_increase"], 0.08)
    y = _jitter(df["aqi_change"], 0.18)
    ax.scatter(x, y, s=120, c=PALETTE["primary"], alpha=0.6)
    ax.axhline(0, color=PALETTE["muted"], lw=1)
    ax.axvline(0, color=PALETTE["muted"], lw=1)
    ax.axhspan(-10, 10, xmin=0.5, xmax=1.0, color=PALETTE["secondary"], alpha=0.12)

    for _, row in df.iterrows():
        ax.text(row["pm25_increase"] + 0.08, row["aqi_change"] + 0.12, row["city"], fontsize=8)

    ax.set_title("Q13: Quadrant Scatter (PM2.5 increase vs AQI change)")
    ax.set_xlabel("PM2.5 increase")
    ax.set_ylabel("AQI change")
    ax.text(0.98, 0.03, "Target zone highlighted: PM2.5 rising, AQI near-stable", transform=ax.transAxes, ha="right", fontsize=8)
    _save(fig, qdir / "plot.png")


def plot_q14() -> None:
    qdir = _qdir(14)
    df = _to_df(14)
    if df.empty:
        return

    df = df.sort_values("difference", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(13, 8))
    for i, row in df.iterrows():
        ax.plot([row["weekday_avg"], row["weekend_avg"]], [i, i], color=PALETTE["muted"], lw=2)

    ax.scatter(df["weekday_avg"], y, c=PALETTE["primary"], s=60, label="Weekday")
    ax.scatter(df["weekend_avg"], y, c=PALETTE["accent"], s=60, label="Weekend")

    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis()
    ax.set_xlabel("Average AQI")
    ax.set_title("Q14: Weekday vs Weekend AQI Dumbbell")
    ax.legend(frameon=False)
    _save(fig, qdir / "plot.png")


def plot_q15() -> None:
    qdir = _qdir(15)
    df = _to_df(15)
    if df.empty:
        return

    df = df.sort_values("volatility_rate", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))
    dc_min = float(df["drastic_changes"].min())
    dc_max = float(df["drastic_changes"].max())
    if dc_max > dc_min:
        sizes = np.interp(df["drastic_changes"], (dc_min, dc_max), (60, 240))
    else:
        sizes = np.full(len(df), 140.0)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.hlines(y=y, xmin=0, xmax=df["volatility_rate"], color=PALETTE["muted"], lw=2)
    ax.scatter(df["volatility_rate"], y, s=sizes, c=PALETTE["primary"], alpha=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis()
    ax.set_xlabel("Volatility rate (%)")
    ax.set_title("Q15: Volatility Lollipop (size = drastic changes)")
    _save(fig, qdir / "plot.png")


def plot_q16() -> None:
    qdir = _qdir(16)
    df = _to_df(16)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    x = _jitter(df["mean_aqi"], 0.2)
    y = _jitter(df["std_aqi"], 0.15)
    ax.scatter(x, y, s=110, alpha=0.6, c=PALETTE["primary"], edgecolors="none")

    x_line = np.linspace(max(1, df["mean_aqi"].min() * 0.8), df["mean_aqi"].max() * 1.1, 120)
    for cv, col in [(0.2, "#64b5f6"), (0.35, "#81c784"), (0.5, "#ffb74d")]:
        ax.plot(x_line, cv * x_line, ls="--", lw=1.2, c=col, alpha=0.6)
        ax.text(x_line[-1], cv * x_line[-1], f"CV={cv:.2f}", fontsize=8, ha="right")

    for _, row in df.iterrows():
        ax.text(row["mean_aqi"] + 0.5, row["std_aqi"] + 0.4, row["city"], fontsize=8)

    ax.set_xlabel("Mean AQI")
    ax.set_ylabel("Std AQI")
    ax.set_title("Q16: Mean-Std Scatter with Iso-Volatility Guides")
    _save(fig, qdir / "plot.png")


def plot_q17() -> None:
    qdir = _qdir(17)
    df = _to_df(17)
    if df.empty:
        return

    df = df.sort_values("avg_spell_days", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.hlines(y=y, xmin=df["avg_spell_days"], xmax=df["max_spell"], color=PALETTE["muted"], lw=2)
    ax.scatter(df["avg_spell_days"], y, s=90, c=PALETTE["primary"], label="Avg spell")
    ax.scatter(df["max_spell"], y, s=70, c=PALETTE["accent"], label="Max spell")

    for i, row in df.iterrows():
        ax.text(float(row["max_spell"]) + 0.3, i, f"spells={int(row['total_spells'])}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis()
    ax.set_xlabel("Days")
    ax.set_title("Q17: Dot-Whisker Polluted Spell Durations")
    ax.legend(frameon=False)
    _save(fig, qdir / "plot.png")


def plot_q18() -> None:
    qdir = _qdir(18)
    df = _to_df(18)
    if df.empty:
        return

    df = df.sort_values("cleansing_impact", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = [PALETTE["secondary"] if v >= 0 else PALETTE["danger"] for v in df["cleansing_impact"]]
    ax.bar(x, df["cleansing_impact"], color=colors, alpha=0.9)
    ax.axhline(0, color=PALETTE["muted"], lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["city"], rotation=35, ha="right")
    ax.set_ylabel("AQI drop (pre - post)")
    ax.set_title("Q18: Sorted Waterfall-Style Cleansing Impact")

    best = df.iloc[0]
    ax.annotate(f"Largest drop: {best['city']} ({best['cleansing_impact']:.1f})", xy=(0, float(best["cleansing_impact"])), xytext=(10, 10), textcoords="offset points", fontsize=8)
    _save(fig, qdir / "plot.png")


def plot_q19() -> None:
    qdir = _qdir(19)
    df = _to_df(19)
    if df.empty:
        return

    df = df.sort_values("avg_recovery_days", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    ev_min = float(df["events"].min())
    ev_max = float(df["events"].max())
    if len(df) > 1 and ev_max > ev_min:
        sizes = np.interp(df["events"], (ev_min, ev_max), (120, 420))
    else:
        sizes = np.full(len(df), 220.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df["avg_recovery_days"], y, s=sizes, c=PALETTE["primary"], alpha=0.5, edgecolors="none")
    for i, row in df.iterrows():
        ax.text(float(row["avg_recovery_days"]) + 0.7, i, f"events={int(row['events'])}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis()
    ax.set_xlabel("Average recovery days")
    ax.set_title("Q19: Bubble Ranking (alpha-blended)")
    _save(fig, qdir / "plot.png")


def plot_q20() -> None:
    qdir = _qdir(20)
    df = _to_df(20)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    x = _jitter(df["avg_particulate"], 0.35)
    y = _jitter(df["avg_gas"], 0.25)
    sc = ax.scatter(x, y, c=df["ratio"], cmap="viridis", s=120, alpha=0.65, edgecolors="none")

    lim = max(float(df["avg_particulate"].max()), float(df["avg_gas"].max())) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", c=PALETTE["muted"], lw=1.2, label="Gas = Particulate")

    for _, row in df.iterrows():
        ax.text(row["avg_particulate"] + 1.0, row["avg_gas"] + 0.8, row["city"], fontsize=7, alpha=0.85)

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Gas/Particulate ratio")
    ax.set_xlabel("Avg particulate pollution")
    ax.set_ylabel("Avg gas pollution")
    ax.set_title("Q20: Gas vs Particulate Scatter + Ratio Scale")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, qdir / "plot.png")


def plot_q21() -> None:
    qdir = _qdir(21)
    df = pd.DataFrame(_load_json(qdir / "data.json").get("onset_data", []))
    if df.empty:
        return

    df["onset_date"] = pd.to_datetime(df["onset_date"], errors="coerce")
    df = df.dropna(subset=["onset_date", "city", "year"])  # type: ignore[arg-type]
    if df.empty:
        return

    df["doy"] = df["onset_date"].dt.dayofyear
    piv = df.pivot_table(index="city", columns="year", values="doy", aggfunc="min")
    piv = piv.sort_index()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(piv, cmap="mako", annot=True, fmt=".0f", linewidths=0.5, linecolor=PALETTE["grid"], cbar_kws={"label": "Onset day-of-year"}, ax=ax)
    ax.set_title("Q21: Onset Timeline Heatmap (city vs year)")
    ax.set_xlabel("Year")
    ax.set_ylabel("City")
    _save(fig, qdir / "plot.png")


def plot_q22() -> None:
    qdir = _qdir(22)
    df = _to_df(22)
    if df.empty:
        return

    df = df.sort_values("p95", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, row in df.iterrows():
        ax.plot([row["p90"], row["p95"]], [i, i], c=PALETTE["muted"], lw=2)

    ax.scatter(df["p90"], y, c=PALETTE["primary"], s=60, label="P90")
    ax.scatter(df["p95"], y, c=PALETTE["danger"], s=60, label="P95")

    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis()
    ax.set_xlabel("AQI percentile")
    ax.set_title("Q22: Percentile Dumbbell (P90 to P95)")
    ax.legend(frameon=False)
    _save(fig, qdir / "plot.png")


def plot_q23() -> None:
    qdir = _qdir(23)
    df = _to_df(23)
    if df.empty:
        return

    df = df.sort_values("summer_spike", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, row in df.iterrows():
        ax.plot([row["winter_o3"], row["summer_o3"]], [i, i], c=PALETTE["muted"], lw=2)
    ax.scatter(df["winter_o3"], y, c=PALETTE["primary"], s=60, label="Winter O3")
    ax.scatter(df["summer_o3"], y, c=PALETTE["accent"], s=60, label="Summer O3")

    for i, row in df.iterrows():
        ax.text(float(row["summer_o3"]) + 0.3, i, f"spike {row['summer_spike']:+.1f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis()
    ax.set_xlabel("O3")
    ax.set_title("Q23: Connected Dot Plot (Winter vs Summer O3)")
    ax.legend(frameon=False)
    _save(fig, qdir / "plot.png")


def plot_q24() -> None:
    qdir = _qdir(24)
    df = _to_df(24)
    if df.empty:
        return

    tr_min = float(df["traffic_ratio"].min())
    tr_max = float(df["traffic_ratio"].max())
    if len(df) > 1 and tr_max > tr_min:
        size = np.interp(df["traffic_ratio"], (tr_min, tr_max), (120, 360))
    else:
        size = np.full(len(df), 220.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    x = _jitter(df["avg_aqi"], 0.35)
    y = _jitter(df["avg_no2"], 0.25)
    sc = ax.scatter(x, y, c=df["traffic_ratio"], s=size, cmap="plasma", alpha=0.58, edgecolors="none")

    for _, row in df.iterrows():
        ax.text(row["avg_aqi"] + 0.7, row["avg_no2"] + 0.5, row["city"], fontsize=7)

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Traffic ratio (NO2/AQI)")
    ax.set_xlabel("Average AQI")
    ax.set_ylabel("Average NO2")
    ax.set_title("Q24: Bubble Scatter (color = traffic ratio)")
    _save(fig, qdir / "plot.png")


def plot_q25() -> None:
    qdir = _qdir(25)
    df = _to_df(25)
    if df.empty:
        return

    df = df.sort_values("rate", ascending=False).reset_index(drop=True)
    df["non_rate"] = 100.0 - df["rate"].astype(float)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(df["city"], df["rate"], color=PALETTE["primary"], label="Co-occurrence %")
    ax.barh(df["city"], df["non_rate"], left=df["rate"], color="#2e3545", label="Non co-occurrence %")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent of winter days")
    ax.set_title("Q25: 100% Stacked Co-occurrence Bars")
    ax.legend(frameon=False)
    _save(fig, qdir / "plot.png")


def _classify_hazard(row: pd.Series) -> str:
    high_o3 = float(row.get("o3", 0)) >= 200.0
    high_co = float(row.get("co", 0)) > 2.5
    if high_o3 and high_co:
        return "Both"
    if high_o3:
        return "High O3"
    return "High CO"


def plot_q26() -> None:
    qdir = _qdir(26)
    data = _load_json(qdir / "data.json")
    df = pd.DataFrame(data.get("hazards", []))
    if df.empty:
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "city"])  # type: ignore[arg-type]
    if df.empty:
        return

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["hazard_type"] = df.apply(_classify_hazard, axis=1)

    monthly = df.groupby(["city", "month"]).size().reset_index(name="count")
    piv = monthly.pivot(index="city", columns="month", values="count").fillna(0)

    type_counts = df.groupby(["city", "hazard_type"]).size().unstack(fill_value=0)
    for col in ["High CO", "High O3", "Both"]:
        if col not in type_counts.columns:
            type_counts[col] = 0
    type_counts = type_counts[["High CO", "High O3", "Both"]]

    fig, axes = plt.subplots(2, 1, figsize=(18, 11), gridspec_kw={"height_ratios": [1.2, 1]})

    sns.heatmap(piv, cmap="rocket", linewidths=0.4, linecolor=PALETTE["grid"], cbar_kws={"label": "Hazard-day count"}, ax=axes[0])
    axes[0].set_title("Q26 Panel A: Hazard Calendar Heatmap (city x month)")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("City")

    axes[1].bar(type_counts.index, type_counts["High CO"], color=PALETTE["primary"], label="High CO")
    axes[1].bar(type_counts.index, type_counts["High O3"], bottom=type_counts["High CO"], color=PALETTE["accent"], label="High O3")
    axes[1].bar(type_counts.index, type_counts["Both"], bottom=type_counts["High CO"] + type_counts["High O3"], color=PALETTE["danger"], label="Both")
    axes[1].set_title("Q26 Panel B: City-level Hazard Type Composition")
    axes[1].set_xlabel("City")
    axes[1].set_ylabel("Days")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(frameon=False)

    fig.suptitle(f"Q26: Hidden Hazard Days (total={int(data.get('total_hazard_days', len(df)))})", y=1.01)
    fig.tight_layout()
    _save(fig, qdir / "plot.png")


def plot_q27() -> None:
    qdir = _qdir(27)
    df = _to_df(27)
    if df.empty:
        return

    df = df.sort_values("avg_recovery", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    avg = df["avg_recovery"].astype(float)
    events = df["events"].astype(float).clip(lower=1.0)
    xerr = np.maximum(0.4, (avg * 0.22) / np.sqrt(events))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.errorbar(avg, y, xerr=xerr, fmt="o", color=PALETTE["primary"], ecolor=PALETTE["muted"], elinewidth=1.6, capsize=3, alpha=0.9)

    for i, row in df.iterrows():
        ax.text(float(row["avg_recovery"]) + 0.7, i, f"events={int(row['events'])}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis()
    ax.set_xlabel("Average recovery days")
    ax.set_title("Q27: Dot Plot with Event-weighted Confidence Cue")
    _save(fig, qdir / "plot.png")


def plot_q28() -> None:
    qdir = _qdir(28)
    df = _to_df(28)
    if df.empty:
        return

    ev_min = float(df["events"].min())
    ev_max = float(df["events"].max())
    if len(df) > 1 and ev_max > ev_min:
        size = np.interp(df["events"], (ev_min, ev_max), (130, 430))
    else:
        size = np.full(len(df), 220.0)

    fig, ax = plt.subplots(figsize=(10, 7))
    x = _jitter(df["avg_warning_days"], 0.06)
    y = _jitter(df["events"], 0.12)
    ax.scatter(x, y, s=size, c=PALETTE["primary"], alpha=0.55, edgecolors="none")

    for _, row in df.iterrows():
        ax.text(row["avg_warning_days"] + 0.03, row["events"] + 0.2, row["city"], fontsize=8)

    ax.set_xlabel("Average warning lead days")
    ax.set_ylabel("Events (reliability)")
    ax.set_title("Q28: Lead-Time vs Reliability Bubble")
    _save(fig, qdir / "plot.png")


def plot_q29() -> None:
    qdir = _qdir(29)
    df = _to_df(29)
    if df.empty:
        return

    df = df.sort_values("seasonal_drop", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(13, 8))
    for i, row in df.iterrows():
        ax.plot([row["summer_avg"], row["winter_avg"]], [i, i], c=PALETTE["muted"], lw=2)

    ax.scatter(df["summer_avg"], y, c=PALETTE["secondary"], s=62, label="Summer")
    ax.scatter(df["winter_avg"], y, c=PALETTE["danger"], s=62, label="Winter")
    for i, row in df.iterrows():
        ax.text(float(row["winter_avg"]) + 0.8, i, f"drop {row['seasonal_drop']:.1f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["city"])
    ax.invert_yaxis()
    ax.set_xlabel("Average AQI")
    ax.set_title("Q29: Seasonal AQI Dumbbell (summer vs winter)")
    ax.legend(frameon=False)
    _save(fig, qdir / "plot.png")


def plot_q30() -> None:
    qdir = _qdir(30)
    df = _to_df(30)
    if df.empty:
        return

    metrics = ["longest_cluster", "total_clusters", "avg_cluster"]
    m = df[metrics].astype(float)
    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-9)

    labels = ["Longest Cluster", "Total Clusters", "Avg Cluster"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, polar=True)

    for i, row in m_norm.iterrows():
        vals = row.values.tolist()
        vals += [vals[0]]
        color = plt.cm.Set2(i % 8)
        ax.plot(angles, vals, lw=2, color=color, label=df.loc[i, "city"])
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Q30: Radar Chart of Burst Metrics (normalized)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=False, fontsize=8)
    _save(fig, qdir / "plot.png")


PLOTTERS: Dict[int, Callable[[], None]] = {
    1: plot_q1,
    2: plot_q2,
    3: plot_q3,
    4: plot_q4,
    5: plot_q5,
    6: plot_q6,
    7: plot_q7,
    8: plot_q8,
    9: plot_q9,
    10: plot_q10,
    11: plot_q11,
    12: plot_q12,
    13: plot_q13,
    14: plot_q14,
    15: plot_q15,
    16: plot_q16,
    17: plot_q17,
    18: plot_q18,
    19: plot_q19,
    20: plot_q20,
    21: plot_q21,
    22: plot_q22,
    23: plot_q23,
    24: plot_q24,
    25: plot_q25,
    26: plot_q26,
    27: plot_q27,
    28: plot_q28,
    29: plot_q29,
    30: plot_q30,
}


def parse_queries(expr: str) -> List[int]:
    expr = expr.strip().lower()
    if expr == "all":
        return list(range(1, 31))

    result: Set[int] = set()
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            for q in range(min(start, end), max(start, end) + 1):
                if 1 <= q <= 30:
                    result.add(q)
        else:
            q = int(part)
            if 1 <= q <= 30:
                result.add(q)
    return sorted(result)


def run_queries(qids: Iterable[int]) -> None:
    for qid in qids:
        func = PLOTTERS.get(qid)
        if func is None:
            continue
        try:
            func()
            print(f"[ok] q{qid}")
        except Exception as exc:
            print(f"[fail] q{qid}: {exc}")


def main() -> None:
    _setup_dark_style()

    parser = argparse.ArgumentParser(description="Visualization-only AQI renderer (Q1-Q30, dark theme)")
    parser.add_argument("--clean", action="store_true", help="Delete existing PNG plots under q1..q30")
    parser.add_argument("--dry-run", action="store_true", help="Preview cleanup count without deleting")
    parser.add_argument("--run", action="store_true", help="Generate plots")
    parser.add_argument("--queries", default="all", help="Query selection: all or list/range, e.g. 1-10,12,14")
    args = parser.parse_args()

    if args.clean:
        n = cleanup_plots(dry_run=args.dry_run)
        mode = "would delete" if args.dry_run else "deleted"
        print(f"[clean] {mode} {n} PNG file(s) under q1..q30")

    should_run = args.run or (not args.clean)
    if should_run:
        selected = parse_queries(args.queries)
        run_queries(selected)


if __name__ == "__main__":
    main()
