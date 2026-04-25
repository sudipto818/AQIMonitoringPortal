"""
Health Risk Predictor — Rule-based system mapping AQI levels to health impacts.

Approach: NAQI (India) + WHO Guidelines
    Health impacts of air pollution are well-established by regulatory
    bodies. Using deterministic rules is MORE TRUSTWORTHY than ML for
    health advisories because:
    1. AQI breakpoints are scientifically defined (not learned)
    2. No risk of model drift / false negatives
    3. Fully interpretable and auditable

    We compute a composite "Health Risk Score" per city using:
    score = Σ(days_in_category × category_weight) / total_days

    This gives a normalized 0–10 score that captures both INTENSITY
    and DURATION of exposure.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

from LordSinghalIsAlwaysSingle.ml.config import AQI_CATEGORIES, PLOT_DPI, PLOT_BG_COLOR, PLOT_TEXT_COLOR, PLOT_ACCENT_COLORS


def _categorize_aqi(aqi):
    """Return (category_name, color, weight) for a given AQI value."""
    for low, high, name, color, weight in AQI_CATEGORIES:
        if low <= aqi <= high:
            return name, color, weight
    if aqi > 500:
        return "Severe", "#8e44ad", 10
    return "Good", "#55a868", 0


def compute_health_risk(df_city, city_name, output_dir):
    """
    Compute health risk score for a city and save:
      - gauge.png: health risk gauge/indicator
      - data.json: category breakdown + risk score + advisory
    """
    os.makedirs(output_dir, exist_ok=True)

    aqi_values = df_city["aqi_daily"].dropna().values

    if len(aqi_values) == 0:
        with open(os.path.join(output_dir, "data.json"), "w") as f:
            json.dump({"city": city_name, "error": "No data"}, f, indent=2)
        return

    # ── Category counts ───────────────────────────────────────────────
    category_counts = {}
    total_weight = 0
    for aqi in aqi_values:
        name, color, weight = _categorize_aqi(aqi)
        if name not in category_counts:
            category_counts[name] = {"count": 0, "color": color, "weight": weight}
        category_counts[name]["count"] += 1
        total_weight += weight

    total_days = len(aqi_values)
    health_score = round(total_weight / total_days, 2) if total_days > 0 else 0

    # Risk tier
    if health_score <= 1:
        risk_tier = "Low Risk"
        tier_color = "#55a868"
        advisory = "Air quality is generally safe. No special precautions needed."
    elif health_score <= 3:
        risk_tier = "Moderate Risk"
        tier_color = "#f5d76e"
        advisory = "Sensitive groups (children, elderly, asthma) should limit prolonged outdoor activity."
    elif health_score <= 5:
        risk_tier = "High Risk"
        tier_color = "#f39c12"
        advisory = "Everyone should reduce outdoor exertion. Sensitive groups should stay indoors."
    elif health_score <= 7:
        risk_tier = "Very High Risk"
        tier_color = "#e74c3c"
        advisory = "Health warnings: serious respiratory effects for everyone. Avoid outdoor activity."
    else:
        risk_tier = "Severe Risk"
        tier_color = "#8e44ad"
        advisory = "EMERGENCY: Health alert for entire population. Stay indoors, use air purifiers."

    # ── Gauge Plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PLOT_BG_COLOR,
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # LEFT: Gauge
    ax = axes[0]
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw gauge arc segments
    gauge_colors = ["#55a868", "#c4e17f", "#f5d76e", "#f39c12", "#e74c3c", "#8e44ad"]
    n_segments = len(gauge_colors)
    for i, gc in enumerate(gauge_colors):
        theta1 = 180 - i * (180 / n_segments)
        theta2 = 180 - (i + 1) * (180 / n_segments)
        arc = plt.matplotlib.patches.Wedge(
            (0, 0), 1.2, theta2, theta1, width=0.3,
            facecolor=gc, edgecolor=PLOT_BG_COLOR, linewidth=2
        )
        ax.add_patch(arc)

    # Needle
    angle = 180 - (health_score / 10) * 180
    angle_rad = np.radians(angle)
    needle_x = 0.85 * np.cos(angle_rad)
    needle_y = 0.85 * np.sin(angle_rad)
    ax.annotate(
        "", xy=(needle_x, needle_y), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="white", lw=3),
    )
    ax.plot(0, 0, "o", color="white", markersize=10, zorder=10)

    ax.text(0, -0.3, f"{health_score}/10",
            ha="center", va="center", fontsize=28, fontweight="bold",
            color=tier_color,
            path_effects=[pe.withStroke(linewidth=2, foreground=PLOT_BG_COLOR)])
    ax.text(0, -0.5, risk_tier,
            ha="center", va="center", fontsize=14, color=tier_color, fontweight="bold")

    ax.set_title(f"{city_name}\nHealth Risk Score",
                 color=PLOT_TEXT_COLOR, fontsize=13, fontweight="bold", pad=10)

    # RIGHT: Category breakdown bar
    ax2 = axes[1]
    ax2.set_facecolor(PLOT_BG_COLOR)

    ordered_cats = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    cat_counts = []
    cat_colors = []
    for cat in ordered_cats:
        if cat in category_counts:
            cat_counts.append(category_counts[cat]["count"])
            cat_colors.append(category_counts[cat]["color"])
        else:
            cat_counts.append(0)
            cat_colors.append("#333")

    bars = ax2.barh(ordered_cats[::-1], [c / total_days * 100 for c in cat_counts[::-1]],
                    color=cat_colors[::-1], edgecolor="#333", height=0.6)

    for bar, count in zip(bars, cat_counts[::-1]):
        if count > 0:
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{count} days ({count/total_days*100:.1f}%)",
                     va="center", color=PLOT_TEXT_COLOR, fontsize=9)

    ax2.set_title("AQI Category Distribution",
                  color=PLOT_TEXT_COLOR, fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("% of Total Days", color=PLOT_TEXT_COLOR, fontsize=10)
    ax2.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color("#444")
    ax2.grid(axis="x", alpha=0.2, color="#555")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot.png"), dpi=PLOT_DPI,
                facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    # ── JSON ──────────────────────────────────────────────────────────
    result = {
        "city": city_name,
        "model": "Rule-Based (NAQI India)",
        "health_score": health_score,
        "max_possible_score": 10,
        "risk_tier": risk_tier,
        "advisory": advisory,
        "total_days_analyzed": total_days,
        "category_breakdown": {
            cat: {
                "days": category_counts[cat]["count"],
                "percentage": round(category_counts[cat]["count"] / total_days * 100, 1),
            }
            for cat in ordered_cats if cat in category_counts
        },
    }

    with open(os.path.join(output_dir, "data.json"), "w") as f:
        json.dump(result, f, indent=2)


def compute_health_risk_all_cities(df, output_dir):
    """Run health risk assessment for all cities and produce a summary."""
    os.makedirs(output_dir, exist_ok=True)
    cities = sorted(df["city"].unique())
    summary = []

    for city in cities:
        df_city = df[df["city"] == city].reset_index(drop=True)
        city_dir = os.path.join(output_dir, city.replace(" ", "_").lower())
        compute_health_risk(df_city, city, city_dir)

        json_path = os.path.join(city_dir, "data.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
                if "health_score" in data:
                    summary.append({
                        "city": city,
                        "health_score": data["health_score"],
                        "risk_tier": data["risk_tier"],
                    })

    # Summary bar chart
    if summary:
        summary.sort(key=lambda x: x["health_score"], reverse=True)

        tier_colors = {
            "Low Risk": "#55a868", "Moderate Risk": "#f5d76e",
            "High Risk": "#f39c12", "Very High Risk": "#e74c3c",
            "Severe Risk": "#8e44ad",
        }

        fig, ax = plt.subplots(figsize=(14, 8), facecolor=PLOT_BG_COLOR)
        ax.set_facecolor(PLOT_BG_COLOR)

        names = [s["city"] for s in summary]
        scores = [s["health_score"] for s in summary]
        colors = [tier_colors.get(s["risk_tier"], "#888") for s in summary]

        bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1],
                       edgecolor="#333", height=0.6)

        for bar, s in zip(bars, summary[::-1]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{s['health_score']} — {s['risk_tier']}",
                    va="center", color=PLOT_TEXT_COLOR, fontsize=8)

        ax.set_title("Health Risk Score by City (NAQI India Standards)",
                     color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Health Risk Score (0 = Safe, 10 = Emergency)",
                      color=PLOT_TEXT_COLOR, fontsize=11)
        ax.set_xlim(0, 12)
        ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.grid(axis="x", alpha=0.2, color="#555")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "summary.png"), dpi=PLOT_DPI,
                    facecolor=PLOT_BG_COLOR, bbox_inches="tight")
        plt.close(fig)

        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump({"cities": summary}, f, indent=2)
