"""
City Clustering Pipeline — Groups cities with similar pollution patterns using K-Means.
Aggregates time-series data into unified city profiles before clustering.
Includes dynamic severity labeling and anti-overlap plotting.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================================================================
# 1. HARDCODED PATHS
# ==============================================================================
ML_DIR = r"C:\Users\kusha\OneDrive\Desktop\Course Projects\DBMS Project\LordSinghalIsAlwaysSingle\ml"
CSV_PATH = r"C:\Users\kusha\OneDrive\Desktop\Course Projects\DBMS Project\LordSinghalIsAlwaysSingle\data\csv\merged\final_merged_aqi_data.csv"
OUTPUT_DIR = r"C:\Users\kusha\OneDrive\Desktop\Course Projects\DBMS Project\LordSinghalIsAlwaysSingle\frontend\public\ml-results\clustering"

# Point Python directly to your 'ml' folder to find config.py
sys.path.append(ML_DIR)

# Import from your config file
try:
    from config import (
        KMEANS_MAX_K,
        PLOT_DPI, PLOT_BG_COLOR, PLOT_TEXT_COLOR, PLOT_ACCENT_COLORS,
    )
except ModuleNotFoundError:
    print(f"Error: Could not find config.py in {ML_DIR}.")
    exit(1)


def cluster_cities(city_features_df, output_dir, n_clusters=None):
    """
    Cluster cities based on their aggregate pollution features.
    Saves:
      - clusters_pca.png: 2D PCA scatter with cluster colors
      - elbow_plot.png: elbow method for optimal K
      - clusters.json: city → cluster mapping + centroids
    """
    os.makedirs(output_dir, exist_ok=True)

    # Feature selection: Grab everything except the 'city' column
    feature_cols = [c for c in city_features_df.columns if c != "city"]
    X = city_features_df[feature_cols].values
    cities = city_features_df["city"].values

    # Scale the aggregated data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Elbow Method ──────────────────────────────────────────────────
    max_k = min(KMEANS_MAX_K, len(cities) - 1)
    inertias = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Auto-detect optimal K using "knee" heuristic
    if n_clusters is None:
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        if len(diffs2) > 0:
            n_clusters = int(np.argmax(diffs2) + 3)  # +2 for K offset, +1 for diff offset
            n_clusters = min(max(n_clusters, 3), max_k)
        else:
            n_clusters = 3

    # Elbow plot
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.plot(list(K_range), inertias, "o-", color=PLOT_ACCENT_COLORS[0],
            linewidth=2, markersize=8)
    ax.axvline(x=n_clusters, color=PLOT_ACCENT_COLORS[1], linestyle="--",
               linewidth=2, label=f"Optimal K = {n_clusters}")
    ax.set_title("Elbow Method — Optimal Number of City Clusters",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Clusters (K)", color=PLOT_TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", color=PLOT_TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR, fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("#444")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "elbow_plot.png"), dpi=PLOT_DPI,
                facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    # ── Final K-Means ─────────────────────────────────────────────────
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # ── DYNAMIC SEVERITY LABELING ─────────────────────────────────────
    centroids = km.cluster_centers_
    # feature_cols[0] is 'avg_aqi'. Sorting by this guarantees Cleanest -> Dirtiest
    sorted_cluster_ids = np.argsort(centroids[:, 0]) 
    
    severity_names = [
        "Clean & Stable",
        "Moderate Seasonal",
        "High Pollution",
        "Severe/Volatile",
        "Extreme Chronic",
        "Hazardous (Level 6)",
        "Hazardous (Level 7)",
        "Hazardous (Level 8)"
    ]
    
    cluster_names = {}
    for rank, cid in enumerate(sorted_cluster_ids):
        # Assign names based on sorted rank (0 = lowest AQI, max = highest AQI)
        cluster_names[cid] = severity_names[min(rank, len(severity_names) - 1)]

    # ── PCA → 2D ──────────────────────────────────────────────────────
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    for cl in range(n_clusters):
        mask = labels == cl
        color = PLOT_ACCENT_COLORS[cl % len(PLOT_ACCENT_COLORS)]
        cname = cluster_names.get(cl, f"Cluster {cl}")
        
        # Anti-Clustering Fix: Add slight alpha for overlap
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=color, s=120, alpha=0.85, edgecolors="white",
                   linewidths=1.5, label=cname, zorder=5)

        for i in np.where(mask)[0]:
            # Cleanly space the labels to prevent text blobs
            ax.annotate(
                cities[i], (X_2d[i, 0], X_2d[i, 1]),
                fontsize=8, color=PLOT_TEXT_COLOR, fontweight="bold",
                xytext=(6, 6), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3),
            )

    ax.set_title("City Clustering by Pollution Profile (K-Means + PCA)",
                 color=PLOT_TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)", color=PLOT_TEXT_COLOR, fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)", color=PLOT_TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
    
    # Move legend completely outside the plot area
    ax.legend(facecolor="#2a2a3e", edgecolor="#444", labelcolor=PLOT_TEXT_COLOR,
              fontsize=11, loc="center left", bbox_to_anchor=(1.02, 0.5), 
              title="Pollution Severity", title_fontsize=12)
              
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(alpha=0.15, color="#555")

    # bbox_inches="tight" ensures the outside legend doesn't get cut off
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clusters_pca.png"), dpi=PLOT_DPI,
                facecolor=PLOT_BG_COLOR, bbox_inches="tight")
    plt.close(fig)

    # ── JSON Export ───────────────────────────────────────────────────
    cluster_map = {}
    for i, city in enumerate(cities):
        cl = int(labels[i])
        if cl not in cluster_map:
            cluster_map[cl] = {
                "cluster_id": cl,
                "cluster_name": cluster_names.get(cl, f"Cluster {cl}"),
                "cities": [],
            }
        cluster_map[cl]["cities"].append({
            "city": city,
            "features": {col: round(float(city_features_df.iloc[i][col]), 2) for col in feature_cols},
        })

    result = {
        "model": "K-Means",
        "n_clusters": n_clusters,
        "features_used": feature_cols,
        "pca_explained_variance": [round(float(v), 4) for v in explained],
        "clusters": list(cluster_map.values()),
    }

    with open(os.path.join(output_dir, "clusters.json"), "w") as f:
        json.dump(result, f, indent=2)

    return labels, n_clusters


# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print(f"Loading raw data from: {CSV_PATH}")
    
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find CSV at {CSV_PATH}. Please check the path.")
        exit(1)

    print("Aggregating time-series data into city profiles...")

    # Create the "City Profile" using CORRECT lowercase CSV column names
    city_profiles = df.groupby('city').agg(
        avg_aqi=('aqi_daily', 'mean'),                 
        avg_pm25=('pm25', 'mean'),              
        avg_pm10=('pm10', 'mean'),               
        volatility=('aqi_daily', 'std'),               
        max_spike=('aqi_daily', 'max')                 
    ).reset_index()

    # Safely handle missing values (Using Median to prevent 0-skew)
    feature_cols = ['avg_aqi', 'avg_pm25', 'avg_pm10', 'volatility', 'max_spike']
    
    for col in feature_cols:
        median_val = city_profiles[col].median()
        city_profiles[col] = city_profiles[col].fillna(median_val)

    print("Running K-Means Clustering & PCA...")
    
    # Run the clustering algorithm and save outputs
    labels, n_clusters = cluster_cities(city_profiles, output_dir=OUTPUT_DIR)

    print(f"✅ Success! Clustered {len(city_profiles)} cities into {n_clusters} groups.")
    print(f"Outputs saved to: {OUTPUT_DIR}")