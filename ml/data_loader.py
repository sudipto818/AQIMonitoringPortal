"""
Data loading and preprocessing utilities.
Reads the merged CSV and provides clean DataFrames for all ML models.
"""
import pandas as pd
import numpy as np
from LordSinghalIsAlwaysSingle.ml.config import CSV_PATH


def load_data():
    """Load the main AQI CSV and return a cleaned DataFrame."""
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    # Fill missing pollutant values with 0 (some cities lack certain sensors)
    pollutant_cols = ["co", "no2", "o3", "pm10", "pm25"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["aqi_daily"] = pd.to_numeric(df["aqi_daily"], errors="coerce").fillna(0)
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    return df


def get_city_daily(df, city_name):
    """Return daily data for a single city, sorted by date."""
    return df[df["city"] == city_name].sort_values("date").reset_index(drop=True)


def get_all_cities(df):
    """Return sorted list of unique city names."""
    return sorted(df["city"].unique().tolist())


def get_monthly_aggregation(df):
    """Aggregate daily data to monthly averages per city."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly = df.groupby(["city", "year", "month"]).agg(
        avg_aqi=("aqi_daily", "mean"),
        avg_pm25=("pm25", "mean"),
        avg_pm10=("pm10", "mean"),
        avg_co=("co", "mean"),
        avg_no2=("no2", "mean"),
        avg_o3=("o3", "mean"),
    ).reset_index()
    return monthly


def get_yearly_aggregation(df):
    """Aggregate daily data to yearly averages per city."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    yearly = df.groupby(["city", "year"]).agg(
        avg_aqi=("aqi_daily", "mean"),
        avg_pm25=("pm25", "mean"),
        avg_pm10=("pm10", "mean"),
        avg_co=("co", "mean"),
        avg_no2=("no2", "mean"),
        avg_o3=("o3", "mean"),
        std_aqi=("aqi_daily", "std"),
    ).reset_index()
    return yearly


def get_city_features(df):
    """
    Build a feature matrix for clustering: one row per city with
    aggregate statistics across the full time range.
    """
    features = df.groupby("city").agg(
        avg_aqi=("aqi_daily", "mean"),
        std_aqi=("aqi_daily", "std"),
        avg_pm25=("pm25", "mean"),
        avg_pm10=("pm10", "mean"),
        avg_co=("co", "mean"),
        avg_no2=("no2", "mean"),
        avg_o3=("o3", "mean"),
        max_aqi=("aqi_daily", "max"),
        min_aqi=("aqi_daily", "min"),
    ).reset_index()

    # Add seasonal ratio (winter avg / summer avg)
    df_copy = df.copy()
    df_copy["month"] = df_copy["date"].dt.month
    winter = df_copy[df_copy["month"].isin([11, 12, 1, 2])].groupby("city")["aqi_daily"].mean()
    summer = df_copy[df_copy["month"].isin([4, 5, 6])].groupby("city")["aqi_daily"].mean()
    seasonal = (winter / summer.replace(0, np.nan)).fillna(1).reset_index()
    seasonal.columns = ["city", "seasonal_ratio"]
    features = features.merge(seasonal, on="city", how="left")
    features["seasonal_ratio"] = features["seasonal_ratio"].fillna(1)

    return features
