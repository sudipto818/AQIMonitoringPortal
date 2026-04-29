import requests
import csv
import time
import sys
import argparse
from collections import defaultdict

# API configuration
API_URL = "https://apiserver.aqi.in/aqi/v2/getAqiCalender"

HEADERS_TEMPLATE = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.aqi.in",
    "Referer": "https://www.aqi.in/",
}

SENSORS = {
    "co": "co",
    "no2": "no2",
    "o3": "o3",
    "pm10": "pm10",
    "pm25": "pm25",
    "aqi": "aqi_daily",
}

YEARS = list(range(2021, 2027))


def fetch_calendar_data(slug, headers, sensor_name, year):
    params = {
        "slug": slug,
        "slugType": "cityId",
        "sensorname": sensor_name,
        "year": str(year),
        "source": "web",
    }

    try:
        resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data.get("data", [])
            else:
                print(f"API error for {sensor_name}/{year}", file=sys.stderr)
        else:
            print(f"HTTP {resp.status_code} for {sensor_name}/{year}", file=sys.stderr)
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)

    return []


def compute_monthly_aqi(daily_data):
    monthly = defaultdict(list)

    for entry in daily_data:
        if entry["value"] is not None:
            month_key = entry["day"][:7]
            monthly[month_key].append(entry["value"])

    return {
        k: round(sum(v) / len(v), 1)
        for k, v in monthly.items()
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--city", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--token", required=True)

    args = parser.parse_args()

    CITY = args.city
    SLUG = args.slug

    headers = HEADERS_TEMPLATE.copy()
    headers["Authorization"] = f"bearer {args.token}"

    print(f"\nFetching AQI data for {CITY}")
    print("=" * 50)

    all_data = defaultdict(lambda: defaultdict(lambda: None))
    aqi_raw_by_year = {}

    for year in YEARS:
        for sensor_api_name, csv_col_name in SENSORS.items():
            print(f"Fetching {sensor_api_name} ({year})...")

            entries = fetch_calendar_data(SLUG, headers, sensor_api_name, year)

            for entry in entries:
                day = entry["day"]
                all_data[day][csv_col_name] = entry["value"]

            if sensor_api_name == "aqi":
                aqi_raw_by_year[year] = entries

            time.sleep(0.4)

    print("\nComputing monthly AQI...")
    monthly_aqi = {}
    for entries in aqi_raw_by_year.values():
        monthly_aqi.update(compute_monthly_aqi(entries))

    output_file = f"{CITY.lower().replace(' ', '_')}_aqi_data_2021_2026.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "city", "date", "co", "no2", "o3",
            "pm10", "pm25", "aqi_daily", "aqi_monthly"
        ])

        for date in sorted(all_data.keys()):
            row = all_data[date]
            month_key = date[:7]

            writer.writerow([
                CITY,
                date,
                row.get("co", ""),
                row.get("no2", ""),
                row.get("o3", ""),
                row.get("pm10", ""),
                row.get("pm25", ""),
                row.get("aqi_daily", ""),
                monthly_aqi.get(month_key, "")
            ])

    print(f"\n✅ Done! File saved: {output_file}")


if __name__ == "__main__":
    main()
