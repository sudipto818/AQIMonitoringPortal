const { executeQuery } = require("../db/connections");

const GET_DAILY_BY_CITY_RANGE_QUERY = `
  SELECT date, aqi_daily AS avgAqi, pm25 AS avgPm25, pm10 AS avgPm10
  FROM aqi_data
  WHERE city_id = ?
    AND date BETWEEN ? AND ?
  ORDER BY date ASC;
`;

const GET_OVERVIEW_QUERY = `
  SELECT
    c.city_id,
    c.city_name,
    c.state,
    latest.date AS latestDate,
    latest.aqi_daily AS currentAqi,
    latest.pm25 AS currentPm25,
    latest.pm10 AS currentPm10,
    month_stats.avgAqi AS monthAvgAqi,
    month_stats.avgPm25 AS monthAvgPm25,
    month_stats.avgPm10 AS monthAvgPm10
  FROM cities c
  LEFT JOIN (
    SELECT d.city_id, d.date, d.aqi_daily, d.pm25, d.pm10
    FROM aqi_data d
    INNER JOIN (
      SELECT city_id, MAX(date) AS max_date
      FROM aqi_data
      GROUP BY city_id
    ) m
      ON m.city_id = d.city_id
     AND m.max_date = d.date
  ) latest
    ON latest.city_id = c.city_id
  LEFT JOIN (
    SELECT
      d.city_id,
      AVG(d.aqi_daily) AS avgAqi,
      AVG(d.pm25) AS avgPm25,
      AVG(d.pm10) AS avgPm10
    FROM aqi_data d
    INNER JOIN (
      SELECT city_id, YEAR(MAX(date)) AS latest_year, MONTH(MAX(date)) AS latest_month
      FROM aqi_data
      GROUP BY city_id
    ) lm
      ON lm.city_id = d.city_id
     AND YEAR(d.date) = lm.latest_year
     AND MONTH(d.date) = lm.latest_month
    GROUP BY d.city_id
  ) month_stats
    ON month_stats.city_id = c.city_id
  ORDER BY c.city_name;
`;

const GET_HISTORICAL_BY_CITY_MONTH_QUERY = `
  SELECT
    YEAR(date) AS year,
    ROUND(AVG(aqi_daily), 0) AS avgAqi,
    ROUND(AVG(pm25), 0) AS avgPm25,
    ROUND(AVG(pm10), 0) AS avgPm10
  FROM aqi_data
  WHERE city_id = ?
    AND MONTH(date) = ?
  GROUP BY YEAR(date)
  ORDER BY YEAR(date);
`;

// 1. 
// Which city experiences the longest consecutive duration of "Severe" AQI days in a year? 
// ➡ Use: window functions (streak calculation) 
// ➡ Insight: emergency intervention planning
const Q1 = `WITH severe_days AS (
    SELECT 
        city_id,
        date,
        YEAR(date) AS year,
        ROW_NUMBER() OVER (PARTITION BY city_id, YEAR(date) ORDER BY date) AS rn
    FROM aqi_data
    WHERE aqi_daily >= 401
),
grouped AS (
    SELECT 
        city_id,
        year,
        date,
        DATE_SUB(date, INTERVAL rn DAY) AS grp
    FROM severe_days
),
streaks AS (
    SELECT 
        city_id,
        year,
        COUNT(*) AS streak_length
    FROM grouped
    GROUP BY city_id, year, grp
)
SELECT 
    c.city_name,
    year,
    MAX(streak_length) AS longest_severe_streak
FROM streaks s
JOIN cities c ON s.city_id = c.city_id
GROUP BY c.city_name, year
ORDER BY longest_severe_streak DESC
LIMIT 15;`;

// -- Which state has shown the most improvement in average AQI over the last 3 years? 
// -- ➡ Use: yearly aggregation + difference 
// -- ➡ Insight: policy effectiveness

const Q2 = `WITH yearly_avg AS (
    SELECT 
        c.state,
        YEAR(a.date) AS year,
        AVG(a.aqi_daily) AS avg_aqi
    FROM aqi_data a
    JOIN cities c ON a.city_id = c.city_id
    GROUP BY c.state, YEAR(a.date)
),

ranked AS (
    SELECT 
        state,
        year,
        avg_aqi,
        ROW_NUMBER() OVER (PARTITION BY state ORDER BY year DESC) AS rn
    FROM yearly_avg
),

comparison AS (
    SELECT 
        s1.state,
        s1.avg_aqi AS latest_avg,
        s3.avg_aqi AS old_avg,
        (s3.avg_aqi - s1.avg_aqi) AS improvement
    FROM ranked s1
    JOIN ranked s3 
        ON s1.state = s3.state
    WHERE s1.rn = 1   -- latest year
      AND s3.rn = 3   -- 3rd latest year
)

SELECT *
FROM comparison
ORDER BY improvement DESC
LIMIT 7;`;

// -- On which dates did extreme AQI spikes occur in each city? 
// -- ➡ Use: anomaly threshold (mean + stddev or percentile) 
// -- ➡ ML tie-in: anomaly detection
// -- Here we have used the (mean + stddev) method to find spikes in each city
const Q3 = `WITH stats AS (
    SELECT 
        city_id,
        AVG(aqi_daily) AS mean_aqi,
        STDDEV(aqi_daily) AS std_aqi
    FROM aqi_data
    GROUP BY city_id
),

anomalies AS (
    SELECT 
        a.city_id,
        a.date,
        a.aqi_daily,
        s.mean_aqi,
        s.std_aqi,
        (s.mean_aqi + 2 * s.std_aqi) AS threshold
    FROM aqi_data a
    JOIN stats s ON a.city_id = s.city_id
    WHERE a.aqi_daily > (s.mean_aqi + 4 * s.std_aqi)
)

SELECT 
    c.city_name,
    a.date,
    a.aqi_daily,
    ROUND(a.mean_aqi, 2) AS mean_aqi,
    ROUND(a.std_aqi, 2) AS std_dev,
    ROUND(a.threshold, 2) AS threshold
FROM anomalies a
JOIN cities c ON a.city_id = c.city_id
ORDER BY c.city_name, a.date;
`;
// -- Which cities consistently remain in the "Good" AQI category across years? 
// -- ➡ Use: HAVING clause + threshold 
// -- ➡ Insight: benchmark cities 
const Q5 = `WITH yearly_avg AS (
    SELECT 
        c.city_name,
        YEAR(a.date) AS year,
        AVG(a.aqi_daily) AS avg_aqi
    FROM aqi_data a
    JOIN cities c ON a.city_id = c.city_id
    GROUP BY c.city_name, YEAR(a.date)
)

SELECT city_name
FROM yearly_avg
GROUP BY city_name
HAVING MAX(avg_aqi) <=150;
`;
// -- Which cities have the highest number of "Severe" or "Very Poor" AQI days annually? 
// -- ➡ Use: category count grouping 
// -- ➡ Insight: pollution hotspots 

const Q4 = `SELECT *
FROM (
    SELECT 
        c.city_name,
        YEAR(a.date) AS year,
        SUM(CASE WHEN a.aqi_daily BETWEEN 401 AND 500 THEN 1 ELSE 0 END) AS very_poor_days,
        SUM(CASE WHEN a.aqi_daily >= 501 THEN 1 ELSE 0 END) AS severe_days
    FROM aqi_data a
    JOIN cities c ON a.city_id = c.city_id
    GROUP BY c.city_name, YEAR(a.date)
) t
WHERE (very_poor_days + severe_days) > 0
ORDER BY (very_poor_days + severe_days) DESC;`;

// -- What is the monthly average AQI for each city over the last 5 years? 
// -- ➡ Use: GROUP BY month, city 
// -- ➡ Visualization: line chart 

const Q6 = `SELECT 
    c.city_name,
    YEAR(a.date) AS year,
    MONTH(a.date) AS month,
    ROUND(AVG(a.aqi_daily), 2) AS avg_monthly_aqi
FROM aqi_data a
JOIN cities c ON a.city_id = c.city_id
GROUP BY 
    c.city_name,
    YEAR(a.date),
    MONTH(a.date)
ORDER BY 
    c.city_name,
    year,
    month;
`;

// What is the monthly average PM2.5 level for each city? 
// ➡ ML: seasonal pattern detection


const Q7 = `SELECT 
    c.city_name,
    YEAR(a.date) AS year,
    MONTH(a.date) AS month,
    ROUND(AVG(a.pm25), 2) AS avg_pm25
FROM aqi_data a
JOIN cities c ON a.city_id = c.city_id
GROUP BY 
    c.city_name,
    YEAR(a.date),
    MONTH(a.date)
ORDER BY 
    c.city_name,
    year,
    month;
`;

// -- What is the yearly growth or decline rate of AQI in each city? 
// -- ➡ Use: (current - previous)/previous 
// -- ➡ Window function: LAG() 
// -- (current − previous) / previous × 100 : growth/decline rate of AQI in each city:

const Q8 = `WITH yearly_avg AS (
    SELECT 
        c.city_name,
        YEAR(a.date) AS year,
        AVG(a.aqi_daily) AS avg_aqi
    FROM aqi_data a
    JOIN cities c ON a.city_id = c.city_id
    GROUP BY c.city_name, YEAR(a.date)
),

with_lag AS (
    SELECT 
        city_name,
        year,
        avg_aqi,
        LAG(avg_aqi) OVER (PARTITION BY city_name ORDER BY year) AS prev_year_aqi
    FROM yearly_avg
)

SELECT 
    city_name,
    year,
    ROUND(avg_aqi, 2) AS current_avg_aqi,
    ROUND(prev_year_aqi, 2) AS prev_year_aqi,
    ROUND(
        (avg_aqi - prev_year_aqi) / prev_year_aqi * 100,
        2
    ) AS growth_rate_percent
FROM with_lag
WHERE prev_year_aqi IS NOT NULL
ORDER BY city_name, year;
`;

// Which month has the worst air quality (highest avg AQI) each year? 
// ➡ Use: MAX over grouped data 
const Q9 = `WITH monthly_avg AS (
    SELECT 
        YEAR(a.date) AS year,
        MONTH(a.date) AS month,
        AVG(a.aqi_daily) AS avg_aqi
    FROM aqi_data a
    GROUP BY 
        YEAR(a.date),
        MONTH(a.date)
),

ranked AS (
    SELECT 
        year,
        month,
        avg_aqi,
        ROW_NUMBER() OVER (
            PARTITION BY year 
            ORDER BY avg_aqi DESC
        ) AS rn
    FROM monthly_avg
)

SELECT 
    year,
    month,
    ROUND(avg_aqi, 2) AS worst_avg_aqi
FROM ranked
WHERE rn = 1
ORDER BY year;


SELECT 
    c.city_name,
    CASE 
        WHEN MONTH(a.date) IN (12, 1, 2) THEN 'Winter'
        WHEN MONTH(a.date) IN (4, 5, 6) THEN 'Summer'
    END AS season,
    ROUND(AVG(a.pm25), 2) AS avg_pm25
FROM aqi_data a
JOIN cities c ON a.city_id = c.city_id
WHERE c.city_name IN ('Delhi', 'Noida', 'Lucknow', 'Chandigarh', 'Jaipur')
AND MONTH(a.date) IN (12, 1, 2, 4, 5, 6)
GROUP BY 
    c.city_name,
    season
ORDER BY 
    c.city_name,
    season;`;

// How do PM2.5 levels fluctuate between summer and winter months in Northern India? 
// ➡ Use: CASE (season classification)

const Q10 = `SELECT 
    c.city_name,
    CASE 
        WHEN MONTH(a.date) IN (12, 1, 2) THEN 'Winter'
        WHEN MONTH(a.date) IN (4, 5, 6) THEN 'Summer'
    END AS season,
    ROUND(AVG(a.pm25), 2) AS avg_pm25
FROM aqi_data a
JOIN cities c ON a.city_id = c.city_id
WHERE c.city_name IN ('Delhi', 'Noida', 'Lucknow', 'Chandigarh', 'Jaipur')
AND MONTH(a.date) IN (12, 1, 2, 4, 5, 6)
GROUP BY 
    c.city_name,
    season
ORDER BY 
    c.city_name,
    season;
`;

// -- Which cities show the fastest AQI recovery after extreme pollution days? 
// -- ➡ Use: time-to-normal calculation 

const Q27= `
WITH severe_days AS (
    SELECT 
        city_id,
        date AS severe_date
    FROM aqi_data
    WHERE aqi_daily >= 401
),

recovery_days AS (
    SELECT 
        s.city_id,
        s.severe_date,
        MIN(a.date) AS recovery_date
    FROM severe_days s
    JOIN aqi_data a 
        ON s.city_id = a.city_id
        AND a.date > s.severe_date
        AND a.aqi_daily <= 100
    GROUP BY s.city_id, s.severe_date
),

recovery_time AS (
    SELECT 
        city_id,
        DATEDIFF(recovery_date, severe_date) AS recovery_days
    FROM recovery_days
)

SELECT 
    c.city_name,
    ROUND(AVG(recovery_days), 2) AS avg_recovery_time
FROM recovery_time r
JOIN cities c ON r.city_id = c.city_id
GROUP BY c.city_name
ORDER BY avg_recovery_time ASC;
`;

// -- 31. How many days before severe AQI does PM2.5 cross critical levels?
// -- ➡ Early warning signal

const Q28 = `
WITH severe_days AS (
    SELECT city_id, date
    FROM aqi_data
    WHERE aqi_daily >= 401
),
pre_spike AS (
    SELECT 
        s.city_id,
        MIN(a.date) AS warning_date,
        s.date AS severe_date
    FROM severe_days s
    JOIN aqi_data a
        ON s.city_id = a.city_id
        AND a.date < s.date
        AND a.pm25 >= 200
    GROUP BY s.city_id, s.date
)
SELECT 
    c.city_name,
    ROUND(AVG(DATEDIFF(severe_date, warning_date)), 2) AS avg_warning_days
FROM pre_spike p
JOIN cities c ON p.city_id = c.city_id
GROUP BY c.city_name
ORDER BY avg_warning_days DESC;
`;

// -- 33. Which cities have the sharpest winter-to-summer AQI drop?
// -- ➡ Seasonal contrast strength

const Q29 = `SELECT 
    c.city_name,
    ROUND(
        AVG(CASE WHEN MONTH(a.date) IN (12,1,2) THEN a.aqi_daily END) -
        AVG(CASE WHEN MONTH(a.date) IN (4,5,6) THEN a.aqi_daily END),
        2
    ) AS seasonal_drop
FROM aqi_data a
JOIN cities c ON a.city_id = c.city_id
GROUP BY c.city_name
ORDER BY seasonal_drop DESC;`;

// -- 34. Do severe AQI days cluster together (burst analysis)?
// -- ➡ Identify clustered extreme events
// -- This will show the longest cluster per city.

const Q30 = `WITH severe_days AS (
    SELECT 
        city_id,
        date,
        ROW_NUMBER() OVER (PARTITION BY city_id ORDER BY date) AS rn
    FROM aqi_data
    WHERE aqi_daily >= 401
),

grouped AS (
    SELECT 
        city_id,
        DATE_SUB(date, INTERVAL rn DAY) AS grp
    FROM severe_days
),

clusters AS (
    SELECT 
        city_id,
        COUNT(*) AS cluster_size
    FROM grouped
    GROUP BY city_id, grp
)

SELECT 
    c.city_name,
    MAX(cluster_size) AS longest_cluster
FROM clusters cl
JOIN cities c ON cl.city_id = c.city_id
GROUP BY c.city_name
ORDER BY longest_cluster DESC;`;


async function findDailyByCityAndRange(cityId, startDate, endDate) {
  return executeQuery(GET_DAILY_BY_CITY_RANGE_QUERY, [cityId, startDate, endDate]);
}

async function findOverviewStats() {
  return executeQuery(GET_OVERVIEW_QUERY);
}

async function findHistoricalByCityMonth(cityId, month) {
  return executeQuery(GET_HISTORICAL_BY_CITY_MONTH_QUERY, [cityId, month]);
}
// Q1
async function getLongestSevereStreak() {
  return executeQuery(Q1);
}

// Q2
async function getStateImprovement() {
  return executeQuery(Q2);
}

// Q3
async function getAQIAnomalies() {
  return executeQuery(Q3);
}

// Q4
async function getPollutionHotspots() {
  return executeQuery(Q4);
}

// Q5
async function getConsistentlyGoodCities() {
  return executeQuery(Q5);
}

// Q6
async function getMonthlyAQI() {
  return executeQuery(Q6);
}

// Q7
async function getMonthlyPM25() {
  return executeQuery(Q7);
}

// Q8
async function getYearlyGrowth() {
  return executeQuery(Q8);
}

// Q9
async function getWorstMonthPerYear() {
  return executeQuery(Q9);
}

// Q10
async function getSeasonalPM25() {
  return executeQuery(Q10);
}

// Q27
async function getRecoveryTime() {
  return executeQuery(Q27);
}

// Q28
async function getEarlyWarningSignal() {
  return executeQuery(Q28);
}

// Q29
async function getSeasonalDrop() {
  return executeQuery(Q29);
}

// Q30
async function getPollutionClusters() {
  return executeQuery(Q30);
}

/* ---------------- EXPORT ---------------- */

module.exports = {
  findDailyByCityAndRange,
  findOverviewStats,
  findHistoricalByCityMonth,

  getLongestSevereStreak,
  getStateImprovement,
  getAQIAnomalies,
  getPollutionHotspots,
  getConsistentlyGoodCities,
  getMonthlyAQI,
  getMonthlyPM25,
  getYearlyGrowth,
  getWorstMonthPerYear,
  getSeasonalPM25,
  getRecoveryTime,
  getEarlyWarningSignal,
  getSeasonalDrop,
  getPollutionClusters,
};
