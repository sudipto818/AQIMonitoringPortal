CREATE DATABASE IF NOT EXISTS aqi_db;
USE aqi_db;

CREATE TABLE cities (
    city_id INT AUTO_INCREMENT PRIMARY KEY,
    city_name VARCHAR(50) UNIQUE,
    state VARCHAR(50),
    region VARCHAR(50)
);

CREATE TABLE aqi_data (
    city_id INT,
    date DATE,
    co FLOAT,
    no2 FLOAT,
    o3 FLOAT,
    pm10 FLOAT,
    pm25 FLOAT,
    aqi_daily FLOAT,
    PRIMARY KEY (city_id, date),
    CONSTRAINT fk_city FOREIGN KEY (city_id) REFERENCES cities(city_id)
);

CREATE INDEX idx_date ON aqi_data(date);
CREATE INDEX idx_city_aqi ON aqi_data(city_id, aqi_daily);
