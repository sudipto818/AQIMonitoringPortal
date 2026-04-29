# AQI Monitoring Portal & Research Engine

**DBMS Project**

---

## 👥 Team Members

* Badri Bishal Das (240150006) - Backend
* Pratham Saluja (240150045) - Frontend and Data
* Kushagra Singhal (240150046) - Machine Learning
* Sudipto Ghosh (240150042) - SQL Queries and Data Collection
* Sujal Patnaik (240150041) - SQL Queries and Database Management

**Date:** 19 March 2026

---

## 📌 1. Introduction

This project aims to develop a web portal for exploring historical Air Quality Index (AQI) and pollution data across major Indian cities (2021–2026). The platform features:
1. **Interactive Dashboard View:** To visualize trends, analyze daily/monthly pollution levels, and compare air quality across cities instantly.
2. **Research Query Engine:** A built-in query system that pairs 30 complex analytical queries from a MariaDB database alongside pre-computed Machine Learning insights and data forecasts.

---

## 📥 2. Data Collection

### 📊 Dataset Overview
The data for this project has been fetched from the aqi.in website and the OpenAQ website. First the data for all the cities was fetched from aqi.in and then the NULL values were filled using the data from the OpenAQ website. The data fetched for this project has been placed in the csv folder inside the data folder. All the data for individual cities have been placed inside the individual_csv subfolder of the csv folder and the entire final dataset built using the data_handling.py has been saved in the final_merged_aqi_data.csv inside the merged subfolder of the csv folder. These data are already available and if it is needed to fetch the data again, the process is explained below.

### 🔑 Generating the Authorization Token
1. Open your browser and navigate to: https://www.aqi.in
2. Open Developer Tools by pressing F12 (or right-click → Inspect).
3. Go to the Network tab.
4. Refresh the page or click on any city to trigger API requests.
5. In the Network panel, locate a request named: `getAqiCalender`
6. Click on this request and open the Headers section.
7. Under Request Headers, find: `Authorization: bearer <token>`
8. Copy only the token string (exclude the word bearer).

### ▶️ Running the Data Extraction Script

After obtaining the token, navigate to the root directory of the project and execute the following command:

```bash
python data/scripts/fetch_aqi_data.py --city "CityName" --slug "state/city-slug" --token "YOUR_TOKEN"
```

**Example:**
```bash
python data/scripts/fetch_aqi_data.py --city "Hyderabad" --slug "india/telangana/hyderabad" --token "YOUR_TOKEN"
```

### 🛠️ Data Handling with OpenAQ

To fill NULL values in the collected data using the OpenAQ API:

1. Login into the OpenAQ website
2. Generate an API key from there
3. Copy and paste the API key in place of `"YOUR_API_KEY"` in line 7 of `data_handling.py`

### ⚠️ Important Notes
* The authorization token is session-based and may expire. If a 401 Unauthorized error occurs, regenerate the token using the above steps.
* Ensure that the correct city slug is provided; otherwise, the API will return empty or invalid data.
* A delay is intentionally added between API requests to avoid rate limiting.

---

## 🧩 3. System Architecture

### Frontend (React + Vite)
* Built using React.js for extremely fast UI updates.
* Uses Chart.js for data visualizations.
* Unified dual-panel structure bridging SQL and Machine Learning outputs visually.

### Backend (Node.js + Express)
* Provides API handler loops.
* Resolves dynamic SQL parameter injections seamlessly.
* Handles MariaDB `BigInt` responses to safely serialize aggregative statistical models (`COUNT()`, `PERCENTILE_CONT()`).

### Database (MariaDB)
* Relational structure linking Cities and Daily AQI entries natively.
* Incorporates Window Functions, Subqueries, Correlation routines, and clustering via highly optimized query formulations (`aqiQueries.js`).

### Machine Learning Pipeline (Python)
* Orchestrates data using `tensorflow` for LSTMs, `scikit-learn` for Isolation Forests & K-Means Clustering, and `statsmodels` for cyclic seasonal projections.
* Generates static outputs (PNG plots and JSON) stored in the `frontend/public/ml-results` directory entirely offline.

---

## 🌆 4. Cities Covered

Delhi, Mumbai, Kolkata, Chennai, Bengaluru, Hyderabad, Ahmedabad, Pune, Jaipur, Lucknow, Chandigarh, Bhopal, Indore, Noida, Guwahati

---

## 🚀 5. Full System Setup & Execution Guide

Follow these steps exactly in order to launch the full system locally. 

### Step 1: Database Setup (MariaDB)

1. **Install MariaDB** Server.
2. Ensure the CSV files are located inside the `/data` folder.
3. Open your Command Prompt/Terminal and navigate to this project's root folder.
4. Launch the MariaDB Client:
   `"C:\Program Files\MariaDB 12.2\bin\mysql.exe" -u root -p`
5. Enable local infile (necessary for bulk CSV loading):
   `SET GLOBAL local_infile = 1;`
6. Run the Schema layout:
   `SOURCE database/schema.sql;`
7. Run the Seed loader:
   `SOURCE database/seed.sql;`

### Step 2: Configure & Run the Backend
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create an environment file (`.env`) in the `backend/` folder and add your database configuration:
   ```env
   DB_HOST=127.0.0.1
   DB_USER=root
   DB_PASSWORD=admin
   DB_NAME=aqi_db
   PORT=4000
   ```
3. Install dependencies and start the server:
   ```bash
   npm install
   npm start
   ```

### Step 3: Run the Machine Learning Pipeline (Python)
*Note: The results are already stored in `frontend/public/` so this is technically optional unless you want to update models or process new database entries.*
1. Navigate to the ML directory:
   ```bash
   cd ml
   ```
2. Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the Master Pipeline Script:
   ```bash
   python generate_all.py
   ```

### Step 4: Run the Frontend Application
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Boot the development server:
   ```bash
   npm run dev
   ```
4. Access the web dashboard at `http://localhost:5173/`. 

*(Toggle between the traditional Dashboard and the Research Query Engine from the top switch bar).*
