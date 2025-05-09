Thought for a couple of seconds

```markdown
# com724_assessment

This project performs a comprehensive analysis of 30 cryptocurrencies by downloading historical OHLCV data, computing returns, applying PCA
for dimensionality reduction, clustering the coins, and performing correlation analysis. It exposes a Flask‐based REST &
WebSocket API for forecasts, signals, EDA and an interactive React dashboard to visualize real‐time data, analysis and forecasts.

---

## 📂 Repository Structure
```

com724_assessment/

├── app.py # Flask + SocketIO backend

├── config.yaml # Tickers, periods, file paths

├── crypto_data.csv # Cached OHLCV data

├── data_collection.py # ETL, PCA, clustering, EDA pipelines

├── forecasting.py # ARIMA & Prophet wrappers

├── signals.py # Trading‐signal generator & backtester

├── requirements.txt # Python dependencies

├── README.md # This file

├── dashboard/

│ └── crypto_dashboard/ # React front-end

│ ├── package.json

│ ├── src/

│ │ ├── App.jsx

│ │ ├── components/

│ │ ├── store/

│ │ └── api/

│ └── public/

└── crypto_venv/ # (optional) Python virtualenv

````

---

## 🔧 Prerequisites

- **Python** ≥ 3.7
- **Node.js & npm** (for the React dashboard)
- Internet connection (to fetch data from Yahoo Finance)

---

## 🐍 Backend Setup (Flask + SocketIO)

1. **Activate your Python virtual environment**
   ```bash
   cd com724_assessment
   python -m venv crypto_venv
   # Windows
   crypto_venv\Scripts\activate
   # Linux/Mac
   source crypto_venv/bin/activate
````

2. **Install Python dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure**

    - Edit `config.yaml` to adjust tickers, period, cache file, etc.

4. **Run the Flask server**

    ```bash
    python app.py
    ```

    By default, the API will listen on `http://0.0.0.0:5000`.

    Endpoints include:

    - `GET /api/tickers`
    - `GET /api/forecast/<TICKER>?model=prophet&horizon=30`
    - `GET /api/signals/<TICKER>?threshold=0.01…`
    - `GET /api/eda/<TICKER>`
    - WebSocket channel for `/api/pipeline` analysis

---

## ⚛️ Frontend Setup (React Dashboard)

1. **Install Node dependencies**

    ```bash
    cd dashboard/crypto_dashboard
    npm install
    ```

2. **Configure API base URL**

    - In `.env` (or `VITE_API_BASE_URL`), point to your Flask server, e.g.:
        ```ini
        VITE_API_BASE_URL=http://localhost:5000
        ```

3. **Run the React app**

    ```bash
    npm run dev
    ```

    By default, the dashboard will be available at `http://localhost:5173`.

---

## 🚀 Usage

1. Start the **backend** :
    ```bash
    cd com724_assessment
    crypto_venv\Scripts\activate  # or source venv
    python app.py
    ```
2. Start the **frontend** (in a separate terminal):
    ```bash
    cd com724_assessment/dashboard/crypto_dashboard
    npm install      # if you haven't already
    npm run dev
    ```
3. Open your browser at `http://localhost:5173`
    - **Real-time Chart** tab: live WebSocket-fed candles
    - **Crypto Analysis** tab: runs the clustering/EDA pipeline via SocketIO
    - **Forecast** tab: Prophet/ARIMA forecasts, target-price & signal performance

---

## 🗒️ Notes

-   Cached data is stored in `crypto_data.csv`. Delete it to force a full re-download.
-   Logs are written to `app.log` (rotating, 10 MB max).
-   To extend the prediction horizon beyond 365 days, adjust `max_horizon` in `config.yaml` and re-start.

Happy analyzing! 🚀
