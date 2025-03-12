# com724_assessment

This project performs a comprehensive analysis of 30 cryptocurrencies by downloading historical OHLCV data, computing returns, applying PCA for dimensionality reduction, clustering the coins, and performing correlation analysis. The analysis is implemented in the `data_collection.py` script.

## Requirements

-   Python 3.7 or higher
-   Internet connection (to download data from Yahoo Finance)

The following Python libraries are required:

-   yfinance
-   pandas
-   numpy
-   scikit-learn
-   matplotlib

## Setup Instructions

### 1. Clone or Download the Project

Clone this repository or download the project files to your local machine.

### 2. Create and Activate the Virtual Environment

#### On Ubuntu/Linux/macOS:

Open a terminal in the project directory and run:

```bash
python3 -m venv crypto_venv
source crypto_venv/bin/activate
```

### On Windows:

Open Command Prompt in the project directory and run:

```bash
python -m venv crypto_venv
crypto_venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Script

```bash
python data_collection.py
```

### 5. Viewing the Output
