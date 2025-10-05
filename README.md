# TX Housing Appreciation Backend

This repository contains an end‑to‑end data pipeline and API for predicting long‑term housing appreciation in Texas cities.  The goal of the project is to help investors identify metropolitan communities that are positioned for strong home value growth based on historical price data.  Education metrics were originally planned as a feature but are optional.  The current version of the pipeline trains on price history alone, making it easier to run without external school performance datasets.

The pipeline is broken into the following stages (education‑free version):

1. **Filter Zillow City Data** (`src/filter_zillow_tx_metros.py`)
   - Reads the raw Zillow *City* ZHVI dataset and filters it down to Texas cities that belong to a metropolitan statistical area (CBSA).  Cities without sufficient history (fewer than 60 months of data) or tiny metros (fewer than three cities) are excluded.  The result is written to `data/processed/zhvi_tx_metros.csv`.

2. **Prepare Price Panel** (`src/join_zillow_education.py`)
   - Converts the wide‑format ZHVI file into a long table of observations (`city`, `date`, `zhvi`) and adds a normalised city name.  The resulting panel is written to `data/processed/panel_tx_city_year.csv`.  This step no longer merges education data.

3. **Create Snapshot Features and Labels** (`src/feature_snapshots.py`)
   - Builds training matrices for specified forecast horizons (5 and 10 years by default).  For each city and each snapshot year *T*, the script computes features using only information available up to and including *T* (price level and three‑year price momentum) and computes the forward compound annual growth rate (CAGR) between *T* and *T + h*.  Two matrices are produced: `data/processed/train_matrix_5y.csv` and `data/processed/train_matrix_10y.csv`.

6. **Train Models** (`src/train_model.py`)
   - Trains gradient boosted tree models (XGBoostRegressor) on the snapshot matrices.  A time‑aware split is used so that early snapshot years are used for training and later years are reserved for validation.  Trained models are saved to `models/model_5y.pkl` and `models/model_10y.pkl`.

7. **Serve Predictions via FastAPI** (`app/main.py`)
   - Implements a simple REST API with endpoints to check service health and to query predictions.  The `/predict` endpoint accepts a city name, snapshot year, and forecast horizon and returns the predicted CAGR along with the features used for the prediction.  The API automatically normalizes city names to match the training data and returns appropriate errors when requested data is not available.

## Project layout

```
re-invest/
│
├── data/
│   ├── raw/             # raw CSVs (Zillow, TEA STAAR exports)
│   └── processed/       # cleaned and merged tables used for modelling
│
├── models/              # trained model artefacts (.pkl)
│
├── src/                 # data ingestion, cleaning, feature creation and training scripts
│   ├── filter_zillow_tx_metros.py
│   ├── fetch_tea_staar.py           # optional; unused for price‑only training
│   ├── build_city_year_education.py # optional; unused for price‑only training
│   ├── join_zillow_education.py     # used to create price panel (no education join)
│   ├── feature_snapshots.py
│   └── train_model.py
│
├── app/
│   └── main.py          # FastAPI server implementation
│
└── README.md            # this file
```

## Running the pipeline

The entire pipeline can be executed from the command line.  You should activate a virtual environment and install the dependencies listed in `requirements.txt` first.  For example:

```bash
# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# install requirements
pip install -r re-invest/requirements.txt

# move into the project directory
cd re-invest

# 1. filter the Zillow data down to Texas metros
python src/filter_zillow_tx_metros.py

# 2. create a long‑format panel from the Zillow data (no education join)
python src/join_zillow_education.py

# 3. build snapshot features and labels
python src/feature_snapshots.py

# 4. train the models
python src/train_model.py

# 5. run the API server
uvicorn app.main:app --reload
```

Follow the comments within each script for details on required input files and processing assumptions.  Several steps (notably TEA data download) require manual intervention or further enhancement to automate fully.  The default settings target Texas and 5‑/10‑year horizons but can be adjusted in the scripts.