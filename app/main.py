"""
FastAPI application for serving housing appreciation predictions.

This module defines a small REST API that exposes two endpoints:

- ``GET /health``: returns a JSON object indicating that the service is alive.
- ``GET /predict``: given a city name, snapshot year, and forecast horizon (5 or 10),
  returns a predicted compound annual growth rate (CAGR) along with the
  features used to compute the prediction.

The API loads precomputed feature matrices and trained models at startup.  In a
production deployment you might prefer to query a database or recompute
features on the fly, but for the hackathon prototype the features are stored
in CSV files created by previous pipeline steps.

To run the API locally::

    uvicorn app.main:app --reload

Once running, test the health endpoint at http://localhost:8000/health and
perform predictions by visiting e.g.::

    http://localhost:8000/predict?city=Austin&year=2015&h=5

Note: If a requested city/year combination is not found in the feature matrix
or if a model is missing for the requested horizon, the API returns an error
response with appropriate status codes.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Ensure the project root is on sys.path so absolute imports work when run via uvicorn
import os as _os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent  # points to the project root (re-invest)
_sys.path.append(str(_ROOT))

# Try to import normalisation from the education module if present; otherwise define fallback
try:
    from src.build_city_year_education import normalize_city  # type: ignore[attr-defined]
except Exception:
    import re

    def normalize_city(name: str) -> str:
        """Fallback normalisation for city names used by the API.

        Converts to upper case, removes punctuation and extra whitespace, and
        preserves alphanumeric characters.
        """
        if not isinstance(name, str) or not name:
            return ""
        s = name.upper()
        s = re.sub(r"[^A-Z0-9 ]", "", s)
        s = " ".join(s.split())
        return s


app = FastAPI(title="Texas Housing Appreciation API")

# Mount static files to serve a simple front‑end.  The files live in ``app/static``.
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Redirect the root URL to the simple front‑end.

    Visiting ``/`` will redirect the browser to ``/static/index.html`` if the file exists.
    Otherwise a small placeholder page is returned.
    """
    index_path = static_dir / "index.html"
    if index_path.exists():
        return RedirectResponse(url="/static/index.html")
    return HTMLResponse("<html><body><h1>Texas Housing Appreciation API</h1><p>The front‑end is not available.</p></body></html>")

# Paths to feature matrices and models
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

FEATURE_MATRICES = {
    5: DATA_DIR / "train_matrix_5y.csv",
    10: DATA_DIR / "train_matrix_10y.csv",
}

MODEL_PATHS = {
    5: MODEL_DIR / "model_5y.pkl",
    10: MODEL_DIR / "model_10y.pkl",
}


@lru_cache(maxsize=None)
def load_feature_matrix(horizon: int) -> pd.DataFrame:
    """Load the feature matrix for a given horizon from disk.

    Parameters
    ----------
    horizon : int
        Forecast horizon in years (5 or 10).

    Returns
    -------
    pandas.DataFrame
        DataFrame loaded from the appropriate CSV.
    """
    path = FEATURE_MATRICES[horizon]
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix for {horizon}y horizon not found at {path}. Ensure the pipeline has been run."
        )
    df = pd.read_csv(path)
    return df


@lru_cache(maxsize=None)
def load_model(horizon: int):
    """Load a trained model for a given horizon."""
    path = MODEL_PATHS[horizon]
    if not path.exists():
        raise FileNotFoundError(
            f"Model for {horizon}y horizon not found at {path}. Train the model first."
        )
    return joblib.load(path)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/predict")
async def predict(
    city: str = Query(..., description="City name (case insensitive)"),
    year: int = Query(..., description="Snapshot year (e.g., 2015)"),
    h: int = Query(5, description="Forecast horizon in years: 5 or 10"),
) -> Dict[str, object]:
    """Predict future housing appreciation for a city and snapshot year.

    Parameters
    ----------
    city : str
        Name of the city.  The API normalises names internally.
    year : int
        Snapshot year for which the prediction is made.  Must be present in
        the feature matrix (i.e., 2000 <= year <= the most recent snapshot year).
    h : int, optional
        Forecast horizon in years (5 or 10).  Defaults to 5.

    Returns
    -------
    dict
        JSON object containing the normalised city, snapshot year, horizon,
        predicted CAGR, and the features used.
    """
    if h not in FEATURE_MATRICES:
        raise HTTPException(status_code=400, detail="Invalid horizon. Choose 5 or 10.")
    try:
        features_df = load_feature_matrix(h)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        model = load_model(h)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    city_norm = normalize_city(city)
    # Filter the features for the requested city and year
    mask = (features_df["city_norm"] == city_norm) & (features_df["snapshot_year"] == year)
    if not mask.any():
        raise HTTPException(status_code=404, detail="No features found for the specified city and year.")
    row = features_df.loc[mask].iloc[0]
    # Extract feature columns
    target_col = f"y_cagr_{h}y"
    feature_cols = [c for c in features_df.columns if c not in {"city_norm", "snapshot_year", target_col}]
    feature_values = row[feature_cols]
    X = feature_values.to_frame().T
    # Predict
    pred = float(model.predict(X)[0])
    # Construct response
    response = {
        "city": city_norm,
        "year": year,
        "horizon_years": h,
        "predicted_cagr": pred,
        "used_features": {col: float(row[col]) for col in feature_cols},
    }
    return response

@app.get("/rank")
async def rank(
    year: int = Query(..., description="Snapshot year to rank"),
    h: int = Query(5, description="Forecast horizon in years: 5 or 10"),
    top: int = Query(50, description="Number of cities to return in descending order"),
) -> list[Dict[str, object]]:
    """Return the top N cities ranked by predicted CAGR for a given snapshot year.

    Parameters
    ----------
    year : int
        Snapshot year to evaluate.
    h : int, optional
        Forecast horizon in years (5 or 10).  Defaults to 5.
    top : int, optional
        Maximum number of cities to return.  Defaults to 50.

    Returns
    -------
    list of dict
        Each entry contains the normalized city name, snapshot year, horizon, and predicted CAGR.
    """
    if h not in FEATURE_MATRICES:
        raise HTTPException(status_code=400, detail="Invalid horizon. Choose 5 or 10.")
    # Load features and model
    try:
        features_df = load_feature_matrix(h)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        model = load_model(h)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    # Filter by year
    df_year = features_df[features_df["snapshot_year"] == year].copy()
    if df_year.empty:
        raise HTTPException(status_code=404, detail=f"No data found for snapshot_year={year}.")
    # Prepare feature matrix
    target_col = f"y_cagr_{h}y"
    feature_cols = [c for c in df_year.columns if c not in {"city_norm", "snapshot_year", target_col}]
    X = df_year[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # Predict
    df_year["predicted_cagr"] = model.predict(X)
    # Sort and slice
    df_year = df_year.sort_values("predicted_cagr", ascending=False).head(top)
    # Build result list with additional feature information
    results: list[Dict[str, object]] = []
    for _, row in df_year.iterrows():
        # Extract numeric features used for the model.  Coerce to float to ensure
        # they can be serialised to JSON.  Exclude the target column.
        feature_dict: Dict[str, float] = {}
        for col in feature_cols:
            try:
                feature_dict[col] = float(row[col])
            except Exception:
                # If the value cannot be converted (e.g. None/NaN), skip it
                continue
        results.append({
            "city": row["city_norm"],
            "year": int(row["snapshot_year"]),
            "horizon_years": h,
            "predicted_cagr": float(row["predicted_cagr"]),
            "features": feature_dict,
        })
    return results

@app.post("/predict_batch")
async def predict_batch(
    payload: Dict[str, object],
) -> list[Dict[str, object]]:
    """Predict for multiple cities at once.

    Expects a JSON body with keys ``cities`` (list of city names), ``year`` and ``h``.

    Returns a list of prediction dictionaries.  If a city/year combination is
    missing, the entry will include an ``error`` message.
    """
    cities = payload.get("cities", [])
    year = payload.get("year")
    h = payload.get("h", 5)
    if not isinstance(cities, list) or not isinstance(year, int) or not isinstance(h, int):
        raise HTTPException(status_code=400, detail="Invalid payload. Expect cities (list), year (int) and h (int).")
    if h not in FEATURE_MATRICES:
        raise HTTPException(status_code=400, detail="Invalid horizon. Choose 5 or 10.")
    # Load features and model
    try:
        features_df = load_feature_matrix(h)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        model = load_model(h)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    target_col = f"y_cagr_{h}y"
    feature_cols = [c for c in features_df.columns if c not in {"city_norm", "snapshot_year", target_col}]
    results = []
    for city in cities:
        city_norm = normalize_city(city)
        mask = (features_df["city_norm"] == city_norm) & (features_df["snapshot_year"] == year)
        if not mask.any():
            results.append({
                "city": city_norm,
                "year": year,
                "horizon_years": h,
                "error": "No features found for the specified city and year",
            })
            continue
        row = features_df.loc[mask].iloc[0]
        X_row = row[feature_cols].to_frame().T
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        pred_val = float(model.predict(X_row)[0])
        results.append({
            "city": city_norm,
            "year": year,
            "horizon_years": h,
            "predicted_cagr": pred_val,
        })
    return results