"""
train_model.py
==============

This module trains gradient boosted tree models on the snapshot feature
matrices produced by ``feature_snapshots.py``.  It builds separate models
for each forecast horizon (5 years and 10 years by default), splits the data
in a time‑aware manner, evaluates on a hold‑out set, and serialises the
trained models to disk.  The script prints basic evaluation metrics (RMSE and
MAE) to help gauge performance.

Usage::

    python src/train_model.py

The feature matrices ``train_matrix_5y.csv`` and ``train_matrix_10y.csv`` must
already exist in ``data/processed``.  They are created by running
``feature_snapshots.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# Paths to feature matrices
FEATURE_MATRIX_PATHS = {
    5: os.path.join("data", "processed", "train_matrix_5y.csv"),
    10: os.path.join("data", "processed", "train_matrix_10y.csv"),
}

# Directory to save trained models
MODEL_DIR = os.path.join("models")


def train_one_horizon(horizon: int) -> Tuple[float, float]:
    """Train a model for a specific horizon and save it.

    Parameters
    ----------
    horizon : int
        Forecast horizon in years (e.g. 5 or 10).

    Returns
    -------
    (float, float)
        Tuple of (RMSE, MAE) on the validation set.
    """
    path = Path(FEATURE_MATRIX_PATHS[horizon])
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix for {horizon}y horizon not found at {path}. Run feature_snapshots.py first."
        )
    df = pd.read_csv(path)
    # Drop rows with any NA values
    df = df.dropna().copy()
    if df.empty:
        raise ValueError(f"No data available to train the {horizon}y model.")

    # Define feature and target columns
    target_col = f"y_cagr_{horizon}y"
    feature_cols = [c for c in df.columns if c not in {"city_norm", "snapshot_year", target_col}]

    # Time-aware split: sort by snapshot_year, then split 70/30
    df = df.sort_values("snapshot_year")
    unique_years = sorted(df["snapshot_year"].unique())
    split_idx = int(len(unique_years) * 0.7)
    train_years = set(unique_years[:split_idx])
    test_years = set(unique_years[split_idx:])
    train_df = df[df["snapshot_year"].isin(train_years)]
    test_df = df[df["snapshot_year"].isin(test_years)]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Define a simple XGBRegressor.  Parameters can be tuned later.
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    print(f"{horizon}y model RMSE: {rmse:.5f}, MAE: {mae:.5f}")

    # Save model
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_{horizon}y.pkl"
    joblib.dump(model, model_path)
    print(f"Saved {horizon}y model to {model_path}")
    return rmse, mae


def main() -> None:
    results: Dict[int, Tuple[float, float]] = {}
    for h in FEATURE_MATRIX_PATHS.keys():
        try:
            results[h] = train_one_horizon(h)
        except Exception as exc:
            print(f"Error training {h}y model: {exc}", file=sys.stderr)
    if results:
        print("Training results summary:")
        for h, (rmse, mae) in results.items():
            print(f"  {h}y horizon — RMSE: {rmse:.5f}, MAE: {mae:.5f}")


if __name__ == "__main__":
    main()