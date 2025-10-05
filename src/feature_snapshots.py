"""
feature_snapshots.py
====================

This module builds snapshot feature matrices for a given forecast horizon.
For each city and snapshot year, it computes a set of features derived solely
from historical price data and a target label equal to the future
compound annual growth rate (CAGR) over ``horizon`` years.  The key point is
that only information available up to the snapshot year is used to form the
features—any observations beyond the snapshot year are excluded to avoid
forward‑looking bias (data leakage).

Two matrices are produced by default: one for a 5‑year horizon and another
for a 10‑year horizon.  They are written to ``data/processed/train_matrix_5y.csv``
and ``train_matrix_10y.csv`` respectively.

Usage::

    python src/feature_snapshots.py

Prerequisites:
  - ``join_zillow_education.py`` must have been run to produce
    ``data/processed/panel_tx_city_year.csv``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Input path for the merged Zillow/Education panel
PANEL_PATH: str = os.path.join("data", "processed", "panel_tx_city_year.csv")

# Horizons (in years) to generate snapshot matrices for
HORIZONS: List[int] = [5, 10]


def compute_cagr(start: float, end: float, years: int) -> float:
    """Compute compound annual growth rate.

    Parameters
    ----------
    start : float
        Starting value.
    end : float
        Ending value.
    years : int
        Number of years between start and end.

    Returns
    -------
    float
        CAGR, computed as ``(end / start)**(1/years) - 1``.  Returns NaN if
        inputs are non‑positive or if ``years`` is zero.
    """
    if years <= 0 or start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1


def build_snapshot_matrix(horizon: int) -> pd.DataFrame:
    """Build a feature matrix for a given forecast horizon.

    Parameters
    ----------
    horizon : int
        The number of years ahead to predict.  For example, ``5`` produces
        5‑year forward CAGRs.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        - ``city_norm``: normalized city name
        - ``snapshot_year``: year at which the prediction is made
        - ``price_level``: ZHVI in the snapshot year
        - ``price_momentum_3y``: 3‑year CAGR of ZHVI up to the snapshot year
        - ``y_cagr_{horizon}y``: target CAGR over the next ``horizon`` years
    """
    panel_file = Path(PANEL_PATH)
    if not panel_file.exists():
        raise FileNotFoundError(
            f"Panel file '{PANEL_PATH}' not found. Run join_zillow_education.py first."
        )
    df = pd.read_csv(panel_file)
    # Ensure correct dtypes
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["zhvi"] = pd.to_numeric(df["zhvi"], errors="coerce")
    # Without education data we only require zhvi to be present
    df = df.dropna(subset=["zhvi"])

    # Sort for reproducibility
    df = df.sort_values(["city_norm", "year"])

    # Group by city for sequential access
    snapshot_records: List[Dict[str, float]] = []
    for city, group in df.groupby("city_norm"):
        group = group.reset_index(drop=True)
        # Build a lookup for year -> row index
        year_to_idx = {int(row["year"]): idx for idx, row in group.iterrows()}
        years_sorted = sorted(year_to_idx.keys())
        for year in years_sorted:
            snapshot_year = year
            future_year = snapshot_year + horizon
            # Require current and future observations to exist
            if snapshot_year not in year_to_idx or future_year not in year_to_idx:
                continue
            current_idx = year_to_idx[snapshot_year]
            future_idx = year_to_idx[future_year]
            current_row = group.loc[current_idx]
            future_row = group.loc[future_idx]
            # Feature: price level at snapshot
            price_level = float(current_row["zhvi"])
            # Feature: 3‑year momentum (CAGR over last 3 years).  Only compute if we have data 3 years prior.
            momentum_year = snapshot_year - 3
            if momentum_year in year_to_idx:
                past_idx = year_to_idx[momentum_year]
                past_row = group.loc[past_idx]
                price_start = float(past_row["zhvi"])
                price_momentum_3y = compute_cagr(price_start, price_level, 3)
            else:
                price_momentum_3y = np.nan
            # Target: future CAGR
            price_end = float(future_row["zhvi"])
            y_cagr = compute_cagr(price_level, price_end, horizon)
            snapshot_records.append(
                {
                    "city_norm": city,
                    "snapshot_year": snapshot_year,
                    "price_level": price_level,
                    "price_momentum_3y": price_momentum_3y,
                    f"y_cagr_{horizon}y": y_cagr,
                }
            )
    return pd.DataFrame(snapshot_records)


def main() -> None:
    for h in HORIZONS:
        try:
            matrix = build_snapshot_matrix(h)
        except Exception as exc:
            print(f"Error building snapshot matrix for horizon {h}: {exc}", file=sys.stderr)
            continue
        out_file = Path(os.path.join("data", "processed", f"train_matrix_{h}y.csv"))
        out_file.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(out_file, index=False)
        print(f"Snapshot feature matrix for {h}y horizon saved to {out_file} ({len(matrix)} rows)")


if __name__ == "__main__":
    main()