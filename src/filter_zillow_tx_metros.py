"""
filter_zillow_tx_metros.py
===========================

This script filters the Zillow *City* ZHVI dataset to retain only Texas cities located
within a recognised metropolitan area (CBSA) and with sufficient historical price
data.  Many rural towns have sparse or missing price history; training a model on
those rows can degrade accuracy.  The filtering rules implemented here ensure that
only cities with at least ``min_months`` non‑null observations and belonging to
metropolitan areas with at least ``min_cities_per_metro`` constituent cities are
preserved.  The resulting dataset is written to ``data/processed/zhvi_tx_metros.csv``.

Usage::

    python src/filter_zillow_tx_metros.py

The raw Zillow file should be placed in ``data/raw/`` and named
``city_level_zillow_data.csv``.  You can adjust the input and output paths by
modifying the module‑level constants ``RAW_PATH`` and ``OUT_PATH``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the raw Zillow City ZHVI CSV.  This file must be downloaded
# separately from the Zillow Research site and placed into ``data/raw``.
RAW_PATH: str = os.path.join("data", "raw", "city_level_zillow_data.csv")

# Path to the filtered output.  The parent directory will be created if it
# does not already exist.
OUT_PATH: str = os.path.join("data", "processed", "zhvi_tx_metros.csv")

# Minimum number of non‑null monthly ZHVI values required for a city to be
# retained.  60 months ≈ 5 years.  Feel free to adjust this threshold.
MIN_MONTHS: int = 60

# Minimum number of cities required for a metro (CBSA) to be retained.  This
# prevents tiny or misreported metros from skewing the data.
MIN_CITIES_PER_METRO: int = 3


def filter_zillow_data(
    raw_path: str = RAW_PATH,
    out_path: str = OUT_PATH,
    min_months: int = MIN_MONTHS,
    min_cities_per_metro: int = MIN_CITIES_PER_METRO,
) -> None:
    """Filter the Zillow City ZHVI dataset for Texas metros.

    Parameters
    ----------
    raw_path : str
        Path to the raw Zillow CSV (City level).  The file must contain at least
        columns for ``State`` or ``StateName``, ``RegionName``, and ``Metro``,
        plus a series of monthly date columns in the form ``YYYY-MM``.
    out_path : str
        Destination for the filtered CSV.  Any parent directories will be
        created if they do not exist.
    min_months : int
        Minimum number of non‑null monthly values a city must have to be
        retained.  Default is 60 months.
    min_cities_per_metro : int
        Minimum number of cities a metro must have to be retained.  Default is 3.

    Notes
    -----
    The script identifies date columns by matching column names to the regular
    expression ``^\d{4}-\d{2}$`` (e.g., ``2010-01``).  If no such columns are
    found, a second pass with a relaxed pattern (``^\d{4}-\d{2}``) is used.
    """

    # Ensure the raw file exists before attempting to read it
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw Zillow file not found at '{raw_path}'. Place the file in the data/raw directory."
        )

    # Read the raw Zillow City dataset
    df = pd.read_csv(raw_file)

    # ---------------------------------------------------------------------
    # Filter to Texas only
    # ---------------------------------------------------------------------
    if "State" in df.columns:
        df = df[df["State"] == "TX"].copy()
    elif "StateName" in df.columns:
        df = df[df["StateName"].astype(str).str.upper() == "TEXAS"].copy()
    else:
        raise KeyError(
            "Expected either a 'State' or 'StateName' column in the Zillow dataset."
        )

    # ---------------------------------------------------------------------
    # Filter to cities that are part of a metro
    # ---------------------------------------------------------------------
    if "Metro" in df.columns:
        non_null_metro = df["Metro"].notna() & df["Metro"].astype(str).str.strip().ne("")
        df = df.loc[non_null_metro].copy()
    else:
        # If there is no Metro column, we cannot determine metropolitan membership
        print(
            "Warning: 'Metro' column not found. All cities will be treated as non‑metro,"
            "which may not be desired."
        )

    # ---------------------------------------------------------------------
    # Identify monthly date columns
    # ---------------------------------------------------------------------
    # Look for columns exactly of the form YYYY-MM
    date_cols = [col for col in df.columns if re.fullmatch(r"\d{4}-\d{2}", str(col))]
    # If none matched, try a relaxed pattern (prefix only)
    if not date_cols:
        date_cols = [col for col in df.columns if re.match(r"\d{4}-\d{2}", str(col))]
    if not date_cols:
        raise RuntimeError(
            "No date columns matching YYYY-MM were found in the Zillow dataset."
        )

    # ---------------------------------------------------------------------
    # Remove cities with too few non‑null months
    # ---------------------------------------------------------------------
    non_null_counts = df[date_cols].notna().sum(axis=1)
    df = df.loc[non_null_counts >= min_months].copy()

    # ---------------------------------------------------------------------
    # Require metros to have a minimum number of cities
    # ---------------------------------------------------------------------
    if "Metro" in df.columns:
        # Count unique cities by metro
        city_counts = df.groupby("Metro")["RegionName"].nunique()
        big_metros = city_counts[city_counts >= min_cities_per_metro].index
        df = df[df["Metro"].isin(big_metros)].copy()

    # ---------------------------------------------------------------------
    # Write the filtered dataset
    # ---------------------------------------------------------------------
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Filtered dataset saved to {out_file}")


if __name__ == "__main__":
    try:
        filter_zillow_data()
    except Exception as exc:
        # Print exceptions to stderr and exit with non‑zero code
        print(f"Error filtering Zillow data: {exc}", file=sys.stderr)
        sys.exit(1)