"""
build_city_year_education.py
============================

This script constructs a city‑year education index for Texas by aggregating
STAAR performance data across campuses and districts.  The output table is
intended to join with the Zillow house price panel for the same cities and
years.  The education index is calculated as a weighted average of
``pct_meets`` and ``pct_masters`` percentages, with weights derived from the
student enrollment at each campus or district.  Additional metrics (such as
separate mean ``pct_meets`` and ``pct_masters``) are also emitted.

Usage::

    python src/build_city_year_education.py

Before running this script, ensure that you have downloaded STAAR datasets
(district and/or campus) into ``data/raw/tea_staar/`` and that they have
consistent column names.  The helper ``standardize_staar_columns`` defined in
``fetch_tea_staar.py`` can be used to rename fields if necessary.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .fetch_tea_staar import standardize_staar_columns


# Directory where raw STAAR files reside
RAW_DIR: str = os.path.join("data", "raw", "tea_staar")

# Output path for the aggregated city‑year education index
OUT_PATH: str = os.path.join("data", "processed", "education_tx_city_year.csv")


def normalize_city(name: str) -> str:
    """Normalize city names to facilitate joins.

    This helper converts a city name to upper case, removes punctuation,
    compresses consecutive whitespace, and strips leading/trailing whitespace.
    It preserves alphanumeric characters and spaces.

    Parameters
    ----------
    name : str
        Raw city name from the STAAR dataset.

    Returns
    -------
    str
        Normalised city name suitable for joining.
    """
    if not isinstance(name, str) or not name:
        return ""
    s = name.upper()
    # Remove punctuation except spaces and alphanumeric
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    # Collapse multiple spaces and strip
    s = " ".join(s.split())
    return s


def extract_year_from_path(path: Path) -> int | None:
    """Infer the year of a STAAR file from its filename.

    The function searches the filename for a four‑digit year between 2000 and 2099.
    If no year is found, it returns None.

    Parameters
    ----------
    path : pathlib.Path
        Path object representing the file.

    Returns
    -------
    int or None
        The inferred year, or None if no year could be parsed.
    """
    m = re.search(r"(20\d{2})", path.stem)
    if m:
        return int(m.group(1))
    return None


def build_city_year_index() -> pd.DataFrame:
    """Aggregate STAAR data to a city–year education index.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``city_norm``, ``year``, ``edu_index``,
        ``staar_meets``, ``staar_masters``, and ``enrollment``.  ``city_norm``
        uses the normalisation defined in :func:`normalize_city`.
    """
    raw_dir = Path(RAW_DIR)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"The STAAR raw directory '{RAW_DIR}' does not exist. Please place STAAR files here."
        )
    files = list(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No STAAR CSV files found in '{RAW_DIR}'. Download and place them in this directory."
        )

    # Accumulate weighted sums per city–year
    city_year_sums: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    city_year_counts: Dict[Tuple[str, int], float] = defaultdict(float)

    for path in files:
        year = extract_year_from_path(path)
        # Load CSV; assume comma delimiter
        df = pd.read_csv(path)
        df = standardize_staar_columns(df)
        # If no year column, infer from filename
        if "year" not in df.columns and year is not None:
            df["year"] = year
        elif "year" not in df.columns:
            raise KeyError(
                f"Could not determine year for file {path}. Add a 'year' column or ensure filename contains the year."
            )

        # Normalise city names
        df["city_norm"] = df["city"].apply(normalize_city)
        # Drop rows without a normalised city or missing metrics
        df = df.dropna(subset=["city_norm", "enrollment", "pct_meets", "pct_masters", "year"])

        # Convert enrollment to numeric
        df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
        df = df[df["enrollment"] > 0]

        # Convert metrics to numeric (assume percentages)
        df["pct_meets"] = pd.to_numeric(df["pct_meets"], errors="coerce")
        df["pct_masters"] = pd.to_numeric(df["pct_masters"], errors="coerce")
        df = df.dropna(subset=["pct_meets", "pct_masters"])

        # Compute a simple performance score as the average of meets and masters percentages
        df["perf_score"] = 0.5 * df["pct_meets"] + 0.5 * df["pct_masters"]

        # Aggregate to city–year using enrollment as weights
        for (_, row) in df.iterrows():
            key = (row["city_norm"], int(row["year"]))
            w = float(row["enrollment"])
            city_year_sums[key]["perf_score"] += row["perf_score"] * w
            city_year_sums[key]["meets"] += row["pct_meets"] * w
            city_year_sums[key]["masters"] += row["pct_masters"] * w
            city_year_counts[key] += w

    # Construct DataFrame from aggregated sums
    records: List[Dict[str, float]] = []
    for (city_norm, year), count in city_year_counts.items():
        sums = city_year_sums[(city_norm, year)]
        total_weight = count
        edu_index = sums["perf_score"] / total_weight
        staar_meets = sums["meets"] / total_weight
        staar_masters = sums["masters"] / total_weight
        records.append(
            {
                "city_norm": city_norm,
                "year": year,
                "edu_index": edu_index,
                "staar_meets": staar_meets,
                "staar_masters": staar_masters,
                "enrollment": total_weight,
            }
        )

    result_df = pd.DataFrame(records)
    return result_df


def main() -> None:
    try:
        edu_df = build_city_year_index()
    except Exception as exc:
        print(f"Failed to build education index: {exc}", file=sys.stderr)
        sys.exit(1)

    # Write output
    out_file = Path(OUT_PATH)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    edu_df.to_csv(out_file, index=False)
    print(f"Education index written to {out_file} ({len(edu_df)} rows)")


if __name__ == "__main__":
    main()